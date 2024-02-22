import copy
import logging
import torch
from torch import nn
from torch.nn import functional as F
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50
from convs.ucir_cifar_resnet import resnet32 as cosine_resnet32
from convs.ucir_resnet import resnet18 as cosine_resnet18
from convs.ucir_resnet import resnet34 as cosine_resnet34
from convs.ucir_resnet import resnet50 as cosine_resnet50
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from convs.modified_represnet import resnet18_rep
from convs.resnet_cbam import resnet18_cbam, resnet34_cbam, resnet50_cbam


def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained, args=args)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained, args=args)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained, args=args)
    elif name == "cosine_resnet18":
        return cosine_resnet18(pretrained=pretrained, args=args)
    elif name == "cosine_resnet32":
        return cosine_resnet32()
    elif name == "cosine_resnet34":
        return cosine_resnet34(pretrained=pretrained, args=args)
    elif name == "cosine_resnet50":
        return cosine_resnet50(pretrained=pretrained, args=args)
    elif name == "resnet18_rep":
        return resnet18_rep(pretrained=pretrained, args=args)
    elif name == "resnet18_cbam":
        return resnet18_cbam(pretrained=pretrained, args=args)
    elif name == "resnet34_cbam":
        return resnet34_cbam(pretrained=pretrained, args=args)
    elif name == "resnet50_cbam":
        return resnet50_cbam(pretrained=pretrained, args=args)

    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        self.convnet.load_state_dict(model_infos["convnet"])
        self.fc.load_state_dict(model_infos["fc"])
        test_acc = model_infos["test_acc"]
        return test_acc


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )


class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat(
                    [
                        weight,
                        torch.zeros(nb_classes - nb_output, self.feature_dim).cuda(),
                    ]
                )
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class Projector(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=None,
        out_dim=None,
        norm_type=nn.LayerNorm,
        act_type=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = norm_type(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.act = act_type()
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.drop(self.act(self.norm1(self.fc1(x))))
        x = self.drop(self.fc2(x))
        return x


class FcHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=10, pre_norm=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pre_norm = pre_norm
        self.norm0 = nn.LayerNorm(self.in_dim, elementwise_affine=False)
        self.fc = self._generate_fc(self.in_dim, self.out_dim)

    def forward(self, x):
        if self.pre_norm:
            x = self.norm0(x)

        x = self.fc(x)
        return {"logits": x}

    def update_fc(self, out_dim):
        self.out_dim = out_dim
        fc = self._generate_fc(self.in_dim, self.out_dim)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    def _generate_fc(self, in_dim, out_dim):
        fc = nn.Linear(in_dim, out_dim)
        return fc


class MLPHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=10, pre_norm=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pre_norm = pre_norm
        self.norm0 = nn.LayerNorm(self.in_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.LayerNorm(self.in_dim),
            nn.GELU(),
        )
        self.fc = self._generate_fc(self.in_dim, self.out_dim)

    def forward(self, x):
        if self.pre_norm:
            x = self.norm0(x)

        feat = self.mlp(x)
        x = self.fc(feat)
        return {"features": feat, "logits": x}

    def update_fc(self, out_dim):
        self.out_dim = out_dim
        fc = self._generate_fc(self.in_dim, self.out_dim)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    def _generate_fc(self, in_dim, out_dim):
        fc = nn.Linear(in_dim, out_dim)
        return fc


class HeadNet(BaseNet):
    def __init__(self, args, pretrained, head):
        super().__init__(args, pretrained)
        self.head = head

    def update_head(self, nb_classes):
        self.head.update_fc(nb_classes)

    def forward(self, x):
        x = self.convnet(x)
        x = self.head(x["features"])
        return x


class FeatureGenerator(nn.Module):
    def __init__(self, in_dim=512, radius=0.5):
        super().__init__()
        self.radius = radius
        self.norm0 = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.res1 = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, in_dim),
        )
        self.res2 = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, in_dim),
        )
        self.proj = nn.Linear(in_dim, in_dim)

    def forward(self, x_bar, noise, pre_norm=False):
        """
        x_bar: [batch_size, feature_dim]
        noise: [batch_size, feature_dim]
        """
        if pre_norm:
            x_bar = self.norm0(x_bar)

        noise = F.normalize(noise, dim=1) * self.radius  # [batch_size, feature_dim]
        x = x_bar + noise

        x = x + self.res1(x)
        x = x + self.res2(x)
        x = self.proj(x)
        return {"features": x}
