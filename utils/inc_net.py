import copy
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


class L2norm(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-12,
        elementwise_affine=False,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        super(L2norm, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.num_features, **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.num_features, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1, eps=self.eps)
        if self.elementwise_affine:
            x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class Reprojector(nn.Module):
    def __init__(self, name: str, in_dim: int, affine: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.name = name.lower()

        if self.name == "layernorm":
            self.reprojector = nn.LayerNorm(self.in_dim, elementwise_affine=affine)
        elif self.name == "batchnorm":
            self.reprojector = nn.BatchNorm1d(self.in_dim, affine=affine)
        elif self.name == "l2norm":
            self.reprojector = L2norm(self.in_dim, affine=affine)

    def forward(self, x):
        """
        x: [batch_size, feature_dim]
        """
        x = self.reprojector(x)
        return {"features": x}


class FcHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=10):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        x = self.fc(x)
        return {"logits": x}

    def update_fc(self, out_dim):
        self.out_dim = out_dim
        fc = nn.Linear(self.in_dim, self.out_dim)
        if self.fc is not None:
            old_out_dim = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:old_out_dim] = weight
            fc.bias.data[:old_out_dim] = bias
        del self.fc
        self.fc = fc


class MLPHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=10, norm_type="batchnorm", act_type="relu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm_type = norm_type.lower()
        self.act_type = act_type.lower()

        self.fc1 = nn.Linear(self.in_dim, self.in_dim)

        if self.norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(self.in_dim)
        elif self.norm_type == "layernorm":
            self.norm = nn.LayerNorm(self.in_dim)
        else:
            raise NotImplementedError("Unknown normlize type {}".format(norm_type))

        if self.act_type == "relu":
            self.act = nn.ReLU()
        elif self.act_type == "gelu":
            self.act = nn.GELU()
        else:
            raise NotImplementedError("Unknown activate type {}".format(act_type))

        self.fc = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        x = self.act(self.norm(self.fc1(x)))
        x = self.fc(x)
        return {"logits": x}

    def update_fc(self, out_dim):
        self.out_dim = out_dim
        fc = nn.Linear(self.in_dim, self.out_dim)
        if self.fc is not None:
            old_out_dim = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:old_out_dim] = weight
            fc.bias.data[:old_out_dim] = bias
        del self.fc
        self.fc = fc


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
