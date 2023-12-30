import logging
import numpy as np
import torch
import copy
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import HeadNet, FcHead, FeatureGenerator
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from torchvision import transforms
from utils.autoaugment import CIFAR10Policy
from utils.ops import Cutout


class Denoise(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = HeadNet(args, False, FcHead(512))
        self._generator = FeatureGenerator(512, radius=.5)
        self._old_generator = None

        self._means = None  # [n * feature_dim], mean of the features of each class

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        if self.args['dataset'] == "cifar100":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255),
                CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
            ]
        elif self.args['dataset'] == "tinyimagenet200":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        elif self.args['dataset'] == "imagenet100":
            self.data_manager._train_trsf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        logging.info(
            "Learning on {}-{}".format(0, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"]
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"]
        )
        # TODO: built feature set

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._generator.to(self._device)
        self._network.to(self._device)
        if self._cur_task == 0:
            logging.info("Load pretrained convnet from {}".format(self.args["convnet_path"]))
            self._network.load_state_dict(torch.load(self.args["convnet_path"], map_location=self._device)["model_state_dict"],
                                          strict=False)
            # freeze the convnet
            for param in self._network.convnet.parameters():
                param.requires_grad = False

        self._compute_means()

        # Train classify head
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._network.head.parameters()),
            lr=self.args["lr"],
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["epochs"])
        self._train_head(train_loader, test_loader, optimizer, scheduler, self.args["epochs"])

        # Train feature generator
        if self._cur_task > 0:
            self._old_generator = copy.deepcopy(self._generator)
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._generator.parameters()),
            lr=self.args["lr"],
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["epochs"])
        self._train_generator(train_loader, test_loader, optimizer, scheduler, self.args["epochs"])

    def _train_head(self, train_loader, test_loader, optimizer, scheduler, epochs):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            self._network.head.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # already normalized in built_feature_set
                logits = self._network.head(inputs, pre_norm=False)
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)

    def _train_generator(self, train_loader, test_loader, optimizer, scheduler, epochs):
        pass

    def _build_feature_set(self, dataset, mode):
        pass

    def _layernorm(self, x, eps=1e-05):
        """
        :param x: Tensor of [n, dim]
        """
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        x = (x - mean) / (std + eps)
        return x

    def _extract_vectors(self, loader):
        self._network.convnet.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            with torch.no_grad():
                _vectors = self._network.convnet(_inputs.to(self._device))["features"]
                _vectors = tensor2numpy(_vectors)

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _compute_means(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                            source='train',
                                                            mode='test')
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["eval_batch_size"], shuffle=False,
                                        num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0, keepdims=True)  # [1 * feature_dim]
                if self._means is None:
                    self._means = class_mean
                else:
                    self._means = np.concatenate((self._means, class_mean), axis=0)

    def _compute_ncm_logits(self, features, means):
        """
        :param features: Tensor of [n, feature_dim]
        :param means: Tensor of [n_classes, feature_dim]
        """
        features = self._layernorm(features)
        means = self._layernorm(means)

        # calc the cosine similarity between features and means
        features = F.normalize(features, dim=1)
        means = F.normalize(means, dim=1)
        logits = torch.mm(features, means.t())  # cosine similarity, [n, n_classes]
        return logits

    def _compute_mse_loss(self, features, target_features):
        """
        :param features: Tensor of [n, feature_dim]
        :param target_features: Tensor of [n, feature_dim]
        """
        features = self._layernorm(features)
        target_features = self._layernorm(target_features)

        loss = F.mse_loss(features, target_features)
        return loss

    def eval_task(self):
        y_pred, y_true = self._eval_ncm(self.test_loader)
        ncm_accy = self._evaluate(y_pred, y_true)
        return {
            "cnn_accy": None,
            "nme_accy": None,
            "ncm_accy": ncm_accy,
        }

    def _eval_ncm(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                features = self._network(inputs)["features"]
            logits = self._compute_ncm_logits(features, torch.from_numpy(self._means).to(self._device))

            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                features = self._network(inputs)["features"]
            logits = self._compute_ncm_logits(features, torch.from_numpy(self._means).to(self._device))

            predicts = torch.max(logits, dim=1)[1]
            correct += torch.sum(torch.eq(predicts.cpu(), targets)).item()
            total += len(targets)

        return np.around(correct * 100 / total, decimals=2)
