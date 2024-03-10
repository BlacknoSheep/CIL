"""
Use reprojector with fc head

reprojector: ["layernorm", "batchnorm", "l2norm", None], layernorm is the best.
affine: bool. If True, enable the affine in reprojector. True is better.
generator: ["oversampling", "noise", "translation"], 生成旧类特征的方法，noise和重投影结合最好
"""

import logging
import numpy as np
import torch
import os
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from utils.inc_net import get_convnet, Reprojector, FcHead
from models.base import BaseLearner
from torchvision import transforms


class MainNet(nn.Module):
    def __init__(self, args, pretrained=False):
        super().__init__()
        self.convnet = get_convnet(args, pretrained)
        self.feature_dim = self.convnet.out_dim
        self.reprojector = None
        if args["reprojector"] is not None:
            self.reprojector = Reprojector(
                args["reprojector"], self.feature_dim, affine=args["affine"]
            )
        self.head = FcHead(self.feature_dim, args["init_cls"])

    def forward(self, x, x_type="image"):
        """
        !!!: 返回的是重投影前的特征
        """
        if x_type != "feature":
            x = self.convnet(x)["features"]
        features = x
        if self.reprojector is not None:
            x = self.reprojector(x)["features"]
        x = self.head(x)["logits"]
        return {"features": features, "logits": x}

    def update_head(self, total_classes):
        self.head.update_fc(total_classes)


class Reprojection(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = MainNet(args)
        self._means: torch.Tensor = None
        self._stds: torch.Tensor = None

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        dataset_name = self.args["dataset"].lower()
        if "cifar" in dataset_name:
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255),
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.RandomErasing(inplace=True),
            ]
        elif "imagenet" in dataset_name:
            self.data_manager._train_trsf = [
                # transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), # https://github.com/pytorch/examples/issues/355
                transforms.RandomResizedCrop(
                    224
                ),  # The default scale (0.08, 1.0) is better for incremental learning
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=63 / 255), # ColorJitter will lower the accuracy for IL
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.RandomErasing(inplace=True),
            ]

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_head(self._total_classes)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            pin_memory=self.args["pin_memory"],
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
            pin_memory=self.args["pin_memory"],
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device, non_blocking=True)
        if self._cur_task == 0:
            if self.args["initial_model_path"] is not None:
                logging.info(
                    "Load initial trained model from {}".format(
                        self.args["initial_model_path"]
                    )
                )
                self._network.load_state_dict(
                    torch.load(
                        self.args["initial_model_path"], map_location=self._device
                    )["model_state_dict"],
                    strict=False,
                )
            else:
                logging.info("Train from scratch")
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, self._network.parameters()),
                    lr=self.args["init_lr"],
                    momentum=0.9,
                    weight_decay=self.args["init_weight_decay"],
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=self.args["init_epochs"]
                )
                self._init_train(
                    train_loader,
                    test_loader,
                    optimizer,
                    scheduler,
                    epochs=self.args["init_epochs"],
                )

                self._save_model("initial_model.pkl")
                self._network.to(self._device, non_blocking=True)

            # Freeze the feature extractor
            logging.info("Freeze convnet")
            for param in self._network.convnet.parameters():
                param.requires_grad = False

            self._compute_means()
        else:
            self._compute_means()

            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lr"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["epochs"]
            )
            self._next_train(
                train_loader,
                test_loader,
                optimizer,
                scheduler,
                epochs=self.args["epochs"],
            )

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, epochs):
        prog_bar = tqdm(range(epochs))
        for i in prog_bar:
            epoch = i + 1
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                preds = torch.max(logits, dim=1)[1]
                correct += torch.sum(torch.eq(preds, targets)).item()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(correct * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
            logging.info(info)

    def _next_train(self, train_loader, test_loader, optimizer, scheduler, epochs):
        # save original train_loader and test_loader, which contains input images
        original_train_loader = train_loader
        original_test_loader = test_loader

        # build feature train_loader and test_loader, which contains features
        test_dataset = self._build_feature_dataset(original_test_loader, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"] * 4,
            shuffle=False,
            num_workers=self.args["num_workers"],
            pin_memory=self.args["pin_memory"],
        )

        prog_bar = tqdm(range(epochs))
        for i in prog_bar:
            epoch = i + 1
            # rebuild train_loader before each epoch to achieve more diversity
            train_dataset = self._build_feature_dataset(
                original_train_loader, mode="train"
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args["batch_size"] * 4,
                shuffle=True,
                num_workers=self.args["num_workers"],
                pin_memory=self.args["pin_memory"],
            )

            self._network.train()
            self._network.convnet.eval()
            losses = 0.0
            correct, total = 0, 0
            correct_new, total_new = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, x_type="feature")["logits"]
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                preds = torch.max(logits, dim=1)[1]
                result = torch.eq(preds, targets)
                correct += torch.sum(result).item()
                total += len(targets)
                # accuracy of new classes
                new_mask = targets >= self._known_classes
                correct_new += torch.sum(result[new_mask]).item()
                total_new += torch.sum(new_mask).item()

            scheduler.step()
            train_acc = np.around(correct * 100 / total, decimals=2)
            train_acc_new = np.around(correct_new * 100 / total_new, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(
                    self._network, test_loader, data_type="feature"
                )
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Train_accy_new {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    train_acc_new,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Train_accy_new {:.2f}".format(
                    self._cur_task,
                    epoch,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    train_acc_new,
                )
            prog_bar.set_description(info)
            logging.info(info)

    def _compute_accuracy(self, model, loader, data_type="image"):
        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, x_type=data_type)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += torch.sum(torch.eq(predicts, targets)).item()
            total += len(targets)

        return np.around(correct * 100 / total, decimals=2)

    def _build_feature_dataset(self, loader, mode: str):
        """
        :param loader: DataLoader of new classes
        :param mode: "train" or "test"
        """
        features, targets = self._extract_features(loader)
        if mode == "train":
            # generate fake old features
            n_per_class = targets.shape[0] // self._new_classes
            # print("Generate {} features for each old class".format(n_per_class))
            # assert 0
            if self.args["generator"] == "oversampling":
                old_features, old_targets = self._generate_by_oversampling(n_per_class)
            elif self.args["generator"] == "noise":
                old_features, old_targets = self._generate_by_noise(n_per_class)
            elif self.args["generator"] == "translation":
                old_features, old_targets = self._generate_by_translation(
                    features, targets, n_per_class
                )
            else:
                raise NotImplementedError(
                    "Unknown generator: {}".format(self.args["generator"])
                )

            features = torch.cat([old_features, features], dim=0)
            targets = torch.cat([old_targets, targets], dim=0)
        elif mode == "test":
            pass
        return DummyDataset(features, targets)

    def _generate_by_oversampling(self, n_per_class):
        features = self._means[0 : self._known_classes]  # e.g: [[0,0],[1,1]]
        features = features.repeat(n_per_class, 1)  # e.g: [[0,0],[1,1],[0,0],[1,1]]
        labels = torch.arange(0, self._known_classes)  # e.g: [0,1]
        labels = labels.repeat(n_per_class)  # e.g: [0,1,0,1]
        return features, labels

    def _generate_by_noise(self, n_per_class):
        features = self._means[0 : self._known_classes]  # e.g: [[0,0],[1,1]]
        stds = self._stds[0 : self._known_classes]  # e.g: [[0,0],[1,1]]
        features = features.repeat(n_per_class, 1)  # e.g: [[0,0],[1,1],[0,0],[1,1]]
        stds = stds.repeat(n_per_class, 1)  # e.g: [[0,0],[1,1],[0,0],[1,1]]
        # sample from normal (Gaussian) distribution
        features = torch.normal(features, stds)  # e.g: [[0,0],[1,1],[0,0],[1,1]]
        labels = torch.arange(0, self._known_classes)  # e.g: [0,1]
        labels = labels.repeat(n_per_class)  # e.g: [0,1,0,1]
        return features, labels

    def _generate_by_translation(
        self, refer_features: torch.Tensor, refer_labels: torch.Tensor, n_per_class: int
    ):
        """
        refer_features: Tensor of [n_refer, feature_dim]
        refer_labels: Tensor of [n_refer]
        """
        refer_means = self._means[refer_labels]  # [n_refer, feature_dim]
        refer_bias = refer_features - refer_means  # [n_refer, feature_dim]
        random_idxs = torch.randint(
            0, refer_bias.shape[0], (self._known_classes * n_per_class,)
        )  # [n]

        features, labels = self._generate_by_oversampling(
            n_per_class
        )  # [n, feature_dim], [n]
        features += refer_bias[random_idxs]  # [n, feature_dim]
        return features, labels

    def eval_task(self, save_result=False):
        res = super().eval_task(save_result)

        # evaluate ensemble (linear+euclidean)
        ensemble_accy = None
        y_pred, y_true = self._eval_ensemble(self.test_loader, ncm_type="euclidean")
        ensemble_accy = self._evaluate(y_pred, y_true)
        res["ensemble_accy"] = ensemble_accy
        if save_result:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self._saved_folder, "ensemble_pred.npy")
            _target_path = os.path.join(self._saved_folder, "ensemble_target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

        # evaluate ensemble (linear+cosine)
        ensemble_cosine_accy = None
        y_pred, y_true = self._eval_ensemble(self.test_loader, ncm_type="cosine")
        ensemble_cosine_accy = self._evaluate(y_pred, y_true)
        res["ensemble_cosine_accy"] = ensemble_cosine_accy
        if save_result:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self._saved_folder, "ensemble_cosine_pred.npy")
            _target_path = os.path.join(
                self._saved_folder, "ensemble_cosine_target.npy"
            )
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

        return res

    def _eval_ensemble(self, loader, ncm_type="euclidean"):
        self._network.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                out = self._network(inputs)
                features, fc_logits = out["features"], out["logits"]
            ncm_logits = self._compute_ncm_logits(
                features,
                self._means.to(self._device),
                ncm_type=ncm_type,
            )
            # softmax
            fc_logits = F.softmax(fc_logits, dim=1)
            ncm_logits = F.softmax(ncm_logits, dim=1)
            logits = fc_logits + ncm_logits

            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_ncm_logits(
        self, features: torch.Tensor, means: torch.Tensor, ncm_type="euclidean"
    ):
        """
        若有重投影，则使用重投影后的特征计算距离
        """
        if self.args["reprojector"] is not None:
            self._network.reprojector.eval()
            features = features.to(self._device)
            means = means.to(self._device)
            with torch.no_grad():
                features = self._network.reprojector(features)["features"]
                means = self._network.reprojector(means)["features"]

        return super()._compute_ncm_logits(features, means, ncm_type)

    def _save_model(self, filename: str):
        self._network.cpu()
        saved_dict = {
            "task_id": self._cur_task,
            "model_state_dict": self._network.state_dict(),
            "means": self._means,
            "stds": self._stds,
        }
        saved_path = os.path.join(self._saved_folder, filename)
        logging.info("Save model to {}".format(saved_path))
        torch.save(saved_dict, saved_path)


class DummyDataset(Dataset):
    def __init__(self, images, labels):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return idx, image, label
