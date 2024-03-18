"""
simple NCM classifier
The distance between the feature and the class mean will be calculated using the euclidean distance or cosine distance.

reprojector: ["layernorm", "batchnorm", "l2norm", None], layernorm is the best.
affine: bool. If True, enable the affine in reprojector. True is better.
head: ["fc", "mlp"], mlp is much better for cifar100.For TinyImageNet, mlp will lower the intial accuracy, which causes the average accuracy to be lower, but the final accuracy is higher.
"""

import logging
import numpy as np
import torch
import os
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import get_convnet, Reprojector, FcHead, MLPHead
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
        if args["head"] == "fc":
            self.head = FcHead(self.feature_dim, args["init_cls"])
        elif args["head"] == "mlp":
            self.head = MLPHead(self.feature_dim, args["init_cls"])
        else:
            raise ValueError("Invalid head type")

    def forward(self, x):
        """
        !!!: 返回的是重投影前的特征
        """
        x = self.convnet(x)["features"]
        features = x
        if self.reprojector is not None:
            x = self.reprojector(x)["features"]
        x = self.head(x)["logits"]
        return {"features": features, "logits": x}

    def update_head(self, total_classes):
        self.head.update_fc(total_classes)


class NCM(BaseLearner):
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
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
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

    def eval_task(self, save_result=False):
        ncm_accy = None
        ncm_cosine_accy = None
        pred_dict = self._eval_ncm(self.test_loader)

        # euclidean
        y_pred, y_true = pred_dict["y_pred"], pred_dict["y_true"]
        ncm_accy = self._evaluate(y_pred, y_true)
        if save_result:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self._saved_folder, "ncm_pred.npy")
            _target_path = os.path.join(self._saved_folder, "ncm_target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

        # cosine
        y_pred, y_true = (
            pred_dict["y_pred_cosine"],
            pred_dict["y_true_cosine"],
        )
        ncm_cosine_accy = self._evaluate(y_pred, y_true)
        if save_result:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self._saved_folder, "ncm_cosine_pred.npy")
            _target_path = os.path.join(self._saved_folder, "ncm_cosine_target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

        return {
            "ncm_accy": ncm_accy,
            "ncm_cosine_accy": ncm_cosine_accy,
        }

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
