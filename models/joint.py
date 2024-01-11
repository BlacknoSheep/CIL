import logging
import numpy as np
import torch
import os
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import HeadNet, FcHead
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from torchvision import transforms


class Joint(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = HeadNet(args, False,
                                FcHead(512, self.args["init_cls"], pre_norm=self.args["pre_norm"]))

        self._saved_prefix = "{}_{}_{}_{}_{}".format(self.args["prefix"], self.args["model_name"],
                                                     self.args["convnet_type"], self.args["dataset"],
                                                     self.args["init_cls"])

    @property
    def _new_classes(self):
        return self._total_classes - self._known_classes

    def after_task(self):
        self._known_classes = self._total_classes

        # # save last predicts
        # y_pred, y_true = self._eval_cnn(self.test_loader)
        # y_pred = y_pred[:, 0]  # [N]
        # predicts = np.stack((y_pred, y_true), axis=1)  # [N, 2]
        # np.save(os.path.join("./saved", self._saved_prefix + "_predicts.npy"), predicts)

        # save the final model
        self._save_all()

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        dataset_name = self.args["dataset"].lower()
        if 'cifar' in dataset_name:
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255),
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.RandomErasing(inplace=True),
            ]
        elif 'imagenet' in dataset_name:
            self.data_manager._train_trsf = [
                # transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), # https://github.com/pytorch/examples/issues/355
                transforms.RandomResizedCrop(224),  # The default scale (0.08, 1.0) is better for incremental learning
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

        # train on all seen classes
        train_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"],
            pin_memory=self.args["pin_memory"]
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"],
            pin_memory=self.args["pin_memory"]
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            if self.args["initial_model_path"] is not None:
                logging.info("Load initial trained model from {}".format(self.args["initial_model_path"]))
                self._network.load_state_dict(torch.load(self.args["initial_model_path"],
                                                         map_location=self._device)["model_state_dict"], strict=True)
            else:
                logging.info("Train from scratch")
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, self._network.parameters()),
                    lr=self.args["init_lr"],
                    momentum=0.9,
                    weight_decay=self.args["init_weight_decay"])
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["init_epochs"])
                self._init_train(train_loader, test_loader, optimizer, scheduler, epochs=self.args["init_epochs"])
                logging.info("Save initial trained model to {}".format(self._saved_prefix + "_0.pkl"))
                self.save_checkpoint(self._saved_prefix)
                self._network.to(self._device)

            # init_acc = self._compute_accuracy(self._network, test_loader)
            # logging.info("Initial accy: {:.2f}".format(init_acc))

            # Freeze the feature extractor
            if self.args["freeze_convnet"]:
                logging.info("Freeze convnet")
                for param in self._network.convnet.parameters():
                    param.requires_grad = False
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lr"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["epochs"])
            self._next_train(train_loader, test_loader, optimizer, scheduler, epochs=self.args["epochs"])

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, epochs):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

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

    def _next_train(self, train_loader, test_loader, optimizer, scheduler, epochs):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            if self.args["freeze_convnet"]:
                self._network.convnet.eval()

            losses = 0.0
            correct, total = 0, 0
            correct_new, total_new = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self.args["freeze_convnet"]:
                    with torch.no_grad():
                        features = self._network.convnet(inputs)["features"]
                else:
                    features = self._network.convnet(inputs)["features"]

                logits = self._network.head(features)["logits"]
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                corrects = torch.eq(preds, targets)
                correct += torch.sum(corrects).item()
                total += len(targets)
                # accuracy of new classes
                new_mask = targets >= self._known_classes
                correct_new += torch.sum(corrects[new_mask]).item()
                total_new += torch.sum(new_mask).item()

            scheduler.step()
            train_acc = np.around(correct * 100 / total, decimals=2)
            train_acc_new = np.around(correct_new * 100 / total_new, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Train_accy_new {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    train_acc_new,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Train_accy_new {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    train_acc_new,
                )
            prog_bar.set_description(info)
            logging.info(info)

    def _save_all(self):
        self._network.cpu()
        saved_dict = {
            "task_id": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        folder_path = "./saved"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        saved_path = os.path.join(folder_path, self._saved_prefix + "_all.pkl")
        torch.save(saved_dict, saved_path)

