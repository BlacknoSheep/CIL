import logging
import numpy as np
import torch
import copy
import os
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from utils.inc_net import HeadNet, FcHead
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from torchvision import transforms


class Momentum(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = HeadNet(
            args,
            False,
            FcHead(512, self.args["init_cls"], pre_norm=self.args["reprojector"]),
        )
        self._means = None
        self._stds = None
        self._shot = None
        self._saved_prefix = "{}_{}_{}_{}_{}".format(
            self.args["prefix"],
            self.args["model_name"],
            self.args["convnet_type"],
            self.args["dataset"],
            self.args["init_cls"],
        )

    @property
    def _new_classes(self):
        return self._total_classes - self._known_classes

    def after_task(self):
        self._known_classes = self._total_classes

        # # save last predicts
        # y_pred, y_true = self._eval_cnn(self._network, self.test_loader)
        # y_pred = y_pred[:, 0]  # [N]
        # predicts = np.stack((y_pred, y_true), axis=1)  # [N, 2]
        # np.save(os.path.join("./saved", self._saved_prefix + "_predicts.npy"), predicts)
        # # save last ncm_cosine predicts
        # y_pred, y_true = self._eval_ncm(self._network.convnet, self.test_loader)
        # y_pred = y_pred[:, 0]  # [N]
        # predicts = np.stack((y_pred, y_true), axis=1)  # [N, 2]
        # np.save(os.path.join("./saved", self._saved_prefix + "_predicts_ncm.npy"), predicts)

        # save the final model, means and stds
        self._save_all()

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
                    strict=True,
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
                logging.info(
                    "Save initial trained model to {}".format(
                        self._saved_prefix + "_0.pkl"
                    )
                )
                self.save_checkpoint(self._saved_prefix)
                self._network.to(self._device, non_blocking=True)

            # init_acc = self._compute_accuracy(self._network, test_loader)
            # logging.info("Initial accy: {:.2f}".format(init_acc))

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
            self._train_head(
                train_loader,
                test_loader,
                optimizer,
                scheduler,
                epochs=self.args["epochs"],
            )

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

                preds = torch.max(logits, dim=1)[1]
                correct += torch.sum(torch.eq(preds, targets)).item()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(correct * 100 / total, decimals=2)

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

    def _train_head(self, train_loader, test_loader, optimizer, scheduler, epochs):
        # save original train_loader and test_loader, which contains input images
        original_train_loader = train_loader
        original_test_loader = test_loader

        # build feature train_loader and test_loader, which contains features
        test_dataset = self._build_feature_dataset(original_test_loader, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
            pin_memory=self.args["pin_memory"],
        )

        # momentum update after each task
        # old_head = copy.deepcopy(self._network.head)

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            # rebuild train_loader before each epoch to achieve more diversity
            train_dataset = self._build_feature_dataset(
                original_train_loader, mode="train"
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args["batch_size"],
                shuffle=True,
                num_workers=self.args["num_workers"],
                pin_memory=self.args["pin_memory"],
            )

            # momentum update after each epoch
            old_head = copy.deepcopy(self._network.head)

            self._network.eval()
            self._network.head.train()
            losses = 0.0
            correct, total = 0, 0
            correct_new, total_new = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                # momentum update after each step
                # old_head = copy.deepcopy(self._network.head)

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network.head(inputs)["logits"]
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

                # momentum update after each step
                # self._momentum_update_head(old_head, momentum=self.args["momentum"])

            # momentum update after each epoch
            self._momentum_update_head(old_head, momentum=self.args["momentum"])

            scheduler.step()
            train_acc = np.around(correct * 100 / total, decimals=2)
            train_acc_new = np.around(correct_new * 100 / total_new, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network.head, test_loader)
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

        # momentum update after each task
        # self._momentum_update_head(old_head, momentum=self.args["momentum"])

    def _extract_vectors(self, loader):
        self._network.convnet.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            with torch.no_grad():
                _vectors = self._network.convnet(_inputs.to(self._device))["features"]
                _vectors = _vectors.cpu().numpy()

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _compute_means(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                idx_dataset = self.data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1), source="train", mode="test"
                )
                idx_loader = DataLoader(
                    idx_dataset,
                    batch_size=self.args["batch_size"],
                    shuffle=False,
                    num_workers=self.args["num_workers"],
                    pin_memory=self.args["pin_memory"],
                )
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(
                    vectors, axis=0, keepdims=True
                )  # [1 * feature_dim]
                class_std = np.std(vectors, axis=0, keepdims=True)  # [1 * feature_dim]
                if self._means is None:
                    self._means = class_mean
                    self._stds = class_std
                else:
                    self._means = np.concatenate((self._means, class_mean), axis=0)
                    self._stds = np.concatenate((self._stds, class_std), axis=0)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits = model(inputs)["logits"]

            predicts = torch.max(logits, dim=1)[1]
            correct += torch.sum(torch.eq(predicts.cpu(), targets)).item()
            total += len(targets)

        return np.around(correct * 100 / total, decimals=2)

    def _compute_ncm_logits(self, features, means):
        """
        :param features: Tensor of [n, feature_dim]
        :param means: Tensor of [n_classes, feature_dim]
        """
        # normalize features
        features = F.normalize(features, dim=1)
        means = F.normalize(means, dim=1)

        logits = torch.mm(features, means.t())  # [n, n_classes]
        return logits

    def eval_task(self):
        # cnn accy
        y_pred, y_true = self._eval_cnn(self._network, self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        # ncm accy
        y_pred, y_true = self._eval_ncm(self._network.convnet, self.test_loader)
        ncm_accy = self._evaluate(y_pred, y_true)
        return {
            "cnn_accy": cnn_accy,
            "ncm_accy": ncm_accy,
        }

    def _eval_cnn(self, model, loader):
        model.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits = model(inputs)["logits"]

            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_ncm(self, model, loader):
        model.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                features = model(inputs)["features"]
            logits = self._compute_ncm_logits(
                features, torch.from_numpy(np.array(self._means)).to(self._device)
            )

            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _momentum_update_head(self, old_head, momentum):
        # except wight and bias of new classes in fc
        old_head.fc.weight.data[self._known_classes :] = (
            self._network.head.fc.weight.data[self._known_classes :]
        )
        old_head.fc.bias.data[self._known_classes :] = self._network.head.fc.bias.data[
            self._known_classes :
        ]
        for param_new, param_old in zip(
            self._network.head.parameters(), old_head.parameters()
        ):
            param_new.data = param_old.data * momentum + param_new.data * (
                1.0 - momentum
            )

    def _build_feature_dataset(self, loader, mode):
        """
        :param loader: DataLoader of new classes
        :param mode: "train" or "test"
        """
        features, targets = self._extract_vectors(loader)
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

            features = np.concatenate(
                (features, old_features), axis=0, dtype=np.float32
            )
            targets = np.concatenate((targets, old_targets), axis=0)
        elif mode == "test":
            pass
        return DummyDataset(torch.from_numpy(features), torch.from_numpy(targets))

    def _generate_by_oversampling(self, n_per_class):
        features = self._means[: self._known_classes]  # e.g: [0,1,2]
        features = np.repeat(features, n_per_class, axis=0)  # e.g: [0,0,1,1,2,2]
        labels = np.arange(self._known_classes)  # e.g: [0,1,2]
        labels = np.repeat(labels, n_per_class, axis=0)  # e.g: [0,0,1,1,2,2]
        return features, labels

    def _generate_by_noise(self, n_per_class):
        features = self._means[: self._known_classes]  # e.g: [0,1,2]
        stds = self._stds[: self._known_classes]  # e.g: [0,1,2]
        # sample from normal (Gaussian) distribution
        features = np.random.normal(
            features, stds, size=(n_per_class, *features.shape)
        )  # e.g: [[0,1,2],[0,1,2]]
        features = np.concatenate(features, axis=0)  # e.g: [0,1,2,0,1,2]
        labels = np.arange(self._known_classes)  # e.g: [0,1,2]
        labels = np.tile(labels, n_per_class)  # e.g: [0,1,2,0,1,2]
        return features, labels

    def _generate_by_translation(self, refer_features, refer_labels, n_per_class):
        """
        refer_features: ndarray of [n, feature_dim]
        refer_labels: ndarray of [n]
        """
        refer_means = self._means[refer_labels]
        refer_bias = refer_features - refer_means
        random_idxs = np.random.randint(
            0, refer_bias.shape[0], size=self._known_classes * n_per_class
        )

        features, labels = self._generate_by_oversampling(n_per_class)
        features += refer_bias[random_idxs]
        return features, labels

    def _layernorm(self, x, eps=1e-05):
        """
        :param x: Tensor of [n, dim]
        :param eps: same as torch.nn.LayerNorm
        """
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        x = (x - mean) / (std + eps)
        return x

    def _save_all(self):
        self._network.cpu()
        saved_dict = {
            "task_id": self._cur_task,
            "model_state_dict": self._network.state_dict(),
            "means": self._means,
            "stds": self._stds,
        }
        folder_path = "./saved"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        saved_path = os.path.join(folder_path, self._saved_prefix + "_all.pkl")
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
