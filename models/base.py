import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.toolkit import accuracy
import os

EPSILON = 1e-8
batch_size = 64


class MainNet(nn.Module):
    def __init__(self, args, pretrained=False):
        super().__init__()
        self.convnet = None
        self.feature_dim = 0
        self.head = None

    def forward(self, x):
        return {"features": None, "logits": None}


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0

        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

        self._network = MainNet(args)
        self._means = (
            None  # [n_classes, feature_dim] of ndarray，类中心，若存在，则计算NCM
        )
        self.topk = 5

        self._saved_folder = "saved/{}_{}_{}/{}_{}_{}".format(
            self.args["prefix"],
            self.args["model_name"],
            self.args["convnet_type"],
            self.args["dataset"],
            self.args["init_cls"],
            self.args["increment"],
        )
        if not os.path.exists(self._saved_folder):
            os.makedirs(self._saved_folder)

    @property
    def _new_classes(self):
        return self._total_classes - self._known_classes

    def after_task(self):
        self._known_classes = self._total_classes

        # save the final model
        self._save_model("all.pkl")

    def incremental_train(self, data_manager):
        self.data_manager = data_manager

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        self.train_loader = None
        self.test_loader = None
        pass

    def _train(self):
        pass

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _inputs = _inputs.to(self._device)
            with torch.no_grad():
                _vectors = self._network(_inputs)["features"]

            vectors.append(_vectors.cpu().numpy())
            targets.append(_targets.numpy())

        return np.concatenate(vectors), np.concatenate(targets)

    def _compute_means(self):
        """
        计算每个类的特征向量的均值和标准差（训练集）
        """
        for class_idx in range(self._known_classes, self._total_classes):
            idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1), source="train", mode="test"
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=self.args["batch_size"],
                shuffle=False,
                num_workers=self.args["num_workers"],
            )
            vectors, _ = self._extract_vectors(idx_loader)
            class_mean = np.mean(vectors, axis=0, keepdims=True)  # [1 * feature_dim]
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
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += torch.sum(torch.eq(predicts.cpu(), targets)).item()
            total += len(targets)

        return np.around(correct * 100 / total, decimals=2)

    def eval_task(self, save_result=False):
        cnn_accy = None
        ncm_accy = None

        # evaluate CNN
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if save_result:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self._saved_folder, "cnn_pred.npy")
            _target_path = os.path.join(self._saved_folder, "cnn_target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

        # evaluate NCM
        if self._means is not None:
            y_pred, y_true = self._eval_ncm(
                self.test_loader, ncm_type=self.args["ncm_type"]
            )
            ncm_accy = self._evaluate(y_pred, y_true)

            if save_result:
                _pred = y_pred.T[0]
                _pred_path = os.path.join(self._saved_folder, "ncm_pred.npy")
                _target_path = os.path.join(self._saved_folder, "ncm_target.npy")
                np.save(_pred_path, _pred)
                np.save(_target_path, y_true)

        return {"cnn_accy": cnn_accy, "ncm_accy": ncm_accy}

    def _evaluate(self, y_pred, y_true):
        """
        计算top1和topk准确率
        """
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits = self._network(inputs)["logits"]
            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_ncm(self, loader, ncm_type="euclidean"):
        self._network.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                features = self._network(inputs)["features"]
            logits = self._compute_ncm_logits(
                features,
                torch.from_numpy(np.array(self._means)).to(self._device),
                ncm_type=ncm_type,
            )

            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_ncm_logits(self, features, means, ncm_type="euclidean"):
        """
        :param features: Tensor of [n, feature_dim]
        :param means: Tensor of [n_classes, feature_dim]
        """
        if ncm_type == "cosine":
            # normalize features
            features = F.normalize(features, dim=1)
            means = F.normalize(means, dim=1)

            logits = torch.mm(features, means.t())  # [n, n_classes]
        elif ncm_type == "euclidean":
            logits = -torch.cdist(features, means, p=2)  # [n, n_classes]
        else:
            raise NotImplementedError("Unknown ncm_type: {}".format(ncm_type))
        return logits

    def _save_model(self, filename: str):
        self._network.cpu()
        saved_dict = {
            "task_id": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        saved_path = os.path.join(self._saved_folder, filename)
        logging.info("Save model to {}".format(saved_path))
        torch.save(saved_dict, saved_path)
