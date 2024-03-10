import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    logfilename = "logs/{}/{}_b{}inc{}/{}_{}_s{}.log".format(
        args["model_name"],
        args["dataset"],
        args["init_cls"],
        args["increment"],
        args["prefix"],
        args["convnet_type"],
        args["seed"],
    )
    if not os.path.exists(os.path.dirname(logfilename)):
        os.makedirs(os.path.dirname(logfilename))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    ncm_curve = {"top1": [], "top5": []}
    ncm_cosine_curve = {"top1": [], "top5": []}
    linear_curve = {"top1": [], "top5": []}
    ensemble_curve = {"top1": [], "top5": []}
    ensemble_cosine_curve = {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        accy = model.eval_task(save_result=True)
        model.after_task()

        # ncm_euclidean
        ncm_accy = accy.get("ncm_accy", None)
        if ncm_accy is not None:
            logging.info("ncm: {}".format(ncm_accy["grouped"]))
            ncm_curve["top1"].append(ncm_accy["top1"])
            ncm_curve["top5"].append(ncm_accy["top5"])
            logging.info("ncm top1 curve: {}".format(ncm_curve["top1"]))
            logging.info("ncm top5 curve: {}".format(ncm_curve["top5"]))
            print(
                "Average Accuracy (ncm):",
                sum(ncm_curve["top1"]) / len(ncm_curve["top1"]),
            )
            logging.info(
                "Average Accuracy (ncm): {}\n".format(
                    sum(ncm_curve["top1"]) / len(ncm_curve["top1"])
                )
            )

        # ncm_cosine
        ncm_cosine_accy = accy.get("ncm_cosine_accy", None)
        if ncm_cosine_accy is not None:
            logging.info("ncm_cosine: {}".format(ncm_cosine_accy["grouped"]))
            ncm_cosine_curve["top1"].append(ncm_cosine_accy["top1"])
            ncm_cosine_curve["top5"].append(ncm_cosine_accy["top5"])
            logging.info("ncm_cosine top1 curve: {}".format(ncm_cosine_curve["top1"]))
            logging.info("ncm_cosine top5 curve: {}".format(ncm_cosine_curve["top5"]))
            print(
                "Average Accuracy (ncm_cosine):",
                sum(ncm_cosine_curve["top1"]) / len(ncm_cosine_curve["top1"]),
            )
            logging.info(
                "Average Accuracy (ncm_cosine): {}\n".format(
                    sum(ncm_cosine_curve["top1"]) / len(ncm_cosine_curve["top1"])
                )
            )

        # linear
        linear_accy = accy.get("linear_accy", None)
        if linear_accy is not None:
            logging.info("linear: {}".format(linear_accy["grouped"]))
            linear_curve["top1"].append(linear_accy["top1"])
            linear_curve["top5"].append(linear_accy["top5"])
            logging.info("linear top1 curve: {}".format(linear_curve["top1"]))
            logging.info("linear top5 curve: {}".format(linear_curve["top5"]))
            print(
                "Average Accuracy (linear):",
                sum(linear_curve["top1"]) / len(linear_curve["top1"]),
            )
            logging.info(
                "Average Accuracy (linear): {}\n".format(
                    sum(linear_curve["top1"]) / len(linear_curve["top1"])
                )
            )

        # ensemble (linear+euclidean)
        ensemble_accy = accy.get("ensemble_accy", None)
        if ensemble_accy is not None:
            logging.info("ensemble: {}".format(ensemble_accy["grouped"]))
            ensemble_curve["top1"].append(ensemble_accy["top1"])
            ensemble_curve["top5"].append(ensemble_accy["top5"])
            logging.info("ensemble top1 curve: {}".format(ensemble_curve["top1"]))
            logging.info("ensemble top5 curve: {}".format(ensemble_curve["top5"]))
            print(
                "Average Accuracy (ensemble):",
                sum(ensemble_curve["top1"]) / len(ensemble_curve["top1"]),
            )
            logging.info(
                "Average Accuracy (ensemble): {}\n".format(
                    sum(ensemble_curve["top1"]) / len(ensemble_curve["top1"])
                )
            )

        # ensemble_cosine (linear+cosine)
        ensemble_cosine_accy = accy.get("ensemble_cosine_accy", None)
        if ensemble_cosine_accy is not None:
            logging.info("ensemble_cosine: {}".format(ensemble_cosine_accy["grouped"]))
            ensemble_cosine_curve["top1"].append(ensemble_cosine_accy["top1"])
            ensemble_cosine_curve["top5"].append(ensemble_cosine_accy["top5"])
            logging.info(
                "ensemble_cosine top1 curve: {}".format(ensemble_cosine_curve["top1"])
            )
            logging.info(
                "ensemble_cosine top5 curve: {}".format(ensemble_cosine_curve["top5"])
            )
            print(
                "Average Accuracy (ensemble_cosine):",
                sum(ensemble_cosine_curve["top1"]) / len(ensemble_cosine_curve["top1"]),
            )
            logging.info(
                "Average Accuracy (ensemble_cosine): {}\n".format(
                    sum(ensemble_cosine_curve["top1"])
                    / len(ensemble_cosine_curve["top1"])
                )
            )


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
