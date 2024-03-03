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

    cnn_curve, ncm_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        accy = model.eval_task(save_result=True)
        cnn_accy = accy.get("cnn_accy", None)
        ncm_accy = accy.get("ncm_accy", None)
        model.after_task()

        if ncm_accy is not None:
            logging.info("NCM: {}".format(ncm_accy["grouped"]))
            ncm_curve["top1"].append(ncm_accy["top1"])
            ncm_curve["top5"].append(ncm_accy["top5"])
            logging.info("NCM top1 curve: {}".format(ncm_curve["top1"]))
            logging.info("NCM top5 curve: {}".format(ncm_curve["top5"]))
            print(
                "Average Accuracy (NCM):",
                sum(ncm_curve["top1"]) / len(ncm_curve["top1"]),
            )
            logging.info(
                "Average Accuracy (NCM): {}\n".format(
                    sum(ncm_curve["top1"]) / len(ncm_curve["top1"])
                )
            )
        if cnn_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])
            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            print(
                "Average Accuracy (CNN):",
                sum(cnn_curve["top1"]) / len(cnn_curve["top1"]),
            )
            logging.info(
                "Average Accuracy (CNN): {}\n".format(
                    sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
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
