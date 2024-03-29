import json
import argparse
from trainer import train


def main():
    cmd_args = setup_parser().parse_args()
    cmd_args = vars(cmd_args)  # Converting argparse Namespace to a dict.
    cmd_args = {
        k: v for k, v in cmd_args.items() if v is not None
    }  # Remove None values.
    # set "null" to None; "true" to True; "false" to False.
    for k, v in cmd_args.items():
        if isinstance(v, str):
            v = v.strip().lower()
            if v == "null":
                cmd_args[k] = None
            elif v == "true":
                cmd_args[k] = True
            elif v == "false":
                cmd_args[k] = False

    json_args = load_json(cmd_args["config"])
    # Overwrite parameters by command line arguments.
    json_args.update(cmd_args)

    train(json_args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce of multiple continual learning algorthms."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Json file of settings.",
        required=True,
    )

    parser.add_argument("--prefix", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--init_cls", type=int)
    parser.add_argument("--increment", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--initial_model_path", type=str)
    parser.add_argument("--init_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--reprojector", type=str)
    parser.add_argument("--affine", type=str)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--enable_momentum_from_task", type=int)
    parser.add_argument("--generator", type=str)
    parser.add_argument("--head", type=str)
    parser.add_argument("--freeze_convnet", type=str)

    return parser


if __name__ == "__main__":
    main()
