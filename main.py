import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    # args.update(param)  # Add parameters from json

    # Overwrite parameters by command line arguments.
    args = {k: v for k, v in args.items() if v is not None}
    param.update(args)
    args = param

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')

    parser.add_argument('--init_cls', type=int)
    parser.add_argument('--increment', type=int)

    return parser


if __name__ == '__main__':
    main()
