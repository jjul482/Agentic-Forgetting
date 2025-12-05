import numpy as np
from trainer import train
import argparse
import json

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./exps/ppo_boxing_one_armed.json',)
    return parser

if __name__ == "__main__":
    main()
