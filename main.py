import os
import importlib
from argparse import ArgumentParser

from src.train_sed import train_valid_test


if __name__ == '__main__':
    parser = ArgumentParser(parents=[])

    parser.add_argument('--config', type=str)

    params = parser.parse_args()

    module = importlib.import_module(params.config, package=None)
    config = module.Config()

    oof, pred = train_valid_test(config)
