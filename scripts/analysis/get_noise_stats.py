# -*- coding: utf-8 -*-
"""Выделение параметров шумов из необработанных данных.

Алгоритм:
- Выделение из каждого события первых бинов, в которых заведомо нет сигнала.
- Построение гистограммы для выделенных шумов.
"""
from argparse import ArgumentParser
from glob import glob

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def __parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Input points wildcard. Example: '
                        '"/home/chernov/data/lan10/**/set_*/p*.df"')
    parser.add_argument('-r', '--recursive', action="store_true",
                        help='Recursive option for wildcard.')
    parser.add_argument('-b', '--bins', default=20,
                        help='Number of first bins to process (default - 20).')
    return parser.parse_args()


def _main():
    args = __parse_args()
    files = glob(args.input, recursive=args.recursive)

    for file in files:
        pass
    print(files)


if __name__ == "__main__":
    sns.set_context("poster")
    _main()
