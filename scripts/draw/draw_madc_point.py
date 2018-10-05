#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drawing df point energy spectrum script.

Point should be processed before drawing.
"""
from argparse import ArgumentParser
from struct import unpack

import dfparser  # Numass files parser
import matplotlib.pyplot as plt  # plotting library
import numpy as np
import seaborn as sns  # matplotlib grahs visual enchancer

from utils import _madc_amps


def __parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Input dataforge point.')
    parser.add_argument('-t', '--ampl-threshold', type=int, default=0,
                        help='Amplitudes threshold (default - 0).')
    parser.add_argument('-m', '--ampl-max', type=int, default=4096,
                        help='Amplitudes max threshold (default - 4096).')
    parser.add_argument('-b', '--bins', type=int, default=100,
                        help='Histogram bins number (default - 100).')
    return parser.parse_args()


def _main():
    # Parse arguments from command line
    args = __parse_args()

    # Read dataforge point
    madc_amps = _madc_amps(args.input)

    _, axes = plt.subplots()
    axes.set_title(args.input)
    axes.set_xlabel("Channels, ch")
    axes.set_ylabel("Counts")

    hist, bins = np.histogram(
        madc_amps, bins=args.bins, range=(
            args.ampl_threshold, args.ampl_max))
    bins_centers = (bins[:-1] + bins[1:]) / 2

    axes.step(bins_centers, hist, where='mid')
    plt.show()


if __name__ == "__main__":
    sns.set_context("poster")
    _main()
