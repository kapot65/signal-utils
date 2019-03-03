#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drawing df point energy spectrum script.

Point should be processed before drawing.
"""
from argparse import ArgumentParser

import dfparser  # Numass files parser
import matplotlib.pyplot as plt  # plotting library
import numpy as np
import seaborn as sns  # matplotlib grahs visual enchancer


def __parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Input dataforge point.')
    parser.add_argument('input2', help='Input dataforge point.')
    parser.add_argument('-t', '--ampl-threshold', type=int, default=496,
                        help='Amplitudes threshold (default - 496).')
    parser.add_argument('-m', '--ampl-max', type=int, default=4016,
                        help='Amplitudes max threshold (default - 4016).')
    parser.add_argument('-b', '--bins', type=int, default=55,
                        help='Histogram bins number (default - 55).')
    return parser.parse_args()


def __extract_amps(filename):
    _, meta, data = dfparser.parse_from_file(filename)

    # Parse Binary data
    point = dfparser.Point()
    point.ParseFromString(data)

    # Extract amlitudes from each block
    amps = {}
    for idx, channel in enumerate(point.channels):
        for block in channel.blocks:
            if idx not in amps:
                amps[idx] = []
            amps[idx].append(np.array(block.events.amplitudes, np.int16))

    for idx in amps:
        amps[idx] = np.hstack(amps[idx])

    return np.hstack(amps.values())


def _main():
    # Parse arguments from command line
    args = __parse_args()

    amps_1 = __extract_amps(args.input)
    amps_2 = __extract_amps(args.input2)

    _, axes = plt.subplots()
    axes.set_title(args.input)
    axes.set_xlabel("Channels, ch")
    axes.set_ylabel("Counts Difference")


    hist_1, bins = np.histogram(
        amps_1, bins=args.bins, range=(
            args.ampl_threshold, args.ampl_max))

    hist_2, _ = np.histogram(
        amps_2, bins=args.bins, range=(
            args.ampl_threshold, args.ampl_max))

    # Calculate bins centers
    bins_centers = (bins[:-1] + bins[1:]) / 2

    axes.step(bins_centers, hist_1 / hist_2, where='mid')

    axes.legend()
    plt.show()


if __name__ == "__main__":
    sns.set_context("poster")
    _main()
