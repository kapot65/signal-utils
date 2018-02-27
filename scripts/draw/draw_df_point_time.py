#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drawing df point events time intervals spectrum script.
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
    parser.add_argument('-t', '--ampl-threshold', type=float, default=0,
                        help='Time difference lower threshold in ns (default '
                        '- 0).')
    parser.add_argument('-m', '--ampl-max', type=float, default=1e+5,
                        help='Time difference upper threshold in ns (default '
                        '- 0).')
    parser.add_argument('-b', '--bins', type=int, default=40,
                        help='Histogram bins number (default - 40).')
    return parser.parse_args()


def _main():
    # Parse arguments from command line
    args = __parse_args()

    # Read dataforge point
    _, _, data = dfparser.parse_from_file(args.input)

    # Parse Binary data
    point = dfparser.Point()
    point.ParseFromString(data)

    # Extract event times from each block
    times = []
    for channel in point.channels:
        for block in channel.blocks:
            times.append(np.array(block.events.times, np.uint64))

    # Combine times into one array
    times = np.hstack(times)

    # Calculate time differences
    diffs = times[1:] - times[:-1]

    # Calculate histogram
    hist, bins = np.histogram(diffs, bins=args.bins, range=(
        args.ampl_threshold, args.ampl_max))

    # Calculate bins centers
    bins_centers = (bins[:-1] + bins[1:]) / 2

    # Drawing graph
    _, axes = plt.subplots()
    axes.set_title(args.input)
    axes.set_xlabel("Time, ns")
    axes.set_ylabel("Counts")
    axes.step(bins_centers, hist, where='mid')
    plt.show()


if __name__ == "__main__":
    sns.set_context("poster")
    _main()
