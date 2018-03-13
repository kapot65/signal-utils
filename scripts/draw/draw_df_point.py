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
    parser.add_argument('-t', '--ampl-threshold', type=int, default=496,
                        help='Amplitudes threshold (default - 496).')
    parser.add_argument('-m', '--ampl-max', type=int, default=4016,
                        help='Amplitudes max threshold (default - 4016).')
    parser.add_argument('-b', '--bins', type=int, default=55,
                        help='Histogram bins number (default - 55).')
    parser.add_argument('-s', '--split-channels', action="store_true",
                        help='Split graphs by channels.')
    return parser.parse_args()


def _main():
    # Parse arguments from command line
    args = __parse_args()

    # Read dataforge point
    _, meta, data = dfparser.parse_from_file(args.input)

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

    if not args.split_channels:
        plots = {
            "all-channels": np.hstack(amps.values())
        }
    else:
        plots = amps

    _, axes = plt.subplots()
    axes.set_title(args.input)
    axes.set_xlabel("Channels, ch")
    axes.set_ylabel("Counts")
    for idx, plot in enumerate(plots):
        # Calculate histogram
        hist, bins = np.histogram(
            plots[plot], bins=args.bins, range=(
                args.ampl_threshold, args.ampl_max))

        # Calculate bins centers
        bins_centers = (bins[:-1] + bins[1:]) / 2

        # Drawing graph
        label = idx
        if "channels" in meta:
            if str(idx) in meta["channels"]:
                label = meta["channels"][str(idx)]

        axes.step(bins_centers, hist, where='mid', label=label)

    axes.legend()
    plt.show()


if __name__ == "__main__":
    sns.set_context("poster")
    _main()
