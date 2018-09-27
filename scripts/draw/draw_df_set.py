#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw df count rate hitsogramm for set.

Set points should be processed before drawing.
"""
import glob
from argparse import ArgumentParser
from os import path

import dfparser  # Numass files parser
import matplotlib.pyplot as plt  # plotting library
import numpy as np
import seaborn as sns  # matplotlib grahs visual enchancer
from natsort import natsorted


def __parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data_root', help='Data root folder.')
    parser.add_argument('set', help='Path to set relative to root.')
    parser.add_argument('-t', '--ampl-threshold', type=int, default=496,
                        help='Point amplitudes threshold (default - 496).')
    parser.add_argument('--x-scale', default=None,
                        choices=['linear', 'log', 'logit', 'symlog'],
                        help='Axis x scale (default - linear).')
    parser.add_argument('--y-scale', default=None,
                        choices=['linear', 'log', 'logit', 'symlog'],
                        help='Axis y scale (default - linear).')
    return parser.parse_args()


def _main():
    # Parse arguments from command line
    args = __parse_args()
    points = natsorted(glob.glob(path.join(args.data_root, args.set, "p*.df")))

    amps = {}

    for p in points:
        # Read dataforge point
        _, meta, data = dfparser.parse_from_file(p)

        # Parse Binary data
        point = dfparser.Point()
        point.ParseFromString(data)

        hv = int(meta['external_meta']['HV1_value'])
        time = meta['params']['events_num'] * meta['params']['b_size'] / \
            meta['params']['sample_freq']

        if hv not in amps:
            amps[hv] = {
                'time': 0,
                'amps': []
            }

        amps[hv]['time'] += time

        for idx, channel in enumerate(point.channels):
            for block in channel.blocks:
                amps[hv]['amps'].append(np.array(
                    block.events.amplitudes, np.int16))

    for hv in amps:
        amps[hv]['amps'] = np.hstack(amps[hv]['amps'])
        amps[hv]['count_rate'] = len(amps[hv]['amps'][
            amps[hv]['amps'] >= args.ampl_threshold]) / amps[hv]['time']

    hvs = sorted(list(amps.keys()))
    crs = [amps[hv]['count_rate'] for hv in hvs]

    _, axes = plt.subplots()

    if args.x_scale:
        axes.set_xscale(args.x_scale)
    if args.y_scale:
        axes.set_yscale(args.y_scale)

    axes.set_title(args.set)
    axes.set_xlabel("High voltage, V")
    axes.set_ylabel("Count rate, Ev/s")
    axes.plot(hvs, crs, label='Set points count rate')

    axes.legend()
    plt.show()


if __name__ == "__main__":
    sns.set_context("poster")
    _main()
