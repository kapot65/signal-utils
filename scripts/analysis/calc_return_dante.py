# -*- coding: utf-8 -*-
"""Выделение близких по времени (возвратных) сигналов и построение гистограммы.

На вход должен подаватся обработанный файл с детектора DANTE.
"""

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

import dfparser
import numpy as np

def _parse_args():
    parser = ArgumentParser(description='Data folder.')
    parser.add_argument('input', help='Input dataforge DANTE file.')
    parser.add_argument(
        '-t', '--threshold', type=float, default=1e4,
        help='Return events interval threshold in ns. Default - 1e4')
    parser.add_argument(
        '--delta-min', type=float, default=0,
        help='Signals delta time minimal value in ns. Default -  0')
    parser.add_argument(
        '--delta-max', type=float, default=None,
        help='Signals delta time max value in ns. Default - threshold value')
    parser.add_argument(
        '--amp-min', type=float, default=0,
        help='Signals amplitudes minimal value in ch. Default -  0')
    parser.add_argument(
        '--amp-max', type=float, default=2000,
        help='Signals amplitudes max value in c. Default - 2000')
    parser.add_argument(
        '--bins-delta', type=int, default=20,
        help='X axis bins. Default -  40')
    parser.add_argument(
        '--bins-amp', type=int, default=20,
        help='Y axis bins. Default -  40')
    return parser.parse_args()


def _main():
    args = _parse_args()
    _, _, data = dfparser.parse_from_file(args.input)
    point = dfparser.Point()
    point.ParseFromString(data)
    del data

    datas = []
    for channel in point.channels:
        amps = channel.blocks[0].events.amplitudes
        times = channel.blocks[0].events.times
        data = np.zeros((len(amps), 3))
        data[:, 0] = list(times)
        data[:, 1] = list(amps)
        datas.append(data)

    datas = np.vstack(datas)
    datas = datas[datas[:, 0].argsort()]
    datas[:, 2][1:] = datas[:, 0][1:] - datas[:, 0][:-1]
    returned = datas[datas[:, 2] < args.threshold, :]

    delta_max = args.delta_max
    if not delta_max:
        delta_max = args.threshold

    returned = returned[np.logical_and(
        returned[:, 1] >= args.amp_min,
        returned[:, 1] <= args.amp_max)]
    returned = returned[np.logical_and(
        returned[:, 2] >= args.delta_min,
        returned[:, 2] <= args.delta_max)]

    plot = sns.jointplot(
        returned[:, 2], returned[:, 1], kind="hex", stat_func=None,
        xlim=(args.delta_min, delta_max), ylim=(args.amp_min, args.amp_max),
        joint_kws={'gridsize': (args.bins_delta, args.bins_amp)}
        )
    plot.set_axis_labels("Time delta, ns", "Amplitude, ch")
    plt.show()


if __name__ == "__main__":
    sns.set_context("poster")
    _main()
