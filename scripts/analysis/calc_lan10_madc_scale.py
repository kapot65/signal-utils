"""Compare spectrums for one point between Lan10-12PCI and MADC."""
from argparse import ArgumentParser
from glob import glob
from os import path
from struct import unpack

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit


def __parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('lan10_root', help='Path to Lan10-12PCI root folder.')
    parser.add_argument('madc_root', help='Path to MADC root folder.')
    parser.add_argument('points', help='Compared points wildcard. '
                        '(examples \n 1. "2017_05/Fill_2/set_*[0-9]/p0*" - ) '
                        '0 index point for all sets in Fill_2.')
    parser.add_argument(
        '--madc-left-border', type=int, default=3000,
        help='MADC histogram generator peak left border (default=3000).')
    parser.add_argument(
        '--madc-right-border', type=int, default=3800,
        help='MADC histogram generator peak right border (default=3800).')
    parser.add_argument(
        '--madc-bins', type=int, default=100,
        help='MADC histogram generator peak right border (default=25).')
    parser.add_argument(
        '--lan-left-border', type=int, default=6000,
        help='Lan10 histogram generator peak left border (default=6000).')
    parser.add_argument(
        '--lan-right-border', type=int, default=7600,
        help='Lan10 histogram generator peak right border (default=7600).')
    parser.add_argument(
        '--lan-bins', type=int, default=50,
        help='Lan10 histogram generator peak right border (default=25).')

    parser.add_argument(
        '-g', '--graph', action='store_true',
        help='Draw graph after fitting.')

    return parser.parse_args()


def _madc_amps(fp):
    _, meta, data = dfparser.parse_from_file(fp)
    amps = np.array([
        unpack('H', bytes(a))[0] for a in (zip(data[0::7], data[1::7]))])
    return amps


def _lan_amps(fp):
    _, meta, data = dfparser.parse_from_file(fp)
    point = dfparser.Point()
    point.ParseFromString(data)

    amps = []

    for idx, channel in enumerate(point.channels):
        for block in channel.blocks:
            amps.append(np.array(block.events.amplitudes, np.int16))
    return np.hstack(amps)


def _gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2. * sigma**2))


if __name__ == "__main__":
    args = __parse_args()
    sns.set_context("poster")

    lan_points = glob(path.join(args.lan10_root, args.points))
    madc_points = glob(path.join(args.madc_root, args.points))

    # Extract data from  files
    madc_gen = []
    for point in madc_points:
        amps = _madc_amps(point)
        amps = amps[np.logical_and(
            amps > args.madc_left_border,
            amps < args.madc_right_border)]
        madc_gen.append(amps)
    madc_gen = np.hstack(madc_gen)

    lan_gen = []
    for point in lan_points:
        amps = _lan_amps(point)
        amps = amps[np.logical_and(
            amps > args.lan_left_border,
            amps < args.lan_right_border)]
        lan_gen.append(amps)
    lan_gen = np.hstack(lan_gen)

    # Calc histograms
    hist_m, bins_m = np.histogram(
        madc_gen, bins=args.madc_bins, density=True,
        range=(args.madc_left_border, args.madc_right_border))
    bins_centers_m = (bins_m[:-1] + bins_m[1:]) / 2

    hist_l, bins_l = np.histogram(
        lan_gen, bins=args.lan_bins, density=True,
        range=(args.lan_left_border, args.lan_right_border))
    bins_centers_l = (bins_l[:-1] + bins_l[1:]) / 2

    popt_m, _ = curve_fit(
        _gauss, bins_centers_m, hist_m,
        p0=[
            1,
            (args.madc_left_border + args.madc_right_border) // 2,
            (args.madc_right_border + args.madc_left_border) // 4
        ])

    popt_l, _ = curve_fit(
        _gauss, bins_centers_l, hist_l,
        p0=[
            1,
            (args.lan_left_border + args.lan_right_border) // 2,
            (args.lan_right_border + args.lan_left_border) // 4
        ])

    a = np.abs(popt_l[2] / popt_m[2])
    b = popt_l[1] - popt_m[1] * a

    if args.graph:
        _, axes = plt.subplots()

        axes.set_title('{} histogram compare'.format(args.points))
        axes.set_xlabel("Channels, ch")
        axes.set_ylabel("Counts")

        axes.step(
            bins_centers_m * a + b, hist_m / a,
            where='mid', label='MADC (a = {}, b = {})'.format(
                np.round(a, 3), np.round(b, 1)))
        axes.step(
            bins_centers_l, hist_l,
            where='mid', label='Lan10')

        axes.legend()

        plt.show()
    else:
        print('MADC (a = {}, b = {})'.format(np.round(a, 3), np.round(b, 1)))
