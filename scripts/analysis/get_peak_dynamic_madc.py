"""Show points peak change in time."""
from argparse import ArgumentParser
from glob import glob
from os import path
from struct import unpack

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dateutil.parser import parse as tparse
from scipy.optimize import curve_fit
from tqdm import tqdm

from utils import _madc_amps


def __parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', help='Path to Lan10-12PCI root folder.')
    parser.add_argument('fill', help='Relative path to fill folder.')
    parser.add_argument('-v', '--voltage', type=int, default=16000,
                        help='Point voltage (default=16000).')

    parser.add_argument('-l', '--left-border', type=int, default=1168,
                        help='Histogram left border. (defailt=2000)')
    parser.add_argument('-r', '--right-border', type=int, default=3600,
                        help='Histogram right border. (default=3600)')
    parser.add_argument('-b', '--bins', type=int, default=37,
                        help='Histogram bins number (default - 37).')

    return parser.parse_args()


def _gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2. * sigma**2))


def _amp_hist_max(amps):
    hist, bins = np.histogram(
        amps, range=(args.left_border, args.right_border),
        bins=args.bins, density=True)

    bins_centers = (bins[:-1] + bins[1:]) / 2

    popt, _ = curve_fit(
        _gauss, bins_centers, hist,
        p0=[
            1,
            (args.left_border + args.right_border) // 2,
            (args.left_border + args.left_border) // 4
        ])

    return popt[1]


def _amp_mean(amps):
    filtered = amps[np.logical_and(
        amps >= args.left_border, amps <= args.right_border)]
    return filtered.mean()


def _amp_mean(amps):
    filtered = amps[np.logical_and(
        amps >= args.left_border, amps <= args.right_border)]
    return filtered.mean()


def _amp_mean(amps):
    filtered = amps[np.logical_and(
        amps >= args.left_border, amps <= args.right_border)]
    return filtered.mean()


def _amp_std(amps):
    filtered = amps[np.logical_and(
        amps >= args.left_border, amps <= args.right_border)]
    return filtered.std()


def _amp_cr(amps):
    filtered = amps[np.logical_and(
        amps >= args.left_border, amps <= args.right_border)]
    return len(filtered)


if __name__ == "__main__":
    args = __parse_args()

    sns.set_context("poster")

    points = glob(path.join(args.root, args.fill, '*/p*)'), recursive=True)

    filtered = []
    for point in points:
        _, meta, _ = dfparser.parse_from_file(point, nodata=True)
        if int(meta['external_meta']['HV1_value']) == args.voltage:
            filtered.append(point)

    peaks = []

    for point in tqdm(filtered):
        try:
            _, meta, data = dfparser.parse_from_file(point, nodata=False)
            amps = _madc_amps(data)

            num = path.basename(path.dirname(point))[4:]
            if num.endswith('_bad'):
                num = num[:-4]

            peak = {
                'point_index': meta['external_meta']['point_index'],
                'time': tparse(meta['start_time'][0]),
                'peak': _amp_hist_max(amps),
                'mean': _amp_mean(amps),
                'std': _amp_std(amps),
                'cr': _amp_cr(amps),
                'color': int(num) % 2
            }
            peaks.append(peak)
        except RuntimeError as err:
            print(err)

    peaks = sorted(peaks, key=lambda v: v['time'])

    palette = sns.color_palette()

    _, ax = plt.subplots()
    ax.set_title('Point peak position dynamic for %s' % args.fill)
    ax.set_xlabel('time')
    ax.set_ylabel('peak, bins')
    for p in peaks:
        ax.scatter(p['time'], p['peak'], c=palette[p['color']])

    _, ax = plt.subplots()
    ax.set_title('Point peak mean dynamic for %s' % args.fill)
    ax.set_xlabel('time')
    ax.set_ylabel('mean, bins')
    for p in peaks:
        ax.scatter(p['time'], p['mean'], c=palette[p['color']])

    _, ax = plt.subplots()
    ax.set_title('Point peak std dynamic for %s' % args.fill)
    ax.set_xlabel('time')
    ax.set_ylabel('std, bins')
    for p in peaks:
        ax.scatter(p['time'], p['std'], c=palette[p['color']])

    _, ax = plt.subplots()
    ax.set_title('Point peak count rate dynamics for %s' % args.fill)
    ax.set_xlabel('time')
    ax.set_ylabel('count rate, ev')
    for p in peaks:
        ax.scatter(p['time'], p['cr'], c=palette[p['color']])

    _, ax = plt.subplots()
    ax.set_title('Point index %s' % args.fill)
    ax.set_xlabel('time')
    ax.set_ylabel('peak, bins')
    ax.scatter([p['time'] for p in peaks], [p['point_index'] for p in peaks])

    plt.show()
