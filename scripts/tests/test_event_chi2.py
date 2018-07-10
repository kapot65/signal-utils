"""Extract events from data and find mean shape.

Script needs raw LAN10-12PCI frame in dataforge-envelope protobuf format.

Script will extract frames from selected points then group them by
amlitude and find mean shape for every group.

Script use frame max length to filter frames with overlapped events.
"""
from argparse import ArgumentParser
from glob import glob
from os.path import abspath, dirname, join

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.optimize import curve_fit
from tqdm import tqdm

from signal_utils.generation_utils import (gen_raw_block, gen_signal,
                                           generate_df)

FREQ = 3125000.0
APPR_POINTS = 4
SIGMA = 10.443 * 16
BINS = 60
RANGE = (0, 600)

THRESHOLD = 700
AREA_L = 100
AREA_R = 200


def _parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        'root',
        help='Acquisition data root (example: "/data/data/lan10/"). '
        'Use "gen" to test generator instead of data.')
    parser.add_argument(
        'wildcard',
        help='Selection wildcard '
        '(example - "2017_11/Fill_1/set_*/p*HV1=16000*.df"). '
        'Generation seconds if root="gen"')
    parser.add_argument(
        '-i',
        '--input',
        default=None,
        help='Input .npy metrics file. '
        'If setted - generation step will be skipped (default - None)')
    parser.add_argument(
        '-o',
        '--output',
        default='chi2s.npy',
        help='Output .npy metrics file (default - "chi2s.npy")')
    parser.add_argument(
        '-l',
        '--left-offset',
        default=50,
        help='Noise extraction left border in bins. (default=50)')
    parser.add_argument(
        '-r',
        '--right-offset',
        default=200,
        help='Noise extraction right border in bins. (default=250)')
    parser.add_argument(
        '-m',
        '--max-len',
        default=308,
        help='Frame max length threshold. (default=308)')
    parser.add_argument(
        '--min-amp',
        default=700,
        help='Minimal amplitude. (default=700)')
    parser.add_argument(
        '--max-amp',
        default=3700,
        help='Maximal amplitude. (default=3700)')
    parser.add_argument(
        '--step',
        default=200,
        help='Size of amplitude group. (default=200)')
    parser.add_argument(
        '-g', '--compare-groups', nargs="+", type=int,
        default=[3, 6, 9, 13],
        help="Indexes of amplitude groups which will be used for comparison "
        "with generated event shape."
    )
    return parser.parse_args()


def _extr_chi2(point):
    chi2s = np.array([])
    for ch in point.channels:
        for bl in ch.blocks:
            for frame in bl.frames:
                data = np.frombuffer(frame.data, np.int16)
                if len(data) <= args.max_len:
                    shape = data[args.left_offset:args.right_offset]
                    if len(shape) == \
                            (args.right_offset - args.left_offset):

                        x_max = shape.argmax()
                        x_l = x_max - APPR_POINTS
                        x_r = x_max + APPR_POINTS

                        if x_l >= 0 and x_r < len(shape):

                            x = np.linspace(
                                0, len(shape) / FREQ, len(shape))

                            a, b, c = np.polyfit(
                                x[x_l: x_r], shape[x_l: x_r], 2)

                            pos = -b / (2 * a)
                            amp = a * pos * pos + b * pos + c

                            shape_gen = gen_signal(x, amp, pos)

                            chi2 = ((shape - shape_gen)**2).sum()
                            chi2 /= len(shape)
                            chi2 /= SIGMA

                            chi2s = np.append(chi2s, [chi2])
    return chi2s


if __name__ == "__main__":
    seaborn.set_context("poster")
    args = _parse_args()

    if not args.input:
        if args.root == 'gen':
            meta, data, _ = generate_df(
                area_l=AREA_L,
                area_r=AREA_R,
                time=float(args.wildcard),
                dist_file=abspath(
                    join(dirname(__file__),
                         '../../signal_utils/data/dist.dat')))
            point = dfparser.Point()
            point.ParseFromString(data)
            chi2s = _extr_chi2(point)
        else:
            files = glob(join(args.root, args.wildcard))
            chi2s = np.array([])
            for p in tqdm(files):
                point = dfparser.Point()
                _, meta, data = dfparser.parse_from_file(p)
                point.ParseFromString(data)
                chi2s = np.append(chi2s, _extr_chi2(point))

        np.save(args.output, chi2s)
    else:
        chi2s = np.load(args.input)

    hist, bins = np.histogram(
        chi2s, bins=BINS, range=RANGE, density=True)
    x_r = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots()

    if args.root == 'gen':
        title = r"$\chi ^ 2$ generated data"
    else:
        title = r"$\chi ^ 2$ %s" % args.wildcard

    ax.set_title(title)
    ax.set_xlabel(r"$\chi ^ 2$")
    ax.step(x_r, hist, where='mid', label='mean = %s' % (
        round(chi2s.mean() / (args.right_offset - args.left_offset), 3)))
    ax.grid(color='lightgray', alpha=0.7)
    ax.legend()
    plt.show()
