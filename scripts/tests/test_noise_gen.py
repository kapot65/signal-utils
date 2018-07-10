"""Noise generation test.

Script will calculate statistics for real and generated noise and
draw compare graphs.

Script needs raw LAN10-12PCI frame in dataforge-envelope protobuf format.
To calculate real noise statistics script will take first extract first
n bins from each frame in selected points.
"""
from argparse import ArgumentParser
from glob import glob
from os.path import join

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.optimize import curve_fit
from tqdm import tqdm

from signal_utils.generation_utils import gen_raw_block

BINS = 50
RANGE = (-800 // 16, 800 // 16)
GEN_SIZE_BLOCKS = 20


def _parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        'root',
        help='Acquisition data root (example: "/data/data/lan10/")')
    parser.add_argument(
        'wildcard',
        help='Selection wildcard '
        '(example - "2017_11/Fill_1/set_*/p*HV1=16000*.df")')
    parser.add_argument(
        '-l',
        '--left-offset',
        default=3,
        help='Noise extraction left border in bins. (default=3)')
    parser.add_argument(
        '-r',
        '--right-offset',
        default=83,
        help='Noise extraction right border in bins. (default=3)')
    return parser.parse_args()


def gaus(x, sigma, mu):
    return (1 / (sigma * (2 * np.pi)**0.5)) * \
        np.exp(-((x - mu)**2 / (2 * sigma**2)))


if __name__ == "__main__":
    seaborn.set_context("poster")

    args = _parse_args()

    noise_real = []
    files = glob(join(args.root, args.wildcard))
    for p in tqdm(files):
        point = dfparser.Point()
        _, meta, data = dfparser.parse_from_file(p)
        point.ParseFromString(data)
        noises = []
        for ch in point.channels:
            for bl in ch.blocks:
                for frame in bl.frames:
                    noise = np.frombuffer(frame.data, np.int16)[
                        args.left_offset:args.right_offset]
                    noises.append(noise)
        noises = np.hstack(noises)
        noise_real.append(noises)

    noise_real = np.hstack(noise_real)
    noise_gen = np.hstack([
        gen_raw_block(freq=0)[0] for _ in range(GEN_SIZE_BLOCKS)])

    hist_r, bins_r = np.histogram(noise_real // 16, bins=BINS,
                                  range=RANGE, density=True)
    x_r = (bins_r[:-1] + bins_r[1:]) / 2
    popt_r, _ = curve_fit(gaus, x_r, hist_r, p0=[10, 0])

    hist_g, _ = np.histogram(noise_gen // 16, bins=BINS,
                             range=RANGE, density=True)
    popt_g, _ = curve_fit(gaus, x_r, hist_g, p0=[10, 0])

    fig, ax = plt.subplots()
    ax.set_title("Noise distribution for %s" % args.wildcard)
    ax.step(x_r, hist_r, where='mid', label="Real")
    ax.plot(
        x_r, gaus(x_r, *popt_r), 'r--',
        label=r"Real" "\n" "($\sigma = %s,$ \n $\mu = %s$" %
        (round(popt_r[0], 3), round(popt_r[1], 3)))
    ax.step(x_r, hist_g, where='mid', label="Generated")
    ax.plot(
        x_r, gaus(x_r, *popt_g), 'g--',
        label=r"Generated" "\n" "($\sigma = %s,$ \n $\mu = %s$" %
        (round(popt_g[0], 3), round(popt_g[1], 3)))
    ax.grid(color='lightgray', alpha=0.7)
    ax.legend()
    plt.show()
