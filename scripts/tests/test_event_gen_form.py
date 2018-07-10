"""Extract events from data and find mean shape.

Script needs raw LAN10-12PCI frame in dataforge-envelope protobuf format.

Script will extract frames from selected points then group them by
amlitude and find mean shape for every group.

Script use frame max length to filter frames with overlapped events.
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

from signal_utils.generation_utils import gen_raw_block, gen_signal

FREQ = 3125000.0


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
        '-i',
        '--input',
        default=None,
        help='Input .npy metrics file. '
        'If setted - generation step will be skipped (default - None)')
    parser.add_argument(
        '-o',
        '--output',
        default='shapes_mean.npy',
        help='Output .npy metrics file (default - "shapes_mean.npy")')
    parser.add_argument(
        '-l',
        '--left-offset',
        default=50,
        help='Noise extraction left border in bins. (default=50)')
    parser.add_argument(
        '-r',
        '--right-offset',
        default=250,
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


if __name__ == "__main__":
    seaborn.set_context("poster")
    args = _parse_args()

    if not args.input:
        shapes = []
        files = glob(join(args.root, args.wildcard))
        for p in tqdm(files):
            point = dfparser.Point()
            _, meta, data = dfparser.parse_from_file(p)
            point.ParseFromString(data)
            shapes_point = []
            for ch in point.channels:
                for bl in ch.blocks:
                    for frame in bl.frames:
                        data = np.frombuffer(frame.data, np.int16)
                        if len(data) <= args.max_len:
                            shape = data[args.left_offset:args.right_offset]
                            if len(shape) == \
                                    (args.right_offset - args.left_offset):
                                shapes_point.append(shape)
            if shapes_point:
                shapes_point = np.vstack(shapes_point)
                shapes.append(shapes_point)

        shapes = np.vstack(shapes)

        shapes_mean = [{
            "min": start,
            "max": start + args.step,
            "shape": shapes[np.logical_and(
                shapes.max(axis=1) >= start,
                shapes.max(axis=1) <= start + args.step)].mean(axis=0)
        } for start in range(args.min_amp, args.max_amp, args.step)]
        np.save(args.output, shapes_mean)
    else:
        shapes_mean = np.load(args.input)[()]

    fig, ax = plt.subplots()
    ax.set_title("Events shapes mean %s" % args.wildcard)
    ax.set_xlabel("Time, bins")
    ax.set_ylabel("Amplitude, channels")
    for group in shapes_mean:
        ax.plot(group['shape'], label="%s - %s" % (group['min'], group['max']))
    ax.grid(color='lightgray', alpha=0.7)
    ax.legend()

    for comp_group in args.compare_groups:
        group = shapes_mean[comp_group]
        amp = group['shape'].max()
        pos = group['shape'].argmax() / FREQ

        shape_gen = gen_signal(
            np.linspace(0, len(group['shape']) / FREQ, len(group['shape'])),
            amp, pos)

        fig, ax = plt.subplots()
        ax.set_title("Mean shape vs generated (%s - %s ch)" % (
            group['min'], group['max']))
        ax.set_xlabel("Time, bins")
        ax.set_ylabel("Amplitude, channels")
        ax.plot(group['shape'], label='averaged')
        ax.plot(shape_gen, 'r--', label='generated')
        ax.set_ylim(-int(args.max_amp * 0.3), args.max_amp)
        ax.grid(color='lightgray', alpha=0.7)
        ax.legend()

    plt.show()
