#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert TRISTAN_11_2017 data to df-envelope format."""
from argparse import ArgumentParser
from os import listdir, path

import dfparser
import numpy as np
from natsort import natsorted
from scipy.io import loadmat

BIN_SIZE_NS = 8


def _parse_args():
    parser = ArgumentParser(description='Data folder.')

    parser.add_argument('input', help='Input dataforge folder.')
    parser.add_argument('-o', '--output', default=None,
                        help='Output dataforge file without extension. '
                        '(default - processed.df in data folder)')
    parser.add_argument('-a', '--ascii', action='store_true',
                        help='Save data in text format')
    parser.add_argument('-e', '--exclude', action='append', default=[],
                        help='Exclude channels. '
                        'The flag can be used repeatedly in command.')

    return parser.parse_args()


def _main():
    args = _parse_args()

    files = natsorted(listdir(args.input))
    point = dfparser.Point()
    channels = {}
    meta = {
        "compression": "zlib",
        "channels": {}
    }

    for filename in files:
        if not filename.endswith('.mat'):
            continue
        data = loadmat(path.join(args.input, filename))
        stats_raw = data["statistics"]
        stats = {str(stats_raw[0, i][0]): stats_raw[1, i].tolist()
                 for i in range(stats_raw.shape[1])}
        for _ in range(2):
            stats = {val: stats[val][0] if stats[val]
                     and isinstance(stats[val], list)
                     else stats[val] for val in stats}

        channel = stats['Board identifier/Spectrum ID']

        if channel in args.exclude:
            continue

        if channel not in channels:
            channels[channel] = point.channels.add()
            channels[channel].blocks.add()
            meta["channels"][len(meta["channels"])] = channel

        amps = channels[channel].blocks[0].events.amplitudes
        times = channels[channel].blocks[0].events.times

        for amp, time in zip([d[0] for d in data['energies']],
                             [d[0] * BIN_SIZE_NS for d in data['timestamps']]):
            amps.append(int(amp))
            times.append(int(time))

    msg = dfparser.create_message(meta, point.SerializeToString())

    out_file = path.join(args.input, 'processed')
    if args.output:
        out_file = args.output
    with open("%s.df" % out_file, 'wb') as out:
        out.write(msg)

    if args.ascii:
        datas = []
        for idx, channel in enumerate(point.channels):
            amps = channel.blocks[0].events.amplitudes
            times = channel.blocks[0].events.times
            data = np.zeros((len(amps), 3))
            data[:, 0] = list(times)
            data[:, 1] = list(amps)
            data[:, 2] = idx
            datas.append(data)

        datas = np.vstack(datas)
        datas = datas[datas[:, 0].argsort()]

        np.savetxt("%s.txt" % out_file, datas, fmt='%.4e')


if __name__ == "__main__":
    _main()
