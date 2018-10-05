"""Parsing utils."""
from glob import glob
from os.path import join

import dfparser
import numpy as np
from natsort import natsorted


def _lan_amps_f(fp):
    _, meta, data = dfparser.parse_from_file(fp)
    point = dfparser.Point()
    point.ParseFromString(data)

    amps = []

    for idx, channel in enumerate(point.channels):
        for block in channel.blocks:
            amps.append(np.array(block.events.amplitudes, np.int16))

    return np.hstack(amps)


def _lan_amps(data):
    point = dfparser.Point()
    point.ParseFromString(data)

    amps = []

    for idx, channel in enumerate(point.channels):
        for block in channel.blocks:
            amps.append(np.array(block.events.amplitudes, np.int16))

    return np.hstack(amps)


def _madc_amps_f(fp):
    _, meta, data = dfparser.parse_from_file(fp)
    amps = np.array([
        unpack('H', bytes(a))[0] for a in (zip(data[0::7], data[1::7]))])
    return amps


def _madc_amps(data):
    amps = np.array(
        [unpack('H', bytes(a))[0] for a in (zip(data[0::7], data[1::7]))])
    return amps


def _get_sets_in_fill(fill_abs_path):
    return natsorted(glob(join(fill_abs_path, "set_*[0-9]")))


def _get_points_in_set(set_abs_path):
    return natsorted(glob(join(set_abs_path, "p*(*s)*.df")))
