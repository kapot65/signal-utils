# -*- coding: utf-8 -*-
"""Определение плохих наборов.

Плохие сеты выделяются на основе отклонений по хи-квадрат от среднего по всем
сетам спектра.

Детектор работает только на преобразованных в события данных с Лан10-12PCI. (
Обработка производится скриптом ./scripts/convert_points.py)

Алгоритм работы:
1. Усреднение всех выбранных спектров.

"""
# TODO: remove hardcode

import glob
from contextlib import closing
from os import path

import dfparser
import numpy as np
from multiprocess import Pool
from natsort import natsorted
from scipy.stats import chisquare

AMPL_THRESH = 500
GROUP_ABS = "/home/chernov/data/lan10_processed/2017_11/Fill_3"


def get_set_spectrum(set_abs_path):
    """Calculate energy spectrum for set."""
    points = glob.glob(path.join(set_abs_path, "p*.df"))

    for point in points:
        _, meta, data = dfparser.parse_from_file(point)
        parsed_data = dfparser.Point()
        parsed_data.ParseFromString(data)
        del data

        global amps
        global times
        amps = []
        times = []
        for channel in parsed_data.channels:
            for block in channel.blocks:
                amps.append(np.array(block.events.amplitudes, np.int16))
                times.append(np.array(block.events.times, np.uint64))

        amps = np.hstack(amps)
        times = np.hstack(times)
        raise Exception


def __main():
    sets = natsorted(glob.glob(path.join(GROUP_ABS, "set_*")))
    with closing(Pool()) as pool:
        pool.map(get_set_spectrum, sets[0:1])


if __name__ == "__main__":
    __main()
