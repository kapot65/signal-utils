# -*- coding: utf-8 -*-
"""Определение плохих наборов.

Плохие сеты выделяются на основе отклонений по хи-квадрат от среднего по всем
сетам спектра.

Детектор работает только на преобразованных в события данных с Лан10-12PCI. (
Обработка производится скриптом ./scripts/convert_points.py)

Алгоритм работы:
1. Усреднение всех выбранных спектров.

- Проверить распределение по отклонениям

"""
# TODO: remove hardcode
# TODO: add time filtering (make as package tool?)

import glob
from contextlib import closing
from functools import partial
from os import path

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from multiprocess import Pool
from natsort import natsorted
from scipy.stats import chisquare

AMPL_THRESH = 496
AMPL_MAX = 4016
BINS = 55
GROUP_ABS = "/home/chernov/data/lan10_processed/2017_11/Fill_3"


def get_set_spectrum(set_abs_path, borders=None, bins=30, normed=True):
    """Calculate energy spectrum for set."""
    points = glob.glob(path.join(set_abs_path, "p*.df"))

    for point in points:
        _, meta, data = dfparser.parse_from_file(point)
        parsed_data = dfparser.Point()
        parsed_data.ParseFromString(data)
        del data

        amps = []
        times = []
        for channel in parsed_data.channels:
            for block in channel.blocks:
                amps.append(np.array(block.events.amplitudes, np.int16))
                times.append(np.array(block.events.times, np.uint64))

        amps = np.hstack(amps)
        times = np.hstack(times)
        hist, bins = np.histogram(amps, bins, range=borders, normed=normed)
        return hist, bins


def __main():
    sets = natsorted(glob.glob(path.join(GROUP_ABS, "set_*")))
    get_spectrum = partial(get_set_spectrum, borders=(
        AMPL_THRESH, AMPL_MAX), bins=BINS)
    with closing(Pool()) as pool:
        out = pool.map(get_spectrum, sets)

    hists = np.array([o[0] for o in out]).T
    bins_bord = np.array([o[1] for o in out]).T
    bins = (bins_bord[1:, :] + bins_bord[:-1, :]) / 2

    _, ax = plt.subplots()
    ax.set_title("Normalized spectrums comparison")
    ax.set_xlabel("Bins, ch")
    ax.set_ylabel("Frequency")
    ax.step(bins, hists)

    bins_mean = bins.mean(axis=1)
    hists_mean = hists.mean(axis=1)
    ax.plot(bins_mean, hists_mean, 'ro', lw=10, label="Mean")
    ax.legend()

    sns.set_palette(DEF_PALLETE)
    _, ax2 = plt.subplots()
    ax2.set_title(r"Spectrums $\chi^2$ deviation")
    ax2.set_xlabel("Bins, ch")
    ax2.set_ylabel("Frequency")
    deviations = ((hists.T - hists_mean)**2).sum(axis=1)/BINS
    #ax2.hist(((hists.T - hists_mean)**2).sum(axis=1), 23)
    ax2.plot(deviations, np.ones(deviations.shape), "ro")


if __name__ == "__main__":
    seaborn.set_context("poster")
    DEF_PALLETE = sns.color_palette()
    sns.set_palette(sns.cubehelix_palette(8))
    __main()
