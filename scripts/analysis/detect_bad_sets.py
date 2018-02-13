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

P_IND_START = 0
P_IND_END = 102
AMPL_THRESH = 496
AMPL_MAX = 4016
BINS = 55
DATA_ROOT = "/home/chernov/data/lan10_processed"
FILL = "2017_11/Fill_3"
GROUP_ABS = path.join(DATA_ROOT, FILL)


def get_set_spectrum(set_abs_path, borders=None, bins=30):
    """Calculate energy spectrum for set."""
    points = natsorted(glob.glob(path.join(set_abs_path, "p*.df")))

    out = {}

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
        hist, bins = np.histogram(amps, bins, range=borders, density=True)
        hist_unnorm, _ = np.histogram(amps, bins, range=borders)
        out[path.relpath(point, set_abs_path)] = {
            "meta": meta,
            "hist": hist,
            "hist_unnorm": hist_unnorm,
            "bins": bins
        }
    return out


def calc_hist_avg(sets_data, point_index):
    """Calculate point average histogram in sets."""
    sets = list(sets_data.values())
    hist_avg = np.zeros(BINS)
    used = 0
    for set_ in sets:
        for point in set_:
            curr_index = int(
                set_[point]["meta"]["external_meta"]["point_index"]
            )
            if curr_index == point_index:
                hist_avg += set_[point]["hist"]
                used += 1
    hist_avg /= used
    return hist_avg


def calc_chi_square(set_, hist_avg, point_index):
    """Calculate chi2 for set (unnormed).

    Parameters
    ----------
    set_ : Set data. Return value from get_set_spectrum().

    hist_avg: Average histogram data. Return value from calc_hist_avg().

    point_index: Index of point.

    Returns
    -------
    chi2: Unnormed chi2 for point.

    See Also
    --------
    get_set_spectrum

    """
    for point in set_:
        curr_index = int(
            set_[point]["meta"]["external_meta"]["point_index"]
        )
        if curr_index == point_index:
            hist = set_[point]["hist"]
            hist_unnorm = set_[point]["hist_unnorm"]
            sigma_raw = hist_unnorm ** 0.5
            sigma = sigma_raw * (hist / hist_unnorm)
            sigma[np.isnan(sigma)] = 1  # replace nans for empty bins
            return ((hist - hist_avg)**2 / sigma**2).sum() / BINS
    return None


def __calc_xticklabels(sets_data):
    labels = []
    for idx in range(P_IND_START, P_IND_END):
        for set_idx in sets_data:
            found = False
            for point in sets_data[set_idx]:
                ext_meta = sets_data[set_idx][point]["meta"]["external_meta"]
                curr_index = int(
                    ext_meta["point_index"]
                )
                if curr_index == idx:
                    found = True
                    hv = format(float(ext_meta["HV1_value"]) / 1000, '.2f')
                    labels.append("%s:%s" % (idx, hv))
                    break
            if found:
                break
    return labels


def __calc_point_sigma(sets_data, point_index):
    """Caluclate sigma based on fitting.

    NOTE: unfinished
    """
    sets = list(sets_data.values())
    global hists
    hists = []
    for set_ in sets:
        for point in set_:
            curr_index = int(
                set_[point]["meta"]["external_meta"]["point_index"]
            )
            if curr_index == point_index:
                hists.append(np.arrayset_[point]["hist"])
    hists = np.vstack(hists)
    #dev_from_mean = (hists - hists.mean(axis=0)).ravel()
    # print(hists)
    raise Exception


def __main():
    sets = natsorted(glob.glob(path.join(GROUP_ABS, "set_*[0-9]")))[5:]
    get_spectrum = partial(get_set_spectrum, borders=(
        AMPL_THRESH, AMPL_MAX), bins=BINS)

    #global out
    with closing(Pool()) as pool:
       out = pool.map(get_spectrum, sets)

    sets_data = {path.relpath(s, GROUP_ABS): d for s, d in zip(sets, out)}

    hists_avg = []
    for idx in range(P_IND_START, P_IND_END):
        hists_avg.append(calc_hist_avg(sets_data, idx))

    sets_points_chi2 = {}
    for set_idx in sets_data:
        x = []
        y = []
        for idx in range(P_IND_START, P_IND_END):
            chi2 = calc_chi_square(sets_data[set_idx], hists_avg[idx], idx)
            if chi2:
                x.append(idx)
                y.append(chi2)
        sets_points_chi2[set_idx] = {"x": x, "y": y}

    fig, ax = plt.subplots()

    ax.set_title(r'$\chi^2$ deviations for each point in %s' % (FILL) +
                 '\nBad sets excluded')
    ax.set_xlabel(r'Point index and HV, V')
    ax.set_ylabel(r'$\chi^2$')
    ax.set_yscale('log')
    ax.set_xticks(list(range(P_IND_START, P_IND_END)))
    ax.set_xticklabels(__calc_xticklabels(sets_data))
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    palette = sns.color_palette("hls", len(sets_points_chi2))
    for idx, set_idx in enumerate(natsorted(sets_points_chi2.keys())):
        ax.scatter(
            sets_points_chi2[set_idx]["x"], sets_points_chi2[set_idx]["y"],
            label=set_idx, s=30, c=palette[idx], edgecolors="face"
        )
    ax.legend()


if __name__ == "__main__":
    sns.set_context("poster")
    sns.set_style(rc={"font.family": "monospace"})
    DEF_PALLETE = sns.color_palette()
    sns.set_palette(sns.cubehelix_palette(8))
    __main()
