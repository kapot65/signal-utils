# -*- coding: utf-8 -*-
"""Определение плохих наборов.

Плохие сеты выделяются на основе отклонений по хи-квадрат от среднего по всем
сетам спектра.

Детектор работает только на преобразованных в события данных с Лан10-12PCI. (
Обработка производится скриптом ./scripts/convert_points.py)

Алгоритм работы:
1. Расчет энергетических спектров для каждой точки для всех сетов в выбранном
   наборе.
   Влияющие параметры:
   - ind-start - минимальный индекс точки в сете, которая будет рассмотрена
   - ind-end - максимальный индекс точки в сете, которая будет рассмотрена
   - ampl-threshold - порог по обрезке каналов снизу. Нижние каналы могут быть
     сильно зашумлены и в итоге плохо повлияют на работу алгоритма.
   - ampl-max - порог по обрезке каналов сверху
   - bins - количество бинов в энергетическом спектре. В силу параметров
     аппаратуры стоит подбирать количество бинов с расчетом на то, чтобы
     координаты краев бинов были кратны 16
   - norming_channels - каналы, по которым будет производится нормировка
     энергетических спектров. Имеет смысл захватывать пик гистограммы. Если
     параметр не указан - нормировка будет производится по интегралу всей
     гистограммы
2. Вычисление усредненных гистограмм для точек с одинаковыми индексами по всем
   сетам в наборе.
3. Вычисление хи-квадрат отклонений от энергетического спектра точки для каждой
   точки во всех сетах.
4. Вывод результата в виде графика.

# TODO: add time filtering (make as package tool?)
# TODO: сделать обработку norming_channels
"""
import glob
from argparse import ArgumentParser
from contextlib import closing
from functools import partial
from os import path

import dfparser
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from multiprocess import Pool
from natsort import natsorted


def __parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data_root', help='Data root folder.')
    parser.add_argument('fill', help='Fill folder relative to data root.')
    parser.add_argument('--ind-start', type=int, default=0,
                        help='Points starting index (default - 0).')
    parser.add_argument('--ind-end', type=int, default=102,
                        help='Points ending index (default - 102).')
    parser.add_argument('-t', '--ampl-threshold', type=int, default=496,
                        help='Amplitudes threshold (default - 496).')
    parser.add_argument('-m', '--ampl-max', type=int, default=4016,
                        help='Amplitudes max threshold (default - 4016).')
    parser.add_argument('-b', '--bins', type=int, default=55,
                        help='Histogram bins number (default - 55).')
    return parser.parse_args()


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
    hist_avg = np.zeros(ARGS.bins)
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
            return ((hist - hist_avg)**2 / sigma**2).sum() / ARGS.bins
    return None


def __calc_xticklabels(sets_data):
    labels = []
    for idx in range(ARGS.ind_start, ARGS.ind_end):
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
    hists = []
    for set_ in sets:
        for point in set_:
            curr_index = int(
                set_[point]["meta"]["external_meta"]["point_index"]
            )
            if curr_index == point_index:
                hists.append(np.arrayset_[point]["hist"])
    hists = np.vstack(hists)


def __main():
    group_abs = path.join(ARGS.data_root, ARGS.fill)
    sets = natsorted(glob.glob(path.join(group_abs, "set_*[0-9]")))[5:]
    get_spectrum = partial(get_set_spectrum, borders=(
        ARGS.ampl_threshold, ARGS.ampl_max), bins=ARGS.bins)

    with closing(Pool()) as pool:
        out = pool.map(get_spectrum, sets)

    sets_data = {path.relpath(s, group_abs): d for s, d in zip(sets, out)}

    hists_avg = []
    for idx in range(ARGS.ind_start, ARGS.ind_end):
        hists_avg.append(calc_hist_avg(sets_data, idx))

    sets_points_chi2 = {}
    for set_idx in sets_data:
        x_values = []
        y_values = []
        for idx in range(ARGS.ind_start, ARGS.ind_end):
            chi2 = calc_chi_square(sets_data[set_idx], hists_avg[idx], idx)
            if chi2:
                x_values.append(idx)
                y_values.append(chi2)
        sets_points_chi2[set_idx] = {"x": x_values, "y": y_values}

    _, axes = plt.subplots()

    axes.set_title(r'$\chi^2$ deviations for each point in %s' % (ARGS.fill) +
                   '\nBad sets excluded')
    axes.set_xlabel(r'Point index and HV, V')
    axes.set_ylabel(r'$\chi^2$')
    axes.set_yscale('log')
    axes.set_xticks(list(range(ARGS.ind_start, ARGS.ind_end)))
    axes.set_xticklabels(__calc_xticklabels(sets_data))
    for tick in axes.get_xticklabels():
        tick.set_rotation(90)
    palette = sns.color_palette("hls", len(sets_points_chi2))
    for idx, set_idx in enumerate(natsorted(sets_points_chi2.keys())):
        axes.scatter(
            sets_points_chi2[set_idx]["x"], sets_points_chi2[set_idx]["y"],
            label=set_idx, s=30, c=palette[idx], edgecolors="face"
        )
    axes.legend()
    plt.show()


if __name__ == "__main__":
    ARGS = __parse_args()
    sns.set_context("poster")
    sns.set_style(rc={"font.family": "monospace"})
    DEF_PALLETE = sns.color_palette()
    sns.set_palette(sns.cubehelix_palette(8))
    __main()
