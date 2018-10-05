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
from itertools import chain
from os import path

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from multiprocess import Pool
from natsort import natsorted
from tqdm import tqdm


def __parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data_root', help='Data root folder.')
    parser.add_argument('fill', help='Fill folder relative to data root.')
    parser.add_argument('-l', '--hist-lower', type=int, default=2592,
                        help='Histogram lower border value (default - 2592).')
    parser.add_argument('-u', '--hist-upper', type=int, default=3072,
                        help='Histogram upper border value(default - 3072).')
    parser.add_argument('-b', '--hist-bins', type=int, default=30,
                        help='Histogram bins number (default - 30).')
    return parser.parse_args()


def split_by_groups(points):
    """Split all points by voltage groups."""
    v_groups = {}

    for p in tqdm(points, desc='grouping by voltage'):
        _, meta, _ = dfparser.parse_from_file(p, nodata=True)
        voltage = int(meta['external_meta']['HV1_value'])
        if voltage not in v_groups:
            v_groups[voltage] = []
        v_groups[voltage].append(p)
    return v_groups


def calc_centered_hist(point):
    amps = _lan_amps_f(point)
    hist, bins = np.histogram(
        amps, range=(ARGS.hist_lower, ARGS.hist_upper), bins=ARGS.hist_bins)


def calc_hist_avg(sets_data, point_index):
    """Calculate point average histogram in sets."""
    sets = list(sets_data.values())
    hist_avg = np.zeros(ARGS.bins)
    used = 0
    for set_ in sets:
        for point in set_:
            curr_index = int(
                set_[point]["meta"]["external_meta"]["point_index"])
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


if __name__ == "__main__":

    ARGS = __parse_args()
    sns.set_context("poster")
    sns.set_style(rc={"font.family": "monospace"})
    DEF_PALLETE = sns.color_palette()
    sns.set_palette(sns.cubehelix_palette(8))

    group_abs = path.join(ARGS.data_root, ARGS.fill)
    sets = natsorted(glob.glob(path.join(group_abs, "set_*[0-9]")))
    points = list(
        chain(
            *(natsorted(glob.glob(path.join(s, "p*(*s)*.df"))) for s in sets)))

    v_groups = split_by_groups(points)

    for voltage in v_groups:
        print(voltage, type(voltage))
        raise Exception

    with closing(Pool()) as pool:
        out = pool.map(get_spectrum, sets)

    sets_data = {path.relpath(s, group_abs): d for s, d in zip(sets, out)}

    hists_avg = []
    for idx in range(ARGS.ind_start, ARGS.ind_end):
        hists_avg.append(calc_hist_avg(sets_data, idx))

    draw_point_diffs(sets_data, 70)

    sets_points_chi2 = {}
    for set_idx in sets_data:
        x_values = []
        y_values = []
        for idx in range(ARGS.ind_start, ARGS.ind_end):
            chi2 = calc_chi_square(sets_data[set_idx], hists_avg[idx], idx)
            if chi2:
                x_values.append(idx)
                y_values.append(chi2)
        sets_points_chi2[set_idx] = {"x": x_values, "y": y_values, }

    _, axes = plt.subplots()

    axes.set_title(r'$\chi^2$ deviations for each point in %s' % (ARGS.fill))
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
            label=set_idx, s=30, c=palette[idx], edgecolors="face")
    axes.legend()
    plt.show()
