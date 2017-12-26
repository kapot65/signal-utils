# -*- coding: utf-8 -*-
"""Скрипт предназначен для вычитания шумов из спектров.

Для определения шума берутся файлы с высоким запирающим напряжением.
На вход должны подаваться обработанные точки.

Input:
fill_dir - path to folder with processed Lan10-12PCI fill data
points_high - high voltage points wildcard
points_high_2 - high voltage points 2 wildcard
points_path - list of low HV points wildcard

Output:
Averaged energy spectrums with and w/o corrected tails for list of intresred
points.
"""
# pylint: disable-msg=R0914

from glob import glob
from os import path

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn


def get_amps(filepath: str) -> np.ndarray:
    """Extract amplitudes from processed file."""
    _, _, data = dfparser.parse_from_file(filepath)
    p_high = dfparser.Point()
    p_high.ParseFromString(data)

    amps = np.hstack([list(block.events.amplitudes)
                      for block in p_high.channels[0].blocks])
    return amps


def _main():
    fill_dir = "/home/chernov/data_processed/2017_05/Fill_3/"
    points_high = "*/p1(30s)(HV1=18600).df"
    points_high_2 = "*/p2(30s)(HV1=18550).df"
    points_high_filepaths = glob(path.join(fill_dir, points_high)) + \
        glob(path.join(fill_dir, points_high_2))

    points_path = ["*/p0(30s)(HV1=16000).df",
                   "*/p36(30s)(HV1=17000).df",
                   "*/p80(30s)(HV1=15000).df",
                   "*/p102(30s)(HV1=14000).df"]

    hist_bins_num = 50
    hist_range = (700, 1500)

    amps_all = np.hstack([get_amps(fp) for fp in points_high_filepaths])
    hist_n, bins_n = np.histogram(amps_all, hist_bins_num, range=hist_range)
    hist_n = hist_n/len(points_high_filepaths)

    # plt.step((bins_n[1:] + bins_n[:-1])/2, hist_n)

    fig_all, ax_all = plt.subplots()
    ax_all.set_yscale("log", nonposx='clip')
    fig_all.canvas.set_window_title("Corrected points tails")
    fig_all.suptitle("Corrected points tails")
    ax_all.set_xlabel("Bins, ch")

    for point_path in points_path:
        points = glob(path.join(fill_dir, point_path))
        amps_p = np.hstack([get_amps(fp) for fp in points])
        hist_p, _ = np.histogram(amps_p, hist_bins_num, range=hist_range)
        hist_p = hist_p/len(points)
        hist_corr = hist_p - hist_n

        fig, axes = plt.subplots()
        fig.canvas.set_window_title(point_path)
        fig.suptitle("Tail with removed noise")
        axes.set_title("File - %s." % (point_path,))
        axes.set_xlabel("Bins, ch")
        axes.set_xlim(*hist_range)
        axes.set_yscale("log", nonposx='clip')
        axes.step((bins_n[1:] + bins_n[:-1])/2, hist_p, label="Original")
        axes.step((bins_n[1:] + bins_n[:-1])/2, hist_corr, label="Corrected")
        ax_all.step((bins_n[1:] + bins_n[:-1])/2, hist_corr, label=point_path)
        axes.legend()

    ax_all.legend()


if __name__ == "__main__":
    seaborn.set_context("poster")
    _main()
