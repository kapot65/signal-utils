# -*- coding: utf-8 -*-
"""Расчет гистограммы амплитуд отброшенных при временной фильтрации событий.

На входе должна быть одна обработанная точка в формате .df.
"""

from os import path

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from test_time_filtering import df_events_to_np

TIME_FILTER_THRESH = 6000
AMPL_THRESH = 496
AMPL_MAX = 4016
BINS = 55
DATA_ROOT = "/home/chernov/data/lan10_processed/"
POINT_PATH = "2017_11/Fill_3/set_1/p81(30s)(HV1=14950).df"


def filter_bad_events(meta, times, events, time_thresh_ns=6000):
    """Фильтрация и вывод массива событий, не прошедших временную фильтрацию.

    @meta - метаданные точки
    @times - массив времен точки
    @events - массив времен точки
    @time_thresh_ns - порог по времени в нс

    @return - массив отфильтрованных плохих событий

    """
    deltas = times[1:] - times[:-1]

    filtered = np.hstack([[0, ], np.where(deltas > time_thresh_ns)[0] + 1])
    filtered_out = np.setdiff1d(np.arange(times.size), filtered)

    return events[filtered_out]


def __main():
    _, meta, data = dfparser.parse_from_file(path.join(DATA_ROOT, POINT_PATH))
    amps, times = df_events_to_np(_, data)

    amps_filtered = filter_bad_events(meta, times, amps, TIME_FILTER_THRESH)

    _, hist_ax = plt.subplots()
    hist_ax.set_title("Rejected events histogramm")
    hist_ax.set_xlabel("Amplitude, ch")
    hist_ax.set_ylabel("Events, num")

    hist, bins = np.histogram(amps_filtered, range=(
        AMPL_THRESH, AMPL_MAX), bins=BINS)
    hist_ax.step((bins[1:] + bins[:-1]) / 2, hist,
                 where="mid", label="Rejected")
    hist, bins = np.histogram(amps, range=(AMPL_THRESH, AMPL_MAX), bins=BINS)
    hist_ax.step((bins[1:] + bins[:-1]) / 2, hist, where="mid", label="All")


if __name__ == "__main__":
    sns.set_context("poster")
    DEF_PALLETE = sns.color_palette()
    __main()
