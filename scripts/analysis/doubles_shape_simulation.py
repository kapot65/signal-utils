#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:16:55 2018.

@author: chernov

Скрипт получает симулированную форму спектра двойных наложений с помощью
генерации только двойных событий и последущей обработки созданного файла.

"""
# pylint: disable-msg=E1101,R0914

from os import path

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from signal_utils.convert_utils import df_frames_to_events
from signal_utils.extract_utils import extract_amps_approx2
from signal_utils.generation_utils import generate_df

TIME_SEC = 30
HIST_DIFF_RANGE = (0, 320 * 10)
HIST_DIFF_BINS_N = 40
HIST_EVS_RANGE = (0, 4096)
HIST_EVS_BINS_N = 64


def gen_doubles(ev_num, bl_time, max_spread=7 * 320e-9):
    """Функция распределения двойных событий.

    Алгоритм может создать и тройные наложения.

    @ev_num - количество событий
    @bl_time - время блока в секундах
    @max_spread - максимальное расстояние между сдвоенными событиями
    """
    times = np.zeros(ev_num + ev_num % 2, np.float128)
    times[0::2] = np.sort(np.random.uniform(
        0, bl_time, (ev_num + ev_num % 2) // 2))
    times[1::2] = times[0::2] + \
        np.random.uniform(0, max_spread, (ev_num + ev_num % 2) // 2)
    return times[:ev_num]


def __main():
    dist_file = path.abspath(path.join(path.dirname(
        __file__), "../../signal_utils/data/dist.dat"))
    meta, data, answer = generate_df(
        time=TIME_SEC, dist_file=dist_file,
        dist_time_func=gen_doubles, freq=10e3)
    ev_times = answer[1::2]
    ev_time_diffs = ev_times[1:] - ev_times[:-1]

    _, dist_hist_ax = plt.subplots()
    dist_hist_ax.set_title("Histogram of distances between generated events")
    dist_hist_ax.set_xlabel("Distance, ns")
    dist_hist_ax.set_ylabel("Events number")
    hist, bins = np.histogram(ev_time_diffs, bins=HIST_DIFF_BINS_N,
                              range=HIST_DIFF_RANGE)
    dist_hist_ax.step((bins[1:] + bins[:-1]) / 2, hist)

    _, data_ext = df_frames_to_events(meta, data, extract_amps_approx2)
    point = dfparser.Point()
    point.ParseFromString(data_ext)
    evs_recovered = []
    for channel in point.channels:
        for block in channel.blocks:
            evs_recovered.append(np.array(block.events.amplitudes))
    evs_recovered = np.hstack(evs_recovered)

    _, ampl_hist_ax = plt.subplots()
    ampl_hist_ax.set_title("Events amplutides histogram")
    ampl_hist_ax.set_xlabel("Amplitude, ch")
    ampl_hist_ax.set_ylabel("Events number")
    hist, bins = np.histogram(answer[0::2], bins=HIST_EVS_BINS_N,
                              range=HIST_EVS_RANGE)
    ampl_hist_ax.step((bins[1:] + bins[:-1]) / 2, hist, where='mid',
                      label="Generated events")
    hist, bins = np.histogram(evs_recovered, bins=HIST_EVS_BINS_N,
                              range=HIST_EVS_RANGE)
    ampl_hist_ax.step((bins[1:] + bins[:-1]) / 2, hist, where='mid',
                      label="Reconstructed events")
    ampl_hist_ax.legend()


if __name__ == "__main__":
    sns.set_context("poster")
    DEF_PALLETE = sns.color_palette()
    __main()
