# -*- coding: utf-8 -*-
"""Effective count rate comparison for real and generated point.

Input:
- dist_path - path to generation distribution
- filename - path to uprocessed dataforge point (14 Kv)

Output:
- Effective count rate for real and generated data for several time thresholds.
"""
# pylint: disable-msg=R0914,C0413
from os import path

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from signal_utils.convert_utils import df_frames_to_events
from signal_utils.extract_utils import extract_amps_approx
from signal_utils.generation_utils import generate_df
from signal_utils.test_utils import _extract_real_frames

TIME_FILTER_THRESH = 6000
AMPL_THRESH = 496
AMPL_MAX = 4016
BINS = 55
DATA_ROOT = "/media/chernov/data/data/lan10_processed/"
# POINT_PATH = "2017_11/Fill_3/set_1/p81(30s)(HV1=14950).df"

# DATA_ROOT = "/media/chernov/data/data/lan10_processed_kotlin/"
POINT_PATH = "2017_11/Fill_2/set_1/p102(30s)(HV1=14000).df"


def prepare_point(time_s=5, freq=40e3, amp_thresh=700,
                  extract_func=extract_amps_approx,
                  dist_path=path.join(path.dirname(__file__),
                                      '../../signal_utils/data/dist.dat')):
    """Фильтрация заведомо недеткетируемых событий из точки."""
    meta, data, block_params = generate_df(time=time_s, threshold=amp_thresh,
                                           dist_file=dist_path, freq=freq)

    real_frames = _extract_real_frames(meta, data, block_params,
                                       frame_l=3, frame_r=5)

    amps_real = real_frames.max(axis=1)
    amps_detectable = amps_real[amps_real > amp_thresh]

    meta_, data_ = df_frames_to_events(meta, data, extract_func)
    meta_["detectable_events"] = amps_detectable.size

    return meta_, data_


def df_events_to_np(_, data):
    """Конвертация массива точек из формата df в numpy array."""
    point = dfparser.Point()
    point.ParseFromString(data)

    amps = []
    times = []

    channel = point.channels[0]
    for block in channel.blocks:
        amps.append(block.events.amplitudes)
        times.append((np.array(block.events.times) + block.time))

    amps = np.hstack(amps)
    times = np.hstack(times)

    return amps, times


def get_time_filtered_crs(meta, times, time_thresh_ns=6000):
    """Фильтрация по времени.

    Фильтрация близких друг к другу событий и поиск эффективной скорости
    счета.

    """
    deltas = (times[1:] - times[:-1])

    filtered = np.hstack([[0, ], np.where(
        np.logical_or(deltas < 0, deltas > time_thresh_ns))[0] + 1])
    filtered_out = np.setdiff1d(np.arange(times.size), filtered)

    print(deltas[filtered_out - 1])

    filtered_time_s = deltas[filtered_out - 1].sum() * np.double(1e-9)

    block_len_s = meta['params']['b_size'] / meta['params']['sample_freq']
    total_time_s = block_len_s * meta['params']['events_num']

    tau = np.double(total_time_s - filtered_time_s) / np.double(filtered.size) - \
        np.double(time_thresh_ns) * np.double(1e-9)

    print(time_thresh_ns, total_time_s, filtered_time_s, tau)
    count_rate = np.double(1) / tau

    return count_rate


def get_crs(meta, times, start=320 * 2, end=320 * 63, step=320 * 2):
    """Вычисление скоростей счета для разных отсечений по времени."""
    time_thrs = np.arange(start, end, step)
    crs = [get_time_filtered_crs(meta, times, time_thresh_ns=time_thr) for
           time_thr in time_thrs]
    return time_thrs, crs


def crs_compare_different_timesteps():
    """Сравнение графиков скоростей счета при разных шагах.

    Тест показывает причину возникновения "пилы" на графиках.
    """
    _, meta, data = dfparser.parse_from_file(path.join(DATA_ROOT, POINT_PATH))
    _, times = df_events_to_np(meta, data)

    deltas = times[1:] - times[:-1]
    hist, bins = np.histogram(deltas, bins=60, range=(-320 * 2, 320 * 63))
    _, axes2 = plt.subplots()
    axes2.step((bins[1:] + bins[:-1])/2, hist, where="mid")
    axes2.grid()
    # plt.show()

    time_thrs_640, crs_gen_640 = get_crs(meta, times)
    fig, axes = plt.subplots()
    fig.canvas.set_window_title('cr_steps_compare')

    axes.set_title("Effective Count Rate / Time threshold")
    axes.set_xlabel("Time Threshold, ns")
    axes.set_ylabel("Effective Count Rate, Hz")

    axes.plot(time_thrs_640, crs_gen_640, label="Step = 640 ns")
    axes.legend(loc=4)


def main():
    """Функция main.

    Сравнение эффективной скорости счета для реальных и сгенерированных данных.

    """
    crs_compare_different_timesteps()
    plt.show()


if __name__ == "__main__":
    seaborn.set_context("poster")
    main()
