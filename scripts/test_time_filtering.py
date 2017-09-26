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
import sys

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn

MAIN_DIR = path.abspath(path.join(path.dirname(__file__), '..'))
if MAIN_DIR not in sys.path:
    sys.path.append(MAIN_DIR)
del MAIN_DIR

from signal_utils.convert_utils import df_frames_to_events
from signal_utils.extract_utils import extract_amps_approx
from signal_utils.generation_utils import generate_df
from signal_utils.test_utils import _extract_real_frames


def prepare_point(time_s=5, freq=40e3, amp_thresh=700,
                  extract_func=extract_amps_approx,
                  dist_path=path.join(path.dirname(__file__),
                                      '../signal_utils/data/dist.dat')):
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


def filter_times(meta, times, time_thresh_ns=6000):
    """Фильтрация по времени.

    Фильтрация близких друг к другу событий и поиск эффективной скорости
    счета.

    """
    deltas = times[1:] - times[:-1]

    filtered = np.hstack([[0, ], np.where(deltas > time_thresh_ns)[0] + 1])
    filtered_out = np.setdiff1d(np.arange(times.size), filtered)

    filtered_time_s = deltas[filtered_out - 1].sum()*np.double(1e-9)

    block_len_s = meta['params']['b_size']/meta['params']['sample_freq']
    total_time_s = block_len_s * meta['params']['events_num']

    tau = np.double(total_time_s - filtered_time_s)/np.double(filtered.size) - \
        np.double(time_thresh_ns)*np.double(1e-9)
    count_rate = np.double(1)/tau

    return count_rate


def get_crs(meta, times):
    """Вычисление скоростей счета для разных отсечений по времени."""
    time_thrs = np.arange(500, 5000, 1000)
    crs = [filter_times(meta, times, time_thresh_ns=time_thr) for
           time_thr in time_thrs]
    return time_thrs, crs


def main():
    """Функция main.

    Сравнение эффективной скорости счета для реальных и сгенерированных данных.

    """
    freq = 42e3
    amp_thresh = 750
    time_s = 30
    dist_path = path.join(path.dirname(__file__),
                          '../signal_utils/data/dist.dat')

    meta_gen, data_gen = prepare_point(time_s=time_s, freq=freq,
                                       amp_thresh=amp_thresh,
                                       extract_func=extract_amps_approx,
                                       dist_path=dist_path)

    _, times_gen = df_events_to_np(meta_gen, data_gen)
    time_thrs, crs_gen = get_crs(meta_gen, times_gen)

    filename = '/home/chernov/data/p102(30s)(HV1=14000).df'
    _, meta_real, data_real = dfparser.parse_from_file(filename)
    meta_real_, data_real_ = df_frames_to_events(meta_real, data_real,
                                                 extract_amps_approx,
                                                 correct_time=True)

    _, times_real = df_events_to_np(meta_real_, data_real_)
    time_thrs, crs_real = get_crs(meta_real_, times_real)

    _, axes = plt.subplots()
    axes.plot(time_thrs, crs_gen, 'ro', label="Generated")
    axes.plot(time_thrs, crs_real, 'bo', label="Real")
    axes.legend()


if __name__ == "__main__":
    seaborn.set_context("poster")
    main()
