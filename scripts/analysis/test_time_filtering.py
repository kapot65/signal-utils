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
    deltas = times[1:] - times[:-1]

    filtered = np.hstack([[0, ], np.where(deltas > time_thresh_ns)[0] + 1])
    filtered_out = np.setdiff1d(np.arange(times.size), filtered)

    filtered_time_s = deltas[filtered_out - 1].sum() * np.double(1e-9)

    block_len_s = meta['params']['b_size'] / meta['params']['sample_freq']
    total_time_s = block_len_s * meta['params']['events_num']

    tau = np.double(total_time_s - filtered_time_s) / np.double(filtered.size) - \
        np.double(time_thresh_ns) * np.double(1e-9)
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
    freq = 42e3
    amp_thresh = 750
    time_s = 1.0
    dist_path = path.join(path.dirname(__file__),
                          '../../signal_utils/data/dist.dat')

    from signal_utils.extract_utils import extract_simple_amps

    meta_gen, data_gen = prepare_point(time_s=time_s, freq=freq,
                                       amp_thresh=amp_thresh,
                                       extract_func=extract_simple_amps,
                                       dist_path=dist_path)

    _, times_gen = df_events_to_np(meta_gen, data_gen)

    time_thrs_640, crs_gen_640 = get_crs(meta_gen, times_gen)
    time_thrs_1000, crs_gen_1000 = get_crs(meta_gen, times_gen, step=1000)
    time_thrs_500, crs_gen_500 = get_crs(meta_gen, times_gen, step=400)

    fig, axes = plt.subplots()

    fig.canvas.set_window_title('cr_steps_compare')

    axes.set_title("Effective Count Rate / Time threshold")
    axes.set_xlabel("Time Threshold, ns")
    axes.set_ylabel("Effective Count Rate, Hz")

    axes.plot(time_thrs_640, crs_gen_640, label="Step = 640 ns")
    axes.plot(time_thrs_500, crs_gen_500, label="Step = 400 ns")
    axes.plot(time_thrs_1000, crs_gen_1000, label="Step = 1000 ns")

    axes.legend(loc=4)


def main():
    """Функция main.

    Сравнение эффективной скорости счета для реальных и сгенерированных данных.

    """
    crs_compare_different_timesteps()


if __name__ == "__main__":
    seaborn.set_context("poster")
    main()
