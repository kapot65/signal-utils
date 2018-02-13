# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 19:49:34 2017

@author: kapot
"""

import io
from datetime import datetime

import numpy as np
from dfparser import dump_to_rsb
from dfparser.df_data.def_values import DEF_RSH_PARAMS
from random_custom_pdf import rand_custom
from scipy.interpolate import interp1d
from tqdm import tqdm

from convert_utils import rsb_to_df


def generate_noise(x, base_freq=3125000.0):
    """
      @x - координаты в секундах
      @base_freq - частота, при которой подбирались параметры сигнала
      @return - шум для заданных координат

    """
    x_min = int(x.min() * base_freq - 2)
    x_max = int(x.max() * base_freq + 2)

    x_orig = np.arange(x_min, x_max)

    noise = np.random.randint(-20, 20, size=len(x_orig))
    c1 = np.convolve(noise, np.full((10), 0.315), 'same')
    c2 = np.convolve(c1, np.full((3), 1 / 3), 'same')
    c3 = np.convolve(c2, np.full((3), 1 / 3), 'same') + 2.5

    noise_conv = np.round(c3)

    interp = interp1d(x_orig / base_freq, noise_conv, kind='nearest')

    return interp(x) * 16


def gen_signal(x, ampl, pos,
               sigma=0.34156352,
               tail_amp=0.37015974,
               tail_factor=2.96038975,
               p=2.20563286,
               s=0.0525933,
               base_freq=3125000.0):
    """
      Генерация сигнала

      Все аргументы, кроме @ampl и @pos подобраны под реальный сигнал.
      Рекомендуется оставить из значения по умолчанию

      @x - координаты в секундах
      @ampl - амплитуда генерируемого сигнала
      @pos - положение генерируемого сигнала в секундах
      @sigma - растяжение сигнала
      @tail_amp - отношение амлитуды выброса к амплитуде сигнала
      @tail_factor - степень резкости пика сигнала
      @p - степень гауссиана амплитуда
      @s - коэффициент растяжения выброса
      @base_freq - частота, при которой подбирались параметры сигнала
      @return - сигнал для заданных координат

    """

    def gauss(x): return np.exp(
        (-1 / 2) * np.power((np.abs(sigma * x * base_freq)), p))

    def gauss_rev(y): return (
        (1 / sigma) * np.power(-2 * np.log(y), 1 / p) / base_freq)

    def spike(x): return (1 / (1 + 2 * x * base_freq * s)**tail_factor - 1.0) * \
        np.exp(-x * base_freq * s)

    spike_offset = gauss_rev(0.1) + pos
    spike_x = x - spike_offset
    spike_x[spike_x < 0] = 0

    return ((gauss(x - pos) + spike(spike_x) * tail_amp) * ampl)


def gen_multiple(x, *args, l_size=10 / 3125000.0, r_size=100 / 3125000.0):
    """
      Функция нескольких событий. Для ускорения работы генерируемый сигнал
      обрезается после l_size от пика слева и на r_size от пика справа.

      @x - упорядоченная сетка
      @a - амплитуда текущего события
      @p - положение пика текущего события
      @args - последовательно указанные амплитуды
      @l_size - граница события слева
      @r_size - граница события справа
      и положения пиков остальных событий

    """
    assert(not len(args) % 2)

    y = np.zeros(len(x), np.float32)

    l_vals = np.searchsorted(x, [args[i + 1] - l_size for
                                 i in range(0, len(args), 2)])
    r_vals = np.searchsorted(x, [args[i + 1] + r_size for
                                 i in range(0, len(args), 2)])

    for i in range(0, len(args), 2):
        n = i // 2
        y[l_vals[n]:r_vals[n]] += gen_signal(x[l_vals[n]:r_vals[n]],
                                             args[i], args[i + 1])

    return y


def gen_raw_block(freq: float=12e+3,
                  sample_freq: float=3125000.0,
                  b_size: int=1048576,
                  dist_file: str=None,
                  dist_time_func=None,
                  min_amp: int=500,
                  max_amp: int=7000):
    """
      Генерация блока данных для тестирования
      @freq - частота событий (на секунду)
      @sample_freq - частота оцифровки Гц
      @b_size - размер блока в бинах
      @dist_file - файл с гистограммой распределения (см data/dist.dat).
      Перекрывает параметры min_amp и max_amp.
      @dist_time_func - Функция, генерирующая времена событий
      (event_num: int, max_time: float) -> np.ndarray(np.float128).
      Если не указано, будет использовано равномерное распределение.
      @max_amp - максимальная амплитуда события
      @min_amp - минимальная амплитуда события
      @return - [data, [amp1, pos1, amp2, pos2, ...]]

    """
    events = int(freq * (b_size / sample_freq))

    params = np.zeros(events * 2, np.float128)

    if dist_time_func is None:
        params[1::2] = np.sort(np.random.uniform(
            0,  b_size / sample_freq, events))
    else:
        params[1::2] = dist_time_func(events, b_size / sample_freq)\
            .astype(np.float128)

    if not dist_file:
        params[0::2] = np.random.uniform(min_amp, max_amp, events)
    else:
        if gen_raw_block.file != dist_file:
            data = np.fromfile(dist_file, sep='\t')
            data = data.reshape((len(data) // 2, 2))
            gen_raw_block.file = dist_file
            gen_raw_block.x = data[:, 0]
            gen_raw_block.y = data[:, 1]

        params[0::2] = rand_custom(gen_raw_block.x, gen_raw_block.y, (events,))

    x = np.arange(b_size) / sample_freq
    data = gen_multiple(x, *params)

    return data + generate_noise(x), params


gen_raw_block.file = None
gen_raw_block.x = None
gen_raw_block.y = None


def generate_multiple_blocks(freq: float=12e+3,
                             sample_freq: float=3125000.0,
                             b_size: int=1048576,
                             dist_file: str=None,
                             dist_time_func=None,
                             min_amp: int=500,
                             max_amp: int=7000,
                             time: float=30.0,
                             block_write_time: float=0.05):
    """
        Генерация множественных блоков

        @freq - частота событий (на секунду)
        @sample_freq - частота оцифровки Гц
        @b_size - размер блока в бинах
        @dist_file - файл с гистограммой распределения (см data/dist.dat).
        Перекрывает параметры min_amp и max_amp.
        @dist_time_func - Функция, генерирующая времена событий
        (event_num: int) -> np.ndarray(np.float128). Если не указано, будет
        использовано равномерное распределение.
        @max_amp - максимальная амплитуда события
        @min_amp - минимальная амплитуда события
        @time - время набора в секундах
        @block_write_time - cкорость записи блока; определяет мертвое время
        (http://elog.mass.inr.ru/nu-mass/319)
        @return - [data, [amp1, pos1, amp2, pos2, ...], start_times];
        здесь data - данные блоков (block_num, block_size)
        [amp1, pos1, amp2, pos2, ...] - массив событий:
            amp1 - амплитуда события в каналах
            pos1 - абсолютное положение события в наносекундах с начала первого
            блока
            start_times - массив в наносекундах стартовых времен для
            каждого блока

    """
    block_time = ((b_size / sample_freq) + block_write_time)
    blocks_num = round(time / block_time + 0.5)

    start_times = np.array([block_time * i for i in range(blocks_num)])
    block_params = []

    data = np.zeros((blocks_num, b_size), np.int16)

    for i in tqdm(range(blocks_num), desc="generating blocks"):
        out = gen_raw_block(freq, sample_freq, b_size,
                            dist_file, dist_time_func, min_amp, max_amp)
        data[i] = out[0]
        out[1][1::2] += start_times[i]
        out[1][1::2] *= 1e+9

        block_params.append(out[1])

    block_params = np.hstack(block_params)

    return data, block_params, start_times * 1e+9


def generate_rsb(freq: float=12e+3,
                 sample_freq: float=3125000.0,
                 b_size: int=1048576,
                 dist_file: str=None,
                 dist_time_func=None,
                 min_amp: int=500,
                 max_amp: int=7000,
                 time: float=30.0,
                 block_write_time: float=0.05):
    """
      Генерация файла rsb с желаемой частотой событий

      @freq - частота событий (на секунду)
      @sample_freq - частота оцифровки Гц
      @b_size - размер блока в бинах
      @dist_file - файл с гистограммой распределения (см data/dist.dat).
      Перекрывает параметры min_amp и max_amp.
      @dist_time_func - Функция, генерирующая времена событий
      (event_num: int) -> np.ndarray(np.float128). Если не указано, будет
      использовано равномерное распределение.
      @max_amp - максимальная амплитуда события
      @min_amp - минимальная амплитуда события
      @time - время набора в секундах
      @block_write_time - cкорость записи блока; определяет мертвое время
      (http://elog.mass.inr.ru/nu-mass/319)
      @return - [file_data, [amp1, pos1, amp2, pos2, ...]];
      здесь file_data - данные сгенерированного rsb файла
      [amp1, pos1, amp2, pos2, ...] - массив событий:
            amp1 - амплитуда события в каналах
            pos1 - абсолютное положение события в наносекундах с начала первого
            блока

    """

    data, block_params, \
        start_times = generate_multiple_blocks(freq, sample_freq, b_size,
                                               dist_file, dist_time_func,
                                               min_amp, max_amp, time,
                                               block_write_time)

    time_start = datetime.now().timestamp() * 1e+9

    rsh_params = DEF_RSH_PARAMS
    rsh_params['aquisition_time'] = time * 1000
    rsh_params['sample_freq'] = sample_freq

    rsb = dump_to_rsb(rsh_params,
                      time_start + start_times,
                      data)
    block_params[1::2] += time_start

    return rsb, block_params


def generate_df(threshold: int=500,
                area_l: int=50,
                area_r: int=100,
                freq: float=12e+3,
                sample_freq: float=3125000.0,
                b_size: int=1048576,
                dist_file: str=None,
                dist_time_func=None,
                min_amp: int=500,
                max_amp: int=7000,
                time: float=30.0,
                block_write_time: float=0.05):
    """
      Генерация файла формата df

      @freq - частота событий (на секунду)
      @sample_freq - частота оцифровки Гц
      @b_size - размер блока в бинах
      @dist_file - файл с гистограммой распределения (см data/dist.dat).
      Перекрывает параметры min_amp и max_amp.
      @dist_time_func - Функция, генерирующая времена событий
      (event_num: int) -> np.ndarray(np.float128). Если не указано, будет
      использовано равномерное распределение.
      @max_amp - максимальная амплитуда события
      @min_amp - минимальная амплитуда события
      @time - время набора в секундах
      @block_write_time - cкорость записи блока; определяет мертвое время
      (http://elog.mass.inr.ru/nu-mass/319)
      @return - [meta, data, [amp1, pos1, amp2, pos2, ...]];
      Здесь:
          meta - метаданные сгенерированного df файла
          data - бинарные данные сгенерированного df файла
          [amp1, pos1, amp2, pos2, ...] - массив событий:
            amp1 - амплитуда события в каналах
            pos1 - абсолютное положение события в наносекундах с начала первого
            блока

    """

    rsb, block_params = generate_rsb(freq, sample_freq, b_size, dist_file,
                                     dist_time_func, min_amp,  max_amp, time,
                                     block_write_time)

    file = io.BytesIO(rsb)
    meta, data = rsb_to_df({'simulation': True,
                            'freq': freq,
                            'min_amp': min_amp,
                            'max_amp': max_amp,
                            'block_write_time': block_write_time},
                           file, threshold,
                           area_l, area_r)

    return meta, data, block_params
