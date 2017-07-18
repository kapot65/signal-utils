"""Misc utils for point processing."""

import collections

import numpy as np
from scipy.signal import argrelextrema
from tqdm import tqdm

from gdrive_utils import load_dataset


def extract_from_dataset(dataset, threshold=700, area_l=50, area_r=100):
    """Получение блоков из датасета [Desperated].

    @dataset - датасет
    @threshold - порог zero-suppression
    @area_l - левая область zero-suppression
    @area_r - правая область zero-suppression
    @return - вырезанные блоки

    """
    frames = []  # кадры из точки
    for i in range(dataset.params["events_num"]):
        try:
            frames.append(dataset.get_event(i)["data"])
        except Exception:
            break

    blocks = []  # вырезанные из кадра события
    for frame in frames:
        peaks = np.where(frame > threshold)[0]
        dists = peaks[1:] - peaks[:-1]
        gaps = np.append(np.array([0]), np.where(dists > area_r)[0] + 1)
        for gap in range(0, len(gaps) - 1):
            l_bord = peaks[gaps[gap]] - area_l
            r_bord = peaks[gaps[gap + 1] - 1] + area_r
            blocks.append(frame[l_bord: r_bord])
    return blocks


def extract_frames(ev_data, peaks, frame_l, frame_r):
    """Выделение событий из кадра по пикам.

    @ev_data - массив кадра
    @peaks - массив положений пиков в кадре в бинах
    @frame_l - размер обрезания события слева от пика
    @frame_r - размер обрезания события справа от пика
    @return - Двумерный массив событий.

    """
    shape = (len(peaks), frame_l + frame_r)
    frames_block = np.zeros(shape, np.int16)
    for j, peak in enumerate(peaks):
        if peak >= frame_l and peak < len(ev_data) - frame_r:
            frames_block[j] = ev_data[int(peak) - frame_l: int(peak) + frame_r]
    return frames_block


def calc_center(event, amp_arg, step=2):
    """Параболическая аппроксимация центра пика.

    @event - массив события
    @amp_arg - положение пика

    """
    x1 = amp_arg
    y1 = event[x1]
    x2 = x1 + step
    y2 = event[x2]
    x3 = x1 - step
    y3 = event[x3]

    a = (y3 - (x3*(y2-y1) + x2*y1 - x1*y2)/(x2 - x1))/ \
      (x3*(x3 - x1 - x2) + x1*x2)
    b = (y2 - y1)/(x2 - x1) - a*(x1 + x2)
    c = (x2*y1 - x1*y2)/(x2 - x1) + a*x1*x2

    x0 = -b/(2*a)
    y0 = a*x0**2 + b*x0 + c

    return y0, x0


def get_peaks(event, threshold):
    """Выделение пиков из блока.

    Условия отбора пика:
    - пик должен быть больше порогового значения
    - пик должен быть локальным экстремумом

    @event - массив события
    @threshold - порог

    """
    extremas = argrelextrema(event, np.greater_equal)[0]
    points = extremas[event[extremas] >= threshold]

    if len(points) >= 2:
        planes = np.where(points[1:] - points[:-1] <= 3)[0] + 1
        points = np.delete(points, planes)

    return points


def get_blocks(points, idxs, threshold=700, area_l=50, area_r=100):
    """
      [Desperated]
      Выделение из файлов google drive блоков алгоритмом zero-suppression
      @points - таблица с точками
      @idxs - индекс или массив индексов интересующих точек
      @threshold - порог zero-suppression
      @area_l - левая область zero-suppression
      @area_r - правая область zero-suppression
      @return - вырезанные блоки

    """
    blocks = []

    def add_blocks(blocks, idx):
        header = points.as_matrix()[idx]
        dataset = load_dataset(header[0], header[1], header[2])
        blocks += extract_from_dataset(dataset, threshold, area_l, area_r)

    if isinstance(idxs, collections.Iterable):
        for i in tqdm(idxs):
            add_blocks(blocks, i)
    else:
        add_blocks(blocks, idxs)

    return blocks


def get_bin_sec(points, idx):
    """
      Получение из файла google drive частоты оцифровки [Desperated].
      
      @points - таблица с точками
      @idx - индекс точки, из которой берется частота оцифровки

    """
    header = points.as_matrix()[idx]
    dataset = load_dataset(header[0], header[1], header[2])
    return dataset.params["sample_freq"]**-1
