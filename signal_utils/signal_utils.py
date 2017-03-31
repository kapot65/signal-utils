import collections

import numpy as np
from tqdm import tqdm

from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

from gdrive_utils import load_dataset, get_points_drom_drive
from generation_utils import gen_multiple

def calc_center(ev, amp_arg, step=2):
    """
      Параболическая аппроксимация центра пика
      
      @ev - массив события
      @amp_arg - положение пика
      
    """
    x1 = amp_arg
    y1 = ev[x1]
    x2 = x1 + step
    y2 = ev[x2]
    x3 = x1 - step
    y3 = ev[x3]

    a = (y3 - (x3*(y2-y1) + x2*y1 - x1*y2)/(x2 - x1))/\
    (x3*(x3 - x1 - x2) + x1*x2)
    b = (y2 - y1)/(x2 - x1) - a*(x1 + x2)
    c = (x2*y1 - x1*y2)/(x2 - x1) + a*x1*x2
    
    x0 = -b/(2*a)
    y0 = a*x0**2 + b*x0 + c
    
    return y0, x0 


def get_peaks(ev, threshold):
    """
      Выделение пиков из блока
      Условия отбора пика:
      - пик должен быть больше порогового значения
      - пик должен быть локальным экстремумом
      
      @ev - массив события
      @threshold - порог
      
    """
    extremas = argrelextrema(ev, np.greater_equal)
    points = extremas[0][ev[extremas] >= threshold]
    
    if len(points) >= 2:
        planes = np.where(points[1:] - points[:-1] == 1)[0] + 1
        points = np.delete(points, planes)
    
    return points


def extract_events_fit(ev, threshold):
    """
      Последовательное выделение событий из блока
      
      В блоке последовательно фитируются событие. Функция следующего события 
      складывается с фунциями уже определенных ранее событий.
      
      Данный метод работает потенциально быстрее и стабильнее по сравнению с 
      одновременным фитированием всех событий в блоке, выдает большую ошибку 
      при выделении близко наложенных событий (событие накладывается на хвост 
      предыдущего события).
      
      @ev - массив события
      @threshold - порог
      @return параметры событий в формате [amp1, pos1, amp2, pos2, ...]
      
    """
    points = get_peaks(ev, threshold)
    values = np.array([], np.float32)

    for i, point in enumerate(points):
        y0, x0 = calc_center(ev, point)

        full_func = lambda x, a, p: gen_multiple(x, a, p, *values)

        popt, pcov = curve_fit(full_func, np.arange(len(ev)), ev, p0=[y0, x0])
        values = np.hstack([values, popt])

    return values


def extract_events_fit_all(ev, threshold):
    """
      Последовательное выделение событий из блока
      Все события фитируются одновременно.
      
      @ev - массив события
      @threshold - порог
      @return параметры событий в формате [amp1, pos1, amp2, pos2, ...]
      
    """
    points = get_peaks(ev, threshold)

    values = np.hstack([calc_center(ev, point) for point in points])
    full_func = lambda x, *values: gen_multiple(x, *values)
    popt, pcov = curve_fit(full_func, np.arange(len(ev)), ev, p0=list(values))

    return popt


def extract_from_dataset(dataset, threshold=700, 
                         area_l=50, area_r=100):
    """
      Получение блоков из датасета [Desperated]
      @dataset - датасет
      @threshold - порог zero-suppression
      @area_l - левая область zero-suppression
      @area_r - правая область zero-suppression
      @return - вырезанные блоки
      
    """
    frames = [] # кадры из точки
    for i in range(dataset.params["events_num"]):
        try:
            frames.append(dataset.get_event(i)["data"])
        except:
            break

    blocks = [] # вырезанные из кадра события
    for frame in frames:
        peaks = np.where(frame > threshold)[0]
        dists = peaks[1:] - peaks[:-1]
        gaps = np.append(np.array([0]), np.where(dists > area_r)[0] + 1)
        for gap in range(0, len(gaps) - 1):
            l_bord = peaks[gaps[gap]] - area_l
            r_bord = peaks[gaps[gap + 1] - 1] + area_r
            blocks.append(frame[l_bord: r_bord])
    return blocks


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
      [Desperated]
      Получение из файла google drive частоты оцифровки
      @points - таблица с точками
      @idx - индекс точки, из которой берется частота оцифровки
      
    """
    header = points.as_matrix()[idx]
    dataset = load_dataset(header[0], header[1], header[2])
    return dataset.params["sample_freq"]**-1


def extract_algo(data, threshold=700):
    """
      Выделенние события из блока. Способ, предложенный Пантуевым В.С.
      @data - массив кадра
      @threshold - порог
      @return - индекс первого бина, превысившего порог,
      индекс перегиба, индекс первого отрицательного бина после перегиба
      
    """
    deriv = data[1:] - data[:-1]
    first_greater = np.argmax(data >= threshold)
    extremum = first_greater + np.argmax(deriv[first_greater-1:] < 0) - 1
    first_negative = first_greater + np.argmax(data[first_greater:] < 0)
    return first_greater, extremum, first_negative


def extract_events(data, threshold=700):
    """
      Выделенние нескольких событий из блока. 
      Способ, предложенный Пантуевым В.С.
      @data - массив кадра
      @threshold - порог
      @return - [индекс первого бина, превысившего порог,
      индекс перегиба, индекс первого отрицательного бина после перегиба]
      
    """
    events = []
    offset = 0
    while(True):
        left, center, right = extract_algo(data[offset:], threshold)
        if not left:
            break
        left += offset
        center += offset
        r_buf = right
        right += offset
        offset += r_buf
        events.append([left, center, right])
        
    return events


def test_functions():
    
    import matplotlib.pyplot as plt
    
    from draw_utils import draw_event, plot_event, plot_multiple_events
    
    points = get_points_drom_drive()
    points.sort_values('time', ascending=False)
    
    bin_sec = get_bin_sec(points, 0)
    blocks = get_blocks(points, 0)
    
    fig, ax = plt.subplots()
    
    draw_event(blocks[0], 0, fig, ax)
    
    fig, ax = plt.subplots()
    event = extract_algo(blocks[0])
    plot_event(blocks[0], 0, event, ax, threshold=700, bin_sec=bin_sec)
    
    fig, ax = plt.subplots()
    events = extract_events(blocks[0])
    plot_multiple_events(blocks[0], events, ax, threshold=700, bin_sec=bin_sec)
