"""Модуль для тестирования параметров алгоритмов."""

from datetime import datetime
from os import path

from dfparser import Point
import numpy as np
from tqdm import tqdm

from signal_utils.draw_utils import draw_metrics
from signal_utils.extract_utils import extract_frames
from signal_utils.generation_utils import generate_df


def _extract_real_frames(meta, data, block_params, frame_l=15, frame_r=25):
    """Выделение кадров реальных событий из сгенерированного df файла.

    @meta - Метаданные df файла.
    @data - Бинарные данные df файла.
    @block_params - Параметры событий.
    @frame_l - Размер сохраняемого кадра в бинах слева.
    @frame_r -Размер сохраняемого кадра в бинах справа.
    @retutn - Массив кадров.
    """
    bin_time = 1e+9 / meta['params']['sample_freq']
    times = block_params[1::2]

    point = Point()
    point.ParseFromString(data)

    frames = np.zeros((times.size, frame_l + frame_r), np.float32)
    for channel in point.channels:
        for block in channel.blocks:
            for event in block.frames:
                event_data = np.frombuffer(event.data, dtype=np.int16)
                ev_l_time = (block.time + event.time)
                ev_r_time = ev_l_time + len(event_data)*bin_time

                l_index = np.searchsorted(times, ev_l_time)
                r_index = np.searchsorted(times, ev_r_time)

                peaks = [int(round((times[idx] - ev_l_time)/bin_time)) for
                         idx in range(l_index, r_index)]

                frames[l_index:r_index] = extract_frames(event_data, peaks,
                                                         frame_l, frame_r)

    return np.vstack(frames)


def _calc_metrics(amps_real, pos_real,
                  amps_extracted, pos_extracted,
                  singles_extracted, max_pos_err):
    """Вычисление метрик.

    В процессе тестирования определяются:
        - время работы алгоритма
        - количество ложных срабатываний
        - количество пропущеных событий
        - отношение классифицированных наложенных событий к реальным
        (реальным наложенным событием является выделенное событие, которое
         соответсвует нескольким реальным событиям)

    Алгоритм расчета метрик:
        1. Задание соответсвия между реальными и выделенными событиями. Для
        этого для каждого события из массива реальных данных ищется ближайшее
        по положению событие из массива выделенных событий.
        2. Подсчет пропущенных событий. Если расстояние между реальным и
        восстановленным событием больше порога (max_pos_err) - событие
        помечается как пропущенное.
        3. Подсчет ложных срабатываний. Ложным срабатыванием считается
        событие из массива восстановленных данных у которого нет ни одного
        соответсвия с реальным событием.
        4. Подсчет реальных наложений. Реальным наложением является событие
        из восстановленного массива, которое соответсвуюет нескольким
        событиям из рельных данных.
        Реальные наложения затем сравниваются с классифицированными
        наложениями, полученными алгоритмом выделения.
        5. Вычисление ошибок восстановления ампитуд (как наложенных так и
        одинарных событий).

    @amps_real -- массив амплитуд реальных событий
    @pos_real -- массив положений реальных событий в наносекундах
    @amps_extracted -- массив амплитуд выделенных событий
    @pos_extracted -- массив амплитуд выделенных положений в наносекундах
    @singles_extracted -- массив классификаций выделенных событий по
    наложенности
    @max_pos_err -- максимальное допустимое отличие положений реального
    события и соответсвующего ему выделенного события в наносекундах

    @return -- Рассчитанные метрики. Пример метрики:
        {
            'amps_extracted': array([ 2463.82592773, ...,  5950.55712891],
                                    dtype=float32),
            'amps_real': array([ 1547.03479004, ...,963.0748291 ],
                               dtype=float32),
            'doubles_detected': array([], dtype=int32),
            'doubles_real': array([  33,  ..., 3719], dtype=int64),
            'false_negatives': array([   0, ..., 4025]),
            'false_positives': array([2593, 2742]),
            'pos_extracted': array([  6.14400000e+04, ..., 3.35321920e+08],
                                   dtype=float32),
            'pos_real': array([  1.50009424e+04, ..., 3.35438528e+08],
                              dtype=float32),
            'real_detected_transitions': array([-1, ..., -1], dtype=int64),
            'singles_extracted': array([ True, ...,  True], dtype=bool),
            'total_detected': 3860,
            'total_real': 4026
         }
    Здесь:
        doubles_detected -- индексы событий, классифицированных как
        наложенные
        doubles_real -- индексы реальных наложенных событий.
        false_negatives -- индексы неопределенных классификатором событий
        false_positives -- индексы ложно определенных событий
        real_detected_transitions -- Массив соответсвия между рельными и
        выделенными событиями. Если соответсвия нету - индекс
        прирванивается -1.
        total_detected -- общее количество выделенных событий
        total_real -- общее количество реальных событий

    """
    metrics = {}
    metrics["total_real"] = len(amps_real)
    metrics["total_detected"] = len(amps_extracted)

    STEP = 1000
    global idxs_raw
    idxs_raw = np.full(pos_real.shape, -1, np.int)
    for i in tqdm(range(0, len(pos_real), STEP), desc="indexing"):
        pos_real_bl = pos_real[i:i+STEP]
        min_ind = np.searchsorted(pos_extracted, pos_real_bl[0] - max_pos_err)
        max_ind = np.searchsorted(pos_extracted, pos_real_bl[-1] + max_pos_err)
        pos_extr_bl = pos_extracted[min_ind:max_ind]

        if len(pos_real_bl) and len(pos_extr_bl):
            dists_ = np.abs(np.subtract.outer(pos_extr_bl, pos_real_bl))

            vals, idxs_rev, counts = np.unique(dists_.argmin(0),
                                               return_inverse=True,
                                               return_counts=True)
            mults = np.where(counts > 1)[0]
            for idx in mults:
                cols = np.where(idxs_rev == idx)[0]
                argsort_idxs = cols[np.argsort(dists_[:, cols].min(axis=0))]
                for j in range(argsort_idxs.size):
                    point = dists_[:, argsort_idxs[j]].argmin()
                    dists_[point, argsort_idxs[j]] += 320*j

            idxs_raw[i: i + STEP] = dists_.argmin(0) + min_ind

    idxs_raw = np.hstack(idxs_raw)

    dists = np.abs(pos_real - pos_extracted[idxs_raw])
    idxs_raw[dists > max_pos_err] = -1

    single_idxs, counts = np.unique(idxs_raw, return_counts=True)

    metrics["real_detected_transitions"] = idxs_raw
    metrics["false_negatives"] = np.arange(len(idxs_raw))[idxs_raw == -1]
    metrics["false_positives"] = np.setdiff1d(np.arange(len(pos_extracted)),
                                              single_idxs)

    doubles_real = single_idxs[counts > 1]
    doubles_real = doubles_real[doubles_real != -1]
    metrics["doubles_real"] = doubles_real
    metrics["doubles_detected"] = np.arange(len(singles_extracted))[
        np.where(singles_extracted == False)]

    return metrics


def extract_frames_big_pos_err(metrics, err):
    """Извлечение кадров событий с большой ошибкой по времени.

    @metrics - Посчитанные метрики. @note test_on_df должа быть выполнена с
    флагом extr_frames=True.
    @err - Минимальная ошибка в нс.
    @retutn - Массив кадров.
    """
    idxs_raw = metrics["real_detected_transitions"]
    pos_err = metrics["pos_real"][idxs_raw != -1] - \
        metrics["pos_extracted"][idxs_raw[idxs_raw != -1]]
    pos_err = metrics["pos_real"][idxs_raw != -1] - \
        metrics["pos_extracted"][idxs_raw[idxs_raw != -1]]
    pos_err = np.abs(metrics["pos_real"][idxs_raw != -1] -
                     metrics["pos_extracted"][idxs_raw[idxs_raw != -1]])
    bad_frames = metrics["frames"][idxs_raw[idxs_raw != -1]][pos_err > 320]

    return bad_frames


def extract_frames_false_pos(metrics):
    """Извлечение кадров ложно положительных событий.

    @metrics - Посчитанные метрики. @note test_on_df должа быть выполнена с
    флагом extr_frames=True.
    @retutn - Массив кадров.
    """
    return metrics['frames'][metrics["false_positives"]]


def extract_false_neg():
    """Извлечение кадров ложно отрицательных событий.

    @metrics - Посчитанные метрики. @note test_on_df должа быть выполнена с
    флагом extr_frames=True.
    @retutn - Массив кадров.
    """
    return metrics['frames_real'][metrics["false_negatives"]]


def test_on_df(meta, data, block_params, algoritm_func, max_pos_err=3200,
               extr_frames=False, frame_l=15, frame_r=25):
    """Тестирование с использованием генерируемых df файлов.

    @note - алгоритм протестирован только на стандартной частоте оцифровки

    @meta - метаданные df файла
    @data - бинарные данные df файла
    @block_params - параметры событий
    @algoritm_func - функция выделения событий из данных.
    Входные аргументы функции:
        - data - данные блока (np.array)
        - start_time - начало блока в наносекундах
        - threshold - порог
        - sample_freq - частота оцифровки данных
    Выходные аргументы функции:
        (params, singles)
        - params - информация о событиях в формате [amp1, pos1, amp2, pos2,
                                                    ...] (np.array).
        Позиция должна быть задана в наносекундах от начала данных
          - singles - массив bool содержащий флаги наложенности событий. Для
          наложенных событий флаг должен иметь значение False, для одинарных
          событий  - True.
     В качестве примера функции выделения см
     signal_utils.extract_utils.extract_amps_approx2.
     @max_pos_err - макисмальная ошибка выделения положения пика (нс)
     @extr_frames - сохранение кадров событий в метрику
     @frame_l - размер сохраняемого кадра в бинах слева
     @frame_r - размер сохраняемого кадра в бинах справа

     @todo изменить фунции детектирования

    """
    threshold = meta['process_params']['threshold']
    sample_freq = meta['params']['sample_freq']

    point = Point()
    point.ParseFromString(data)

    start = datetime.now()
    frames = []
    amps = []
    times = []
    singles = []
    for channel in point.channels:
        for i, block in enumerate(channel.blocks):
            amps_block = []
            times_block = []
            singles_block = []
            for event in block.frames:
                ev_data = np.frombuffer(event.data, np.int16)
                params, singles_raw = algoritm_func(ev_data, event.time,
                                                    threshold, sample_freq)

                if extr_frames:
                    peaks = np.round((params[1::2] -
                                      event.time) * sample_freq / 1e+9)

                    frames_block = extract_frames(ev_data, peaks,
                                                  frame_l, frame_r)
                    frames.append(frames_block)

                amps_block.append(params[0::2])
                times_block.append(params[1::2])
                singles_block.append(singles_raw)

            amps_block = np.hstack(amps_block).astype(np.float16)
            times_block = np.hstack(times_block)
            times_block = times_block + block.time
            singles_block = np.hstack(singles_block).astype(np.bool)

            amps.append(amps_block)
            times.append(times_block)
            singles.append(singles_block)

    amps = np.hstack(amps)
    times = np.hstack(times)
    singles = np.hstack(singles)
    if extr_frames:
        frames = np.vstack(frames)

    end = datetime.now()
    delta = (end - start).total_seconds()

    amps_real = block_params[0::2]
    times_real = block_params[1::2]

    metrics = _calc_metrics(amps_real, times_real,
                            amps, times,
                            singles, max_pos_err)

    out = {"amps_real": amps_real,
           "pos_real": times_real,
           "amps_extracted": amps,
           "pos_extracted": times,
           "singles_extracted": singles,
           "time_elapsed": delta,
           **metrics}

    if extr_frames:
        out['frames'] = frames
        out['frames_real'] = _extract_real_frames(meta, data, block_params,
                                                  frame_l, frame_r)

    return out


def __test():
    import seaborn

    seaborn.set_context("poster")

    from extract_utils import extract_amps_approx
    from pylab import rcParams

    rcParams['figure.figsize'] = 10, 10

    dist_path = path.join(path.dirname(__file__), 'data/dist.dat')

    meta, data, block_params = generate_df(time=1, threshold=700,
                                           dist_file=dist_path, freq=40e3)

    global metrics
    metrics = test_on_df(meta, data, block_params, extract_amps_approx,
                         extr_frames=True)
    draw_metrics(metrics)


if __name__ == '__main__':
    __test()
