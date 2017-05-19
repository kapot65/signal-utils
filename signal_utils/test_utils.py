# -*- coding: utf-8 -*-

from datetime import datetime
from functools import partial

import numpy as np
from dfparser import Point

from draw_utils import draw_metrics
from generation_utils import gen_raw_block


def _extract_events(data, positions, ev_pre_l, ev_pre_r, sample_freq):
    """
      Вырезание кадров событий из данных
      
      @data - массив данных сигнала
      @positions - массив положений пиков в секундах с начала кадра
      @ev_pre_l - размер кадра сохраняемого события слева от пика
      @ev_pre_r - размер кадра сохраняемого события справа от пика
      @sample_freq
      
      @return - массив кадров событий
    """
    
    frames_extr = np.zeros((len(positions), ev_pre_l + ev_pre_r), 
                           np.float32)
    
    for i, pos in enumerate((positions*(sample_freq*1e-9)).astype(np.int)):
        if 0 < pos - ev_pre_l and pos + ev_pre_r < data.shape[0]:
            frames_extr[i] = data[pos - ev_pre_l: pos + ev_pre_r]
            
    return frames_extr


def process_data(df_meta, df_data, block_params,
                 algoritm_func, 
                 max_pos_err,
                 ev_pre_l: int=10,
                 ev_pre_r: int=50):
    
    p = Point()
    p.ParseFromString(df_data)
    
    sample_freq = df_meta['params']['sample_freq']
    threshold = df_meta['process_params']['threshold']
    
    params_extracted = []
    singles_extracted = []
    
    time_start = datetime.now()
    
    for ch in p.channels:
        for block in ch.blocks:
            print(block.time)
            for event in block.events:
                
                data = np.frombuffer(event.data, np.int16)
                events, singles = algoritm_func(data, block.time + event.time, 
                                                threshold, sample_freq)
                params_extracted.append(events)
                singles_extracted.append(singles) 
        
    delta = (datetime.now() - time_start).total_seconds()
    
    params_extracted = np.hstack(params_extracted)
    singles_extracted = np.hstack(singles_extracted)
    
    amps_extracted = params_extracted[0::2]
    pos_extracted = params_extracted[1::2]
    
    amps_real = block_params[0::2]
    pos_real = block_params[1::2]
    
    frames_real = []
    frames_extr = []
    
    for ch in p.channels:
        for block in ch.blocks:
            for event in block.events:
                #pos_extracted[np.searchsorted()]
                print(block.time)
                raise Exception
                pass
                           
    '''
    frames_real = _extract_events(data, pos_real, 
                                  ev_pre_l, ev_pre_r, 
                                  sample_freq)
    
    
    
    frames_extr = _extract_events(data, pos_extracted, 
                                  ev_pre_l, ev_pre_r, 
                                  sample_freq)
    '''
    return amps_real, pos_real, amps_extracted,  pos_extracted,
    singles_extracted, delta
    
          

#from extract_utils import extract_simple_amps
#out = process_data(meta, data, block_params, extract_simple_amps, 5000)

def _generate_block(b, algoritm_func, max_pos_err, 
                    threshold, area_l, area_r, 
                    freq, sample_freq, b_size, 
                    min_amp, max_amp,
                    ev_pre_l: int=10,
                    ev_pre_r: int=50):
    """
      Генерация кадра и выделение из него событий.
      
      @b - номер блока
      @algoritm_func - фукнция выделения
      @max_pos_err - макисмальная ошибка выделения положения пика (нс)
      @threshold - порог zero-suppression
      @area_l - левая область zero-suppression
      @area_r - правая область zero-suppression
      @freq - скорость счета в секунду
      @sample_freq - частота оцифровки тестовых данных 
      @b_size - размер кадра
      @min_amp - минимальная амлитуда генерируемого события 
      @max_amp - максимальная амлитуда генерируемого события
      @ev_pre_l - размер кадра сохраняемого события слева от пика
      @ev_pre_r - размер кадра сохраняемого события справа от пика
      
      @return (amps_real, pos_real, frames_real, 
      amps_extracted, pos_extracted, frames_extr, singles_extracted, 
      delta)
      amps_real, amps_extracted - массив реальных и восстановленных амплитуд
      pos_real, pos_extracted - массив реальных и восстановленных положений 
      пиков в наносекундах
      frames_real, frames_extr - массив реальных и восстановленных кадров 
      событий
      singles_extracted - массив классов восстановленных событий
      delta - время работы алгоритма выделения в секундах
      
    """
    
    block_off = b*((b_size/sample_freq)*1e+9 + 2*max_pos_err)
    
    data, params = gen_raw_block(freq, sample_freq, b_size, 
                                 min_amp, max_amp)
    blocks = np.array(list(apply_zsupression(data, threshold, 
                                             area_l, area_r)))
    times = (blocks[:,0]/sample_freq)*1e+9  #наносекунды
    
    params_extracted = []
    singles_extracted = []
    
    
    time_start = datetime.now()
    for i in range(len(blocks)):
        block = data[blocks[i][0]:blocks[i][1]]
        events, singles = algoritm_func(block, times[i], 
                                        threshold, sample_freq)
        params_extracted.append(events)
        singles_extracted.append(singles)
        
    delta = (datetime.now() - time_start).total_seconds()
    params_extracted = np.hstack(params_extracted)
    singles_extracted = np.hstack(singles_extracted)
    
    
    amps_real = params[0::2]
    pos_real = params[1::2]*1e+9
    frames_real = _extract_events(data, pos_real, 
                                  ev_pre_l, ev_pre_r, 
                                  sample_freq)
    
    amps_extracted = params_extracted[0::2]
    pos_extracted = params_extracted[1::2] 
    
    frames_extr = _extract_events(data, pos_extracted, 
                                  ev_pre_l, ev_pre_r, 
                                  sample_freq)
    
    return amps_real, pos_real + block_off, frames_real,\
           amps_extracted,  pos_extracted + block_off, frames_extr,\
           singles_extracted, delta
           

def _calc_metrics(amps_real, pos_real, frames_real, 
                  amps_extracted, pos_extracted, frames_extracted, 
                  singles_extracted, delta, max_pos_err):
    """
      Вычисление метрик.
      
      Описание алгоритма см в signal_utils.test_utils.test_algoritm
      
      @amps_real -- массив амплитуд реальных событий
      @pos_real -- массив положений реальных событий в наносекундах
      @frames_real -- кадры реальных событий
      @amps_extracted -- массив амплитуд выделенных событий
      @pos_extracted -- массив амплитуд выделенных положений в наносекундах
      @frames_extracted -- кадры выделенных событий
      @singles_extracted -- массив классификаций выделенных событий по 
      наложенности
      @delta -- время выполнения алгоритма в секундах
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
            'frames_extracted': array([[ 256. ..., 28.80752563],   
                                       ..., 
                                       [-144., ..., -334.03234863]], 
                                      dtype=float32),
            'frames_real': array([[  96., ..., -45.58765411],
                                  ..., 
                                  [-624.06677246, ..., -64.64994812]], 
                                 dtype=float32),
            'pos_extracted': array([  6.14400000e+04, ..., 3.35321920e+08], 
                                   dtype=float32),
            'pos_real': array([  1.50009424e+04, ..., 3.35438528e+08], 
                              dtype=float32),
            'real_detected_transitions': array([-1, ..., -1], dtype=int64),
            'singles_extracted': array([ True, ...,  True], dtype=bool),
            'time_elapsed': 0.081999,
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
             time_elapsed -- время выполнения алгоритма в секундах
             total_detected -- общее количество выделенных событий
             total_real -- общее количество реальных событий
      
    """
    metrics = {}
    metrics["time_elapsed"] = delta
    metrics["total_real"] = len(amps_real)
    metrics["total_detected"] = len(amps_extracted)

    
    idxs_raw = np.abs(np.subtract.outer(pos_extracted, pos_real)).argmin(0)
    dists = np.abs(pos_real - pos_extracted[idxs_raw])
    idxs_raw[dists > max_pos_err] = -1
    single_idxs, counts = np.unique(idxs_raw, return_counts=True)
    
    metrics["real_detected_transitions"] = idxs_raw
    metrics["false_negatives"] = np.arange(len(idxs_raw))[idxs_raw==-1]
    metrics["false_positives"] = np.setdiff1d(np.arange(len(pos_extracted)), \
                                              single_idxs)
    
    doubles_real = single_idxs[counts > 1]
    doubles_real = doubles_real[doubles_real != -1]
    metrics["doubles_real"] = doubles_real
    metrics["doubles_detected"] = np.arange(len(singles_extracted))\
                                  [np.where(singles_extracted==False)]
                                  
    return metrics
    

def test_algoritm(algoritm_func,
                  max_pos_err = 5000,
                  threshold: int=700, 
                  area_l: int=50, 
                  area_r: int=100,
                  freq: float=12e+3, 
                  sample_freq: float=3125000.0, 
                  b_size: int=1048576,
                  n_blocks: int=1,
                  min_amp: int=700, 
                  max_amp: int=7000):
    """
      Тестирование выделения событий из сигнала и вывод метрик
      
      @note - алгоритм протестирован только на стандартной частоте оцифровки
      
      В процессе тестирования определяются:
          - время работы алгоритма
          - количество ложных срабатываний 
          - количество пропущеных событий
          - отношение классифицированных наложенных событий к реальным
          (реальным наложенным событием является выделенное событие, которое
           соответсвует нескольким реальным событиям)
          
      Также выводятся:    
          - гистограмма расстояний между пиками в наложенных событиях
          - гистограмма ошибок по амплитудам
          - гистограммы реального и восстановленного спектров
          - гистограмма реальных амплитуд пропущенных событий (без шумов)
          
      Алгоритм тестирования:
          1. Генерация кадра с учетом скорости счета, частоты оцифровки, 
          диапазона амлитуд. События распределяются равномерно по амплитудам
          и положениям.
          2. Применение алгоритма zero-suppression для сгенерированного кадра.
          Выделение блоков.
          4. Применение алгоритма выделения событий для каждого блока. Сбор 
          всех событий в один массив.
          5. Задание соответсвия между реальными и выделенными событиями. Для 
          этого для каждого события из массива реальных данных ищется ближайшее
          по положению событие из массива выделенных событий.
          6. Подсчет пропущенных событий. Если расстояние между реальным и 
          восстановленным событием больше порога (max_pos_err) - событие 
          помечается как пропущенное.
          7. Подсчет ложных срабатываний. Ложным срабатыванием считается 
          событие из массива восстановленных данных у которого нет ни одного 
          соответсвия с реальным событием.
          8. Подсчет реальных наложений. Реальным наложением является событие
          из восстановленного массива, которое соответсвуюет нескольким 
          событиям из рельных данных.
          Реальные наложения затем сравниваются с классифицированными
          наложениями, полученными алгоритмом выделения.
          9 Вычисление ошибок восстановления ампитуд (как наложенных так и 
          одинарных событий).      
      
      @algoritm_func - функция выделения событий из данных.
      Входные аргументы функции:
          - data - данные блока (np.array)
          - start_time - начало блока в наносекундах
          - threshold - порог
          - sample_freq - частота оцифровки данных
      Выходные аргументы функции:  
          (params, singles)
          - params - информация о событиях в формате [amp1, pos1, amp2, pos2,
          ...] (np.array). позиция должна быть задана в наносекундах от начала
          данных
          - singles - массив bool содержащий флаги наложенности событий. Для
          наложенных событий флаг должен иметь значение False, для одинарных
          событий  - True.
      
      @max_pos_err - макисмальная ошибка выделения положения пика (нс)
      @threshold - порог zero-suppression
      @area_l - левая область zero-suppression
      @area_r - правая область zero-suppression
      @freq - скорость счета в секунду
      @sample_freq - частота оцифровки тестовых данных 
      @b_size - размер кадра
      @n_blocks - количество кадров
      @min_amp - минимальная амлитуда генерируемого события 
      @max_amp - максимальная амлитуда генерируемого события 
      
    """
    
    func = partial(_generate_block, algoritm_func=algoritm_func, 
                   max_pos_err=max_pos_err, threshold=threshold, area_l=area_l, 
                   area_r=area_r, freq=freq, sample_freq=sample_freq, 
                   b_size=b_size, min_amp=min_amp, max_amp=max_amp)
    
    events = np.array([func(b) for b in range(n_blocks)])
    
    amps_real = np.hstack(events[:, 0])
    pos_real = np.hstack(events[:, 1])
    frames_real = np.vstack(events[:, 2])
    amps_extracted = np.hstack(events[:, 3])
    pos_extracted = np.hstack(events[:, 4])
    frames_extracted = np.vstack(events[:, 5])
    singles_extracted = np.hstack(events[:, 6])
    delta = events[:, 7].sum()
    
    metrics = _calc_metrics(amps_real, pos_real, frames_real, 
                           amps_extracted, pos_extracted, frames_extracted, 
                           singles_extracted, delta, max_pos_err)
    
    return {"amps_real": amps_real,
            "pos_real": pos_real, 
            "frames_real": frames_real,     
            "amps_extracted": amps_extracted, 
            "pos_extracted": pos_extracted, 
            "frames_extracted": frames_extracted,         
            "singles_extracted": singles_extracted, 
            **metrics}


def test_convertion_speed():
    """
    @todo: сделать функцию
    """
    pass
    

if __name__ == '__main__':
    
    from pylab import rcParams
    from extract_utils import extract_simple_amps
    
    rcParams['figure.figsize'] = 10, 10
    res = test_algoritm(extract_simple_amps, b_size=int(1048576), n_blocks=1)
    draw_metrics(res)
    
