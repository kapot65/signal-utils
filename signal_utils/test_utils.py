# -*- coding: utf-8 -*-

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from generation_utils import gen_raw_block
from extract_utils import apply_zsupression
    

def test_algoritm(algoritm_func,
                  max_pos_err = 5000,
                  threshold: int=700, 
                  area_l: int=50, 
                  area_r: int=100,
                  freq: float=12e+3, 
                  sample_freq: float=3125000.0, 
                  b_size: int=1048576,
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
          - threshold - порог
          - start_time - начало блока в наносекундах
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
      @min_amp - минимальная амлитуда генерируемого события 
      @max_amp - максимальная амлитуда генерируемого события 
      
    """

    amps_real = np.array([])
    pos_real = np.array([])
    
    amps_extracted = np.array([])
    pos_extracted = np.array([])
    
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
        events, singles = algoritm_func(block, threshold, 
                                        times[i], sample_freq)
        params_extracted.append(events)
        singles_extracted.append(singles)
    delta = (datetime.now() - time_start).total_seconds()
    
    params_extracted = np.hstack(params_extracted)
    singles_extracted = np.hstack(singles_extracted)
    
    amps_real = np.hstack([amps_real, params[0::2]])
    pos_real = np.hstack([pos_real,(params[1::2]/sample_freq)*1e+9])
    
    amps_extracted = np.hstack([amps_extracted, params_extracted[0::2]])
    pos_extracted = np.hstack([pos_extracted, params_extracted[1::2]])
    
    # Анализ
    print("Summary: ")
    print("algoritm time: %s s"%(delta))
    print("total events: %s"%(len(amps_real)))
    print("detected events: %s"%(len(amps_extracted)))
    
    idxs_raw = np.abs(np.subtract.outer(pos_extracted, pos_real)).argmin(0)
    dists = np.abs(pos_real - pos_extracted[idxs_raw])
    idxs_raw[dists > max_pos_err] = -1
    single_idxs, counts = np.unique(idxs_raw, return_counts=True)
    
    print("Positives/negatives: ")
    false_neg = len(idxs_raw[idxs_raw==-1])
    print("false negatives %s (%s)"%(false_neg, false_neg/len(idxs_raw)))
    
    false_pos = len(np.setdiff1d(np.arange(len(pos_extracted)), single_idxs))
    print("false positives %s (%s)"%(false_pos, false_pos/len(pos_real)))
    
    print("Singles/doubles: ")
    singles_detected = idxs_raw[np.where(singles_extracted==False)]
    singles_real = single_idxs[counts > 1]
    singles_real = singles_real[singles_real != -1]
    
    singles_real_dists = []
    for idx in singles_real:
        doubles_real_pos = pos_real[idxs_raw == idx]
        singles_real_dists.append(doubles_real_pos[1:] - doubles_real_pos[:-1])
        
    singles_real_dists = np.hstack(singles_real_dists)
    
    
    fig, ax = plt.subplots(2, 2)
    ax[0][0].set_title("real doubles dists")
    ax[0][0].hist(singles_real_dists, 40)
    
    print("%s detected \n"\
          "%s real \n"\
          "%s intersection"%(len(singles_detected),
                             len(singles_real),
                             len(np.intersect1d(singles_detected,
                                                singles_real))))
    
    print("Amplitude accuracy:")
    error = amps_real[idxs_raw != -1] - \
            amps_extracted[idxs_raw[idxs_raw != -1]]
            
    ax[0][1].set_title("amplitude error")
    ax[0][1].hist(error, 40)
    
    ax[1][1].set_title("negative_amplitudes error")
    ax[1][1].hist(amps_real[np.where(idxs_raw == -1)[0]], 40)
    
    
    idxs_raw[idxs_raw==-1]
    
    range_ = (min(amps_real.min(), amps_extracted.min()),
              max(amps_real.max(), amps_extracted.max()))
    
    ax[1][0].set_title("amp hists")
    ax[1][0].hist(amps_real, 80, fc=(1,0,0,0.5), label="real", range=range_)
    ax[1][0].hist(amps_extracted, 80, fc=(0,0,1,0.5),
            label="extracted", range=range_)
    ax[1][0].legend()
    

if __name__ == '__main__':
    
    from pylab import rcParams
    from extract_utils import extract_amps_approx
    rcParams['figure.figsize'] = 20, 10
    test_algoritm(extract_amps_approx)
