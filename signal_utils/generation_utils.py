# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 19:49:34 2017

@author: kapot
"""

import numpy as np

def generate_noise(size):
    noise = np.random.randint(-20, 20, size=(size + 15))
    c1 = np.convolve(noise, np.full((10), 0.315), 'valid')
    c2 = np.convolve(c1, np.full((3), 1/3), 'valid')
    c3 = np.convolve(c2, np.full((3), 1/3), 'valid') + 2.5
    return np.round(c3)
    
    
def gen_signal(x, ampl, pos, 
          sigma=0.34156352, 
          tail_amp=0.37015974, 
          tail_factor=2.96038975,
          p=2.20563286,
          s=0.0525933):
    """
      Генерация сигнала
      
      Все аргументы, кроме @ampl и @pos подобраны под реальный сигнал.
      Рекомендуется оставить из значения по умолчанию
      
      @x - координаты
      @ampl - амплитуда генерируемого сигнала 
      @pos - положение генерируемого сигнала
      @sigma - растяжение сигнала
      @tail_amp - отношение амлитуды выброса к амплитуде сигнала
      @tail_factor - степень резкости пика сигнала
      @p - степень гауссиана амплитуда
      @s - коэффициент растяжения выброса
      @return сигнал для заданных координат
      
    """
    gauss = lambda x: np.exp((-1/2)*np.power((np.abs(sigma*x)), p))
    gauss_rev = lambda y: (1/sigma)*np.power(-2*np.log(y), 1/p)
    spike = lambda x: (1/(1+2*x*s)**tail_factor - 1.0)*np.exp(-x*s)
    
    spike_offset = gauss_rev(0.1) + pos
    spike_x = x - spike_offset
    spike_x[spike_x < 0] = 0
    
    return ((gauss(x - pos) + spike(spike_x)*tail_amp)*ampl)


def gen_multiple(x, *args, l_size=50, r_size=100):
    """
      Функция нескольких событий. Для ускорения работы генерируемый сигнал
      обрезается после l_size от пика слева и на r_size от пика справа.
      
      @a - амплитуда текущего события
      @p - положение пика текущего события
      @args - последовательно указанные амплитуды 
      @l_size - граница события слева
      @r_size - граница события справа
      и положения пиков остальных событий
    
    """
    assert(not len(args)%2)

    y = np.zeros(len(x), np.float32)

    for i in range(0, len(args), 2):
        block = np.logical_and(x > args[i + 1] - l_size, 
                               x < args[i + 1] + r_size)
        y[block] = y[block] + gen_signal(x[block], args[i], args[i + 1])

    return y


def gen_raw_block(freq: float=12e+3, 
                  sample_freq: float=3125000.0, 
                  b_size: int=1048576,
                  min_amp: int=500, 
                  max_amp: int=7000):
    """
      Генерация блока данных для тестирования
      @freq - частота событий (на секунду)
      @sample_freq - частота оцифровки Гц
      @b_size - размер блока в бинах
      @max_amp - максимальная амплитуда события
      @min_amp - минимальная амплитуда события
      @return - [data, [amp1, pos1, amp2, pos2, ...]]
      
    """
    
    events = int(freq*(b_size/sample_freq))
    
    params = np.zeros(events*2, np.float32)
    
    params[1::2] = np.sort(np.random.uniform(0,  b_size, events))
    params[0::2] = np.random.uniform(min_amp, max_amp, events)
    
    x = np.arange(b_size)
    data = gen_multiple(x, *params)
    
    return data + generate_noise(b_size - 2)*16, params
