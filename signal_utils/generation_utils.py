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
      Генерацмя сигнала
      
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
    
    return (gauss(x - pos) + spike(spike_x)*tail_amp)*ampl


def gen_multiple(x, a, p, *args):
    """
      Функция нескольких событий
      
      @a - амплитуда текущего события
      @p - положение пика текущего события
      @args - последовательно указанные амплитуды 
      и положения пиков остальных событий
    
    """
    assert(not len(args)%2)

    y = gen_signal(x, a, p)
    for i in range(0, len(args), 2):
        y += gen_signal(x, args[i], args[i + 1])
    return y
