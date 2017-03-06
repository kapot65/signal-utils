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
    spike = lambda x: (1/(1+2*x*s)**tail_factor - 1.0)*np.exp(-x*s)
    
    y = gauss(x - pos)
    spike_offset = np.where(y > 0.1)[0][-1]
    y[spike_offset:] += spike(x[spike_offset:] - x[spike_offset])*tail_amp
        
    return y*ampl

