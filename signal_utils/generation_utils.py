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


def gen_peak(ampl=1000, use_offset=True, p=1, 
             sigma=0.6, n_bins=20, left=-2, right=2):
    """
      Генерация пика сигнала
      
      В качестве пика используется форма [обобщенного гауссиана](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.general_gaussian.html#scipy.signal.general_gaussian)
      Все аргументы, кроме @ampl и @use_offset подобраны под реальный сигнал.
      Рекомендуется оставить из значения по умолчанию
      
      @ampl - апплитуда пика в каналах
      @use_offset - производить смещение пика в пределах бина. Смещение нужно для того, 
      чтобы пик не попадал в бин, как в реальных данных
      @p - параметр гауссиана
      @sigma - параметр гауссиана
      @n_bins - количество бинов 
      @left - левая граница гауссиана 
      @right - правая граница гауссиана
      @return - массив с пиком
      
    """
    
    offset = 0
    if use_offset:
        bin_error = (right - left)/n_bins
        offset = np.random.uniform(-bin_error, bin_error)
        
    off = lambda x: x + offset
    gauss_gen = lambda x: np.exp((-1/2)*np.power((np.abs(x/sigma)), 2*p))
    data = np.array([gauss_gen(off(x)) for x in np.linspace(-2, 2, 20)])
    return data*ampl


def generate_spike(ampl=400, a=-0.19009746026101096, 
                   b=-56.524973694092296, lenght=42):
    """
      Генерация выброса сигнала
      
      Все аргументы, кроме @ampl подобраны под реальный сигнал.
      Рекомендуется оставить из значения по умолчанию
      
      @ampl - апплитуда пика в каналах
      @a - коэффициент соотношения амплитуды пика и выброса
      @b - коэффициент соотношения амплитуды пика и выброса
      @lenght - длина выброса
      @return - массив с выбросом
      
    """
    
    spike_amp = np.abs(a*ampl + b)
    
    formula = lambda x, s: (1/(1+2*x*s)**3 - 0.9)*np.exp(-x*s) 
    data = np.array([formula(x, 0.048) for x in np.linspace(0, 100, lenght*2)])
    data = data/np.abs(data.min())*spike_amp
    
    return data


def gen_signal(peak_amp, l_offset=45, add_noise=False):
    """
      Генерация сигнала
      
      @peak_amp - апплитуда пика в каналах
      @l_offset - положение сигнала
      @add_noise - Добавление шума
      @return - сигнал
      
    """
    peak = gen_peak(peak_amp)
    spike = generate_spike(peak_amp)

    signal = np.zeros(l_offset + len(peak) + len(spike))
    signal[l_offset:l_offset + len(peak)] = peak
    signal[l_offset + round(len(peak)*0.76): 
           l_offset + round(len(peak)*0.76) + len(spike)] = spike

    if add_noise:
        noise = generate_noise(len(signal) - 2)
        return signal + noise*16
    else:
        return signal

