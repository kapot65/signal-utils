# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:53:00 2017

@author: kapot
"""

import numpy as np
from mpld3 import plugins


def draw_event(data, num, fig, ax):
    """
      Отрисовка первого события из блока. Размер события задан 128 бинам.
      @data - событие
      @num - номер события
      @fig - график
      @ax - оси
      
    """
    data = data[0:128]
    line = ax.plot(data)
    tooltip = plugins.LineLabelTooltip(line[0], label='event: %s'%(num))
    plugins.connect(fig, tooltip)


def plot_event(data, num, event, ax, threshold=700, bin_sec=3.2e-07):
    """
      Отрисовка события и его пареметров
      @data - событие
      @num - номер события
      @event - событие в формате [индекс первого бина, превысившего порог,
      индекс перегиба, индекс первого отрицательного бина после перегиба]
      @ax - оси, на которые будут открисованы события
      @threshold - порог
      @bin_sec - коэфициент перевода бинов в секунды
      
    """
    y_range = (np.min(data), np.max(data))

    first_greater, extremum, first_negative = event

    lenght = first_negative - extremum
    ax.set_title("ampl: %s, size: %s (%s s)"%(data[extremum], 
                 lenght, lenght*bin_sec))

    ax.plot(data, label="event")
    ax.plot((0, len(data)), (threshold, threshold), "--", 
            color="green", label="first greater")
    ax.plot((first_greater, first_greater), y_range, "--", 
            color="orange", label="first greater")
    ax.plot((extremum, extremum), y_range, "--", 
            color="red", label="extremum")
    ax.plot((first_negative, first_negative), y_range, "--", 
            color="#00AAFF", label="first negative")

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('time')
    ax.set_ylabel('ampl')
    ax.legend();


def plot_multiple_events(data, events, ax, threshold=700, bin_sec=3.2e-07):
    """
      Отрисовка события и его пареметров
      @data - массив кадра
      @ax - оси, на которые будут открисованы события
      @events - события в формате [[индекс первого бина, превысившего порог,
      индекс перегиба, индекс первого отрицательного бина после перегиба],...]
      @threshold - порог
      @bin_sec - коэфициент перевода бинов в секунды
      
    """
    y_range = (np.min(data), np.max(data))

    ax.plot(data)
    ax.plot((0, len(data)), (threshold, threshold), "--", color="green")

    for ev in events:
        ax.plot((ev[0], ev[0]), y_range, "--", color="orange")
        ax.plot((ev[1], ev[1]), y_range, "--", color="red")
        ax.plot((ev[2], ev[2]), y_range, "--", color="#00AAFF")

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('time')
    ax.set_ylabel('ampl')