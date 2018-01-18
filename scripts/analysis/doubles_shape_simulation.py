#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:16:55 2018.

@author: chernov

Скрипт получает симулированную форму спектра двойных наложений с помощью
генерации только двойных событий и последущей обработки созданного файла.

"""

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def dist_time_func(ev_num, bl_time, max_spread=3*320e-9):
    """Функция распределения двойных событий.
    
    Алгоритм может создать и тройные наложения.
    
    @ev_num - количество событий
    @bl_time - время блока в секундах
    @max_spread - максимальное расстояние между сдвоенными событиями
    """
    times = np.zeros(ev_num + ev_num % 2)
    times[0::2] = np.sort(np.random.uniform(0,  bl_time, ev_num // 2))
    times[1::2] = times[0::2] + \
        np.random.uniform(0, max_spread, ev_num // 2)
    return times


def __main():
    print("Unimplemented")


if __name__ == "__main__":
    sns.set_context("poster")
    DEF_PALLETE = sns.color_palette()
    __main()
