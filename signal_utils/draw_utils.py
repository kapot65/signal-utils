# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:53:00 2017

@author: kapot
"""

import numpy as np
plt = None
plugins = None

def import_graphics():
    global plt
    global plugins
    if not plt:
        plt = __import__('matplotlib').pyplot
    if not plugins:
        plugins = __import__('mpld3').plugins

class graphics(object):

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        import_graphics()
        self.f(*args, **kwargs)
    

@graphics
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


@graphics
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


@graphics
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


@graphics
def draw_metrics(metrics: dict):
    """
      Отрисовка метрик
      
      Выводятся:
          - гистограмма расстояний между пиками в наложенных событиях
          - гистограмма ошибок по амплитудам
          - гистограммы реального и восстановленного спектров
          - гистограмма реальных амплитуд пропущенных событий (без шумов)
      
      @metrics -- Метрики. Cм. вывод signal_utils.test_utils._calc_metrics
      
    """
    print("Summary: ")
    print("Algoritm time: %s s"%(metrics["time_elapsed"]))
    print("Total events: %s"%(metrics["total_real"]))
    print("Detected events: %s"%(metrics["total_detected"]))
    print("Positives/negatives: ")
    print("False negatives %s"%(len(metrics["false_negatives"])))
    print("False positives %s"%(len(metrics["false_positives"])))

    idxs_raw = metrics["real_detected_transitions"]

    def get_dbl_amps_dists(idxs):

        dists, amps = [], []

        for idx in idxs:
            doubles_pos = metrics["pos_real"][idxs_raw == idx]
            doubles_amps = metrics["amps_real"][idxs_raw == idx]
            dists.append(doubles_pos[1:] - doubles_pos[:-1])
            amps.append(np.log(doubles_amps[1:], 
                               doubles_amps[:-1]))

        if len(dists):
            dists = np.hstack(dists)
            amps = np.hstack(amps)
            return amps, dists
        else:
            return np.array([]), np.array([])

    dbl_amps_real, dbl_dists_real = get_dbl_amps_dists(metrics["doubles_real"])
    dbl_amps_det, \
    dbl_dists_det = get_dbl_amps_dists(metrics["doubles_detected"])

    fig, ax = plt.subplots(4, 2)
    hist, bins = np.histogram(dbl_dists_real, 40)
    hist_det, bins = np.histogram(dbl_dists_det, bins=bins)
    ax[2][0].set_title("real doubles")
    ax[2][0].hist2d(dbl_dists_real, dbl_amps_real)
    ax[2][1].set_title("detected doubles")
    ax[2][1].hist2d(dbl_dists_det, dbl_amps_det)
    ax[1][1].set_title("real/detected doubles dists cumsums")
    ax[1][1].plot(bins[1:], np.cumsum(hist), label='real')
    ax[1][1].plot(bins[1:], np.cumsum(hist_det), label='detected')
    ax[1][1].legend()

    print("%s detected \n"\
          "%s real \n"\
          "%s intersection"%(len(metrics["doubles_detected"]),
                             len(metrics["doubles_real"]),
                             len(np.intersect1d(metrics["doubles_detected"],
                                                metrics["doubles_real"]))))

    print("Amplitude accuracy:")
    error = metrics["amps_real"][idxs_raw != -1] - \
            metrics["amps_extracted"][idxs_raw[idxs_raw != -1]]

    ax[0][1].set_title("amplitude error")
    ax[0][1].hist(error, 40)

    ax[0][0].set_title("negative_amplitudes error")
    ax[0][0].hist(metrics["amps_real"][np.where(idxs_raw == -1)[0]], 40)


    idxs_raw[idxs_raw==-1]

    range_ = (min(metrics["amps_real"].min(), 
                  metrics["amps_extracted"].min()),
              max(metrics["amps_real"].max(), 
                  metrics["amps_extracted"].max()))

    ax[1][0].set_title("real vs generated amps hists")
    ax[1][0].hist(metrics["amps_real"], 80, \
                  fc=(1,0,0,0.5), label="real", range=range_)
    ax[1][0].hist(metrics["amps_extracted"], 80, fc=(0,0,1,0.5),
            label="extracted", range=range_)
    ax[1][0].legend()

    range_ = (min(metrics["pos_real"].min(), 
                  metrics["pos_extracted"].min()),
              max(metrics["pos_real"].max(), 
                  metrics["pos_extracted"].max()))

    ax[3][0].set_title("real vs generated pos hists")
    ax[3][0].hist(metrics["pos_real"], 80, \
                  fc=(1,0,0,0.5), label="real", range=range_)
    ax[3][0].hist(metrics["pos_extracted"], 80, fc=(0,0,1,0.5),
            label="extracted", range=range_)
    ax[3][0].legend()


@graphics
def draw_real_frame(meta, metrics, idx, real=True, frame_l=15, frame_r=25):
    """Отрисовка кадра реального события. На графике также указываются
    амплитуды и времена реальных и выделенных событий.

    @meta - Метаданные df.
    @metrics - Посчитанные метрики. @note test_on_df должа быть выполнена с
    флагом extr_frames=True.
    @idx - Индекс кадра.
    @real - Если True - будет выведен кадр реального события, если False -
    кадр восстановленного события.
    @frame_l - Размер кадра в бинах слева.
    @frame_r -Размер кадра в бинах справа.
    """
    freq = 1/meta['params']['sample_freq']

    if real:
        data = metrics['frames_real'][idx]
        pos = metrics['pos_real'][idx]
    else:
        data = metrics['frames'][idx]
        pos = metrics['pos_extracted'][idx]

    ev_l_time = pos - frame_l*freq*1e+9
    ev_r_time = pos + frame_r*freq*1e+9

    l_index_r = np.searchsorted(metrics['pos_real'], ev_l_time)
    r_index_r = np.searchsorted(metrics['pos_real'], ev_r_time)

    l_index_e = np.searchsorted(metrics['pos_extracted'], ev_l_time)
    r_index_e = np.searchsorted(metrics['pos_extracted'], ev_r_time)

    poses_real = metrics['pos_real'][l_index_r: r_index_r] - ev_l_time
    poses_extr = metrics['pos_extracted'][l_index_e: r_index_e] - ev_l_time


    fig, ax = plt.subplots()
    ax.plot(data, label='frame data')
    ax.plot(poses_real/(1e+9*freq),
             metrics['amps_real'][l_index_r: r_index_r], 'ro',
             label='real events')
    ax.plot(poses_extr/(1e+9*freq),
             metrics['amps_extracted'][l_index_e: r_index_e], 'bo',
             label='extracted events')

    for i in range(l_index_r, r_index_r):
        if metrics['real_detected_transitions'][i] != -1:

            x = (metrics['pos_real'][i] - ev_l_time) / (1e+9*freq)
            y = metrics['amps_real'][i]

            trans_ind = metrics['real_detected_transitions'][i]
            trans_x = (metrics['pos_extracted'][trans_ind] - ev_l_time) / \
                      (1e+9*freq)
            trans_y = metrics['amps_extracted'][trans_ind]
            ax.arrow(x, y, trans_x - x, trans_y - y, fc='k', ec='k')


    ax.legend()