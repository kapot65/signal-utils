from os import path

import numpy as np
from scipy.optimize import curve_fit

from generation_utils import gen_multiple
from generation_utils import gen_signal
from process_utils import calc_center
from process_utils import extract_frames
from process_utils import get_peaks


def extract_fit_all(data, start_time, threshold, sample_freq,
                    pos_step=2.0, amp_step=500.0):
    """
      Выделение событий с помощью фитирования всех событий одновременно.
      Классификация наложений производится полносвязной нейронной сетью.

      Алгоритм требует высокой производительности, но дает наиболее точный
      результат.

      Алгоритм:
          1. Выделение локальных максимумов в кадре выше порога. Локальные
          максимумы считаются предварительными пиками.
          2. Классификация наложенных событий с помощью НС по предварительным
          пикам.
          3. Одновременное фитирование всех пиков. В качестве начальных данных
          берутся предварительные пики и соответсвующие им амплитуды. Диапазоны
          фитирования задаются значениями pos_step и amp_step.

    """
    if not extract_fit_all.model:
        if not 'load_model' in locals():
            from keras.models import load_model
        extract_fit_all.model = load_model(path.join(path.dirname(__file__),
                                           "data/mlp_classifier.h5"))

    peaks = get_peaks(data, threshold)

    if not len(peaks):
        return np.array([]), np.array([])

    prep_peaks = np.vstack([data[peak - extract_fit_all.l_off:
                                 peak + extract_fit_all.r_off]
                           for peak in peaks])
    singles = extract_fit_all.model.predict(prep_peaks/extract_fit_all.x_max)\
                           .argmax(axis=1).astype(np.bool)

    params = extract_events_fit_all(data, threshold, amp_step, pos_step)

    params[1::2] = ((params[1::2]/sample_freq)*1e+9) + start_time

    return params, singles

extract_fit_all.model = None
extract_fit_all.frame_len = 50
extract_fit_all.l_off = 14
extract_fit_all.r_off = extract_fit_all.frame_len - extract_fit_all.l_off
extract_fit_all.x_min = -32768
extract_fit_all.x_max = 32768


def extract_simple_amps(data, start_time, threshold, sample_freq):
    """
      Выделение событий поиском локальных максимумов выше порога.

      Алгоритм является наиболее быстрым, однако он не учитывает форму сигнала,
      что приводит к систематической ошибке выделения амлитуд близких событий
      (событие попадает на хвост другого события).

    """
    peaks = get_peaks(data, threshold)

    coeff = np.float128(1e+9/sample_freq)

    params = np.zeros(len(peaks)*2, np.float128)
    params[0::2] = data[peaks]
    params[1::2] = (np.float128(peaks)*coeff) + start_time
    singles = np.ones(peaks.shape, np.bool)

    return params, singles


def extract_amps_approx(data, start_time, threshold, sample_freq):
    """
      Последовательное выделение событий из блока с вычитанием предыдущих
      событий.

      Алгоритм имеет меньшую скорость работы по сравнению с
      extract_simple_amps, однако учитывет форму сигнала.

      Алгоритм:
          1. Выделение локальных максимумов в кадре выше порога.
          2. Последовательная обработка пиков:
              1. Сохранение амплитуды и положения текущего пика.
              2. Вычитание из данных формы сигнала, соответсвующей выделенным
              амплитуде и положению текущего события.
              3. Переход к следующему событию.

    """
    data = data.copy().astype(np.float32)
    peaks = get_peaks(data, threshold)

    params = np.zeros(len(peaks)*2, np.float32)
    params[1::2] = ((peaks/sample_freq)*1e+9) + start_time

    x = np.linspace(0, data.size/sample_freq, data.size)

    for i in range(len(peaks)):
        peak = peaks[i]
        amp = data[peak]
        params[i*2] = amp
        data -= gen_multiple(x, amp, peak/sample_freq)

    singles = np.ones(peaks.shape, np.bool)

    return params, singles


def extract_amps_approx2(data, start_time, threshold, sample_freq,
                         classify=False, frame_l=15, frame_r=25):
    """
      Последовательное выделение событий из блока с вычитанием предыдущих
      событий.

      Ускоренный вариант extract_amps_approx. Вместо вычета события из всего
      кадра, событие вычитается только из пиков.

      Примерно на 30% медленее extract_simple_amps.

      @todo: Разобраться с сохранением времени в наносекундах
    """

    peaks = get_peaks(data, threshold)
    if classify:
        if not extract_amps_approx2.bst:
            if not 'load_model' in locals():
                global xgb
                xgb = __import__('xgboost')
            extract_amps_approx2.bst = xgb.Booster()
            model_path = path.join(path.dirname(__file__), 'data/xgb.dat')
            extract_amps_approx2.bst.load_model(model_path)

        frames = extract_frames(data.copy(), peaks, frame_l, frame_r)
        preds = extract_amps_approx2.bst.predict(xgb.DMatrix(frames))
        singles = preds > 0.97
    else:
        singles = np.ones(peaks.shape, np.bool)

    params = np.zeros(len(peaks)*2, np.float32)
    params[0::2] = data[peaks]
    params[1::2] = ((peaks/sample_freq)*1e+9) + start_time

    for i, peak in enumerate(peaks[:-1]):
        params[(i + 1)*2::2] -= gen_signal(peaks[i + 1:], params[i*2], peak)

    return params, singles
extract_amps_approx2.bst = None


def extract_amps_approx3(data, start_time, threshold, sample_freq):
    """
      Последовательное выделение событий из блока с вычитанием предыдущих
      событий.
    """
    data = data.copy().astype(np.float64)
    x = np.linspace(0, data.size/sample_freq, data.size)

    params = []
    while True:
        peaks = get_peaks(data, threshold)
        if not len(peaks):
            break

        peak = peaks[0]

        params.append(data[peak])
        params.append(((peak/sample_freq)*1e+9) + start_time)

        data -= gen_signal(x, data[peak], peak/sample_freq)

    params = np.array(params)
    singles = np.ones((params.shape[0]//2), np.bool)

    return params, singles


def extract_amps_front_fit(data, start_time, threshold, sample_freq,
                           l_step=2, r_step=5):
    """Выделение событий с помощью последовательного фитирования фронта.


    @l_step - Количество точек слева от порогового бина, по которым будет
    проведено фитирование.
    @r_step - Количество точек справа от порогового бина, по которым будет
    проведено фитирование.
    """
    params = []
    while True:
        data, popt, pcov = extract_event(data, threshold, l_step=l_step,
                                         r_step=r_step, freq=1.0/sample_freq)
        if popt is None:
            break
        else:
            params.append(popt)

    params = np.hstack(params)
    singles = np.ones(params.shape[0]//2, np.bool)


    params[1::2] = params[1::2]*1e+9 + start_time
    return params, singles


def extract_event(data, threshold, l_step, r_step, freq, def_amp=2000):
    """Выделение первого события в кадре с помощью фитирования по фронту.

    @data - Кадр события.
    @threshold - Порог.
    @def_amp - Амплитуда по-умолчанию при фитировании.
    @l_step - Количество точек слева от порогового бина, по которым будет
    проведено фитирование.
    @r_step - Количество точек справа от порогового бина, по которым будет
    проведено фитирование.
    @freq - Частота оцифровки.
    @return data, popt, pcov, где:
        data - Кадр, из которого вычтено определившееся событие.
        popt - [amp, pos] подобранные при фитировании амплитуда и положение
        события
        pcov - параметры сходимости (см scipy.optimize.curve_fit)
    """
    above_thr = np.where(data > threshold)[0]
    if not above_thr.size:
        return data, None, None

    bord = above_thr[0]

    l_ind = max(0, bord - l_step)
    r_ind = min(data.size, bord + r_step)

    part = data[l_ind:r_ind]

    x = np.linspace(0, part.shape[0]*freq, part.shape[0])

    p0 = [def_amp, part.argmax()*freq]
    bounds = ([500, part.argmax()*freq - 3.2e-06],
              [5700, part.argmax()*freq + 3.2e-06])

    popt, pcov = curve_fit(gen_signal, x, part,
                           p0=p0, bounds=bounds)

    x_big = np.linspace(0, data.shape[0]*freq, data.shape[0])

    return data - gen_signal(x_big - l_ind*freq, *popt), \
           popt + [0, l_ind*freq], pcov


def extract_events_fit(ev, threshold):
    """
      Последовательное выделение событий из блока

      В блоке последовательно фитируются событие. Функция следующего события
      складывается с фунциями уже определенных ранее событий.

      Данный метод работает потенциально быстрее и стабильнее по сравнению с
      одновременным фитированием всех событий в блоке, выдает большую ошибку
      при выделении близко наложенных событий (событие накладывается на хвост
      предыдущего события).

      @ev - массив события
      @threshold - порог
      @return параметры событий в формате [amp1, pos1, amp2, pos2, ...]

    """
    points = get_peaks(ev, threshold)
    values = np.array([], np.float32)

    for i, point in enumerate(points):
        y0, x0 = calc_center(ev, point)

        full_func = lambda x, a, p: gen_multiple(x, a, p, *values)

        popt, pcov = curve_fit(full_func, np.arange(len(ev)), ev, p0=[y0, x0])
        values = np.hstack([values, popt])

    return values


def extract_events_fit_all(ev, threshold, pos_step=2.0, amp_step=500.0):
    """
      Последовательное выделение событий из блока
      Все события фитируются одновременно.

      @ev - массив события
      @threshold - порог
      @pos_step - максимальное отклонение по положению при фитировании
      @amp_step - максимальное положительное отклонение по амплитуде при
      фитировании. Нижняя граница всегда равна 0.
      @return параметры событий в формате [amp1, pos1, amp2, pos2, ...]

    """
    points = get_peaks(ev, threshold)

    values = np.zeros(len(points)*2)
    values[0::2] = ev[points]
    values[1::2] = points

    upper_bounds = values.copy()
    upper_bounds[1::2] = upper_bounds[1::2] + pos_step
    upper_bounds[0::2] = upper_bounds[0::2] + amp_step

    lower_bounds = values.copy()
    lower_bounds[1::2] = lower_bounds[1::2] - pos_step
    lower_bounds[0::2] = 0

    full_func = lambda x, *values: gen_multiple(x, *values)
    popt, pcov = curve_fit(full_func, np.arange(len(ev)), ev, p0=list(values))

    return popt


def extract_algo(data, threshold=700):
    """
      Выделенние события из блока. Способ, предложенный Пантуевым В.С.
      @data - массив кадра
      @threshold - порог
      @return - индекс первого бина, превысившего порог,
      индекс перегиба, индекс первого отрицательного бина после перегиба

    """
    deriv = data[1:] - data[:-1]
    first_greater = np.argmax(data >= threshold)
    extremum = first_greater + np.argmax(deriv[first_greater-1:] < 0) - 1
    first_negative = first_greater + np.argmax(data[first_greater:] < 0)
    return first_greater, extremum, first_negative


def extract_events(data, threshold=700):
    """
      Выделенние нескольких событий из блока.
      Способ, предложенный Пантуевым В.С.
      @data - массив кадра
      @threshold - порог
      @return - [индекс первого бина, превысившего порог,
      индекс перегиба, индекс первого отрицательного бина после перегиба]

    """
    events = []
    offset = 0
    while(True):
        left, center, right = extract_algo(data[offset:], threshold)
        if not left:
            break
        left += offset
        center += offset
        r_buf = right
        right += offset
        offset += r_buf
        events.append([left, center, right])

    return events
