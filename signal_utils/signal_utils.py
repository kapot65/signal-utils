import collections

import numpy as np
from tqdm import tqdm


from gdrive_utils import load_dataset, get_points_drom_drive


def extract_from_dataset(dataset, threshold=700, 
                         area_l=50, area_r=100):
    """
      Получение блоков из датасета
      @dataset - датасет
      @threshold - порог zero-suppression
      @area_l - левая область zero-suppression
      @area_r - правая область zero-suppression
      @return - вырезанные блоки
      
    """
    frames = [] # кадры из точки
    for i in range(dataset.params["events_num"]):
        try:
            frames.append(dataset.get_event(i)["data"])
        except:
            break

    blocks = [] # вырезанные из кадра события
    for frame in frames:
        peaks = np.where(frame > threshold)[0]
        dists = peaks[1:] - peaks[:-1]
        gaps = np.append(np.array([0]), np.where(dists > area_r)[0] + 1)
        for gap in range(0, len(gaps) - 1):
            l_bord = peaks[gaps[gap]] - area_l
            r_bord = peaks[gaps[gap + 1] - 1] + area_r
            blocks.append(frame[l_bord: r_bord])
    return blocks


def get_blocks(points, idxs):
    blocks = []
    
    def add_blocks(blocks, idx):
        header = points.as_matrix()[idx]
        dataset = load_dataset(header[0], header[1], header[2])
        blocks += extract_from_dataset(dataset)
    
    if isinstance(idxs, collections.Iterable):
        for i in tqdm(idxs):
            add_blocks(blocks, i)
    else:
        add_blocks(blocks, idxs)
    
    return blocks


def get_bin_sec(points, idx):
    header = points.as_matrix()[idx]
    dataset = load_dataset(header[0], header[1], header[2])
    return dataset.params["freq"]**-1


def extract_algo(data, threshold=700):
    """
      Выделенние события из блока
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
      Выделенние нескольких событий из блока
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


def test_functions():
    
    import matplotlib.pyplot as plt
    
    from draw_utils import draw_event, plot_event, plot_multiple_events
    
    points = get_points_drom_drive()
    points.sort_values('time', ascending=False)
    
    bin_sec = get_bin_sec(points, 0)
    blocks = get_blocks(points, 0)
    
    fig, ax = plt.subplots()
    
    draw_event(blocks[0], 0, fig, ax)
    
    fig, ax = plt.subplots()
    event = extract_algo(blocks[0])
    plot_event(blocks[0], 0, event, ax, threshold=700, bin_sec=bin_sec)
    
    fig, ax = plt.subplots()
    events = extract_events(blocks[0])
    plot_multiple_events(blocks[0], events, ax, threshold=700, bin_sec=bin_sec)
