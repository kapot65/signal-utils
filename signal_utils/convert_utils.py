"""Data files conversion utils."""
# pylint: disable-msg=C0413

from os import path
import sys

from dateutil.parser import parse
import dfparser
import numpy as np

CUR_DIR = path.dirname(path.realpath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)
del CUR_DIR

from process_utils import extract_frames


def apply_zsupression(data: np.ndarray, threshold: int=500, area_l: int=50,
                      area_r: int=100) -> tuple:
    """Обрезание шумов в файле данных платы Лан10-12PCI.

    Функция расчитана на файлы данных с максимальным размером кадра
    (непрерывное считывание с платы).

    @data - данные кадра (отдельный канал)
    @threshold - порог амплитуды события
    @area_l - область около события, которая будет сохранена
    @area_r - область около события, которая будет сохранена

    @return список границ события

    """
    peaks = np.where(data > threshold)[0]
    dists = peaks[1:] - peaks[:-1]
    gaps = np.append(np.array([0]), np.where(dists > area_r)[0] + 1)

    events = ((peaks[gaps[gap]] - area_l, peaks[gaps[gap + 1] - 1] + area_r)
              for gap in range(0, len(gaps) - 1))

    return events


def rsb_to_df(ext_meta: dict, rsb_file, threshold: int=500,
              area_l: int=50, area_r: int=100) -> (dict, bytearray, int):
    """Конвертировние данных формата rsb в формат df.

    @meta - метаданные сообщения с точками
    @rsb_file - файл с платы Руднева-Шиляева
    @threshold - порог амплитуды события (параметр zero-suppression)
    @area_l - область около события, которая будет сохранена (параметр
    zero-suppression)
    @area_r - область около события, которая будет сохранена (параметр
    zero-suppression)

    @return - (meta, data, data_type)

    """
    sec_coef = 1e+9

    rsb_ds = dfparser.RshPackage(rsb_file)

    meta = {}
    meta["external_meta"] = ext_meta
    meta["params"] = rsb_ds.params
    meta["process_params"] = {
        "threshold": threshold,
        "area_l": area_l,
        "area_r": area_r
    }

    begin_time = parse(rsb_ds.params["start_time"]).timestamp()*sec_coef
    end_time = parse(rsb_ds.params["end_time"]).timestamp()*sec_coef
    bin_time = (rsb_ds.params["sample_freq"]**-1)*sec_coef
    b_size = rsb_ds.params["b_size"]

    if rsb_ds.params["events_num"] == -1:
        meta["recalc_events_num"] = True
        rsb_ds.params["events_num"] = np.iinfo(int).max
        for i in range(np.iinfo(int).max):
            try:
                rsb_ds.get_event(i)
            except Exception as exception:
                rsb_ds.params["events_num"] = i
                break

    events_num = rsb_ds.params["events_num"]
    ch_num = rsb_ds.params["channel_number"]

    use_time_corr = False
    if events_num > 0:
        event = rsb_ds.get_event(0)
        if "ns_since_epoch" not in event:
            use_time_corr = True
            times = list(np.linspace(begin_time, end_time -
                                     int(bin_time*b_size),
                                     events_num))
            meta["correcting_time"] = "linear"

    point = dfparser.Point()
    channels = [point.channels.add(id=channel) for channel in range(ch_num)]
    for i in range(events_num):
        event_data = rsb_ds.get_event(i)

        if use_time_corr:
            time = times[i]
        else:
            time = event_data["ns_since_epoch"]

        for channel in range(ch_num):
            block = channels[channel].blocks.add(time=int(time), )

            ch_data = event_data["data"][channel::ch_num]
            for frame in apply_zsupression(ch_data, threshold, area_l, area_r):
                frame = np.clip(frame, 0, ch_data.shape[0] - 1)
                event = block.frames.add()
                event.time = int(frame[0]*bin_time)
                event.data = ch_data[frame[0]:frame[1]].astype(np.int16) \
                    .tobytes()

    meta["bin_offset"] = 0
    meta["bin_size"] = point.ByteSize()
    data = point.SerializeToString()

    return meta, data


def df_frames_to_events(meta, data, extract_func, frame_l=15, frame_r=25,
                        correct_time=False):
    """Convert frames to events in dataforge points."""
    threshold = meta['process_params']['threshold']
    sample_freq = meta['params']['sample_freq']

    point = dfparser.Point()
    point.ParseFromString(data)

    for channel in point.channels:
        for block in channel.blocks:

            events = block.events
            for _ in range(len(block.frames)):
                frame = block.frames.pop(0)

                if correct_time:
                    frame.time = (frame.time//np.uint64(10)).astype(np.uint64)

                ev_data = np.frombuffer(frame.data, np.int16)
                params, singles_raw = extract_func(ev_data, frame.time,
                                                   threshold, sample_freq)

                singles = np.where(singles_raw == True)[0]

                events.times.extend(np.round(params[singles*2 + 1])
                                    .astype(np.uint64))
                events.amplitudes.extend(np.round(params[singles*2])
                                         .astype(np.uint64))

                doubles = np.where(singles_raw == False)[0]
                events_bins = np.round((params[doubles*2 + 1] -
                                        frame.time) * sample_freq / 1e+9)

                frames_block = extract_frames(ev_data, events_bins,
                                              frame_l, frame_r)
                for idx, frame in enumerate(frames_block):
                    time = int(np.round(params[doubles[idx]*2 + 1]) +
                               frame_l / sample_freq * 1e+9)
                    frame_ser = frame.astype(np.int16).tobytes()
                    block.frames.add(time=time, data=frame_ser)

    meta['process_params']['extracted'] = True
    meta['process_params']['extracted_frame_l'] = frame_l
    meta['process_params']['extracted_frame_r'] = frame_r

    return meta, point.SerializeToString()
