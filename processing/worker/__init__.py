"""Distributed processing tasks."""
from os import environ
from os.path import join

import dfparser
import numpy as np
from celery import Celery
from dateutil.parser import parse as tparse
from dotenv import load_dotenv

from app import cache
from utils import _get_points_in_set, _lan_amps

load_dotenv()
app = Celery('tasks', config_source='celeryconfig')


@app.task
# @cache.memoize(timeout=0)
def get_point_amps(filename):
    """Get point amplitudes array.

    Parameters
    ----------
    filename : relative path to point.

    """
    _, meta, data = dfparser.parse_from_file(
        join(environ['LAN10_DATA_PATH'], filename))
    try:
        return _lan_amps(data)
    except ValueError:
        return np.array([], dtype=np.int16)


@app.task
@cache.memoize(timeout=0)
def get_point_meta(filename):
    """Get point metadata.

    Parameters
    ----------
    filename : relative path to point.

    """
    _, meta, _ = dfparser.parse_from_file(
        join(environ['LAN10_DATA_PATH'], filename), nodata=True)
    return meta


@app.task
@cache.memoize(timeout=0)
def get_point_cr(filename, range_l, range_r):
    """Calculate point count rate.

    Returns count rate [counts/s].

    Parameters
    ----------
    filename : relative path to point.
    range_l : amplitude window left border.
    range_r : amplitude window right border.

    """
    amps = get_point_amps(filename)
    meta = get_point_meta(filename)

    event_len = int(meta['params']['b_size']) / int(
        meta['params']['sample_freq'])
    time_total = event_len * int(meta['params']['events_num'])
    counts = amps[np.logical_and(amps < range_r, amps > range_l)].shape[0]

    if time_total == 0:
        return 0
    else:
        return counts / time_total


@app.task
@cache.memoize(timeout=0)
def get_set_orientation(set_path):
    """Calculate set orientation.

    Method will take 2 points and compare treir indices and start_times.
    Returns True in set has direct orientation, False otherwise.

    Parameters
    ----------
    set_path : relative path to set.

    """
    points = _get_points_in_set(join(environ['LAN10_DATA_PATH'], set_path))

    _, meta1, _ = dfparser.parse_from_file(points[0], nodata=True)
    _, meta2, _ = dfparser.parse_from_file(points[1], nodata=True)

    ind1 = int(meta1['external_meta']['point_index'])
    time1 = tparse(meta1['params']['start_time'])
    ind2 = int(meta2['external_meta']['point_index'])
    time2 = tparse(meta2['params']['start_time'])

    if ind1 > ind2:
        if time1 > time2:
            return True
        else:
            return False
    else:
        if time1 > time2:
            return False
        else:
            return True
