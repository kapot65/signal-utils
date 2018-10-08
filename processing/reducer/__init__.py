"""Reduce methods."""
import time
from glob import glob
from os import environ
from os.path import join, relpath

import numpy as np
from celery import group
from tqdm import tqdm

from app import cache
from defaults import DEF_BINS, DEF_RANGE_L, DEF_RANGE_R
from worker import get_point_amps, get_point_cr, get_point_meta


@cache.cached(timeout=60, key_prefix='point_index')
def get_points_index():
    """Return full points list."""
    data_dir = environ['LAN10_DATA_PATH']
    return [relpath(point, data_dir) for point in glob(
        join(data_dir, '*/*/*/*.df'), recursive=True)]


def filter_points(filter_fp=lambda f: True, filter_meta=lambda m: True,):
    """Get filtered points from data folder.

    Returns points list.

    Parameters
    ----------
    filter_fp : filepath filter function: lambda(filename) -> bool.
    filter_meta : metadata filter function: lambda(meta) -> bool.

    """
    index = get_points_index()
    filtered_fp = [f for f in index if filter_fp(f)]
    job = group([get_point_meta.s(f) for f in filtered_fp])
    metas = job.apply_async().join()

    filtered_meta = [
        [f, m] for f, m in zip(filtered_fp, metas) if filter_meta(m)]
    return filtered_meta


def get_points_cr(
        points, range_l=DEF_RANGE_L, range_r=DEF_RANGE_R):
    """Calculate points count rate.

    Returns count rate [counts/s].

    Parameters
    ----------
    points : points list.
    range_l : amplitude left threshold.
    range_r : amplitude right threshold.

    """
    job = group([get_point_cr.s(p, range_l, range_r) for p in points])
    crs = job.apply_async().join()

    return crs


def get_points_hist(
        points, range_l=DEF_RANGE_L, range_r=DEF_RANGE_R, bins=DEF_BINS):
    """Calculate points histogram.

    Parameters
    ----------
    points : points list.
    range_l : amplitude left threshold.
    range_r : amplitude right threshold.
    bins : histogram bins.

    """
    amps = (get_point_amps.delay(p).wait() for p in points)

    return [np.histogram(a, range=(range_l, range_r), bins=bins) for a in amps]
