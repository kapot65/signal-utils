"""Extracting frames testing algoritm"""
from multiprocessing import Pool
from os.path import abspath, dirname, join
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from pylab import rcParams
from signal_utils import test_utils
from signal_utils import extract_utils
from signal_utils.generation_utils import generate_df


def _parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-i',
        '--input',
        default=None,
        help='Input .npy metrics file. '
        'If setted - generation step will be skipped (default - None)')
    parser.add_argument(
        '-o',
        '--output',
        default='metrics.npy',
        help='Output .npy metrics file (default - "metrics.npy")')
    parser.add_argument(
        '-m',
        '--method',
        default='extract_simple_amps',
        help='Output .npy '
        'extracting method name (default - "extract_simple_amps")')
    parser.add_argument(
        '--total-time',
        default=300,
        type=int,
        help='Length of generated data in seconds (default - 300)')
    parser.add_argument(
        '--iter-time',
        type=int,
        default=5,
        help='Length per iteration in seconds (default - 5)')
    parser.add_argument(
        '-t',
        '--threshold',
        default=700,
        type=float,
        help='Extracting threshold (default - 700)')
    parser.add_argument(
        '-f',
        '--frequency',
        default=40e3,
        type=float,
        help='Events frequency in Hz (default - 40e3)')

    dist_def = abspath(
        join(dirname(__file__), '../../signal_utils/data/dist.dat'))
    parser.add_argument(
        '-d',
        '--dist-path',
        default=dist_def,
        help='Events distribution path (default - "%s")' % dist_def)

    return parser.parse_args()


def __iter(_):
    meta, data, block_params = generate_df(
        time=ARGS.iter_time,
        threshold=ARGS.threshold,
        dist_file=ARGS.dist_path,
        freq=ARGS.frequency)

    metrics = test_utils.test_on_df(
        meta, data, block_params, METHOD, extr_frames=False)
    return metrics


def __append_part(metrics, part):
    total = metrics['total_real']
    total_det = metrics['total_detected']

    metrics['amps_real'] = np.append(metrics['amps_real'].astype(np.float16),
                                     part['amps_real'].astype(np.float16))

    metrics['amps_extracted'] = np.append(
        metrics['amps_extracted'].astype(np.float16),
        part['amps_extracted'].astype(np.float16))

    metrics['pos_real'] = np.append(
        metrics['pos_real'].astype(np.float32),
        (part['pos_real'].astype(np.float32) - part['pos_real'].astype(
            np.float32)[0]) * ARGS.iter_time * 1e+9)

    metrics['pos_extracted'] = np.append(
        metrics['pos_extracted'].astype(np.float32),
        (part['pos_extracted'].astype(np.float32) -
         part['pos_extracted'].astype(np.float32)[0]) * ARGS.iter_time * 1e+9)

    metrics['singles_extracted'] = np.append(
        metrics['singles_extracted'].astype(np.bool),
        part['singles_extracted'].astype(np.bool))

    metrics['time_elapsed'] += part['time_elapsed']

    metrics['total_real'] += part['total_real']

    metrics['total_detected'] += part['total_detected']

    trans_part = part['real_detected_transitions'].astype(np.int32).copy()
    trans_part[trans_part >= 0] += int(total_det)

    metrics['real_detected_transitions'] = np.append(
        metrics['real_detected_transitions'].astype(np.int32), trans_part)

    metrics['false_negatives'] = np.append(
        metrics['false_negatives'].astype(np.int32),
        part['false_negatives'].astype(np.int32) + total)

    metrics['doubles_real'] = np.append(
        metrics['doubles_real'].astype(np.int32),
        part['doubles_real'].astype(np.int32) + total)

    metrics['false_positives'] = np.append(
        metrics['false_positives'].astype(np.int32),
        part['false_positives'].astype(np.int32) + total_det)

    metrics['doubles_detected'] = np.append(
        metrics['doubles_detected'].astype(np.int32),
        part['doubles_detected'].astype(np.int32) + total_det)


if __name__ == '__main__':
    ARGS = _parse_args()

    if not ARGS.input:
        METHOD = getattr(extract_utils, ARGS.method)
        p = Pool()
        metrics_parts = p.map(__iter, range(ARGS.total_time // ARGS.iter_time))
        metrics_all = {
            'amps_real': np.array([], dtype=np.float16),
            'pos_real': np.array([], dtype=np.float32),
            'amps_extracted': np.array([], dtype=np.float16),
            'pos_extracted': np.array([], dtype=np.float32),
            'singles_extracted': np.array([], dtype=np.bool),
            'time_elapsed': 0,
            'total_real': 0,
            'total_detected': 0,
            'real_detected_transitions': np.array([], dtype=np.int32),
            'false_negatives': np.array([], dtype=np.int32),
            'false_positives': np.array([], dtype=np.int32),
            'doubles_real': np.array([], dtype=np.int32),
            'doubles_detected': np.array([], dtype=np.int32)
        }

        for m in metrics_parts:
            __append_part(metrics_all, m)

        metrics_all['method'] = ARGS.method
        np.save(ARGS.output, metrics_all)
    else:
        metrics_all = np.load(ARGS.input)[()]


    # seaborn.set_context("poster")
    # rcParams['figure.figsize'] = 10, 10
