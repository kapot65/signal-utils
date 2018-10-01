"""Find dependency between algorint accuracy and threshold."""
from argparse import ArgumentParser
from multiprocessing import Pool
from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import numpy as np
import seaborn

from signal_utils import extract_utils, test_utils
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
    parser.add_argument(
        '--draw-graphs', action='store_true', help='Draw metrics graphs')

    return parser.parse_args()


def _gen_hists_plot(
        amps_orig, amps_extr, false_pos, false_neg, mult_det, bins, range_):
    fig, ax = plt.subplots()

    hist_orig, bins_orig = np.histogram(
        amps_orig, bins=bins, range=range_)

    hist_extr, bins_extr = np.histogram(
        amps_extr, bins=bins, range=range_)

    hist_pos, bins_pos = np.histogram(
        amps_extr[false_pos],
        bins=bins, range=range_)

    hist_neg, bins_neg = np.histogram(
        amps_orig[false_neg],
        bins=bins, range=range_)

    hist_mult, bins_mult = np.histogram(
        amps_extr[mult_det],
        bins=bins, range=range_)

    ax.set_title("Extracted events histograms")
    ax.set_xlabel("Amplitude, ch")

    ax.step((bins_pos[:-1] + bins_pos[1:]) / 2, hist_pos,
            where='mid', label='false positives')

    ax.step((bins_neg[:-1] + bins_neg[1:]) / 2, hist_neg,
            where='mid', label='undetected')

    ax.step((bins_orig[:-1] + bins_orig[1:]) / 2, hist_orig,
            where='mid', label='original')

    ax.step((bins_extr[:-1] + bins_extr[1:]) / 2, hist_extr,
            where='mid', label='extracted')

    ax.step((bins_mult[:-1] + bins_mult[1:]) / 2, hist_mult,
            where='mid', label='overlapped')

    ax.grid(color='lightgray', alpha=0.7)
    ax.legend()

    return fig


def _gen_amps_errs_plot(
        amps_orig, singles_or, amps_extr, singles_det, bins, range_):
    fig, ax = plt.subplots()

    hist_errs_amp, bins_errs_amp = np.histogram(
        amps_orig[singles_or] - amps_extr[singles_det],
        bins=bins, range=range_)

    ax.set_title("Amplitude errors")
    ax.set_xlabel("Error, ch")
    ax.step((bins_errs_amp[:-1] + bins_errs_amp[1:]) / 2,
            hist_errs_amp, where='mid')
    ax.grid(color='lightgray', alpha=0.7)
    return fig


def _gen_times_errs_plot(
        pos_orig, singles_or, pos_extr, singles_det, bins, range_):
    fig, ax = plt.subplots()

    hist_errs_amp, bins_errs_amp = np.histogram(
        pos_orig[singles_or] - pos_extr[singles_det],
        bins=bins,
        range=range_
    )

    ax.set_title("Time errors")
    ax.set_xlabel("Error, bins")

    ax.step((bins_errs_amp[:-1] + bins_errs_amp[1:]) / 2,
            hist_errs_amp, where='mid')
    ax.grid(color='lightgray', alpha=0.7)
    return fig


def __iter(_):
    meta, data, block_params = generate_df(
        time=ARGS.iter_time,
        threshold=ARGS.threshold,
        dist_file=ARGS.dist_path,
        freq=ARGS.frequency)

    metrics = test_utils.test_on_df(
        meta, data, block_params, METHOD, extr_frames=False)

    return metrics


def __append_part(metrics, part, idx):
    total = metrics['total_real']
    total_det = metrics['total_detected']

    metrics['amps_real'] = np.append(metrics['amps_real'].astype(np.float16),
                                     part['amps_real'].astype(np.float16))

    metrics['amps_extracted'] = np.append(
        metrics['amps_extracted'].astype(np.float16),
        part['amps_extracted'].astype(np.float16))

    metrics['pos_real'] = np.append(
        metrics['pos_real'].astype(np.int64),
        (part['pos_real'] - part['pos_real'][0]) +
        (ARGS.iter_time * 1e+9) * idx).astype(np.int64)

    metrics['pos_extracted'] = np.append(
        metrics['pos_extracted'].astype(np.int64),
        (part['pos_extracted'] - part['pos_extracted'][0])
        + (ARGS.iter_time * 1e+9) * idx).astype(np.int64)

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
    seaborn.set_context("poster")

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

        for idx, m in enumerate(metrics_parts):
            __append_part(metrics_all, m, idx)

        metrics_all['method'] = ARGS.method
        np.save(ARGS.output, metrics_all)
    else:
        metrics_all = np.load(ARGS.input)[()]

    if ARGS.draw_graphs:
        # seaborn.set_context("poster")
        rcParams['figure.figsize'] = 8, 6
        hist_range = (0, 8000)
        hist_bins = 200

        amps_errs_range = (-2000, 2000)
        amps_errs_bins = 200

        times_errs_range = (-1000, 1000)
        times_errs_bins = 200

        amps_orig = metrics_all['amps_real']
        pos_orig = metrics_all['pos_real']
        amps_extr = metrics_all['amps_extracted']
        pos_extr = metrics_all['pos_extracted']
        false_neg = metrics_all['false_negatives']
        false_pos = metrics_all['false_positives']

        trans = metrics_all['real_detected_transitions']
        _, ind, cnts = np.unique(
            trans, return_index=True, return_counts=True)

        singles_or = ind[cnts == 1]
        singles_det = trans[ind[cnts == 1]]

        mult_or = ind[cnts > 1]
        mult_det = trans[ind[cnts > 1]]

        fig_hist = _gen_hists_plot(
            amps_orig, amps_extr, false_pos, false_neg, mult_det,
            hist_bins, hist_range)

        fig_amps_errs = _gen_amps_errs_plot(
            amps_orig, singles_or, amps_extr, singles_det,
            amps_errs_bins, amps_errs_range)

        fig_times_errs = _gen_times_errs_plot(
            pos_orig, singles_or, pos_extr, singles_det,
            times_errs_bins, times_errs_range)

        md = markdown(OUT_TEMPLATE)
        html = Environment(loader=BaseLoader()).from_string(md).render(
            method=metrics_all['method'],
            total=len(amps_orig),
            extr_all=len(amps_extr),
            extr_all_perc=len(amps_extr) / len(amps_orig) * 100,
            extr_fpos=len(false_pos),
            extr_fpos_perc=len(false_pos) / len(amps_extr) * 100,
            extr_ok=len(singles_or),
            extr_ok_perc=len(singles_or) / len(amps_extr) * 100,
            extr_ovlpd=len(mult_or),
            extr_ovlpd_perc=len(mult_or) / len(amps_extr) * 100,
            hist_plot=mpld3.fig_to_html(fig_hist),
            amp_errs_plot=mpld3.fig_to_html(fig_amps_errs),
            times_errs_plot=mpld3.fig_to_html(fig_times_errs)
        )

        fig_hist.show()
        fig_amps_errs.show()
        fig_times_errs.show()
        plt.show()

        serve(html)
