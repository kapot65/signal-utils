"""Comparison of MADC and Lan10-12PCI full set spectrum.

Input:
- data - path to MADC data
- df_data_root - path to Lan10-12PCI root folder
- points_path - list for 14-17 kV points.

Output:
- Graph of full spectrum for MADC and Lan10-12PCI data.
- Graph of full spectrum ratio between MADC and Lan10-12PCI data.
  Formula: ratio[bin] = 1.0 - counts[bin]/counts_madc[bin]
"""
# pylint: disable-msg=R0914,R0915

from glob import glob
from os import path

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn


def parse_madc_binary(data):
    """Extract amplitudes from madc binary data."""
    amps = np.zeros(len(data)//7, np.uint16)
    for amp_idx, idx in enumerate(range(0, len(data), 7)):
        amps[amp_idx] = np.frombuffer(data[idx:idx+2], np.uint16)[0]

    return amps


def main():
    """Execute main function."""
    df_madc_data_root = "/home/chernov/data_on_server_madc"
    df_data_root = "/home/chernov/data_processed"
    set_path = "2017_05/Fill_2/set_8"

    threshold_madc = 450
    threshold_madc_h = 3100
    threshold = threshold_madc*2.03
    threshold_h = threshold_madc_h*2.03

    points = sorted(glob(path.join(df_data_root, set_path, "p**")))
    points_madc = sorted(glob(path.join(df_madc_data_root, set_path, "p**")))

    counts_madc = {}

    for point in points_madc:
        _, meta, data = dfparser.parse_from_file(point)
        hv_val = int(meta['external_meta']['HV1_value'])

        if hv_val not in counts_madc:
            counts_madc[hv_val] = []

        amps = parse_madc_binary(data)

        filt = np.logical_and(amps > threshold_madc,
                              amps < threshold_madc_h)
        counts_madc[hv_val].append(amps[filt].size)

    counts = {}

    for point in points:
        _, meta, data = dfparser.parse_from_file(point)

        hv_val = int(meta['external_meta']['HV1_value'])

        time = meta['params']['b_size'] * meta['params']['events_num'] / \
            meta['params']['sample_freq']

        point_ds = dfparser.Point()
        point_ds.ParseFromString(data)

        amps = np.hstack([list(block.events.amplitudes)
                          for block in point_ds.channels[0].blocks])

        if hv_val not in counts:
            counts[hv_val] = []
        filt = np.logical_and(amps > threshold,
                              amps < threshold_h)
        counts[hv_val].append(float(amps[filt].size)/time*30.0)

    counts = {key: np.mean(counts[key]) for key in counts}
    counts_madc = {key: np.mean(counts_madc[key]) for key in counts_madc}

    counts_x = list(counts.keys())
    counts_y = [counts[key] for key in counts.keys()]

    counts_madc_x = list(counts_madc.keys())
    counts_madc_y = [counts_madc[key] for key in counts_madc.keys()]

    fig, axes = plt.subplots()
    fig.canvas.set_window_title("energy_spectrum_compare")
    fig.suptitle("CAMAC MADC vs. Lan10-12PCI energy spectrums \n"
                 "Set - %s" % (set_path))
    axes.plot(counts_x, counts_y, 'ro', label='Lan10-12PCI')
    axes.plot(counts_madc_x, counts_madc_y, 'bo', label='MADC')
    axes.set_xlabel("Voltage, V")
    axes.set_ylabel("Efficient counts")
    axes.legend()

    print(counts_x)
    print(counts_madc_x)
    # assert counts_x == counts_madc_x

    fig, axes = plt.subplots()
    fig.canvas.set_window_title("energy_spectrum_compare_ratio")
    fig.suptitle("CAMAC MADC vs. Lan10-12PCI energy spectrums ratio \n"
                 "Set - %s" % (set_path))
    axes.plot(counts_x, 1 - np.array(counts_y) / np.array(counts_madc_y),
              'ro', label='')
    axes.set_xlabel("Voltage, V")
    axes.set_ylabel("MADS vs Lan10_12PCI ratio")


if __name__ == "__main__":
    seaborn.set_context("poster")
    main()
