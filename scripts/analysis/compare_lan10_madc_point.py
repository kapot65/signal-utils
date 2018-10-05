"""Calculate scale factor and offset between Lan10-12PCI and MADC spectrums."""
from argparse import ArgumentParser
from glob import glob
from os import path
from struct import unpack

import dfparser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from utils import _lan_amps_f, _madc_amps_f


def __parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('lan10_root', help='Path to Lan10-12PCI root folder.')
    parser.add_argument('madc_root', help='Path to MADC root folder.')
    parser.add_argument('set', help='Path to set to compare.')

    parser.add_argument(
        '-l', '--lan-border', type=int, default=0,
        help='histogram left border (default=0).')
    parser.add_argument(
        '-r', '--right-border', type=int, default=4096,
        help='histogram right border (default=4096).')
    parser.add_argument(
        '-b', '--bins', type=int, default=100,
        help='histogram bins number (default=100).')

    return parser.parse_args()


def _main():
    args = __parse_args()

    sns.set_context("poster")

    madc_amps = np.array(_madc_amps_f(path.join(args.madc_root, args.point)))
    lan_amps = _lan_amps_f(path.join(args.lan10_root, args.point + '.df'))

    hist_m, bins = np.histogram(
        madc_amps, bins=args.bins,
        range=(args.left_border, args.right_border))
    hist_l, bins = np.histogram(
        (lan_amps + args.offset) * args.scale, bins=args.bins,
        range=(args.left_border, args.right_border))

    # amps2 = (lan_amps + args.offset) * args.scale
    # print(np.sum(np.logical_and(amps2 > 500, amps2 < 2000)))
    # print(np.sum(np.logical_and(madc_amps > 500, madc_amps < 2000)))

    bins_centers = (bins[:-1] + bins[1:]) / 2

    _, axes = plt.subplots()

    axes.set_title('{} histogram compare'.format(args.point))
    axes.set_xlabel("Channels, ch")
    axes.set_ylabel("Counts")

    axes.step(bins_centers, hist_m, where='mid', label='MADC')
    axes.step(
        bins_centers, hist_l, where='mid',
        label='Lan10-12PCI ({} scale, {} offset)'.format(
            args.scale, args.offset))

    axes.legend()

    plt.show()

    # data = np.genfromtxt("/home/chernov/Downloads/set_43_detector.out")
    #
    # x_data = data[:, 0]
    #
    # y_points = data[:, 1:].transpose()
    # y_points = (y_points.T / np.max(y_points, axis=1)).T
    #
    # df_data_root = "/home/chernov/data_processed"
    #
    # points_path = ["2017_05/Fill_3/set_43/p0(30s)(HV1=16000).df",
    #                "2017_05/Fill_3/set_43/p36(30s)(HV1=17000).df",
    #                "2017_05/Fill_3/set_43/p80(30s)(HV1=15000).df",
    #                "2017_05/Fill_3/set_43/p102(30s)(HV1=14000).df"]
    #
    # bins = 500
    # range_ = (0, 8000)
    #
    # for idx, point_rel in enumerate(points_path):
    #     _, _, data = dfparser.parse_from_file(path.join(df_data_root,
    #                                                     point_rel))
    #
    #     point = dfparser.Point()
    #     point.ParseFromString(data)
    #
    #     amps = np.hstack([list(block.events.amplitudes)
    #                       for block in point.channels[0].blocks])
    #
    #     hist, x_point = np.histogram(amps, bins=bins, range=range_)
    #     hist = hist / np.max(hist[np.where(x_point > 3000)[0][0]:])
    #
    #     func = interp1d(x_point[1:] + x_point[:-1], hist,
    #                     bounds_error=False, fill_value=0)
    #
    #     def func_mod(x, a, b, c): return c * func(a * x + b)
    #
    #     x_peak = np.where(np.logical_and(x_data > 1000, x_data < 1600))
    #     popt, _ = curve_fit(func_mod, x_data[x_peak], y_points[idx][x_peak],
    #                         p0=[3.68, 700, 1])
    #
    #     fig, axes = plt.subplots()
    #     fig.canvas.set_window_title(point_rel)
    #     fig.suptitle("CAMAC MADC vs. Lan10-12PCI spectrums")
    #     axes.set_title("File - %s. \nOptimized parameters: a=%s, b=%s, c=%s" %
    #                    (point_rel, *np.round(popt, 2)))
    #     axes.set_xlabel("Bins, ch")
    #     axes.set_xlim(0, 2000)
    #     # axes.set_yscale("log", nonposx='clip')
    #     x_interp = np.linspace(0, 2000, 500)
    #     axes.plot(x_interp, func_mod(x_interp, *popt), label="Lan10-12PCI")
    #     axes.plot(x_data, y_points[idx], label="CAMAC MADC")
    #     axes.legend()


if __name__ == "__main__":
    _main()
