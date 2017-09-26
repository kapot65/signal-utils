"""Comparison of MADC and Lan10-12PCI acquired spectrum for several points.

Input:
- MADC spectrum - https://drive.google.com/open?id=0B0Ux_fvsLMdAT0cxSm1qZzFKcDQ
- Lan10-12PCI spectrum - set_43, 14 - 17 kV

Variables:
data - path to MADC data
df_data_root - path to Lan10-12PCI root folder
points_path - list for 14-17 kV points.


Output:
- points spectrum diffrerence graphs for every point.
"""
# pylint: disable-msg=R0914,W0640

from os import path

import dfparser
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import seaborn


def main():
    """Compare Lan10-12PCI points energy spectrum for several points."""
    seaborn.set_context("poster")

    data = np.genfromtxt("/home/chernov/Downloads/set_43_detector.out")

    x_data = data[:, 0]

    y_points = data[:, 1:].transpose()
    y_points = (y_points.T / np.max(y_points, axis=1)).T

    df_data_root = "/home/chernov/data_processed"

    points_path = ["2017_05/Fill_3/set_43/p0(30s)(HV1=16000).df",
                   "2017_05/Fill_3/set_43/p36(30s)(HV1=17000).df",
                   "2017_05/Fill_3/set_43/p80(30s)(HV1=15000).df",
                   "2017_05/Fill_3/set_43/p102(30s)(HV1=14000).df"]

    bins = 500
    range_ = (0, 8000)

    for idx, point_rel in enumerate(points_path):
        _, _, data = dfparser.parse_from_file(path.join(df_data_root,
                                                        point_rel))

        point = dfparser.Point()
        point.ParseFromString(data)

        amps = np.hstack([list(block.events.amplitudes)
                          for block in point.channels[0].blocks])

        hist, x_point = np.histogram(amps, bins=bins, range=range_)
        hist = hist / np.max(hist[np.where(x_point > 3000)[0][0]:])

        func = interp1d(x_point[1:] + x_point[:-1], hist,
                        bounds_error=False, fill_value=0)

        func_mod = lambda x, a, b, c: c*func(a*x + b)

        x_peak = np.where(np.logical_and(x_data > 1000, x_data < 1600))
        popt, _ = curve_fit(func_mod, x_data[x_peak], y_points[idx][x_peak],
                            p0=[3.68, 700, 1])

        fig, axes = plt.subplots()
        fig.canvas.set_window_title(point_rel)
        fig.suptitle("CAMAC MADC vs. Lan10-12PCI spectrums")
        axes.set_title("File - %s. \nOptimized parameters: a=%s, b=%s, c=%s" %
                       (point_rel, *np.round(popt, 2)))
        axes.set_xlabel("Bins, ch")
        axes.set_xlim(0, 2000)
        # axes.set_yscale("log", nonposx='clip')
        x_interp = np.linspace(0, 2000, 500)
        axes.plot(x_interp, func_mod(x_interp, *popt), label="Lan10-12PCI")
        axes.plot(x_data, y_points[idx], label="CAMAC MADC")
        axes.legend()


if __name__ == "__main__":
    main()
