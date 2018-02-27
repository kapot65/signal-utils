#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drawing events from raw dataforge point.
"""
from argparse import ArgumentParser

import dfparser  # Numass files parser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # matplotlib grahs visual enchancer
from matplotlib.widgets import Slider


def __parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Input raw dataforge point.')
    return parser.parse_args()


def _main():
    # Parse arguments from command line
    args = __parse_args()

    # Read dataforge point
    _, _, data = dfparser.parse_from_file(args.input)

    # Parse Binary data
    point = dfparser.Point()
    point.ParseFromString(data)

    # Extract raw frames from each block
    frames = []
    for channel in point.channels:
        for block in channel.blocks:
            for frame in block.frames:
                frames.append(np.frombuffer(frame.data, np.int16))

    # Draw graph
    fig, axes = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax_frame = plt.axes([0.1, 0.10, 0.8, 0.03])
    # Add slider on graph
    s_freq = Slider(ax_frame, 'Frame', 1, len(frames) - 1, valinit=0)

    cur_plot, = axes.plot(frames[0])

    def update(_):
        """Update frame in graph."""
        # Get current frame number from slider
        frame = int(round(s_freq.val))
        # Change frame data
        cur_plot.set_ydata(frames[frame])
        cur_plot.set_xdata(np.arange(len(frames[frame])))
        # Redraw plot
        fig.canvas.draw_idle()

    # Bind update function to slider
    s_freq.on_changed(update)
    plt.show()


if __name__ == "__main__":
    sns.set_context("poster")
    _main()
