"""Zero supperssion testing."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from signal_utils.convert_utils import apply_zsupression
from signal_utils.generation_utils import gen_raw_block

if __name__ == "__main__":
    seaborn.set_context("poster")
    threshold = 700
    area_l = 50
    area_r = 100

    x, params = gen_raw_block(b_size=1500)
    filtered = list(apply_zsupression(x, threshold, area_l, area_r))

    fig, ax = plt.subplots()
    ax.plot(x)
    ax.set_title("Zsuppression th = %s area_r = %s area_l = %s" % (
        threshold, area_l, area_r
    ))
    ax.set_xlabel("Position, bins")
    ax.set_ylabel("Amplitude, labels")
    ax.axhline(threshold, label="Threshold", ls='--',
               color=seaborn.xkcd_rgb["pale red"])
    ax.grid(color='lightgray', alpha=0.7)
    for area in filtered:
        ax.axvspan(area[0], area[1], alpha=0.5, color='green')
    ax.legend()

    plt.show()
