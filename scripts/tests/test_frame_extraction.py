from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from pylab import rcParams

from signal_utils import extract_utils
from signal_utils.generation_utils import gen_multiple, generate_noise


def _parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-m',
        '--method',
        default='extract_simple_amps',
        help='extracting method name (default - "extract_simple_amps")')

    return parser.parse_args()


FRAME_LEN = 200
THRESHOLD = 700
FREQ = 3125000.0
EVENTS = np.array([
    2000, 20 / FREQ,
    1000, 26 / FREQ,
    1000, 36 / FREQ,
    1000, 60 / FREQ,
])

if __name__ == '__main__':
    # seaborn.set_context("poster")
    np.random.seed(0)

    ARGS = _parse_args()
    method = getattr(extract_utils, ARGS.method)

    x = np.linspace(0, FRAME_LEN / FREQ, FRAME_LEN)
    y = gen_multiple(x, *EVENTS) + generate_noise(x)

    events, _ = method(y, 0, THRESHOLD, FREQ)
    print(events)

    fig, ax = plt.subplots()

    ax.set_title(ARGS.method)
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Amplitude, ch")

    ax.plot(x, y)
    ax.plot(EVENTS[1::2], EVENTS[0::2], 'bo', label="Event")
    ax.plot(events[1::2] * 1e-9, events[0::2], 'ro', label="Extracted")

    for idx in range(len(events) // 2):
        ax.plot(x, gen_multiple(
            x, events[idx * 2], events[idx * 2 + 1] * 1e-9))

    ax.grid(color='lightgray', alpha=0.7)
    ax.legend()

    plt.show()
