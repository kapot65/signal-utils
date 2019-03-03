import pandas as pd
import matplotlib.pyplot as plt  # plotting library
import seaborn as sns  # matplotlib grahs visual enchancer


def read_envelope(filename):
    with open(filename) as fin:
        nSkip = 0
        names = []
        for line in fin:
            nSkip += 1
            if line.startswith("#f"):
                names = line.split()[1:]
                break
    return pd.read_table(filename, skiprows=nSkip, sep="\\s+", names=names)


if __name__ == "__main__":

    sns.set_context("poster")

    fig, axes = plt.subplots()

    data = read_envelope("/home/chernov/Downloads/group_6_up.out")

    for point in (p for p in data.columns.values if p != "channel"):
        axes.step(data["channel"], data[point], where='mid', label=point)

    axes.grid(which='minor', alpha=0.3)
    axes.grid(which='major', alpha=0.7)
    axes.xaxis.set_major_locator(plt.MultipleLocator(512))
    axes.xaxis.set_minor_locator(plt.MultipleLocator(64))
    axes.legend()
    plt.show()
