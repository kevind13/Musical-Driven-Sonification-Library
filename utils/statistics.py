import statistics
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


def find_max_non_outlier(list):
    mean = statistics.mean(list)
    stdev = statistics.stdev(list)

    outliers = []
    for value in list:
        if value < mean - 4 * stdev or value > mean + 4 * stdev:
            outliers.append(value)

    max_non_outlier = None
    for value in list:
        if value not in outliers:
            if max_non_outlier is None or value > max_non_outlier:
                max_non_outlier = value

    return max_non_outlier


def draw_histogram(list, title: Optional[str]= None, x_label: Optional[str]= None, y_label:Optional[str]=None):
    n, bins, patches = plt.hist(x=list, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)

    if x_label is not None: plt.xlabel(x_label)
    if y_label is not None: plt.ylabel(y_label)
    if title is not None: plt.title(title)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
