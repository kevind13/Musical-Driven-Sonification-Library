from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from utils.constants import MIDI_NOTE_RANGE


def remove_ending_zeros(arr):
    index = len(arr) - 1
    while index >= 0 and arr[index] == 0:
        index -= 1
    return arr[:index + 1]


def plot_midi_array(array,
                    title: Optional[str] = None,
                    x_label: Optional[str] = None,
                    y_label: Optional[str] = None,
                    legend: Optional[bool] = None):
    fig = plt.figure()

    colors = ['b', 'g', 'orange', 'r']
    for i in range(array.shape[0]):
        notes = np.multiply(np.where(array[i] > 0, 1, 0), range(1, MIDI_NOTE_RANGE + 1)).max(axis=1)
        print(notes)
        notes = remove_ending_zeros(notes)
        plt.plot(range(notes.shape[0]), notes, marker='_', linestyle='', color=colors[i], label=f'Track {i+1}')

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    if legend:
        plt.legend()
    plt.show()


def compare_midi_arrays(real,
                        reconstructed,
                        titles = None,
                        x_label: Optional[str] = None,
                        y_label: Optional[str] = None,
                        legend: Optional[bool] = None):

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

    colors = ['b', 'g', 'orange', 'r']
    
    for index, data in enumerate([real, reconstructed]):
        for i in range(data.shape[0]):
            notes = np.multiply(np.where(data[i] > 0, 1, 0), range(1, MIDI_NOTE_RANGE + 1)).max(axis=1)
            notes = remove_ending_zeros(notes)
            axs[index].plot(range(notes.shape[0]), notes, marker='_', linestyle='', color=colors[i], label=f'Track {i+1}')

        if x_label is not None:
            axs[index].set_xlabel(x_label)
        if y_label is not None:
            axs[index].set_ylabel(y_label)
        if titles is not None:
            axs[index].set_title(titles[index])
        if legend:
            axs[index].legend()


    plt.tight_layout()
    plt.show()