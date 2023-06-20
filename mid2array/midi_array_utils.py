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
                        title = None,
                        titles = None,
                        x_label: Optional[str] = None,
                        y_label: Optional[str] = None,
                        legend: Optional[bool] = None):

    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    colors = ['b', 'g', 'orange', 'r']
    
    for index, data in enumerate([real, reconstructed]):
        for i in range(data.shape[0]):
            notes = np.multiply(np.where(data[i] > 0, 1, 0), range(1, MIDI_NOTE_RANGE + 1)).max(axis=1)
            notes = remove_ending_zeros(notes)
            axs[index].plot(range(notes.shape[0]), notes, marker='_', linestyle='', color=colors[i], label=f'Track {i+1}')
            diff_notes = np.multiply(np.where(np.abs(real[i] - reconstructed[i]) > 0, 1, 0), range(1, MIDI_NOTE_RANGE + 1)).max(axis=1)
            print(diff_notes)
            diff_indices = np.where(diff_notes >= 1)[0]
            # if len(diff_indices) > 0:
            #     axs[index].plot(diff_indices, diff_notes[diff_indices], marker='s', markersize=10, linestyle='', 
            #     markerfacecolor='none', markeredgecolor='red', label='Differences')

        if x_label is not None:
            axs[index].set_xlabel(x_label, fontsize=14)
        if y_label is not None:
            axs[index].set_ylabel(y_label, fontsize=14)
        if titles is not None:
            axs[index].set_title(titles[index], fontsize=16)
        if legend:
            handles, labels = axs[index].get_legend_handles_labels()

            if 'Differences' in labels:
                diff_index = labels.index('Differences')
                # Rearrange the labels and handles to move 'Differences' to the end
                labels = labels[:diff_index] + labels[diff_index+1:] + [labels[diff_index]]
                handles = handles[:diff_index] + handles[diff_index+1:] + [handles[diff_index]]

            axs[index].legend(handles, labels, fontsize=12)
    
        axs[index].tick_params(axis='both', which='major', labelsize=12)

    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.95, va='baseline', multialignment='center')
    
    plt.tight_layout(pad=3)
    plt.show()
