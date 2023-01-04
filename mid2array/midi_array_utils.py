from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from utils.constants import MIDI_NOTE_RANGE

def remove_ending_zeros(arr):
    index = len(arr) - 1
    while index >= 0 and arr[index] == 0:
        index -= 1
    return arr[:index+1]

def plot_midi_array(array, title: Optional[str]= None, x_label: Optional[str]= None, y_label:Optional[str]=None):
    fig = plt.figure()
    
    colors = ['b', 'g', 'orange', 'r']
    for i in range(array.shape[0]):
        notes = np.multiply(np.where(array[i]>0, 1, 0), range(1, MIDI_NOTE_RANGE + 1)).sum(axis=1)
        notes = remove_ending_zeros(notes)
        plt.plot(range(notes.shape[0]), notes, marker='_', linestyle='', color=colors[i])
    ## TODO: need to add the axis labels and each variable labels
    plt.title(title)
    plt.show()