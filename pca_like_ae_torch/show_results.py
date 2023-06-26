import os
import sys
sys.path.append("/Users/kevindiaz/Desktop/SonificationThesis")
from mid2array.mid2array import mid2arry
from mid2array.midi_array_utils import compare_midi_arrays
from mid2matrix.matrix2mid import matrix2mid

import numpy as np

import warnings
warnings.filterwarnings('ignore')

import pickle

with open('evaluation/evaluation.pickle', 'wb') as handle:
        pickle.dump(test_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)