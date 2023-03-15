import matplotlib.pyplot as plt

import numpy as np
import pickle

with open('pcaae_models/evaluation_dense.pickle', 'rb') as f:
    evaluation = pickle.load(f)

test_number = 2

real = evaluation[test_number]['input'][0]
pred = evaluation[test_number]['output']

print(pred.shape)
print(real.shape)

real_midi =  matrix2mid(real)
pred_midi =  matrix2mid(pred)

real_midi.save('midi_test/real.mid')
pred_midi.save('midi_test/pred.mid')