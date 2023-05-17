import os
import sys
sys.path.append("/Users/kevindiaz/Desktop/SonificationThesis")
from mid2array.mid2array import mid2arry
from mid2array.midi_array_utils import compare_midi_arrays
from mid2matrix.matrix2mid import matrix2mid
import numpy as np

import warnings

from utils.constants import MIDI_BOTTOM_NOTE, MIDI_GCD_TIME, MIDI_TOP_NOTE
warnings.filterwarnings('ignore')

import torch
from torch import nn

from torch import optim
import itertools

import scipy.io

import argparse

import pickle

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import random

gpu=False
path_data = 'dataset'


class Encoder(nn.Module):
    def __init__(self, code_size=1, input_shape=(128, 4, 89)):
    # def __init__(self, code_size=1, input_shape=(128, 4, 91)):
        super(Encoder, self).__init__()        
        self.latent_dim = code_size
        self.input_shape = input_shape

        input_size = input_shape[0] * input_shape[1] * input_shape[2]

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 512*20)
        self.fc2 = nn.Linear(512*20, 512*5)
        self.fc3 = nn.Linear(512*5, 512)
        self.fc4 = nn.Linear(512, code_size)

        self.zero_mean = nn.BatchNorm1d(self.latent_dim, affine=False, eps=0)
        self.relu = nn.ReLU()
    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(batch_size, -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        z = x.view((batch_size, -1))

        z = self.zero_mean(z)
        return z  
    
class Decoder(nn.Module):
    def __init__(self, code_size=1, output_shape=(128, 4, 89)):
    # def __init__(self, code_size=1, output_shape=(128, 4, 91)):
        super(Decoder, self).__init__()
        # Shape required to start transpose convs
        self.output_shape = output_shape

        output_size = output_shape[0] * output_shape[1] * output_shape[2]

        # Fully connected layers
        self.fc1 = nn.Linear(code_size, 512)
        self.fc2 = nn.Linear(512, 512*5)
        self.fc3 = nn.Linear(512*5, 512*20)
        self.fc4 = nn.Linear(512*20, output_size)

        self.relu = nn.ReLU()
        
    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        # Reshape output
        x = x.view(batch_size, *self.output_shape)
        x = nn.Sigmoid()(x)
        return x


train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

if gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# X_train = np.array(scipy.io.loadmat(f'{path_data}/all_4_channels_data_01_short.mat')['train_data'], dtype=np.uint8)
# X_train = np.array(scipy.io.loadmat(f'{path_data}/timeseries_midi_dataset_with_transpose.mat')['train_data'], dtype=np.uint8)
X_train = np.array(scipy.io.loadmat(f'{path_data}/timeseries_midi_dataset_with_same_key.mat')['train_data'], dtype=np.uint8)
# X_train = np.where(X_train == 90, 1, 0)
X_train = to_categorical(X_train, num_classes=np.max(X_train)+1)

x_train, x_test =train_test_split(X_train, test_size=0.2, random_state=69)
x_val_loader, x_test_loader = train_test_split(x_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=69) 


x_train = np.array(random.choices(x_train, k=20))

print(x_train.shape)

testing_dataset = torch.stack([torch.Tensor(i) for i in x_train])
testing_dataset = torch.utils.data.DataLoader(testing_dataset,batch_size=1,shuffle=True)

input_size = x_train.shape[1]

PCAAE_E = []

latent_dimension = 130
weight = 'weights_ae'
evaluation = 'evaluation_ae'

checkpoint = torch.load(f'{weight}/AE_midi',map_location=device)

model_PCAAE_E = Encoder(code_size=latent_dimension).to(device)
model_PCAAE_E.load_state_dict(checkpoint['AE_E_state_dict']) 

model_PCAAE_D = Decoder(code_size=latent_dimension).to(device)
model_PCAAE_D.load_state_dict(checkpoint['AE_D_state_dict']) 


test_eval = {}
for index, x in enumerate(testing_dataset):
    with torch.no_grad():
        latent_space = model_PCAAE_E.eval()(x)

    recon_data = model_PCAAE_D(latent_space)
    output = recon_data.detach().numpy()
    # output = np.rint(output).astype(int)
    test_eval[index] = {'input': x.detach().numpy(), 'latent_space': latent_space.detach().numpy(), 'output': output}

    print(test_eval[index]['input'].shape)
    assert False, 'a'
    real_midi =  matrix2mid(X_test)
    pred_midi =  matrix2mid(X_pred)

    _, real = mid2arry(real_midi, block_size=MIDI_GCD_TIME, truncate_range=(MIDI_BOTTOM_NOTE,MIDI_TOP_NOTE))
    _, pred = mid2arry(pred_midi, block_size=MIDI_GCD_TIME, truncate_range=(MIDI_BOTTOM_NOTE,MIDI_TOP_NOTE))

    compare_midi_arrays(real, pred, x_label='Ticks / GCD', y_label='MIDI Notes', titles=['Real MIDI', 'Reconstructed MIDI'], legend=True, title='VAE - Comparison between Real and Reconstructed MIDI')
    print(f'{index}')

with open(f'evaluation/{evaluation}.pickle', 'wb') as handle:
        pickle.dump(test_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)


print(test_eval[3])