import numpy as np

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn

from torch import optim
import itertools

import scipy.io

import argparse

import pickle

from sklearn.model_selection import train_test_split

import random

gpu=False
path_data = 'dataset'

class Encoder(nn.Module):
    def __init__(self, code_size=1, input_shape=(4, 256, 68)):
        super(Encoder, self).__init__()        
        self.latent_dim = code_size
        self.input_shape = input_shape

        input_size = input_shape[0] * input_shape[1] * input_shape[2]

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, code_size)

        self.zero_mean = nn.BatchNorm1d(self.latent_dim, affine=False, eps=0)
        self.leaky = nn.LeakyReLU(0.2)
    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(batch_size, -1)

        x = self.leaky(self.fc1(x))
        x = self.fc2(x)
        
        z = x.view((batch_size, -1))

        z = self.zero_mean(z)
        return z  
    
class Decoder(nn.Module):
    def __init__(self, code_size=1, output_shape=(4, 256, 68)):
        super(Decoder, self).__init__()
        # Shape required to start transpose convs
        self.output_shape = output_shape

        output_size = output_shape[0] * output_shape[1] * output_shape[2]

        # Fully connected layers
        self.fc1 = nn.Linear(code_size, 512)
        self.fc2 = nn.Linear(512, output_size)

        self.leaky = nn.LeakyReLU(0.2)
        
    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = self.leaky(self.fc1(z))
        x = self.fc2(x)

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


X_train = np.array(scipy.io.loadmat(f'{path_data}/all_4_channels_data_01_short.mat')['train_data'], dtype=np.uint8)
X_train = np.where(X_train == 90, 1, 0)
x_train, x_test =train_test_split(X_train, test_size=0.2, random_state=69)
x_val_loader, x_test_loader = train_test_split(x_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=69) 


x_train = np.array(random.choices(x_train, k=20))

print(x_train.shape)

testing_dataset = torch.stack([torch.Tensor(i) for i in x_train])
testing_dataset = torch.utils.data.DataLoader(testing_dataset,batch_size=1,shuffle=True)

input_size = x_train.shape[1]

latent_dimension = 40
PCAAE_E = []

for dim in range(1,latent_dimension + 1):
    checkpoint_PCAAE = torch.load(f'weights/PCAAE{dim}midi_lambdacov1',map_location=device)
    model_PCAAE_E = Encoder().to(device)
    model_PCAAE_E.load_state_dict(checkpoint_PCAAE['PCAAE_E_state_dict']) 
    PCAAE_E.append(model_PCAAE_E)


checkpoint_PCAAE = torch.load('weights/PCAAE40midi_lambdacov1',map_location=device)
PCAAE_D = Decoder(code_size=latent_dimension).to(device)
PCAAE_D.load_state_dict(checkpoint_PCAAE['PCAAE_D_state_dict']) 


test_eval = {}
for index, x in enumerate(testing_dataset):
    with torch.no_grad():
        latent_space = torch.cat((
            PCAAE_E[0].eval()(x),
            PCAAE_E[1].eval()(x),
            PCAAE_E[2].eval()(x),
            PCAAE_E[3].eval()(x),
            PCAAE_E[4].eval()(x),
            PCAAE_E[5].eval()(x),
            PCAAE_E[6].eval()(x),
            PCAAE_E[7].eval()(x),
            PCAAE_E[8].eval()(x),
            PCAAE_E[9].eval()(x),
            PCAAE_E[10].eval()(x),
            PCAAE_E[11].eval()(x),
            PCAAE_E[12].eval()(x),
            PCAAE_E[13].eval()(x),
            PCAAE_E[14].eval()(x),
            PCAAE_E[15].eval()(x),
            PCAAE_E[16].eval()(x),
            PCAAE_E[17].eval()(x),
            PCAAE_E[18].eval()(x),
            PCAAE_E[19].eval()(x),
            PCAAE_E[20].eval()(x),
            PCAAE_E[21].eval()(x),
            PCAAE_E[22].eval()(x),
            PCAAE_E[23].eval()(x),
            PCAAE_E[24].eval()(x),
            PCAAE_E[25].eval()(x),
            PCAAE_E[26].eval()(x),
            PCAAE_E[27].eval()(x),
            PCAAE_E[28].eval()(x),
            PCAAE_E[29].eval()(x),
            PCAAE_E[30].eval()(x),
            PCAAE_E[31].eval()(x),
            PCAAE_E[32].eval()(x),
            PCAAE_E[33].eval()(x),
            PCAAE_E[34].eval()(x),
            PCAAE_E[35].eval()(x),
            PCAAE_E[36].eval()(x),
            PCAAE_E[37].eval()(x),
            PCAAE_E[38].eval()(x),
            PCAAE_E[39].eval()(x),
        ),dim=-1)

    recon_data = PCAAE_D(latent_space)
    output = recon_data.detach().numpy()
    # output = np.rint(output).astype(int)
    test_eval[index] = {'input': x.detach().numpy(), 'latent_space': latent_space.detach().numpy(), 'output': output}

    print(f'{index}')

with open('evaluation/evaluation.pickle', 'wb') as handle:
        pickle.dump(test_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)


print(test_eval[3])