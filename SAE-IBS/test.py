import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from keras.utils.np_utils import to_categorical
import torch
import scipy.io
import numpy as np
import copy

gpu = True
if gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('Device: ', device)

ref_data_dir = '/home/kevin/autoencoder/SAE-IBS/dataset/timeseries_midi_dataset_all.mat'

savepath = '/home/kevin/autoencoder/SAE-IBS/results/AE/AE_3_Layer_130_Latent_10240_2560_512'
checkpoint = torch.load(savepath + '/bestcheckpoint_AE.pt')  # reload best checkpoint
model = checkpoint['model']
model.load_state_dict(checkpoint['model_state_dict'])

# Load data
def load_data(dir, subdir='train_data', data_type=np.uint8):
    return np.array(scipy.io.loadmat(f'{dir}')[subdir], dtype=np.uint8)

X_train = load_data(ref_data_dir)
X_train = to_categorical(X_train, num_classes=np.max(X_train)+1)

train_loader = DataLoader(torch.from_numpy(X_train), batch_size=256, shuffle=True)


for b, (x_data) in enumerate(train_loader):
    x = x_data.to(device)
    emb = model.encoder(x)
    if b == 0:
        embedding = copy.deepcopy(emb)
    else:
        embedding = torch.cat([embedding, emb], 0)

print('PROBANDO SVD')
print(torch.svd_lowrank(embedding - nn.Parameter(torch.mean(embedding, 0)), 130))
del x, embedding