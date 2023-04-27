import os
import sys
import torch
import scipy.io as sio
import numpy as np
from functions import *
from keras.utils.np_utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt


out_dir = 'results/SAE/SAE_3_Layer_7_Latent_9216_2560_512'
path_model = f'{out_dir}/SAE_3_Layer_7_Latent_9216_2560_512.tar'
path_latent = f'{out_dir}/latent/latent_project.mat'
path_complete_latent = f'{out_dir}/latent/all_latent_project.mat'

device = torch.device('cpu')

project_mat = sio.loadmat(path_latent)


complete_project_mat = sio.loadmat(path_complete_latent)['latent_project']

complete_project_mat = np.transpose(complete_project_mat, (1,0))

print(complete_project_mat.shape)

print(complete_project_mat[0][:10])

from scipy.stats import zscore

mean = np.mean(complete_project_mat[0])
std = np.std(complete_project_mat[0])
z_scores = zscore(complete_project_mat[0])

x = z_scores * std + mean

print(z_scores[:10])

print(x[:10])

print(np.mean(z_scores))
print(np.std(z_scores))
print(np.min(z_scores), np.max(z_scores))

print(mean, std)
print(np.min(complete_project_mat[0]), np.max(complete_project_mat[0]))


print(f'Cinco estaría mapeado a: {(np.mean(z_scores) + np.std(z_scores) * 5) * std + mean}')
print(f'Cuatro estaría mapeado a: {(np.mean(z_scores) + np.std(z_scores) * 4) * std + mean}')
print(f'Tres estaría mapeado a: {(np.mean(z_scores) + np.std(z_scores) * 3) * std + mean}')
print(f'Dos estaría mapeado a: {(np.mean(z_scores) + np.std(z_scores) * 2) * std + mean}')
print(f'Uno estaría mapeado a: {(np.mean(z_scores) + np.std(z_scores) * 1) * std + mean}')
print(f'Cero estaría mapeado a: {(np.mean(z_scores) + np.std(z_scores) * 0) * std + mean}')


print(f'Menos Uno estaría mapeado a: {(np.mean(z_scores) - np.std(z_scores) * 1) * std + mean}')
print(f'Menos Dos estaría mapeado a: {(np.mean(z_scores) - np.std(z_scores) * 2) * std + mean}')
print(f'Menos Tres estaría mapeado a: {(np.mean(z_scores) - np.std(z_scores) * 3) * std + mean}')
print(f'Menos Cuatro estaría mapeado a: {(np.mean(z_scores) - np.std(z_scores) * 4) * std + mean}')
print(f'Menos Cinco estaría mapeado a: {(np.mean(z_scores) - np.std(z_scores) * 5) * std + mean}')

labels = ['Comp. 1', 'Comp. 2', 'Comp. 3', 'Comp. 4', 'Comp. 5', 'Comp. 6', 'Comp. 7']
df = pd.DataFrame(data=np.array(complete_project_mat).T, columns=labels)
sns.boxplot(data=df)

plt.xlabel('Principal components', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(fontsize=10)
plt.savefig(out_dir + '/Latent' + '_box_plot' + "_" + '3_Layer_7_Latent_9216_2560_512' + '.jpg')

plt.show()
'''
# This is to get all the latent space for all the dataset

X_train = load_data('dataset/timeseries_midi_dataset_all.mat')
X_train = to_categorical(X_train, num_classes=np.max(X_train)+1)

checkpoint = torch.load(path_model, map_location=torch.device('cpu'))  
model = checkpoint['model']
model.load_state_dict(checkpoint['model_state_dict'])

X_pred, latent_project = projectSAE(model, X_train, 256, 7, device)

all_latent_path = out_dir + '/latent'
if not os.path.exists(all_latent_path):
    os.makedirs(all_latent_path)

# save latent
savemat(all_latent_path + '/all_latent_project.mat', mdict={'latent_project': latent_project})
'''

'''
# This is to test the latent space with the real and pred data already saved when the model was trained
latent_project = project_mat['latent_project']
test_data = project_mat['test_data']
pred_data = project_mat['pred_data']

for z in latent_project:
    latent_test = torch.from_numpy(np.array([z]))

    if 'SAE' in type(model).__name__:
        X_recon = model.decoder_svd(latent_test)
    else:
        X_recon = model.decoder(latent_test)

    X_recon = X_recon.detach().numpy() 
    X_recon = np.argmax(X_recon.reshape((-1, X_recon.shape[-1])), axis=-1).reshape((X_recon.shape[1],X_recon.shape[2]))


    t = np.argmax(test_data[0], axis=-1).reshape((test_data[0].shape[0],test_data[0].shape[1]))
    print(t[:10])
    print(X_recon[:10])

    break
'''

# python main_SAE.py --pretrain_epochs 150 --ref_data_dir /home/kevin/autoencoder/SAE-IBS/dataset/timeseries_midi_dataset_all.mat --batch_size 256 --latent_dim 7 --model_name "SAE" --maxNum_epochs 150 --patience 100



