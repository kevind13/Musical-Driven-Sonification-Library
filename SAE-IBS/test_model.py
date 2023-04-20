import os
import sys

sys.path.append("/Users/kevindiaz/Desktop/SonificationThesis/SAE-IBS/dataset/")
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch.backends.cudnn as cudnn
import pandas
from functions import *
from model import Autoencoder
from plot_functions import plot_latent
from arguments import parse_args, save_args
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import scipy.io as sio

gpu = True
if gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('Device: ', device)
# set deterministic
torch.manual_seed(69)
np.random.seed(69)
cudnn.deterministic = True


def main():
    args = parse_args()
    args.work_dir = os.path.dirname(os.path.realpath(__file__))
    run_parameters = str(len(args.hidden_dim)) + '_Layer_' + str(args.latent_dim) + '_Latent_' + "_".join(str(x) for x in args.hidden_dim)
    args.out_dir = os.path.join(args.work_dir, 'results/' + args.model_name + '/' + args.model_name + '_' + run_parameters)
    args.checkpoints_dir = os.path.join(args.out_dir, 'checkpoints')

    path_model = args.out_dir + '/' + args.model_name + '_' + run_parameters + '.tar'

    print(path_model)
    checkpoint = torch.load(path_model) 
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])

    latent_path = args.out_dir + '/latent'

    project_mat = sio.loadmat(latent_path + '/latent_project.mat')

    latent_project = project_mat['latent_project']
    test_data = project_mat['test_data']
    pred_data = project_mat['pred_data']

    testing_dataset = DataLoader(torch.from_numpy(test_data),batch_size=1,shuffle=False)

    for index, (x) in enumerate(testing_dataset):
        data = x.to(device)
        if 'VariationalAutoencoder' in type(model).__name__:
            X_pred, z, _, _ = model(data)
        else:
            X_pred, z = model(data)
        X_pred = X_pred.to('cpu').detach().numpy()
        X_pred = np.argmax(X_pred.reshape((-1, X_pred.shape[-1])), axis=-1).reshape((X_pred.shape[1],X_pred.shape[2]))

        X_test = data.to('cpu').detach().numpy()
        X_test = np.argmax(X_test.reshape((-1, X_test.shape[-1])), axis=-1).reshape((X_test.shape[1],X_test.shape[2]))

        print(X_test[:3])
        print(X_pred[:3])

        print('Now from the lastent space that is saved from the training')

        latent_test = torch.from_numpy(np.array([latent_project[index]])).to(device)

        if 'SAE' in type(model).__name__:
            X_recon = model.decoder_svd(latent_test)
        else:
            X_recon = model.decoder(latent_test)

        X_recon = X_recon.to('cpu').detach().numpy()
        X_recon = np.argmax(X_recon.reshape((-1, X_recon.shape[-1])), axis=-1).reshape((X_recon.shape[1],X_recon.shape[2]))

        print(X_recon[:3])

        if index == 3:
            break

if __name__ == '__main__':
    main()

#python test_model.py --pretrain_epochs 10 --ref_data_dir /home/kevin/autoencoder/SAE-IBS/dataset/timeseries_midi_dataset_all.mat --batch_size 256 --latent_dim 130 --model_name "AE" --maxNum_epochs 100 --patience 100