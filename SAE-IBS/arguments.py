# Utility functions & configuration constants
import os
import argparse
import json
import torch.cuda

def parse_args():
    base = '/Users/kevindiaz/Desktop/SonificationThesis/SAE-IBS/dataset/'

    parser = argparse.ArgumentParser(description='PyTorch SAE-IBS')

    # general
    parser.add_argument('--model_name', default='SAEIBS', type=str, help='name of experiment')
    parser.add_argument('--seed', default=69, type=int, help='set random seed')
    parser.add_argument('--train', default=True, type=bool, help='training mode')
    parser.add_argument('--test', default=False, type=bool, help='test mode')
    parser.add_argument('--cuda', action='store_true', default=True and torch.cuda.is_available(),
                        help='enables CUDA training')

    # data
    parser.add_argument('--ref_data_dir',
                        default=base + 'timeseries_midi_dataset_all.mat',
                        type=str, metavar='PATH', help='training data path')

    parser.add_argument('--pretrain_model_type', default='AE',
                        type=str, help='pretrain model type')

    # training hyperparameters
    parser.add_argument('--actFn', default='sigmoid', type=str, help='activation function of last layer')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxNum_epochs', default=3000, type=int)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--loss_opt', default='BCE', type=str)

    # network hyperparameters
    parser.add_argument('--latent_dim', default=130, type=int)
    parser.add_argument('--hidden_dim', default=[512*20, 512*5, 512], type=int)
    parser.add_argument('--drop_rate', default=0.1, type=int)
    parser.add_argument('--cond_dropout', default=False, type=bool, help='add dropout layer')

    # optimizer hyperparmeters
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_decay', default=0.9999, type=float)
    parser.add_argument('--decay_step', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-6, type=float)

    args = parser.parse_args()
    return args


def save_args(args, path):
    """Saves parameters to json file"""
    json_path = "{}/args.json".format(path)
    with open(json_path, 'w') as f:
        json.dump(vars(args), f, indent=4)


def load_args(json_path):
    """Loads parameters from json file"""
    with open(json_path) as f:
        params = json.load(f)
    return argparse.Namespace(**params)
