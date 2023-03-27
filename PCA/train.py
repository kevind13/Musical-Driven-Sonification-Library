import argparse
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import scipy.io
import random

DIMENSIONS = 200

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def train_PCA(y, dimension):
    pca = PCA(n_components=dimension)
    pca.fit(y)

    pca_vectors = pca.components_
    pca_values = pca.explained_variance_

    latent_means = pca.mean_
    latent_stds = np.sqrt(pca_values)

    return pca, pca_vectors, pca_values, latent_means, latent_stds

def train(latent_means_path, latent_stds_path, latent_pca_values_path, latent_pca_vectors_path, dimensions, dataset_mat_path, sample_midi_vectors_path):
    data = np.array(scipy.io.loadmat(dataset_mat_path)['train_data'], dtype=np.float32)
    shape_data = data.shape                                                               # store shape of y before reshaping it
    data = np.reshape(data,[shape_data[0],shape_data[1]*shape_data[2]])/255 
    X_train, _ =train_test_split(data, test_size=0.05)

    pca, pca_vectors, pca_values, latent_means, latent_stds = train_PCA(X_train,dimensions)  


    random_indexs = np.random.randint(low=0, high=X_train.shape[0], size=40)
    samples = X_train[random_indexs]


    np.save(latent_means_path, latent_means)
    np.save(latent_stds_path, latent_stds)
    np.save(latent_pca_values_path, pca_values)
    np.save(latent_pca_vectors_path, pca_vectors)

    np.save(sample_midi_vectors_path, samples)

    return pca



if __name__ == "__main__":
    # configure parser and parse arguments
    parser = argparse.ArgumentParser(description='PCA fit for midi files')
    parser.add_argument('--latent_means_path', default='weights/latent_means.npy', type=str, help='Path to latent means numpy array.')
    parser.add_argument('--latent_stds_path', default='weights/latent_stds.npy', type=str, help='Path to latent stds numpy array.')
    parser.add_argument('--latent_pca_values_path', default='weights/latent_pca_values.npy', type=str, help='Path to pca values numpy array.')
    parser.add_argument('--latent_pca_vectors_path', default='weights/latent_pca_vectors.npy', type=str, help='Path to pca vectors numpy array.')
    parser.add_argument('--dimensions', default=DIMENSIONS, type=int, help='The number of dimensions of latent space for PCA.')
    parser.add_argument('--dataset_mat_path', default='dataset/timeseries_midi_dataset_with_transpose.mat', type=str, help='Path to .mat formatted midis dataset')
    parser.add_argument('--sample_midi_vectors_path', default='samples/samples.npy', type=str, help='Path to store some samples of midi to reproduce later.')

    args = parser.parse_args()
    latent_means_path = args.latent_means_path
    latent_stds_path = args.latent_stds_path
    latent_pca_values_path = args.latent_pca_values_path
    latent_pca_vectors_path = args.latent_pca_vectors_path
    dimensions = args.dimensions
    dataset_mat_path = args.dataset_mat_path
    sample_midi_vectors_path = args.sample_midi_vectors_path
    train(latent_means_path, latent_stds_path, latent_pca_values_path, latent_pca_vectors_path, dimensions, dataset_mat_path, sample_midi_vectors_path)



    # latent_means = np.load(dir_name + sub_dir_name + '/latent_means.npy')
    # latent_stds = np.load(dir_name + sub_dir_name + '/latent_stds.npy')
    # latent_pca_values = np.load(dir_name + sub_dir_name + '/latent_pca_values.npy')
    # latent_pca_vectors = np.load(dir_name + sub_dir_name + '/latent_pca_vectors.npy')