import tensorflow as tf
from tensorflow.keras import layers, activations, losses, optimizers
import numpy as np
import scipy.io
import itertools
from sklearn.model_selection import train_test_split
import time
import pickle

batchsize = 4
gpu = True
is_training = True
num_epoch = 30
lr = 0.001
lambda_cov = 1
lambda_rec = 1

autoencoder = False


class Encoder(tf.keras.Model):

    def __init__(self, latent_space_dimension):
        super(Encoder, self).__init__()
        self.latent_space_dimension = latent_space_dimension

        self.lstm1 = tf.keras.layers.LSTM(16, return_sequences=True, activation='selu')
        self.lstm2 = tf.keras.layers.LSTM(self.latent_space_dimension, return_sequences=False, activation='selu')
        
        self.bn = layers.BatchNormalization()


    def call(self, x):
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.bn(x)
        return x


class Decoder(tf.keras.Model):

    def __init__(self, latent_space_dimension, output_size):
        super(Decoder, self).__init__()
        # Shape required to start transpose convs
        self.reshape = input_shape
        self.lstm1 = layers.RepeatVector(128)
        self.lstm2 = tf.keras.layers.LSTM(16, return_sequences=True, activation='selu')
        self.lstm3 = layers.TimeDistributed(tf.keras.layers.Dense(4, activation='linear'))

    def call(self, x):
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        return x


class AutoEncoder(tf.keras.Model):

    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x, training=None, **kwargs):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def reconstruction_loss(data, recon_data, MSE=False):
    if MSE:
        return losses.MeanSquaredError()(data, recon_data)
    else:
        return losses.MeanAbsoluteError()(data, recon_data)


def cov_loss(z, step):
    if step > 1:
        loss = 0
        for idx in range(step - 1):
            loss += tf.reduce_mean(tf.multiply(z[idx], z[-1]))**2
        loss = loss / (step - 1)
    else:
        loss = tf.zeros_like(z)
    return tf.reduce_mean(loss)


def train_AE(AE, optimizer, epoch, train_ds, test_ds):
    train_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()

    for _, data in enumerate(train_ds):
        with tf.GradientTape() as tape:
            recon_data = AE(data)
            loss_value = reconstruction_loss(recon_data, data)
            print(f'\r{loss_value.numpy():.5f}', end='')
        grads = tape.gradient(loss_value, AE.trainable_variables)
        optimizer.apply_gradients(zip(grads, AE.trainable_variables))
        train_loss.update_state(loss_value)

    for _, data in enumerate(test_ds):
        recon_data = AE(data)
        loss_value = reconstruction_loss(recon_data, data)
        test_loss.update_state(loss_value)

    print('====> AE Epoch: {}, Train loss: {:.6f}, Test loss: {:.6f}'.format(epoch, train_loss.result(),
                                                                             test_loss.result()))
    return {'train_loss': tf.keras.backend.eval(train_loss.result()), 'test_loss': tf.keras.backend.eval(test_loss.result())}


def train_PCA_AE(PCAAE_E, PCAAE_D, optimizer, epoch, step, train_ds, test_ds):
    train_loss = tf.keras.metrics.Mean()
    train_content_loss = tf.keras.metrics.Mean()
    train_cov_loss = tf.keras.metrics.Mean()

    for _, data in enumerate(train_ds):
        with tf.GradientTape() as tape:
            z = []
            for jd in range(step - 1):
                z.append(PCAAE_E[jd](data, training=True))

            z_i = PCAAE_E[step - 1](data, training=True)
            z.append(z_i)
            latent_space = tf.concat(z, axis=-1)
            latent_array = latent_space.numpy().reshape(-1)
            recon_data = PCAAE_D[step - 1](latent_space, training=True)
            loss_data = lambda_rec * reconstruction_loss(recon_data, data)
            loss_cov = lambda_cov * cov_loss(latent_array, step)
            loss = loss_data + loss_cov
        grads = tape.gradient(loss, PCAAE_E[step - 1].trainable_variables + PCAAE_D[step - 1].trainable_variables)
        optimizer.apply_gradients(
            zip(grads, PCAAE_E[step - 1].trainable_weights + PCAAE_D[step - 1].trainable_weights))

        train_loss.update_state(loss)
        train_content_loss.update_state(loss_data)
        if step > 1:
            train_cov_loss.update_state(loss_cov)

    test_loss = tf.keras.metrics.Mean()
    test_content_loss = tf.keras.metrics.Mean()
    test_cov_loss = tf.keras.metrics.Mean()
    for _, data in enumerate(test_ds):
        z = []
        for jx in range(step):
            z.append(PCAAE_E[jx](data, training=False))
        latent_space = tf.concat(z, axis=-1)
        latent_array = latent_space.numpy().reshape(-1)
        recon_data = PCAAE_D[step - 1](latent_space, training=False)
        loss_data = lambda_rec * reconstruction_loss(recon_data, data)
        loss_cov = lambda_cov * cov_loss(latent_array, step)

        test_loss.update_state(loss_data + loss_cov)
        test_content_loss.update_state(loss_data)
        if step > 1:
            test_cov_loss.update_state(loss_cov)

    print('PCAAE{} Epoch: {} Train loss: {:.6f},\t Train Data loss: {:.6f},\t Train Cov loss: {:.8f},'.format(
        step, epoch, train_loss.result(), train_content_loss.result(), train_cov_loss.result()))

    print('PCAAE{} Epoch: {} Test Data loss: {:.6f},\t Test Cov loss: {:.8f},'.format(
        step, epoch, test_loss.result(), test_content_loss.result(), test_cov_loss.result()))

    return {
        'train_loss': tf.keras.backend.eval(train_loss.result()),
        'train_content_loss': tf.keras.backend.eval(train_content_loss.result()),
        'train_cov_loss': tf.keras.backend.eval(train_cov_loss.result()),
        'test_loss': tf.keras.backend.eval(test_loss.result()),
        'test_content_loss': tf.keras.backend.eval(test_content_loss.result()),
        'test_cov_loss': tf.keras.backend.eval(test_cov_loss.result())
    }



X_train = np.array(scipy.io.loadmat('timeseries_midi_dataset.mat')['train_data'], dtype=np.float32)
X_train, X_test =train_test_split(X_train, test_size=0.2)
X_train=np.array(X_train); X_test=np.array(X_test);


latent_space_dimension = 12

input_shape = X_train[0].shape

training_features = X_train.astype('float32')
training_dataset = tf.data.Dataset.from_tensor_slices(training_features).batch(batchsize)
# training_dataset = training_dataset.shuffle(train_loader.shape[0])

testing_features = X_test.astype('float32')
testing_dataset = tf.data.Dataset.from_tensor_slices(testing_features).batch(batchsize)
# testing_dataset = testing_dataset.shuffle(test_loader.shape[0])

if autoencoder:
    AE_E = Encoder(latent_space_dimension)
    AE_D = Decoder(latent_space_dimension, input_shape)
    ae = AutoEncoder(AE_E, AE_D)

    AE_optim = optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)

    history = {'train_loss': [], 'test_loss': []}
    for epoch in range(1, num_epoch + 1):
        temp_history = train_AE(ae, AE_optim, epoch, training_dataset, testing_dataset)
        history['train_loss'].append(temp_history.get('train_loss', 0))
        history['test_loss'].append(temp_history.get('test_loss', 0))
        
    print(history)

else:
    start_time = time.time()

    PCAAE_E = []
    PCAAE_D = []
    for id_m in range(latent_space_dimension):
        PCAAE_E_i = Encoder(latent_space_dimension=1)
        PCAAE_D_i = Decoder(latent_space_dimension=id_m + 1, output_size=input_shape)

        PCAAE_E.append(PCAAE_E_i)
        PCAAE_D.append(PCAAE_D_i)

    history = {
        'train_loss': [],
        'train_content_loss': [],
        'train_cov_loss': [],
        'test_loss': [],
        'test_content_loss': [],
        'test_cov_loss': []
    }

    for model in range(1, latent_space_dimension + 1):
        optim_temp = optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
        for epoch in range(1, num_epoch + 1):
            temp_history = train_PCA_AE(PCAAE_E, PCAAE_D, optim_temp, epoch, model, training_dataset, testing_dataset)
            history['train_loss'].append(temp_history.get('train_loss', 0))
            history['train_content_loss'].append(temp_history.get('train_content_loss', 0))
            history['train_cov_loss'].append(temp_history.get('train_cov_loss', 0))
            history['test_loss'].append(temp_history.get('test_loss', 0))
            history['test_content_loss'].append(temp_history.get('test_content_loss', 0))
            history['test_cov_loss'].append(temp_history.get('test_cov_loss', 0))
    
    end_training_time = time.time()
    print('Execution time before save:', end_training_time - start_time, 'seconds')

    with open('pcaae_models/lstm/loss_all.pickle', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    for model in range(latent_space_dimension):
        PCAAE_E[model].save(f'pcaae_models/lstm/encoder_all_{model+1}',save_format='tf')
        PCAAE_D[model].save(f'pcaae_models/lstm/decoder_all_{model+1}',save_format='tf')


    end_time = time.time()
    print('Execution time:', end_time - start_time, 'seconds')