import tensorflow as tf
from tensorflow.keras import layers, activations, losses, optimizers
import numpy as np
import scipy.io
import itertools

batchsize = 4
gpu = True
is_training = True
num_epoch = 2
lr = 0.001
lambda_cov = 1
lambda_rec = 1

autoencoder = False


class Encoder(layers.Layer):

    def __init__(self, input_size, hidden_size1, hidden_size2, latent_space_dimension):
        super(Encoder, self).__init__()
        self.leaky = layers.LeakyReLU(0.2)

        self.fc1 = layers.Dense(units=hidden_size1, activation=self.leaky, input_shape=(input_size,))
        self.fc2 = layers.Dense(units=hidden_size2, activation=self.leaky)
        self.fc3 = layers.Dense(latent_space_dimension)
        self.bn = layers.BatchNormalization()

    def call(self, input_features):
        x = self.fc1(input_features)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.bn(x)
        return x


class Decoder(layers.Layer):

    def __init__(self, latent_space_dimension, hidden_size2, hidden_size1, output_size):
        super(Decoder, self).__init__()
        self.leaky = layers.LeakyReLU(0.2)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

        self.fc1 = layers.Dense(units=hidden_size2, activation=self.leaky)
        self.fc2 = layers.Dense(units=hidden_size1, activation=self.leaky)
        self.fc3 = layers.Dense(units=output_size)

    def call(self, code):
        x = self.fc1(code)
        x = self.fc2(x)
        x = self.sigmoid(self.fc3(x))

        return x


def reconstruction_loss(data, recon_data, MSE=False):
    if MSE:
        return losses.MeanSquaredError()(data, recon_data)
    else:
        return losses.BinaryCrossentropy()(data, recon_data)


def cov_loss(z, step):
    if step > 1:
        loss = 0
        for idx in range(step - 1):
            loss += tf.reduce_mean(z[:, idx] * z[:, -1])**2
        loss = loss / (step - 1)
    else:
        loss = tf.zeros_like(z)
    return tf.reduce_mean(loss)


def train_AE(E, D, optimizer, epoch, train_ds, test_ds):
    train_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()

    for _, data in enumerate(train_ds):
        with tf.GradientTape() as tape:
            recon_data = D(E(data))
            loss_value = reconstruction_loss(recon_data, data)
        grads = tape.gradient(loss_value, E.trainable_variables + D.trainable_variables)
        optimizer.apply_gradients(zip(grads, E.trainable_variables + D.trainable_variables))
        train_loss.update_state(loss_value)

    for _, data in enumerate(test_ds):
        recon_data = D(E(data))
        loss_value = reconstruction_loss(recon_data, data)
        test_loss.update_state(loss_value)

    print('====> AE Epoch: {}, Train loss: {:.6f}, Test loss: {:.6f}'.format(epoch, train_loss.result(),
                                                                             test_loss.result()))


def train_PCA_AE(PCAAE_E, PCAAE_D, optimizer, epoch, step, train_dataset, test_dataset):
    train_loss = 0
    train_content_loss = 0
    train_cov_loss = 0
    for _, data in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            z = []
            for jd in range(step - 1):
                z.append(PCAAE_E[jd](data, training=True))

            z_i = PCAAE_E[step - 1](data, training=True)
            z.append(z_i)
            latent_space = tf.concat(z, axis=-1)
            recon_data = PCAAE_D[step - 1](latent_space, training=True)
            loss_data = lambda_rec * reconstruction_loss(recon_data, data)
            loss_cov = lambda_cov * cov_loss(latent_space, step)
            loss = loss_data + loss_cov
        grads = tape.gradient(loss, PCAAE_E[step - 1].trainable_variables + PCAAE_D[step - 1].trainable_variables)

        del tape

        optimizer.apply_gradients(
            zip(grads, PCAAE_E[step - 1].trainable_variables + PCAAE_D[step - 1].trainable_variables))

        train_loss += loss
        train_content_loss += loss_data
        if step > 1:
            train_cov_loss += loss_cov

    test_loss = 0
    test_content_loss = 0
    test_cov_loss = 0
    for _, data in enumerate(train_dataset):
        z = []
        for jx in range(step):
            z.append(PCAAE_E[jx](data, training=False))
        latent_space = tf.concat(z, axis=-1)
        recon_data = PCAAE_D[step - 1](latent_space, training=False)
        loss_data = lambda_rec * reconstruction_loss(recon_data, data)
        loss_cov = lambda_cov * cov_loss(latent_space, step)
        test_loss += loss_data + loss_cov

        test_content_loss += loss_data
        if step > 1:
            test_cov_loss += loss_cov

    print('PCAAE{} Epoch: {} Train loss: {:.6f},\t Train Data loss: {:.6f},\t Train Cov loss: {:.8f},'.format(
        step, epoch, train_loss / len(train_dataset), train_content_loss / len(train_dataset),
        train_cov_loss / len(train_dataset)))

    print('PCAAE{} Epoch: {} Test Data loss: {:.6f},\t Test Cov loss: {:.8f},'.format(
        step, epoch, test_loss / len(test_dataset), test_content_loss / len(test_dataset),
        test_cov_loss / len(test_dataset)))


X_train = np.array(scipy.io.loadmat('exploratory_data.mat')['train_data'], dtype=np.float32)
test_loader = X_train[:8]
train_loader = X_train[8:]

print(test_loader.shape)

input_size = X_train.shape[1]
hidden_size1 = 1000
hidden_size2 = 100
latent_space_dimension = 12

training_features = train_loader.astype('float32')
training_dataset = tf.data.Dataset.from_tensor_slices(train_loader)
training_dataset = training_dataset.batch(1)
training_dataset = training_dataset.shuffle(train_loader.shape[0])
training_dataset = training_dataset.prefetch(1 * 4)

testing_features = test_loader.astype('float32')
testing_dataset = tf.data.Dataset.from_tensor_slices(test_loader)
testing_dataset = testing_dataset.batch(1)
testing_dataset = testing_dataset.shuffle(test_loader.shape[0])
testing_dataset = testing_dataset.prefetch(1 * 4)

if autoencoder:
    AE_E = Encoder(input_size, hidden_size1, hidden_size2, latent_space_dimension)
    AE_D = Decoder(latent_space_dimension, hidden_size2, hidden_size1, input_size)

    # Define the loss function and optimizer
    criterion = losses.MeanSquaredError()
    AE_optim = optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)

    num_epoch = 10
    for epoch in range(1, num_epoch + 1):
        train_AE(AE_E, AE_D, AE_optim, epoch, training_dataset, testing_dataset)

else:
    PCAAE_E = []
    PCAAE_D = []
    for id_m in range(latent_space_dimension):
        PCAAE_E_i = Encoder(input_size, hidden_size1, hidden_size2, latent_space_dimension=1)
        PCAAE_D_i = Decoder(latent_space_dimension=id_m + 1,
                            hidden_size2=hidden_size2,
                            hidden_size1=hidden_size1,
                            output_size=input_size)

        PCAAE_E.append(PCAAE_E_i)
        PCAAE_D.append(PCAAE_D_i)

    for model in range(1, latent_space_dimension + 1):
        optim_temp = optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
        for epoch in range(1, num_epoch + 1):
            train_PCA_AE(PCAAE_E, PCAAE_D, optim_temp, epoch, model, training_dataset, testing_dataset)