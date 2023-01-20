import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import scipy.io

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
from collections import deque

batchsize = 4
gpu = True
is_training = True
EPOCHS = 20
lr = 0.001
lambda_cov = 1
lambda_rec = 1

# list_ds = tf.data.Dataset.list_files('*/*.jpg')
# load_img = lambda x: tf.io.decode_image(tf.io.read_file(x), channels=3, dtype=tf.float32)
# ds = list_ds.map(load_img).batch(64)

X_train = np.array(scipy.io.loadmat('exploratory_4_channels_data.mat')['train_data'], dtype=np.float32)
test_loader = X_train[:8]
train_loader = X_train[8:]

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


class Encoder(K.Model):

    def __init__(self, input_size, hidden_size1, hidden_size2, latent_space_dimension):
        super(Encoder, self).__init__()
        self.conv1 = K.layers.Conv2D(hidden_size1, kernel_size=(3, 3), padding='same')
        self.maxp1 = K.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = K.layers.Conv2D(hidden_size2, kernel_size=(3, 3), padding='same')
        self.maxp2 = K.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = K.layers.Conv2D(latent_space_dimension, kernel_size=(3, 3), padding='same')
        self.maxp3 = K.layers.MaxPooling2D(pool_size=(2, 2))

        self.leaky = K.layers.LeakyReLU(0.2)
        self.bn = K.layers.BatchNormalization()
        self.latent_space_dimension = latent_space_dimension

    def call(self, x, training=None, **kwargs):
        x = self.leaky(self.conv1(x))
        x = self.maxp1(x)
        x = self.leaky(self.conv2(x))
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.maxp3(x)
        x = tf.reshape(x, (-1, self.latent_space_dimension))
        x = self.bn(x)
        return x


class Decoder(K.Model):

    def __init__(self, latent_space_dimension, hidden_size2, hidden_size1, output_size, input_shape):
        super(Decoder, self).__init__()
        self.convtrans1 = K.layers.Conv2DTranspose(hidden_size2,
                                                   strides=(2, 2),
                                                   kernel_size=(3, 3),
                                                   padding='valid',
                                                   input_shape=(12, 12, 256))
        self.convtrans2 = K.layers.Conv2DTranspose(hidden_size1, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.convtrans3 = K.layers.Conv2DTranspose(output_size, kernel_size=(3, 3), strides=(2, 2), padding='same')

        self.reshape = K.layers.Reshape(input_shape)
        self.leaky = K.layers.LeakyReLU(0.2)
        self.sigmoid = K.layers.Activation('sigmoid')

    def call(self, x, training=None, **kwargs):
        x = self.leaky(self.convtrans1(x))
        x = self.leaky(self.convtrans2(x))
        x = self.sigmoid(self.convtrans3(x))
        x = self.reshape(x)
        return x


class AutoEncoder(K.Model):

    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x, training=None, **kwargs):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


ae = AutoEncoder(Encoder(input_size, hidden_size1, hidden_size2, latent_space_dimension),
                 Decoder(latent_space_dimension, hidden_size2, hidden_size1, input_size, [input_size]))

print(ae.summary)

binary_crossentropy = K.losses.BinaryCrossentropy(from_logits=False)

reconstruction_loss = K.metrics.Mean(name='reconstruction_loss')

optimizer = K.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)

TEMPLATE = '\rEpoch {:2} Reconstruction Loss {:.4f}'
monitor = deque(maxlen=11)


for x in training_dataset:
    output = ae(x)
    break

print(output)
# if __name__ == '__main__':
#     for epoch in range(EPOCHS):
#         reconstruction_loss.reset_states()
#         print(f'Starting Epoch {epoch}')
#         for data in training_dataset:
#             with tf.GradientTape() as tape:
#                 output = ae(data)
#                 loss = binary_crossentropy(data, output)
#                 print(f'\r{loss.numpy():.5f}', end='')

#             gradients = tape.gradient(loss, ae.trainable_variables)
#             optimizer.apply_gradients(zip(gradients, ae.trainable_variables))

#             reconstruction_loss(loss)
#         print(data)
#         print(TEMPLATE.format(epoch + 1, reconstruction_loss.result()))
#         monitor.append(reconstruction_loss.result())

#         if epoch >= monitor.maxlen and monitor.popleft() < min(monitor):
#             print(f'Early stopping. No reconstruction loss decrease in {monitor.maxlen - 1} epochs.')
#             break