import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import scipy.io
import itertools
import pickle
from sklearn.model_selection import train_test_split


class Encoder(tf.keras.Model):

    def __init__(self, latent_space_dimension):
        super(Encoder, self).__init__()
        self.latent_space_dimension = latent_space_dimension

        self.conv1 = layers.Conv2D(154,3,1,activation='elu',padding="valid", input_shape=(128,4,1))
        self.conv2 = layers.Conv2D(195,2,1,activation='selu',padding="valid")
        self.conv3 = layers.Conv2D(221,1,1,activation='relu',padding="same")
        self.flat1 = layers.Flatten(input_shape=(1, 125, 1, 221))
        self.dense1 = tf.keras.layers.Dense(units=356, activation='selu')
        self.dense2 = tf.keras.layers.Dense(units=379, activation='selu')
        self.dense3 = tf.keras.layers.Dense(units=self.latent_space_dimension, activation='selu')        
        self.bn = layers.BatchNormalization()


    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.bn(x)
        return x


class Decoder(tf.keras.Model):

    def __init__(self, latent_space_dimension, output_size):
        super(Decoder, self).__init__()
        # Shape required to start transpose convs
        self.dense1 = tf.keras.layers.Dense(units=356, activation='selu')
        self.dense2 = tf.keras.layers.Dense(units=27625, activation='selu')
        self.rs = layers.Reshape((125,1,221))
        self.conv1 = layers.Conv2DTranspose(195,1,(1,1),activation='relu',output_padding=0, padding="same")
        self.conv2 = layers.Conv2DTranspose(154,(2,2),1,activation='selu',output_padding=0,padding="valid")
        self.conv3 = layers.Conv2DTranspose(1,(3,3),1,activation='elu',output_padding=0,padding='valid')
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.rs(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x



train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10


x_train = np.array(scipy.io.loadmat('timeseries_midi_dataset.mat')['train_data'], dtype=np.float32)
x_train_loader, x_test = train_test_split(x_train, test_size=1-train_ratio, random_state=13)
x_val_loader, x_test_loader = train_test_split(x_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=13) 

x_train_loader = x_train_loader.reshape([x_train_loader.shape[0],x_train_loader.shape[1],x_train_loader.shape[2], 1])
x_test_loader = x_test_loader.reshape([x_test_loader.shape[0],x_test_loader.shape[1],x_test_loader.shape[2], 1])
x_val_loader = x_val_loader.reshape([x_val_loader.shape[0],x_val_loader.shape[1],x_val_loader.shape[2], 1])

print(x_train_loader.shape, x_val_loader.shape, x_test_loader.shape)

input_size = x_train.shape[1]
latent_space_dimension = 12

input_shape = x_train[0].shape

training_features = x_train_loader.astype('float32')
training_dataset = tf.data.Dataset.from_tensor_slices(training_features).batch(1)
# training_dataset = training_dataset.shuffle(train_loader.shape[0])


testing_features = x_test_loader.astype('float32')
testing_dataset = tf.data.Dataset.from_tensor_slices(testing_features).batch(1)
# testing_dataset = testing_dataset.shuffle(test_loader.shape[0])

validation_features = x_val_loader.astype('float32')
validation_dataset = tf.data.Dataset.from_tensor_slices(validation_features).batch(1)


latent_dimension = 12
PCAAE_E = []

for dim in range(1,latent_dimension + 1):
    PCAAE_E.append(models.load_model(
        f'pcaae_models/conv/encoder_conv_{dim}', compile=False
    ))

print(f'pcaae_models/conv/decoder_conv_{latent_dimension}')
PCAAE_D = models.load_model(
        f'pcaae_models/conv/decoder_conv_{latent_dimension}', compile=False
    )


test_eval = {}
for index, x in enumerate(testing_dataset):
    z = PCAAE_E[0](x).numpy()[0]
    for dim in range(1,latent_dimension):
        z = np.append(z, PCAAE_E[dim](x).numpy()[0])\

    latent_space = tf.concat(z, axis=-1)
    latent_array = latent_space.numpy().reshape(-1)

    recon_data = PCAAE_D([latent_space]).numpy()
    recon_data = np.rint(recon_data[0]).astype(int)

    test_eval[index] = {'input': x.numpy(), 'latent_space': z, 'output': recon_data}

with open('pcaae_models/conv/evaluation_conv.pickle', 'wb') as handle:
        pickle.dump(test_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)


print(test_eval[3])