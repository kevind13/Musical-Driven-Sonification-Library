import keras
from keras import layers
import tensorflow as tf

import scipy.io
import numpy as np

X_train = np.array(scipy.io.loadmat('X_train.mat')['X_train'],dtype=np.float32)
X_test = X_train[30:,]
X_train = X_train[:30,]

encoding_dim = 12

in_dim=int(X_train.shape[1])
input = keras.Input(shape=(in_dim,))
encoded = layers.Dense(541, activation='tanh')(input)
encoded = layers.Dense(54, activation='tanh')(encoded)
encoded = layers.Dense(encoding_dim, activation='sigmoid')(encoded)

decoded = layers.Dense(54, activation='tanh')(encoded)
decoded = layers.Dense(541, activation='tanh')(decoded)
decoded = layers.Dense(in_dim, activation='tanh')(decoded)

autoencoder = keras.Model(input, decoded)

encoder = keras.Model(input, encoded)

encoded_input = keras.Input(shape=(encoding_dim,))
# decoder_layer = autoencoder.layers[-1]
decoder_layer = autoencoder.layers[-3](encoded_input)
decoder_layer = autoencoder.layers[-2](decoder_layer)
decoder_layer = autoencoder.layers[-1](decoder_layer)
decoder = keras.Model(encoded_input, decoder_layer)

# with tf.device("/cpu:0"):
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError())

autoencoder.fit(X_train, X_train,epochs=100,batch_size=4,shuffle=True,validation_data=(X_test, X_test))

encoder.compile(optimizer='adam', loss='binary_crossentropy')
decoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.save('autoencoder.h5')
encoder.save('encoder.h5')
decoder.save('decoder.h5')