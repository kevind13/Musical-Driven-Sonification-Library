from keras.datasets import mnist
from keras.layers import Input, Dense
from keras import regularizers, models, optimizers
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.io

# Analytical PCA of the training set
def AnalyticalPCA(y, dimension):
    pca = PCA(n_components=dimension)
    pca.fit(y)
    loadings = pca.components_
    return pca, loadings

# Linear Autoencoder
def LinearAE(y, dimension, learning_rate = 0.001, regularization = 5e-4, epochs=15):
    input = Input(shape=(y.shape[1],))
    encoded = Dense(dimension, activation='linear',
                    kernel_regularizer=regularizers.l2(regularization))(input)
    decoded = Dense(y.shape[1], activation='linear',
                    kernel_regularizer=regularizers.l2(regularization))(encoded)
    autoencoder = models.Model(input, decoded)
    autoencoder.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='mean_squared_error')
    autoencoder.fit(y, y, epochs=epochs, batch_size=4, shuffle=True)
    (w1,b1,w2,b2)=autoencoder.get_weights()
    return (autoencoder, w1,b1,w2,b2)

dimension = 40                                                                  # feel free to change this, but you may have to tune hyperparameters
(y, _), (_, _) = mnist.load_data()   
data = np.array(scipy.io.loadmat('timeseries_midi_dataset_with_transpose.mat')['train_data'], dtype=np.float32)
shape_data = data.shape                                                         # store shape of y before reshaping it
shape_y = data.shape                                                              # store shape of y before reshaping it
y = np.reshape(data,[shape_data[0],shape_data[1]*shape_data[2]]).astype('float32')/255 

p_analytical = AnalyticalPCA(y,dimension)                                       # PCA by applying SVD to y
(autoencoder_model,_, _, w2, _) = LinearAE(y, dimension)                                          # train a linear autoencoder
(p_linear_ae, _, _) = np.linalg.svd(w2.T, full_matrices=False)                    # PCA by applying SVD to linear autoencoder weights
p_analytical = np.reshape(p_analytical,[dimension,shape_y[1],shape_y[2]])       # reshape loading vectors before plotting
w2 = np.reshape(w2,[dimension,shape_y[1],shape_y[2]])                         # reshape autoencoder weights before plotting
p_linear_ae = np.reshape(p_linear_ae.T, [dimension, shape_y[1], shape_y[2]])    # reshape loading vectors before plotting
print(p_linear_ae.shape)
x_test = y[5].reshape(1,512)
a = autoencoder_model.predict(x_test)

x_test = x_test.reshape(128,4)
a = a.reshape(128,4)

print(x_test[:5]*255)
print(a[:5]*255)