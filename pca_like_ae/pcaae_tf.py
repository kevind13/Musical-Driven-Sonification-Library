# from tensorflow.keras.models import Model

# class Encoder(Model):
#     def __init__(self, code_size=1, kernel_size=4, n_chan=1):
#         super(Encoder, self).__init__()        
#         self.latent_dim = code_size
        
#         # Convolutional layers
#         cnn_kwargs = dict(strides=2, padding='same')
#         self.conv1 = tf.keras.layers.Conv2D(n_chan, int(code_size*32), kernel_size, **cnn_kwargs)
#         self.conv2 = tf.keras.layers.Conv2D(int(code_size*32), kernel_size, **cnn_kwargs)
#         self.conv3 = tf.keras.layers.Conv2D(int(code_size*16), kernel_size, **cnn_kwargs)
#         self.conv4 = tf.keras.layers.Conv2D(int(code_size*8), kernel_size, **cnn_kwargs)
#         self.conv5 = tf.keras.layers.Conv2D(int(code_size*4), kernel_size, **cnn_kwargs)
#         self.conv6 = tf.keras.layers.Conv2D(int(code_size*2), kernel_size, **cnn_kwargs)
#         self.zero_mean = tf.keras.layers.BatchNormalization(affine=False, epsilon=0)
#         self.leaky = tf.keras.layers.LeakyReLU(alpha=0.2)

#     def call(self, x):
#         # Convolutional layers with ReLu activations
#         x = self.leaky(self.conv1(x))
#         x = self.leaky(self.conv2(x))
#         x = self.leaky(self.conv3(x))
#         x = self.leaky(self.conv4(x))
#         x = self.leaky(self.conv5(x))
#         x = self.conv6(x)
#         z = tf.reshape(x, (-1, self.latent_dim))
#         z = self.zero_mean(z)
#         return z


# class Decoder(Model):
#     def __init__(self, code_size=1, kernel_size = 4, n_chan = 1):
#         super(Decoder, self).__init__()
#         # Shape required to start transpose convs
#         self.reshape = (code_size, 1, 1)
         
#         # Convolutional layers
#         cnn_kwargs = dict(stride=2, padding=1)
#         self.convT6 = tf.keras.layers.Conv2DTranspose(code_size, kernel_size, **cnn_kwargs)
#         self.convT5 = tf.keras.layers.Conv2DTranspose(int(code_size*2), kernel_size, **cnn_kwargs)
#         self.convT4 = tf.keras.layers.Conv2DTranspose(int(code_size*4), kernel_size, **cnn_kwargs)
#         self.convT3 = tf.keras.layers.Conv2DTranspose(int(code_size*8), kernel_size, **cnn_kwargs)
#         self.convT2 = tf.keras.layers.Conv2DTranspose(int(code_size*16), kernel_size, **cnn_kwargs)
#         self.convT1 = tf.keras.layers.Conv2DTranspose(n_chan, kernel_size, **cnn_kwargs)
#         self.leaky = tf.keras.layers.LeakyReLU(0.2)
        
#     def call(self, z):
#         batch_size = z.shape[0]
#         x = tf.reshape(z, (batch_size, *self.reshape))
        
#         # Convolutional layers with ReLu activations
#         x = self.leaky(self.convT6(x))
#         x = self.leaky(self.convT5(x))
#         x = self.leaky(self.convT4(x))
#         x = self.leaky(self.convT3(x))
#         x = self.leaky(self.convT2(x))
#         # Sigmoid activation for final conv layer
#         x = tf.sigmoid(self.convT1(x))

#         return x