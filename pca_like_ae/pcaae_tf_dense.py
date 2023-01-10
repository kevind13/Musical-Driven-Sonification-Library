import tensorflow as tf
import numpy as np


class Encoder(tf.keras.Model):

    def __init__(self, encoding_dim, in_dim):
        super(Encoder, self).__init__()
        self.encoding_dim = encoding_dim
        self.in_dim = in_dim

        self.fc1 = tf.keras.layers.Dense(541, input_shape=self.in_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(54, activation='relu')
        self.fc3 = tf.keras.layers.Dense(self.encoding_dim, activation='relu')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Decoder(tf.keras.Model):

    def __init__(self, encoding_dim, in_dim):
        super(Decoder, self).__init__()
        self.encoding_dim = encoding_dim
        self.in_dim = in_dim

        self.fc1 = tf.keras.layers.Dense(54, input_shape=(self.encoding_dim,), activation='relu')
        self.fc2 = tf.keras.layers.Dense(541, activation='relu')
        self.fc3 = tf.keras.layers.Dense(self.in_dim[0], activation='sigmoid')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Autoencoder(tf.keras.Model):
  def __init__(self, encoding_dim, in_dim):
    super(Autoencoder, self).__init__()
    self.encoder = Encoder(encoding_dim, in_dim)
    self.decoder = Decoder(encoding_dim, in_dim)

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

encoding_dim = 32
in_dim = (784,)

autoencoder = Autoencoder(encoding_dim, in_dim)

# Load and preprocess your data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_test.shape)
# Define the loss function and the optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

# Define the metric for evaluation
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Define the training loop
@tf.function
def train_step(x):
  with tf.GradientTape() as tape:
    logits = autoencoder(x)
    loss_value = loss_fn(x, logits)
  grads = tape.gradient(loss_value, autoencoder.trainable_variables)
  optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables))
  accuracy_metric(x, logits)
  return loss_value

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
  for x_batch in x_train:
    loss_value = train_step(x_batch)
  accuracy = accuracy_metric.result()
  accuracy_metric.reset_states()
  print('Epoch {}: loss = {}, accuracy = {}'.format(epoch, loss_value, accuracy))
