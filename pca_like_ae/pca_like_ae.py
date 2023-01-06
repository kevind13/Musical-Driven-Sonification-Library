# import tensorflow as tf
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, code_size=1, kernel_size = 4, n_chan = 4):
        super(Encoder, self).__init__()        
        self.latent_dim = code_size
        
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, int(code_size*32), kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(int(code_size*32), int(code_size*16), kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(int(code_size*16), int(code_size*8), kernel_size, **cnn_kwargs)
        self.conv4 = nn.Conv2d(int(code_size*8), int(code_size*4), kernel_size, **cnn_kwargs)
        self.conv5 = nn.Conv2d(int(code_size*4), int(code_size*2), kernel_size, **cnn_kwargs)
        self.conv6 = nn.Conv2d(int(code_size*2), code_size, kernel_size, **cnn_kwargs)
        self.zero_mean = nn.BatchNorm1d(self.latent_dim, affine=False, eps=0)
        self.leaky = nn.LeakyReLU(0.2)
    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        x = self.leaky(self.conv3(x))
        x = self.leaky(self.conv4(x))
        x = self.leaky(self.conv5(x))
        x = self.conv6(x)
        z = x.view((batch_size, -1))
        z = self.zero_mean(z)
        return z  

class Decoder(nn.Module):
    def __init__(self, code_size=1, kernel_size = 4, n_chan = 4):
        super(Decoder, self).__init__()
        # Shape required to start transpose convs
        self.reshape = (code_size, 1, 1)
         
            # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.convT6 = nn.ConvTranspose2d(code_size, int(code_size*2), kernel_size, **cnn_kwargs)
        self.convT5 = nn.ConvTranspose2d(int(code_size*2), int(code_size*4), kernel_size, **cnn_kwargs)
        self.convT4 = nn.ConvTranspose2d(int(code_size*4), int(code_size*8), kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(int(code_size*8), int(code_size*16), kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(int(code_size*16), int(code_size*32), kernel_size, **cnn_kwargs)
        self.convT1 = nn.ConvTranspose2d(int(code_size*32), n_chan, kernel_size, **cnn_kwargs)
        self.leaky = nn.LeakyReLU(0.2)
        
    def forward(self, z):
        batch_size = z.size(0)
        x = z.view(batch_size, *self.reshape)
        
        # Convolutional layers with ReLu activations
        x = self.leaky(self.convT6(x))
        x = self.leaky(self.convT5(x))
        x = self.leaky(self.convT4(x))
        x = self.leaky(self.convT3(x))
        x = self.leaky(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = nn.Sigmoid()(self.convT1(x))

        return x
