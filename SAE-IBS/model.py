import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


def flatten(t):
    return [item for sublist in t for item in sublist]


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, activation_fn=F.relu):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dropout = cond_dropout
        self.drop_rate = drop_rate
        self.activation_fn = activation_fn

        self.zero_mean = nn.BatchNorm1d(self.latent_dim, affine=False, eps=0) ## NOTA: AGREGAR ESTO DISMINUYÓ MUCHISIMO EL LOSS

        if type(input_dim) is tuple:
            self.input_dim = reduce(lambda x, y: x * y, input_dim)

        neurons = [self.input_dim, *hidden_dim, latent_dim]
        if cond_dropout:
            layers = flatten(
                [[nn.Linear(neurons[i - 1], neurons[i]), nn.Dropout(p=drop_rate)] for i in range(1, len(neurons))])
            self.hidden = nn.ModuleList(layers[:-1])
        else:
            linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
            self.hidden = nn.ModuleList(linear_layers)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        for layer in self.hidden[:-1]:
            x = self.activation_fn(layer(x))
        x = self.hidden[-1](x)
        x = x.view((batch_size, -1))

        x = self.zero_mean(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, actFn, activation_fn=F.relu):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_shape = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dropout = cond_dropout
        self.drop_rate = drop_rate
        self.actFn = actFn
        self.activation_fn = activation_fn

        if type(input_dim) is tuple:
            self.input_dim = reduce(lambda x, y: x * y, input_dim)

        neurons = [latent_dim, *reversed(hidden_dim)]
        if cond_dropout:
            layers = flatten(
                [[nn.Linear(neurons[i-1], neurons[i]), nn.Dropout(p=drop_rate)] for i in range(1, len(neurons))])
            self.hidden = nn.ModuleList(layers)
        else:
            linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
            self.hidden = nn.ModuleList(linear_layers)
        self.fc = nn.Linear(neurons[-1], self.input_dim)

    def forward(self, x):
        batch_size = x.size(0)

        for layer in self.hidden:
            x = self.activation_fn(layer(x))
        
        x = self.fc(x)

        # Reshape output
        x = x.view(batch_size, *self.output_shape)
        
        if self.actFn == 'sigmoid':
            x = torch.sigmoid(x)  # last layer before output is sigmoid, transfer between 0 and 1
        elif self.actFn == 'tanh':
            x = torch.tanh(x)
        elif self.actFn == 'leakyRelu':
            x = F.leaky_relu(x)
        elif self.actFn == 'relu':
            x = F.relu(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, actFn):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dropout = cond_dropout
        self.drop_rate = drop_rate
        self.actFn = actFn
        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.latent_dim, self.cond_dropout, self.drop_rate)
        self.decoder = Decoder(self.input_dim, self.hidden_dim, self.latent_dim, self.cond_dropout, self.drop_rate, self.actFn)

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon, latent


class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, activation_fn=F.relu):
        super(VAE_Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dropout = cond_dropout
        self.drop_rate = drop_rate
        self.activation_fn = activation_fn

        if type(input_dim) is tuple:
            self.input_dim = reduce(lambda x, y: x * y, input_dim)

        neurons = [self.input_dim, *hidden_dim]

        if cond_dropout:
            layers = flatten(
                [[nn.Linear(neurons[i - 1], neurons[i]), nn.Dropout(p=drop_rate)] for i in range(1, len(neurons))])
            self.hidden = nn.ModuleList(layers[:-1])
        else:
            linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
            self.hidden = nn.ModuleList(linear_layers)
        self.fc_mu = nn.Linear(in_features=hidden_dim[-1], out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_dim[-1], out_features=latent_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        for layer in self.hidden:
            x = self.activation_fn(layer(x))
        x_mu = self.fc_mu(x)
        x_mu = x_mu.view((batch_size, -1))
        x_logvar = self.fc_logvar(x)
        x_logvar = x_logvar.view((batch_size, -1))
        return x_mu, x_logvar


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, actFn):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dropout = cond_dropout
        self.drop_rate = drop_rate
        self.actFn = actFn
        self.encoder = VAE_Encoder(self.input_dim, self.hidden_dim, self.latent_dim, self.cond_dropout, self.drop_rate)
        self.decoder = Decoder(self.input_dim, self.hidden_dim, self.latent_dim, self.cond_dropout, self.drop_rate, self.actFn)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, actFn):
        super(SAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dropout = cond_dropout
        self.drop_rate = drop_rate
        self.actFn = actFn
        self.encoder = Encoder(self.input_dim, self.hidden_dim[:-1], self.hidden_dim[-1], self.cond_dropout, self.drop_rate)
        self.decoder = Decoder(self.input_dim, self.hidden_dim[:-1], self.hidden_dim[-1], self.cond_dropout, self.drop_rate, self.actFn)

        # svd related
        self.V = nn.Parameter(torch.empty(self.latent_dim, self.latent_dim))
        self.rank = self.latent_dim
        self.emb = None
        self.mean_emb = nn.Parameter(torch.empty(self.latent_dim,), requires_grad=False)

    def initialize_svd(self, x):
        self.emb = x
        self.mean_emb = nn.Parameter(torch.mean(self.emb, 0))
        _, _, V = torch.svd_lowrank(self.emb - self.mean_emb, self.rank)
        max_ind = torch.argmax(torch.abs(V), 0)
        colsign = torch.sign(V[max_ind, torch.arange(V.shape[1])])
        self.V = nn.Parameter(V * colsign)

    def update_svd(self, x):
        z = torch.matmul(x - self.mean_emb, self.V)
        x_hat = torch.matmul(z, torch.transpose(self.V, 0, 1)) + self.mean_emb
        return z, x_hat

    def encoder_svd(self, x):
        x_enc = self.encoder(x)
        z, x_hat = self.update_svd(x_enc)
        return x_enc, z, x_hat
    
    def decoder_svd(self, z):
        x_hat = torch.matmul(z, torch.transpose(self.V, 0, 1)) + self.mean_emb
        x_recon = self.decoder(x_hat)
        return x_recon

    def forward(self, x):
        x_enc, z, x_hat = self.encoder_svd(x)
        x_recon = self.decoder(x_hat)
        return x_recon, z
