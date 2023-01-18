import torch
import torch.nn as nn
import numpy as np
import scipy.io
from torch import optim
import itertools

batchsize = 1
gpu = True
is_training = True
num_epoch = 100
lr = 0.001
lambdacov = 1
lambdarec = 1

if gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, latent_space_dimension):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, latent_space_dimension)
        self.zero_mean = nn.BatchNorm1d(latent_space_dimension, affine=False, eps=0)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.fc1(x))
        x = self.leaky(self.fc2(x))
        x = self.fc3(x)
        x = self.zero_mean(x)
        return x


class Decoder(nn.Module):

    def __init__(self, latent_space_dimension, hidden_size2, hidden_size1, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_space_dimension, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, output_size)
        self.leaky = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky(self.fc1(x))
        x = self.leaky(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x


def reconstruction_loss(data, recon_data, MSE=False):
    if MSE:
        return nn.MSELoss()(data, recon_data)
    else:
        return nn.BCELoss()(data, recon_data)


def cov_loss(z, step):
    if step > 1:
        loss = 0
        for idx in range(step - 1):
            loss += ((z[:, idx] * z[:, -1]).mean())**2
        loss = loss / (step - 1)
    else:
        loss = torch.zeros_like(z)
    return loss.mean()


def train_AE(E, D, optimizer, epoch, train_loader, test_loader):
    train_loss = 0
    test_loss = 0

    E.train()
    D.train()
    for batch_idx, (data) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_data = D(E(data))
        loss = reconstruction_loss(recon_data, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    E.eval()
    D.eval()
    for batch_idx, (data) in enumerate(test_loader):
        data = data.to(device)
        recon_data = D(E(data))
        loss = reconstruction_loss(recon_data, data)
        test_loss += loss.item()

    print('====> AE Epoch: {}, Train loss: {:.6f}, Test loss: {:.6f}'.format(epoch, train_loss / len(train_loader),
                                                                             test_loss / len(test_loader)))


def train_PCA_AE(PCAAE_E, PCAAE_D, optimizer, epoch, step, train_loader, test_loader, lambda_rec, lambda_cov, device):
    train_loss = 0
    train_content_loss = 0.
    train_cov_loss = 0.

    test_loss = 0
    test_content_loss = 0.
    test_cov_loss = 0.

    # Training part:
    for idx_step in range(1, step + 1):
        PCAAE_E[idx_step - 1].train()
    PCAAE_D[step - 1].train()

    for id_step in range(1, step):
        model_temp = PCAAE_E[id_step - 1]
        for param in model_temp.parameters():
            param.required_grad = False

    for batch_idx, (data) in enumerate(train_loader):
        data = data.to(device)

        z = []
        with torch.no_grad():
            for jd in range(step - 1):
                z.append(PCAAE_E[jd](data))

        optimizer[step - 1].zero_grad()
        z_i = PCAAE_E[step - 1](data)
        z.append(z_i)
        latent_space = torch.cat(z, dim=-1)
        recon_data = PCAAE_D[step - 1](latent_space)
        loss_data = lambda_rec * reconstruction_loss(recon_data, data)
        loss_cov = lambda_cov * cov_loss(latent_space, step)
        loss = loss_data + loss_cov
        loss.backward()
        train_loss += loss.item()
        train_content_loss += loss_data.item()
        if step > 1:
            train_cov_loss += loss_cov.item()
        optimizer[step - 1].step()

    # Testing part:
    for idx_step in range(1, step + 1):
        PCAAE_E[idx_step - 1].eval()
    PCAAE_D[step - 1].eval()

    for batch_idx, (data) in enumerate(test_loader):
        data = data.to(device)
        z = []
        for jx in range(step):
            z.append(PCAAE_E[jx](data))
        latent_space = torch.cat(z, dim=-1)
        recon_data = PCAAE_D[step - 1](latent_space)
        loss_data = lambda_rec * reconstruction_loss(recon_data, data)
        loss_cov = lambda_cov * cov_loss(latent_space, step)
        test_loss += loss_data + loss_cov
        test_content_loss += loss_data
        if step > 1:
            test_cov_loss += loss_cov

    print('PCAAE{} Epoch: {} Train loss: {:.6f},\t Train Data loss: {:.6f},\t Train Cov loss: {:.8f},'.format(
        step, epoch, train_loss / len(train_loader), train_content_loss / len(train_loader),
        train_cov_loss / len(train_loader)))

    print('PCAAE{} Epoch: {} Test loss: {:.6f},\t Test Data loss: {:.6f},\t Tes Cov loss: {:.8f},'.format(
        step, epoch, test_loss / len(test_loader), test_content_loss / len(test_loader),
        test_cov_loss / len(test_loader)))


X_train = np.array(scipy.io.loadmat('exploratory_data.mat')['train_data'], dtype=np.uint8)
X_test = X_train[:8]
X_train = X_train[8:]

print(X_test.shape)
print(X_train.shape)

input_size = X_train.shape[1]
hidden_size1 = 1000
hidden_size2 = 100
latent_space_dimension = 12

train_dataset = torch.stack([torch.Tensor(i) for i in X_train])
test_dataset = torch.stack([torch.Tensor(i) for i in X_test])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, num_workers=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize)

# * Autoencoder

# AE_E = Encoder(input_size, hidden_size1, hidden_size2, latent_space_dimension).to(device)
# AE_D = Decoder(latent_space_dimension, hidden_size2, hidden_size1, input_size).to(device)

# # Define the loss function and optimizer
# criterion = nn.MSELoss()
# AE_optim = optim.Adam(itertools.chain(AE_E.parameters(), AE_D.parameters()), lr=lr, betas=(0.5, 0.999))

# num_epoch = 100
# for epoch in range(1, num_epoch + 1):
#     train_AE(AE_E, AE_D, AE_optim, epoch, train_loader, test_loader)

# * PCA Like Autoencoder

PCAAE_E = []
PCAAE_D = []
for id_m in range(latent_space_dimension):
    PCAAE_E_i = Encoder(input_size, hidden_size1, hidden_size2, latent_space_dimension=1).to(device)
    PCAAE_D_i = Decoder(latent_space_dimension=id_m + 1,
                        hidden_size2=hidden_size2,
                        hidden_size1=hidden_size1,
                        output_size=input_size).to(device)

    PCAAE_E.append(PCAAE_E_i)
    PCAAE_D.append(PCAAE_D_i)

PCAAE_optim = []
for id_m in range(latent_space_dimension):
    optim_temp = optim.Adam(itertools.chain(PCAAE_E[id_m].parameters(), PCAAE_D[id_m].parameters()),
                            lr=lr,
                            betas=(0.5, 0.999))
    PCAAE_optim.append(optim_temp)

is_training = True
if is_training:
    for model in range(1, latent_space_dimension + 1):
        for epoch in range(1, num_epoch + 1):
            train_PCA_AE(PCAAE_E, PCAAE_D, PCAAE_optim, epoch, model, train_loader, lambdarec, lambdacov, device)
            # torch.save( {'PCAAE_E_state_dict': PCAAE_E[model-1].state_dict(),
            #              'PCAAE_D_state_dict': PCAAE_D[model-1].state_dict(),
            #              'PCAAE_optim_state_dict': PCAAE_optim[model-1].state_dict(),})
