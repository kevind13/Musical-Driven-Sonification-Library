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

if gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.zero_mean = nn.BatchNorm1d(hidden_size3, affine=False, eps=0)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.fc1(x))
        x = self.leaky(self.fc2(x))
        x = self.fc3(x)
        return x

class Decoder(nn.Module):

    def __init__(self, hidden_size3, hidden_size2, hidden_size1, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_size3, hidden_size2)
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


def train_AE(E, D, optimizer, epoch, train_data, test_loader):
    train_loss = 0
    test_loss = 0

    E.train()
    D.train()
    for batch_idx, (data) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        z = E(data)
        recon_data = D(z)
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

    print('====> AE Epoch: {}, Train loss: {:.6f}, Test loss: {:.6f}'.format(epoch, train_loss / len(train_data),
                                                                             test_loss / len(test_loader)))


X_train = np.array(scipy.io.loadmat('X_train.mat')['X_train'], dtype=np.float32)
X_test = X_train[:5]
X_train = X_train[5:]

print(X_test.shape)
print(X_train.shape)

input_size = X_train.shape[1]
hidden_size1 = 541
hidden_size2 = 54
hidden_size3 = 12

train_dataset = torch.stack([torch.Tensor(i) for i in X_train])
test_dataset = torch.stack([torch.Tensor(i) for i in X_test])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, num_workers=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize)

# autoencoder = Autoencoder(input_size, hidden_size1, hidden_size2, hidden_size3, input_size)

AE_E = Encoder(input_size, hidden_size1, hidden_size2, hidden_size3).to(device)
AE_D = Decoder(hidden_size3, hidden_size2, hidden_size1, input_size).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
AE_optim = optim.Adam(itertools.chain(AE_E.parameters(),AE_D.parameters()), lr=lr, betas=(0.5, 0.999))

num_epoch = 100
for epoch in range(1, num_epoch + 1):
    train_AE(AE_E, AE_D, AE_optim, epoch, train_loader, test_loader)
