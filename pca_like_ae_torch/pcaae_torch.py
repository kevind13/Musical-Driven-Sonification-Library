import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

import torchvision
import torch
from torch import nn

from torch import optim
import itertools

import scipy.io

import argparse

from sklearn.model_selection import train_test_split


## Get some parametres

np.random.seed(0) 

save_training_image = False
path_data = 'dataset'

batchsize = 128
gpu = False
folder4weights = 'weights'
train_latent = 40
lambdacov = 1
lambdarec = 1
num_epoch = 100
lr = 0.001

if gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


X_train = np.array(scipy.io.loadmat(f'{path_data}/all_4_channels_data_01_short.mat')['train_data'], dtype=np.uint8)
X_train = np.where(X_train == 90, 1, 0)
X_train, X_test =train_test_split(X_train, test_size=0.2, random_state=69)
X_train=np.array(X_train); X_test=np.array(X_test);

print(X_train.shape)
print(np.min(X_train),np.max(X_train))

train_dataset = torch.stack([torch.Tensor(i) for i in X_train])
test_dataset = torch.stack([torch.Tensor(i) for i in X_test])
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batchsize,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batchsize)

# Create the networks: encoder and decoder

import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, a, b, eps=1e-8, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        self._dtype_min_gt_0 = torch.tensor(torch.finfo(self.a.dtype).eps, dtype=self.a.dtype)
        self._dtype_max_lt_1 = torch.tensor(1 - torch.finfo(self.a.dtype).eps, dtype=self.a.dtype)
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * self.b - self._little_phi_a * self.a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)

    def expand(self, batch_shape, _instance=None):
        # TODO: it is likely that keeping temporary variables in private attributes violates the logic of this method
        raise NotImplementedError


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'loc': constraints.real,
        'scale': constraints.positive,
        'a': constraints.real,
        'b': constraints.real,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, a, b, eps=1e-8, validate_args=None):
        self.loc, self.scale, self.a, self.b = broadcast_all(loc, scale, a, b)
        a_standard = (a - self.loc) / self.scale
        b_standard = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a_standard, b_standard, eps=eps, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale


def truncated_normal(loc=0., scale=0.05, a=-1., b = 1.):
    return TruncatedNormal(loc, scale, a, b)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
       # print("Initialisation with truncated normal !")
        truncated_normal(m.weight)
        m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self, code_size=1, input_shape=(4, 256, 68)):
        super(Encoder, self).__init__()        
        self.latent_dim = code_size
        self.input_shape = input_shape

        input_size = input_shape[0] * input_shape[1] * input_shape[2]

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, code_size)

        self.zero_mean = nn.BatchNorm1d(self.latent_dim, affine=False, eps=0)
        self.leaky = nn.LeakyReLU(0.2)
    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(batch_size, -1)

        x = self.leaky(self.fc1(x))
        x = self.fc2(x)
        
        z = x.view((batch_size, -1))

        z = self.zero_mean(z)
        return z  
    
class Decoder(nn.Module):
    def __init__(self, code_size=1, output_shape=(4, 256, 68)):
        super(Decoder, self).__init__()
        # Shape required to start transpose convs
        self.output_shape = output_shape

        output_size = output_shape[0] * output_shape[1] * output_shape[2]

        # Fully connected layers
        self.fc1 = nn.Linear(code_size, 512)
        self.fc2 = nn.Linear(512, output_size)

        self.leaky = nn.LeakyReLU(0.2)
        
    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = self.leaky(self.fc1(z))
        x = self.fc2(x)

        # Reshape output
        x = x.view(batch_size, *self.output_shape)
        x = nn.Sigmoid()(x)
        return x


# Reconstruction loss

def reconstruction_loss(data, recon_data, MSE=False):
    if MSE:
        return nn.MSELoss()(data,recon_data)
    else:
        return nn.BCELoss()(data,recon_data)
    
# We define covariance loss

def cov_loss(z,step):
    if step>1:
        loss = 0
        for idx in range(step-1):
            loss += ((z[:,idx]*z[:,-1]).mean())**2
        loss = loss/(step-1)
    else:
        loss = torch.zeros_like(z)
    return loss.mean()

# We train our PCA-like AE

def train_PCA_AE(PCAAE_E,PCAAE_D, optimizer,
                 epoch, step, train_loader,
                 lambda_rec,lambda_cov,
                 device):
    train_loss = 0      
    train_content_loss = 0.
    train_cov_loss = 0.
       
    test_loss = 0      
    test_content_loss = 0.
    test_cov_loss = 0.       
    
    # Training part:
    for idx_step in range(1,step+1):
        PCAAE_E[idx_step-1].train()
    PCAAE_D[step-1].train()
    
    for id_step in range(1,step):
        model_temp = PCAAE_E[id_step-1]
        for param in model_temp.parameters():
            param.required_grad = False
            
    for batch_idx, (data) in enumerate(train_loader):
        data = data.to(device)
    
        z = []
        with torch.no_grad():
            for jd in range(step-1):
                z.append(PCAAE_E[jd](data))
        
        optimizer[step-1].zero_grad()
        z_i = PCAAE_E[step-1](data)
        z.append(z_i)           
        latent_space = torch.cat(z,dim=-1)
        recon_data = PCAAE_D[step-1](latent_space)
        loss_data = lambda_rec*reconstruction_loss(recon_data, data)
        loss_cov = lambda_cov*cov_loss(latent_space,step)
        loss = loss_data + loss_cov
        loss.backward()
        train_loss += loss.item()
        train_content_loss += loss_data.item()
        if step>1:
            train_cov_loss +=  loss_cov.item()
        optimizer[step-1].step()
    
    # Testing part:
    for idx_step in range(1,step+1):
        PCAAE_E[idx_step-1].eval()
    PCAAE_D[step-1].eval()
    
    for batch_idx, (data) in enumerate(test_loader):
        data = data.to(device)
        z = []
        for jx in range(step):
            z.append(PCAAE_E[jx](data))   
        latent_space = torch.cat(z,dim=-1)
        recon_data = PCAAE_D[step-1](latent_space)            
        loss_data = lambda_rec*reconstruction_loss(recon_data, data)
        loss_cov = lambda_cov*cov_loss(latent_space,step)
        test_loss += loss_data + loss_cov
        test_content_loss += loss_data
        if step>1:
            test_cov_loss +=  loss_cov    
            
    print('====> PCAAE{} Epoch: {} Train loss: {:.6f},\t Train Data loss: {:.6f},\t Train Cov loss: {:.8f},'.format(
            step,
            epoch, 
            train_loss / len(train_dataset), 
            train_content_loss / len(train_dataset), 
            train_cov_loss / len(train_dataset)))
    
    print('====> PCAAE{} Epoch: {} Test loss: {:.6f},\t Test Data loss: {:.6f},\t Tes Cov loss: {:.8f},'.format(
            step,
            epoch, 
            test_loss / len(test_dataset), 
            test_content_loss / len(test_dataset), 
            test_cov_loss / len(test_dataset)))

PCAAE_E = []
PCAAE_D = []
for id_m in range(train_latent):  
    PCAAE_E_i = Encoder(code_size=1).to(device)
    PCAAE_D_i = Decoder(code_size=id_m+1).to(device)
    PCAAE_E_i.apply(init_weights)
    PCAAE_D_i.apply(init_weights)

    PCAAE_E.append(PCAAE_E_i)
    PCAAE_D.append(PCAAE_D_i)

PCAAE_optim = []
for id_m in range(train_latent):
    optim_temp = optim.Adam(itertools.chain(PCAAE_E[id_m].parameters(), 
                                            PCAAE_D[id_m].parameters()), 
                            lr=lr, betas=(0.5, 0.999))
    PCAAE_optim.append(optim_temp)


if os.path.exists(folder4weights) is False:
    os.makedirs(folder4weights)

print("Training PCA AE for midi files")

for model in range(1, train_latent+1):
    weightname = folder4weights+'/PCAAE'+str(model)+'midi_'+'lambdacov'+str(lambdacov)
    for epoch in range(1, num_epoch + 1):
        train_PCA_AE(PCAAE_E, PCAAE_D, PCAAE_optim, 
                        epoch, model, train_loader, 
                        lambdarec, lambdacov, device)
        torch.save( {'PCAAE_E_state_dict': PCAAE_E[model-1].state_dict(),
                        'PCAAE_D_state_dict': PCAAE_D[model-1].state_dict(),
                        'PCAAE_optim_state_dict': PCAAE_optim[model-1].state_dict(),}, 
                    weightname)  
  
## We train a PCAAE without covariance loss

train_with_no_cov = False

if train_with_no_cov: 
    PCAAE_E_noCov = []
    PCAAE_D_noCov = []
    for id_m in range(train_latent):
        PCAAE_E_i = Encoder(code_size=1).to(device)
        PCAAE_D_i = Decoder(code_size=id_m+1).to(device)
        PCAAE_E_i.apply(init_weights)
        PCAAE_D_i.apply(init_weights)
    
        PCAAE_E_noCov.append(PCAAE_E_i)
        PCAAE_D_noCov.append(PCAAE_D_i)

    PCAAE_optim_noCov = []
    for id_m in range(train_latent):
        optim_temp = optim.Adam(itertools.chain(PCAAE_E_noCov[id_m].parameters(), 
                                                PCAAE_D_noCov[id_m].parameters()), 
                                lr=lr, betas=(0.5, 0.999))
        PCAAE_optim_noCov.append(optim_temp)
        
    print("Training PCA AE for midi files without covariance: ")
    
    for model in range(1, train_latent+1):
        weightname = folder4weights+'/PCAAE'+str(model)+'midi_'+'lambdacov0'
        for epoch in range(1, num_epoch + 1):
            train_PCA_AE(PCAAE_E_noCov, PCAAE_D_noCov, PCAAE_optim_noCov, 
                        epoch, model, train_loader, 
                        lambdarec, 0, device)
            torch.save( {'PCAAE_E_noCov_state_dict': PCAAE_E_noCov[model-1].state_dict(),
                        'PCAAE_D_noCov_state_dict': PCAAE_D_noCov[model-1].state_dict(),
                        'PCAAE_noCov_optim_state_dict': PCAAE_optim_noCov[model-1].state_dict(),}, 
                        weightname)  
                


# if __name__ == "__main__":
#     # configure parser and parse arguments
#     parser = argparse.ArgumentParser(description='PCA fit for midi files')
#     parser.add_argument('--gpu', default=False, type=bool, help='Use gpu')

#     args = parser.parse_args()
#     gpu = args.gpu