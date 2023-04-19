
import os
import torch
import numpy as np
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from scipy.io import savemat
from torch.utils.data import Dataset, DataLoader
import copy
from tools import EarlyStopping
import random
import pandas as pd
import scipy.io


def load_data(dir, subdir='train_data', data_type=np.uint8):
    return np.array(scipy.io.loadmat(f'{dir}')[subdir], dtype=np.uint8)


def load_ibs(dir_ibs, fname_ibs):
    hf = h5py.File(dir_ibs, 'r')
    ibs = np.array(hf[fname_ibs][:]).astype(np.float32)
    return ibs


def categorical_cross_entropy(y_pred, y_true):
    return -(y_true * torch.log(y_pred)).sum(dim=1).sum()


def recon_loss(recon_x, x, loss_opt=None):
    if loss_opt =='MAE':
        recon_loss = F.l1_loss(recon_x, x, reduction='sum')
    elif loss_opt == 'BCE':
        recon_loss = F.binary_cross_entropy(recon_x, x) ##, reduction='sum')
    else:
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    return recon_loss


def vae_loss(recon_x, x, mu, logvar, variational_beta, loss_opt=None):
    # recon
    rcon_loss = recon_loss(recon_x, x, loss_opt)
    # KL-divergence
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rcon_loss + variational_beta * kldivergence


def save_checkpoint(model, optimizer, scheduler, epoch, path, V=None):
    if V is None:
        torch.save(
            {'model': model,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'scheduler_state_dict': scheduler.state_dict()},
            os.path.join(path,'checkpoint_{:04d}.pt'.format(epoch)))
    else:
        torch.save(
            {'model': model,
             'model_state_dict': model.state_dict(),
             'V': V,
             'optimizer_state_dict': optimizer.state_dict(),
             'scheduler_state_dict': scheduler.state_dict()},
            os.path.join(path,'checkpoint_SAEIBS_{:04d}.pt'.format(epoch)))


def run_AE(model, train_data, val_data, batch_size, optimizer, scheduler, device, num_epochs, savepath, num_patience, loss_opt=None):    
    train_loader = DataLoader(torch.from_numpy(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.from_numpy(val_data), batch_size=batch_size, shuffle=True)
    
    print('Training AE...')
    train_losses = []
    val_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(path=savepath + '/best_checkpoint_AE.pt', patience=num_patience, verbose=True, delta=0.00001)

    for epoch in range(num_epochs):
        model, train_loss = train_AE(model, train_loader, optimizer, scheduler, device, loss_opt)
        val_loss = validate_AE(model, val_loader, device, loss_opt)
        print('Epoch [%d / %d] training loss: %f validation loss: %f' % (epoch + 1, num_epochs, train_loss, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        early_stopping(val_loss, model, epoch)
        if epoch > 50 and epoch % 100 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, savepath + '/checkpoints/')
        if early_stopping.early_stop:
            print('Early stopping at epoch: %d' % epoch)
            break
    checkpoint = torch.load(savepath + '/best_checkpoint_AE.pt')  # reload the best checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    val_loss_min = early_stopping.val_loss_min
    return model, train_losses, val_losses, val_loss_min


def run_VAE(model, train_data, val_data, batch_size, optimizer, scheduler, device, num_epochs, savepath, num_patience, variational_beta, loss_opt=None):
    # load data in batch
    train_loader = DataLoader(torch.from_numpy(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.from_numpy(val_data), batch_size=batch_size, shuffle=True)

    print('Training AE...')
    train_losses = []
    val_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(path=savepath + '/checkpoint.pt', patience=num_patience, verbose=True, delta=0.00001)

    for epoch in range(num_epochs):
        model, train_loss = train_VAE(model, train_loader, optimizer, scheduler, device,variational_beta, loss_opt)
        val_loss = validate_VAE(model, val_loader, device, variational_beta, loss_opt)
        print('Epoch [%d / %d] training loss: %f validation loss: %f' % (epoch + 1, num_epochs, train_loss, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    checkpoint = torch.load(savepath + '/checkpoint.pt')  # reload the best checkpoint
    model.load_state_dict(checkpoint)
    val_loss_min = early_stopping.val_loss_min
    return model, train_losses, val_losses, val_loss_min


def run_SAEIBS(model, train_data, val_data, batch_size, optimizer, scheduler, device, num_epochs, savepath, num_patience, ref_ibs=None, loss_opt=None):

    train_loader = DataLoader(torch.from_numpy(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.from_numpy(val_data), batch_size=batch_size, shuffle=True)

    print('Training SAEIBS...')

    if 'SAE' in type(model).__name__:
        if model.emb is None:
            with torch.no_grad():
                for b, (x_data) in enumerate(train_loader):
                    x = x_data.to(device)
                    emb = model.encoder(x)
                    if b == 0:
                        embedding = copy.deepcopy(emb)
                    else:
                        embedding = torch.cat([embedding, emb], 0)
                model.initialize_svd(embedding)
            del x, embedding

    train_losses = []
    val_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(path=savepath + '/bestcheckpoint_SAEIBS.pt', patience=num_patience, verbose=True, delta=0.00001)

    for epoch in range(num_epochs):
        model, train_loss, z = train_SAEIBS(model, train_loader, optimizer, scheduler, device, loss_opt)
        val_loss = validate_SAEIBS(model, val_loader, device, loss_opt)
        print('Epoch [%d / %d] training loss: %f validation loss: %f' % (epoch + 1, num_epochs, train_loss, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        early_stopping(val_loss, model, epoch, model.V, model.mean_emb)
        if epoch % 100 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, savepath+'/checkpoints/', model.V)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    checkpoint = torch.load(savepath + '/bestcheckpoint_SAEIBS.pt')  # reload best checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    V = checkpoint['V']
    val_loss_min = early_stopping.val_loss_min
    return model, train_losses, val_losses, val_loss_min, V


def train_AE(model, data_loader, optimizer, scheduler, device, loss_opt=None):
    # set to training mode
    model.train()
    total_loss = 0
    for batch_idx, (x_batch) in enumerate(data_loader):
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        
        x_batch_recon, h = model(x_batch)
        # reconstruction error
        loss = recon_loss(x_batch_recon, x_batch, loss_opt)
        # backpropagation
        loss.backward()
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        scheduler.step()
        # loss
        total_loss += loss.item()

    return model, total_loss / len(data_loader.dataset)


def train_VAE(model, dataloader, optimizer, scheduler, device, variational_beta, loss_opt=None):
    # set to training mode
    model.train()
    total_loss = 0
    for b, (x_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        x_batch_recon, h, latent_mu, latent_logvar = model(x_batch)
        # vae loss
        loss = vae_loss(x_batch_recon, x_batch, latent_mu, latent_logvar, variational_beta, loss_opt)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        scheduler.step()
        # loss
        total_loss += loss.item()
    return model, total_loss/len(dataloader.dataset)


def train_SAEIBS(model, dataloader, optimizer, scheduler, device, loss_opt=None):
    # set to training mode
    model.train()
    total_loss = 0
    for b, (x_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        # reconstruction
        x_batch_recon, z = model(x_batch)
        # reconstruction error
        loss = recon_loss(x_batch_recon, x_batch, loss_opt)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        scheduler.step()
        # loss
        total_loss += loss.item()
    return model, total_loss/len(dataloader.dataset), z


def validate_AE(model, data_loader, device, loss_opt=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for b, (x_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device)
            x_batch_recon, h = model(x_batch)
            loss = recon_loss(x_batch_recon, x_batch, loss_opt)
            total_loss += loss.item()
    return total_loss/len(data_loader.dataset)


def validate_VAE(model, dataloader, device, variational_beta, loss_opt=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for b, (x_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            x_batch_recon, h, latent_mu, latent_logvar = model(x_batch)
            loss = vae_loss(x_batch_recon, x_batch, latent_mu, latent_logvar, variational_beta, loss_opt)
            total_loss += loss.item()
    return total_loss/len(dataloader.dataset)


def validate_SAEIBS(model, dataloader, device, loss_opt=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for b, (x_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            x_batch_recon, _ = model(x_batch)
            loss = recon_loss(x_batch_recon, x_batch, loss_opt)
            total_loss += loss.item()
    return total_loss/len(dataloader.dataset)


def project(model, data, batch_size, latent_dim, device):
    dataloader = DataLoader(torch.from_numpy(data), batch_size=batch_size, shuffle=False)
    # set to evaluation mode
    model.eval()
    latent = torch.zeros(0, latent_dim)
    for b, (x_batch) in enumerate(dataloader):
        with torch.no_grad():
            x_batch = x_batch.to(device)
            # latent space
            if 'VariationalAutoencoder' in type(model).__name__:
                X_pred, z, _, _ = model(x_batch)
            else:
                X_pred, z = model(x_batch)
            latent = torch.cat((latent.to(device), z))
    latent = latent.to('cpu').detach().numpy()
    return X_pred, latent


def projectSAEIBS_traindata(model, data, V, batch_size, device):
    dataloader = DataLoader(MyDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=False)
    # set to evaluation mode
    model.eval()
    with torch.no_grad():
        for b, (x_batch, _) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            emb = model.encoder(x_batch)
            if b == 0:
                embedding = copy.deepcopy(emb)
            else:
                embedding = torch.cat([embedding, emb], 0)

        mean_emb = torch.mean(embedding, 0)
        latent = torch.matmul(embedding - mean_emb, V)
    return latent.to('cpu').detach().numpy(), mean_emb.to('cpu').detach().numpy()

def projectSAEIBS_single(model, data, device):
    dataloader = DataLoader(torch.from_numpy(data), batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for b, (x_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            x_enc, z, x_hat, V, mean_emb = model.encoder_svd(x_batch)

            print(z)

            x_hat_v2 = torch.matmul(z, torch.transpose(V, 0, 1)) + mean_emb
            
            print(x_hat.shape)
            print(x_hat_v2,shape)

            x_recon = model.decoder(x_hat_v2)

            print(x_recon.shape)
            print(x_batch.shape)

            # if b == 0:
            #     latent = copy.deepcopy(x_enc)
            # else:
            #     embedding = torch.cat([embedding, emb], 0)

    # return latent.to('cpu').detach().numpy(), mean_emb.to('cpu').detach().numpy()


def projectSAEIBS_newdata(model, data, ibs_connect, V, mean_emb, batch_size, device):
    dataloader = DataLoader(MyDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=False)
    # set to evaluation mode
    model.eval()
    with torch.no_grad():
        for b, (x_batch, _) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            emb = model.encoder(x_batch)
            if b == 0:
                embedding = copy.deepcopy(emb)
            else:
                embedding = torch.cat([embedding, emb], 0)

        embedding = torch.mm(torch.from_numpy(ibs_connect).to(device), embedding)
        latent = torch.matmul(embedding - torch.from_numpy(mean_emb).to(device), V)
    return latent.to('cpu').detach().numpy()


def load_project_save(dir, fname, model, savename, key, savepath, batch_size, scale_opt, latent_dim, device):
    data = load_data(dir, fname, scale_opt)
    latent = project(model, data, batch_size, latent_dim, device)
    savemat(savepath + '/' + savename + '.mat', mdict={key: latent})


def load_projectSAEIBS_save(dir, fname, dir_ibs, fname_ibs, model, V, mean_emb, savename, key, savepath, batch_size, scale_opt, device):
    data = load_data(dir, fname, scale_opt)
    ibs_connect = load_ibs(dir_ibs, fname_ibs)
    latent = projectSAEIBS_newdata(model, data, ibs_connect, V, mean_emb, batch_size, device)
    savemat(savepath + '/' + savename + '.mat', mdict={key: latent})


def orthogonality(latent, path, run_parameters, modelname):
    # covariance of embedding
    cov_emb = np.cov(np.transpose(latent))
    plt.imshow(cov_emb, cmap='seismic', interpolation='nearest')
    plt.xlabel("Dimension")
    plt.ylabel("Dimension")
    plt.suptitle("Covariance matrix\n (1 KG " + modelname + run_parameters + ")", fontsize=12)
    plt.savefig(path + '/Cov_' + run_parameters + '.jpg')
    plt.show()
    return cov_emb
