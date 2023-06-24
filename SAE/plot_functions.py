

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot_loss(train_loss, val_loss, path, title=None):
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if title is not None:
        plt.savefig(path + '/loss' + title + '.jpg')
    else:
        plt.savefig(path + '/loss.jpg')
    plt.show()


def plot_latent(latent, figure_title, figure_name, path, run_parameters):
    ncol = latent.shape[1]
    if ncol == 2:
        fig, ax = plt.subplots()
        ax.scatter(latent[:, 0], latent[:, 1], s=2)
        ax.legend()
    else:
        w, h = plt.figaspect(0.2)
        fig, ax = plt.subplots(1, int(ncol / 2), figsize=(w, h))
        for i, j in zip(range(0, ncol, 2), range(0, int(ncol / 2))):
            ax[j].scatter(latent[:, i], latent[:, i + 1], s=2)
            #ax[j].legend()
            ax[j].set_xlabel('Axis' + str(i + 1), fontsize=10)
            ax[j].set_ylabel('Axis' + str(i + 2), fontsize=10)
    fig.suptitle("Latent Space \n (" + figure_title + "_" + run_parameters + ")", fontsize=12)
    plt.tight_layout()
    plt.savefig(path + '/Latent' + figure_name + "_" + run_parameters + '.jpg')
    plt.show()
