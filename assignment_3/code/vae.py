import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from torchvision.utils import make_grid

from datasets.bmnist import bmnist

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.fc1 = nn.Linear(784, hidden_dim)    # hidden layer
        self.fc21 = nn.Linear(hidden_dim, z_dim) # mean
        self.fc22 = nn.Linear(hidden_dim, z_dim) # variance

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        h = F.relu(self.fc1(input)) # hidden
        mean, std = self.fc21(h), self.fc22(h) # latent space: mean and variance

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.fc1 = nn.Linear(z_dim, hidden_dim) # hidden layer
        self.fc2 = nn.Linear(hidden_dim, 784)   # output layer

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """

        h = F.relu(self.fc1(input))
        mean = torch.sigmoid(self.fc2(h))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        # reshape
        input = input.view(-1, 784) 

        # mean and variance
        mean, std = self.encoder.forward(input) 

        # reparameterization
        epsilon = torch.randn_like(std)
        z = epsilon.mul(std).add_(mean)

        # reconstruct digit
        output = self.decoder(z)

        # loss
        L_recon = F.binary_cross_entropy(output, input.view(-1, 784), reduction='sum') # reconstruction loss
        L_reg = -0.5 * torch.sum(1 + std - mean.pow(2) - std.exp()) # regularization loss
        average_negative_elbo = (L_recon + L_reg)/input.shape[0] # total loss

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        # sample from normal distribution
        z_sample = torch.randn(n_samples, self.z_dim)

        # pixel means from sampled latent space
        im_means = self.decoder(z_sample)

        # pixelwise bernoulli to sample image
        sampled_ims = torch.bernoulli(im_means)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """

    total_epoch_elbo = 0

    for step, batch in enumerate(data):
        if model.training:
            # Set gradients to zero
            optimizer.zero_grad()
        batch_elbo = model(batch)
        total_epoch_elbo += batch_elbo
        if model.training:
            # The optimization process
            batch_elbo.backward() # perform backward pass
            optimizer.step() # update weights

    average_epoch_elbo = batch_elbo / step

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    # create output directory
    path = os.path.join('output', 'vae')
    os.makedirs(path, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename))
    plt.close()


def plot_samples(sampled_ims, im_means, epoch=0):
    # create output directory
    ims_path = os.path.join('output', 'vae', 'samples', 'images')
    means_path = os.path.join('output', 'vae', 'samples', 'means')
    os.makedirs(ims_path, exist_ok=True)
    os.makedirs(means_path, exist_ok=True)

    sampled_ims_grid = make_grid(sampled_ims.view(-1, 1, 28, 28))
    im_means_grid = make_grid(im_means.view(-1, 1, 28, 28))

    plt.imshow(sampled_ims_grid[0])
    plt.tight_layout()
    plt.savefig(os.path.join(ims_path, 'epoch_' + str(epoch) + '.png'))
    
    plt.imshow(im_means_grid[0].detach().numpy())
    plt.tight_layout()
    plt.savefig(os.path.join(means_path, 'epoch_' + str(epoch) + '.png'))


def plot_manifold(model, filename='manifold.pdf'):
    # Create output image directory
    path = os.path.join('output', 'vae')
    os.makedirs(path, exist_ok=True)

    ppf = norm.ppf(torch.linspace(0, 1, steps=20))
    X,Y = np.meshgrid(ppf, ppf)
    z = torch.tensor([X.flatten(),Y.flatten()]).to(torch.float).t()

    im_means = model.decoder(z)
    plt.imshow(make_grid(im_means.view(-1, 1, 28, 28), nrow=20).detach().numpy()[0])
    plt.savefig(os.path.join(path, filename))


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=args.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    # Sample initial
    sampled_ims, im_means = model.sample(64)
    plot_samples(sampled_ims, im_means, epoch=-1)

    train_curve, val_curve = [], []
    for epoch in range(args.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        # --------------------------------------------------------------------

        sampled_ims, im_means = model.sample(64)
        plot_samples(sampled_ims, im_means, epoch=epoch)

    # --------------------------------------------------------------------
    #  Add functionality to plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')
    if args.zdim == 2:
        plot_manifold(model, 'manifold.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    args = parser.parse_args()

    main()