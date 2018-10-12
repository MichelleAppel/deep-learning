import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from tensorboardX import SummaryWriter

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        self.layers = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        return self.layers(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    if args.summary:
        # Tensorboard writer
        writer = SummaryWriter(os.path.join('output', 'gan', 'summary', args.model_name))

    adversarial_loss = torch.nn.BCELoss()

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            imgs.cuda()
            adversarial_loss.cuda()
            discriminator.cuda()
            generator.cuda()

            # Adversarial ground truths
            valid = torch.Tensor(imgs.shape[0], 1).fill_(1.0).cuda()
            fake = torch.Tensor(imgs.shape[0], 1).fill_(0.0).cuda()

            z = torch.randn(imgs.shape[0], args.latent_dim).cuda()
            g = generator(z)

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()

            g_loss = adversarial_loss(discriminator(g), valid)

            # The optimization process
            g_loss.backward() # Perform backward pass
            optimizer_G.step() # Update weights

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(imgs.view(imgs.shape[0], -1).cuda()), valid)
            fake_loss = adversarial_loss(discriminator(g.detach()), fake)
            d_loss = (real_loss + fake_loss)/2

            d_loss.backward() # Perform backward pass
            optimizer_D.step() # Update weights

            # Print progress test
            if i % 10 == 0:
                print("Epoch {}, Train Step {:03d}, Batch Size = {}, "
                    "G_loss = {:.3f}, D_loss = {:.3f}".format(
                        epoch, i,
                        args.batch_size, g_loss, d_loss
                ))   

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i

            if args.summary:
                writer.add_scalar('g_loss', g_loss, batches_done)
                writer.add_scalar('d_loss', d_loss, batches_done)

            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:

                save_image(g[:25].view(25, 1, 28, 28), os.path.join('output', 'gan', 'images', args.model_name, '{}.png').format(batches_done), nrow=5, normalize=True)

                
def main():
    # create output image directory
    path = os.path.join('output', 'gan')
    os.makedirs(os.path.join(path, 'images', args.model_name), exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # itart training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # save generator
    torch.save(generator.state_dict(), os.path.join(path, args.model_name + '.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--summary', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, 
                        help='Make summary')
    parser.add_argument('--model_name', type=str, default='gan_1d',
                        help='dir to save stuff')                        
    args = parser.parse_args()

    main()