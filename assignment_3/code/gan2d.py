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

        self.fc_layers = nn.Sequential(
            nn.Linear(args.latent_dim, 128*7**2)
        )

        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.shape[0], -1)
        h = self.fc_layers(z)
        out = self.conv_layers(h.view(h.shape[0], 128, 7, 7))
        out = out.view(out.shape[0], 1, 28, 28)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=8 ,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                    ),                      # output shape (16, 28, 28)
            nn.LeakyReLU(0.2),              # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (8, 14, 14)
            
            nn.Conv2d(
                in_channels=8,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                    ),                      # output shape (32, 14, 14)
            nn.LeakyReLU(0.2),              # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 7, 7)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(16*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        h = self.conv_layers(img)
        # print(h.shape)
        return self.fc_layers(h.view(h.shape[0], -1))
        


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

            z = torch.randn(imgs.shape[0], args.latent_dim, 1, 1).cuda()
            g = generator(z)
            # print(g.shape)
            # print(g)

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

            real_loss = adversarial_loss(discriminator(imgs.cuda()), valid)
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
    parser.add_argument('--model_name', type=str, default='gan_2d',
                        help='dir to save stuff')                        
    args = parser.parse_args()

    main()