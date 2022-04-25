import argparse
import os
from pickletools import optimize
from cv2 import CAP_PROP_XI_TEST_PATTERN_GENERATOR_SELECTOR
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets
from torch.autograd import Variable

from data import get_miniImagenet_dataset
from model import Generator, Discriminator, get_resnet_D, get_unet_G

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=300, help="interval between image samples")
parser.add_argument("--saving_interval", type=int, default=10, help="interval between model saving")
parser.add_argument("--c", type=float, default=0.0025, help="constant of L1 norm")
parser.add_argument("--gp", type=float, default=10, help="constant of gp")
args = parser.parse_args()

use_GPU = True if torch.cuda.is_available() else False
tensorType = torch.cuda.FloatTensor if use_GPU else torch.FloatTensor

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = tensorType(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(tensorType(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train():
    default_shape = (128, 128)

    train_loader = get_miniImagenet_dataset(batchsize=args.batch_size, shape=default_shape, phase='train')
    val_loader = get_miniImagenet_dataset(batchsize=args.batch_size, shape=default_shape, phase='val')
    test_loader = get_miniImagenet_dataset(batchsize=args.batch_size, shape=default_shape, phase='test')

    # G = Generator(default_shape)
    G = get_unet_G()
    optimizerG = torch.optim.Adam(G.parameters(), lr=args.lr)
    # D = Discriminator(default_shape)
    D = get_resnet_D()
    optimizerD = torch.optim.Adam(D.parameters(), lr=args.lr)

    if use_GPU:
        G = G.cuda()
        D = D.cuda()
    
    batches_done = 0
    adloss = torch.nn.BCELoss()

    for epoch in tqdm(range(args.n_epochs)):

        for i, (gray, colorized) in enumerate(train_loader):
            try: 
                # z = Variable(tensorType(np.random.normal(0, 1, gray.shape)))
                # z = (z - z.max()) / (z.max() - z.min())

                if use_GPU:
                    gray = gray.cuda()
                    colorized = colorized.cuda()

                # Configure input
                gray = Variable(gray.type(tensorType))
                gray = torch.cat([gray, gray, gray], dim = 1)
                # gray = gray.view(gray.shape[0], -1)

                # Train Discriminator

                if i % args.n_critic == 0:
                    optimizerD.zero_grad()

                    colored_by_G = G(gray).detach()

                    penalty = compute_gradient_penalty(D, colorized, colored_by_G)

                    # Adversarial loss

                    loss_D = -torch.mean(D(colorized)) + torch.mean(D(colored_by_G)) + args.gp * penalty
                    # loss_D = (adloss(D(colorized), valid) + adloss(D(colored_by_G), fake)) / 2 + args.gp * penalty

                    loss_D.backward()

                    optimizerD.step()

                    # Clip weights of discriminator
                    for p in D.parameters():
                        p.data.clamp_(-args.clip_value, args.clip_value)
                    
                # Train Generator

                optimizerG.zero_grad()

                # Generate a batch of images
                gen_imgs = G(gray)

                # Adversarial loss
                L = args.c * torch.norm(gen_imgs - colorized, p=1)
                loss_G = -torch.mean(D(gen_imgs)) + L
                # loss_G = adloss(D(gen_imgs), valid) + L

                loss_G.backward()
                optimizerG.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [L: %f]"
                    % (epoch, args.n_epochs, batches_done % len(train_loader), len(train_loader), loss_D.item(), loss_G.item(), L.item())
                )

                if batches_done % args.sample_interval == 0:
                    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                batches_done += 1

            except KeyboardInterrupt:
                print('interrupted. try saving model now..')
                torch.save(G, f'G_conv_epoch{epoch}.pth')
                print('saved')
                exit(0)

        if (epoch + 1) % args.saving_interval == 0:
            print("Saving model...")
            torch.save(G, f'G_conv_epoch{epoch}.pth')

    print("Saving model...")
    torch.save(G, 'G_conv.pth')

if __name__ == "__main__":
    train()