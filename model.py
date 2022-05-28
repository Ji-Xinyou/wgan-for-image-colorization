import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import functools

from loss import GANLoss, compute_gradient_penalty

class Sobel(nn.Module):
    '''
    Edge detection using Sobel operator:
        input: depth image
        output: 
            out[:, 0, :, :] = dx
            out[:, 1, :, :] = dy
    
    The output of Sobel operator will be used to
    compute terms **loss_grad** and **loss_normal**
    '''
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        # 2(dx, dy) x 1(depth) x (3 x 3) (filter size)
        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        return out

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        map = self.model(input)
        return torch.sum(map, dim=(2, 3))

class MainModel(nn.Module):
    def __init__(self, net_G=None, net_D=None, lr_G=5e-5, lr_D=5e-5, 
                 beta1=0.2, beta2=0.9, lambda_L1=100., 
                 lambda_gp=10., clip_value=0.01, weight_decay=1e-4,
                 clip=False, use_edge=False,
                 ):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.lambda_gp = lambda_gp
        self.clip = clip_value
        self.wgan = clip
        self.use_edge = use_edge

        self.net_G = net_G.to(self.device)
        self.net_D = net_D.to(self.device)

        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.RMSprop(self.net_G.parameters(), lr=lr_G)
        self.opt_D = optim.RMSprop(self.net_D.parameters(), lr=lr_D)
        self.edge_detector = Sobel().to(self.device)
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, l, ab):
        self.L = l.to(self.device)
        if self.use_edge:
            self.edge = self.edge_detector(self.L)
        self.ab = ab.to(self.device)
        
    def forward(self):
        input = self.L
        if self.use_edge:
            input = torch.cat([self.L, self.edge], dim=1)
        self.fake_color = self.net_G(input)
    
    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)

        penalty = 0
        if self.wgan:
            penalty = compute_gradient_penalty(self.net_D, real_image.data, fake_image.data)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + penalty * self.lambda_gp
        self.loss_D.backward()
    
    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G = self.loss_G_GAN
        if self.lambda_L1 != 0:
            self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
            if self.use_edge:
                edge_fake_a = self.edge_detector(self.fake_color[:, 0, :, :].unsqueeze(dim=1))
                edge_fake_b = self.edge_detector(self.fake_color[:, 1, :, :].unsqueeze(dim=1))
                edge_a = self.edge_detector(self.ab[:, 0, :, :].unsqueeze(dim=1))
                edge_b = self.edge_detector(self.ab[:, 1, :, :].unsqueeze(dim=1))
                self.loss_edge_l1 = (self.L1criterion(edge_a, edge_fake_a) + \
                                    self.L1criterion(edge_b, edge_fake_b)) * self.lambda_L1 / 3
                self.loss_G += self.loss_edge_l1
            self.loss_G += self.loss_G_L1

        self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        # if self.wgan:
        #     for p in self.net_D.parameters():
        #         p.data.clamp_(-self.clip, self.clip)
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
    
    def get_l(self):
        return self.L
    
    def get_ab(self):
        return self.fake_color
    
    def fake_result(self):
        return torch.cat([self.L, self.fake_color], dim=1)
    
    def loss_item(self):
        return {'D': self.loss_D.item(), 'G_GAN': self.loss_G_GAN.item(), 'G_L1': self.loss_G_L1.item()}
    
    def get_G(self):
        return self.net_G
    
    def get_D(self):
        return self.net_D
    