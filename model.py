import torch.nn as nn
import torchvision.models as models
import numpy as np

from unet import ResNetUNet

class Generator(nn.Module):
    '''
    Generator of Wassersterin Gan
        input: gray scale image
        output: colorized image
    '''
    def __init__(self, img_shape=(224, 224), latent_dim=100):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        def conv_block(inchan, outchan, kern_size=3):
            layers = []
            if kern_size == 3:
                layers.append(nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1)) # scale preserving
            else:
                layers.append(nn.Conv2d(inchan, outchan, kernel_size=1, stride=1)) # scale preserving
            layers.append(nn.BatchNorm2d(outchan))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        
        self.img_shape = (3, *img_shape)

        # self.model = nn.Sequential(
        #     *block(int(np.prod(img_shape)), 128, normalize=False),
        #     *block(128, 256),
        #     *block(256, 512),
        #     *block(512, 1024),
        #     nn.Linear(1024, 3 * int(np.prod(img_shape))),
        #     nn.Tanh()
        # )
        self.conv1 = nn.Sequential(*conv_block(1, 32, kern_size=1))
        self.conv2 = nn.Sequential(*conv_block(32, 64, kern_size=1))
        self.conv3 = nn.Sequential(*conv_block(64, 128, kern_size=1))
        self.conv5 = nn.Sequential(*conv_block(128, 128, kern_size=1))
        self.conv5 = nn.Sequential(*conv_block(128, 3, kern_size=1))

    def forward(self, gray, z):
        # img = self.model(gray_image)
        # img = img.view(img.shape[0], *self.img_shape)
        # return img

        # method 1
        # img = gray + z
        # img = self.conv1(gray)
        # img = self.conv2(img)
        # img = self.conv3(img)
        # img = self.conv4(img)
        # img = img + gray
        # return img
        
        # method 2
        gray = self.conv1(gray)
        z = self.conv1(z)
        
class Discriminator(nn.Module):
    '''
    Discriminator of Wasserstein GAN
        input: colorized image
        output: validity of the image
    '''
    def __init__(self, img_shape=(64, 64)):
        super(Discriminator, self).__init__()

        def conv_block(inchan, outchan):
            layers = []
            layers.append(nn.Conv2d(inchan, outchan, kernel_size=4, stride=2, padding=1))  # halfscale
            layers.append(nn.BatchNorm2d(outchan))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.convs = nn.Sequential(
            # nn.Linear(3 * int(np.prod(img_shape)), 512),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 1),
            *conv_block(3, 64), # 224 -> 112
            *conv_block(64, 128), # -> 56
            *conv_block(128, 256), # -> 28
            *conv_block(256, 512), # -> 14 x 14
        )

        self.fc = nn.Linear(512 * ((img_shape[0] // 16) ** 2), 1)

    def forward(self, img):
        # img_flat = img.view(img.shape[0], -1)
        # validity = self.model(img_flat)
        # return validity
        img = self.convs(img)
        img = img.view(img.shape[0], -1)
        return self.fc(img)
    
def get_unet_G():
    return ResNetUNet(n_class=3)

def get_resnet_D():
    return models.resnet18(pretrained=False, num_classes=1)
    # class Resnet_D(nn.Module):
    #     def __init__(self):
    #         super(Resnet_D, self).__init__()
    #         self.resnet = models.resnet18(pretrained=False, num_classes=1)
    #         self.out = nn.Sigmoid()

    #     def forward(self, x):
    #         return self.out(self.resnet(x))
    # return Resnet_D()
            