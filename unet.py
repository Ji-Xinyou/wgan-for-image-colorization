import torch
import torch.nn as nn
import torchvision.models as models

class SABLock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.f = nn.Conv2d(channel, channel, kernel_size=1, stride=1)
        self.g = nn.Conv2d(channel, channel, kernel_size=1, stride=1)
        self.h = nn.Conv2d(channel, channel, kernel_size=1, stride=1)
        self.v = nn.Conv2d(channel, channel, kernel_size=1, stride=1)
    def forward(self, x):
        fx = self.f(x)
        gx = self.g(x)
        attention = nn.Softmax(dim=3)(fx.permute(0, 1, 3, 2).matmul(gx))
        hx = self.h(x)
        return self.v(hx.matmul(attention))

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )

class ResNetUNet(nn.Module):
    def __init__(self, in_class, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0[0] = nn.Conv2d(in_class, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)

        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        # self.SA1 = SABLock(64)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)

        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        # self.SA2 = SABLock(128)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)

        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        # self.SA3 = SABLock(256)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)

        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.SA = SABLock(512)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(in_class, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.SA(self.layer4(layer3))

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out