import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    '''(convolution => [BN] => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, init_features=32):
        super(UNet2D, self).__init__()

        features = init_features
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)

        self.outconv = nn.Conv2d(features, out_channels, kernel_size=1)

    def upCheckCat(self, x, d):
        if x.shape[-2:] == d.shape[-2:]:
            return torch.cat((x, d), dim=1)
        else:
            return torch.cat((x, F.interpolate(d, size=x.shape[-2:], mode='bilinear', align_corners=True)), dim=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool1(x1))
        x3 = self.encoder3(self.pool2(x2))
        x4 = self.encoder4(self.pool3(x3))

        # Bottleneck
        x5 = self.bottleneck(self.pool4(x4))

        # Decoder
        d4 = self.upconv4(x5)
        d4 = self.upCheckCat(x4, d4)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = self.upCheckCat(x3, d3)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.decoder1(d1)
        out = self.outconv(d1)
        return out
