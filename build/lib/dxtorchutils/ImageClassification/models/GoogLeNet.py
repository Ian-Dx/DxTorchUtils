"""GoogLeNet v1"""
from collections import OrderedDict
import torch
from torch.nn import *


class GoogLeNet(Module):
    def __init__(self, num_classes=21):
        super(GoogLeNet, self).__init__()
        self.conv0 = _Conv(3, 64, 7, 2, 4)
        self.conv1 = Sequential(
            OrderedDict([
                ("LRN0", LocalResponseNorm(5)),
                ("conv0", _Conv(64, 192, 1)),
                ("conv1", _Conv(192, 192, 3, 1, 1)),
                ("LRN1", LocalResponseNorm(5))
            ])
        )

        self.pool0 = MaxPool2d(3, 2, ceil_mode=True)
        self.pool1 = MaxPool2d(3, 2, ceil_mode=True)
        self.pool2 = MaxPool2d(3, 2, ceil_mode=True)
        self.pool3 = MaxPool2d(3, 2, ceil_mode=True)
        self.pool4 = AvgPool2d(7, 1)

        self.inception1a = _Inception(192, 96, 16, 64, 128, 32, 32)
        self.inception1b = _Inception(256, 128, 32, 128, 192, 96, 64)
        self.inception2a = _Inception(480, 96, 16, 192, 208, 48, 64)
        self.inception2b = _Inception(512, 112, 24, 160, 224, 64, 64)
        self.inception2c = _Inception(512, 128, 24, 128, 256, 64, 64)
        self.inception2d = _Inception(512, 144, 32, 112, 288, 64, 64)
        self.inception2e = _Inception(528, 160, 32, 256, 320, 128, 128)
        self.inception3a = _Inception(832, 160, 32, 256, 320, 128, 128)
        self.inception3b = _Inception(832, 192, 48, 384, 384, 128, 128)
        self.aux1 = _InceptionAux(512)
        self.aux2 = _InceptionAux(528)
        self.fc = Linear(1024, 1000)
        self.out = Linear(1000, num_classes)

    def forward(self, input):
        # (n, 3, 224, 224)
        x = self.conv0(input)
        # (n, 64, 112, 112)
        x = self.pool0(x)
        # (n, 64, 56, 56)
        x = self.conv1(x)
        # (n, 192, 56, 56)
        x = self.pool1(x)
        # (n, 192, 28, 28).
        x = self.inception1a(x)
        # (n, 256, 28, 28)
        x = self.inception1b(x)
        # (n, 480, 28, 28)
        x = self.pool2(x)
        # (n, 480, 14, 14)
        x = self.inception2a(x)
        # (n, 512, 14, 14)
        aux1 = self.aux1(x)
        x = self.inception2b(x)
        # (n, 512, 14, 14)
        x = self.inception2c(x)
        # (n, 512, 14, 14)
        x = self.inception2d(x)
        # (n, 528, 14, 14)
        aux2 = self.aux2(x)
        x = self.inception2e(x)
        # (n, 832, 14, 14)
        x = self.pool3(x)
        # (n, 832, 7, 7)
        x = self.inception3a(x)
        # (n, 832, 7, 7)
        x = self.inception3b(x)
        # (n, 1024, 7, 7)
        x = self.pool4(x)
        x = torch.dropout(x, 0.7, self.training)
        # (n, 1024, 1, 1)
        x = x.view(x.shape[0], -1)
        # (n, 1024)
        x = self.fc(x)
        # (n, 1000)
        if self.training:
            x = x + 0.3 * aux1 + 0.3 * aux2

        output = self.out(x)
        # (n, num_classes)

        return output


class _Inception(Module):
    def __init__(self, in_channel, mid_channel2, mid_channel3, out_channel1, out_channel2, out_channel3, out_channel4):
        super(_Inception, self).__init__()
        self.path1 = _Conv(in_channel, out_channel1, 1)
        self.path2 = Sequential(
            OrderedDict([
                ("conv0", _Conv(in_channel, mid_channel2, 1)),
                ("conv1", _Conv(mid_channel2, out_channel2, 3, 1, 1))
            ])
        )
        self.path3 = Sequential(
            OrderedDict([
                ("conv0", _Conv(in_channel, mid_channel3, 1)),
                ("conv1", _Conv(mid_channel3, out_channel3, 5, 1, 2))
            ])
        )
        self.path4 = Sequential(
            OrderedDict([
                ("pool", MaxPool2d(3, 1, 1)),
                ("conv", _Conv(in_channel, out_channel4, 1))
            ])
        )

    def forward(self, input):
        x1 = self.path1(input)
        x2 = self.path2(input)
        x3 = self.path3(input)
        x4 = self.path4(input)
        output = torch.cat((x1, x2, x3, x4), 1)

        return output


class _InceptionAux(Module):
    def __init__(self, in_channel):
        super(_InceptionAux, self).__init__()
        self.conv = Sequential(
            OrderedDict([
                ("pool", AvgPool2d(5, 3)),
                ("conv", _Conv(in_channel, 128, 1))
            ])
        )
        self.fc = _FC(2048, 1024, 1000)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x





class _Conv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_Conv, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, input):
        x = self.conv(input)
        output = torch.relu(x)

        return output


class _FC(Module):
    def __init__(self, in_features, mid_features, out_features):
        super(_FC, self).__init__()
        self.fc0 = Linear(in_features, mid_features)
        self.fc1 = Linear(mid_features, out_features)

    def forward(self, input):
        x = self.fc0(input)
        x = torch.relu(x)
        output = self.fc1(x)

        return output