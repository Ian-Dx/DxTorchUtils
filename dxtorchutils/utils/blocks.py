import time

from dxtorchutils.utils.layers import *
from torch.nn import functional as F


class SeBlock(Module):
    def __init__(self, channels, reduction=16):
        super(SeBlock, self).__init__()

        self.channels = channels
        mid_channel = channels // reduction if channels // reduction > 0 else 1

        self.pool = AdaptiveAvgPool2d(1)
        self.fc0 = fc_relu(channels, mid_channel, False)
        self.fc1 = fc_sigmoid(mid_channel, channels, False)

    def forward(self, input):
        batch_size, channels, _, _ = input.shape
        assert channels == self.channels, "Channel mismatch"

        x = self.pool(input)
        x = x.view(batch_size, channels)
        x = self.fc0(x)
        x = self.fc1(x)
        attention = x.view(batch_size, channels, 1, 1)

        output = input * attention

        return output


class NonLocalBlock(Module):
    def __init__(self, in_channels, mid_channels=None, sub_sample=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        if mid_channels is None:
            mid_channels = in_channels // 2 if in_channels // 2 != 0 else 1

        self.mid_channels = mid_channels

        self.g = Conv2d(in_channels, mid_channels, 1)
        self.phi = Conv2d(in_channels, mid_channels, 1)
        self.theta = Conv2d(in_channels, mid_channels, 1)

        self.w = Sequential(
            OrderedDict([
                ("W", Conv2d(mid_channels, in_channels, 1, 1)),
                ("normalization", BatchNorm2d(in_channels))
            ])
        )

        self.sub_sample = sub_sample

        if sub_sample:
            self.g = Sequential(
                OrderedDict([
                    ("G", self.g),
                    ("pool", MaxPool2d(2))
                ])
            )
            self.phi = Sequential(
                OrderedDict([
                    ("Phi", self.phi),
                    ("pool", MaxPool2d(2))
                ])
            )
            self.pool = MaxPool2d(2)

    def forward(self, input):
        batch_size, _, h, w = input.shape

        x_g = self.g(input).view(batch_size, self.mid_channels, -1)
        x_g = x_g.permute(0, 2, 1)

        x_theta = self.theta(input).view(batch_size, self.mid_channels, -1)
        x_theta = x_theta.permute(0, 2, 1)

        x_phi = self.phi(input).view(batch_size, self.mid_channels, -1)

        x_f = torch.matmul(x_theta, x_phi)
        x_f = F.softmax(x_f, dim=-1)

        x = torch.matmul(x_f, x_g)
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, -1, h, w)
        x_w = self.w(x)

        output = x_w + input

        return output


class GlobalContextBlock(Module):
    def __init__(self, in_channels, mid_channels=None):
        super(GlobalContextBlock, self).__init__()

        self.in_channels = in_channels
        if mid_channels is None:
            self.mid_channels = in_channels // 16 if in_channels // 16 > 0 else 1

        self.conv_mask = Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.channel_conv = Sequential(
            OrderedDict([
                ("conv0", Conv2d(self.in_channels, self.mid_channels, kernel_size=1)),
                ("normalization", LayerNorm([self.mid_channels, 1, 1])),
                ("activation", ReLU(inplace=True)),
                ("conv1", Conv2d(self.mid_channels, self.in_channels, kernel_size=1))
            ])
        )

    def forward(self, input):
        batch_size, c, h, w = input.shape
        # input [N, C, H, W]
        x = input.view(batch_size, c, h * w)
        # x [N, C, H * W]
        x = x.unsqueeze(1)
        # x [N, 1, C, H * W]
        x_mask = self.conv_mask(input)
        # mask [N, 1, H, W)
        x_mask = x_mask.view(batch_size, 1, h * w)
        # mask [N, 1, H * W)
        x_mask = self.softmax(x_mask)
        x_mask = x_mask.unsqueeze(-1)
        # mask [N, 1, H * W, 1]
        x_context = torch.matmul(x, x_mask)
        # context [N, 1, C, 1]
        x_context = x_context.view(batch_size, c, 1, 1)
        # context [N, C, 1, 1]
        x_context = self.channel_conv(x_context)
        # context [N, C, 1, 1]
        output = input + x_context
        # context [N, C, H, W]

        return output


if __name__ == "__main__":
    in_tensor = torch.ones((12, 64, 128, 128))

    cb = GlobalContextBlock(64)
    cv = NonLocalBlock(64)
    time0 = time.time()
    out_tensor = cb(in_tensor)
    time1 = time.time()
    hh = cv(in_tensor)

    time2 = time.time()

    print(time1-time0)
    print(time2-time1)

    print(in_tensor.shape)
    print(out_tensor.shape)
    print(hh.shape)