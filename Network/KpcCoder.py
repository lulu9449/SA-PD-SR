import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, kernel_width, out_channels=10):
        super(Encoder, self).__init__()
        print("---", kernel_width)
        self.conv1 = nn.Conv2d(kernel_width ** 2, out_channels*2, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, kernels):
        kpc = self.tanh(self.conv1(kernels))
        kpc = self.conv2(kpc)
        return kpc

class Decoder(nn.Module):
    def __init__(self, kernel_width, in_channels=10):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels*2, kernel_width ** 2, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, kernels):
        kpc = self.tanh(self.conv1(kernels))
        kpc = self.conv2(kpc)
        return kpc
