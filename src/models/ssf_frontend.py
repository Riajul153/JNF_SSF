import torch
from torch import nn

from models.erb import ERB


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.depth = nn.Conv2d(channels, channels, (kernel_size, kernel_size), padding=(pad, pad), groups=channels)
        self.point = nn.Conv2d(channels, channels, 1)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.point(self.depth(x)))


class SSFErbTemporalFrontend(nn.Module):
    def __init__(
        self,
        channels: int,
        use_erb: bool = True,
        erb_subband_1: int = 65,
        erb_subband_2: int = 64,
        nfft: int = 512,
        high_lim: int = 8000,
        fs: int = 16000,
        tconv_layers: int = 3,
        tconv_kernel: int = 3,
    ):
        super().__init__()
        self.use_erb = use_erb
        self.erb = None
        if use_erb:
            self.erb = ERB(erb_subband_1, erb_subband_2, nfft=nfft, high_lim=high_lim, fs=fs)

        self.tconv = nn.Identity()
        if tconv_layers and tconv_layers > 0:
            blocks = [TemporalConvBlock(channels, kernel_size=tconv_kernel) for _ in range(tconv_layers)]
            self.tconv = nn.Sequential(*blocks)

    def forward(self, x):
        # x: [B, C, F, T]
        if self.use_erb:
            x = x.permute(0, 1, 3, 2)  # [B, C, T, F]
            x = self.erb.bm(x)
            x = self.erb.bs(x)
            x = x.permute(0, 1, 3, 2)  # [B, C, F, T]

        x = x.permute(0, 1, 3, 2)  # [B, C, T, F]
        x = self.tconv(x)
        x = x.permute(0, 1, 3, 2)  # [B, C, F, T]
        return x
