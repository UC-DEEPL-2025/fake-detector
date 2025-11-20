from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, kernel: int, padding: int, norm: str, activation: str, neg_slope: float):
        super().__init__()
        acts = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(negative_slope=neg_slope, inplace=True),
        }
        norm_layer = (lambda c: nn.BatchNorm2d(c)) if norm == "bn" else (lambda c: nn.Identity())
        act = acts[activation]
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            norm_layer(out_ch),
            act,
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            norm_layer(out_ch),
            act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, up_kernel: int, up_stride: int, conv_kernel: int, conv_padding: int, norm: str, activation: str, neg_slope: float):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=up_kernel, stride=up_stride)
        self.conv = ConvBlock(in_ch, out_ch, kernel=conv_kernel, padding=conv_padding, norm=norm, activation=activation, neg_slope=neg_slope)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad if needed due to odd sizes
        if x.shape[-2] != skip.shape[-2] or x.shape[-1] != skip.shape[-1]:
            diff_y = skip.shape[-2] - x.shape[-2]
            diff_x = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# important: depth, base_channels, out_channels
class UNetClassifier(nn.Module):
    def __init__(self, *, depth: int, base_channels: int, out_channels: int, conv_kernel: int, conv_padding: int, norm: str, activation: str, activation_negative_slope: float, up_kernel: int, up_stride: int, pool_kernel: int, pool_stride: int, classifier: str):
        super().__init__()
        assert depth >= 2, "depth >= 2 required"
        chs: List[int] = [base_channels * (2 ** i) for i in range(depth)]

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = 3
        for c in chs:
            self.enc_blocks.append(ConvBlock(in_ch, c, kernel=conv_kernel, padding=conv_padding, norm=norm, activation=activation, neg_slope=activation_negative_slope))
            self.pools.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))
            in_ch = c

        self.bottleneck = ConvBlock(chs[-1], chs[-1] * 2, kernel=conv_kernel, padding=conv_padding, norm=norm, activation=activation, neg_slope=activation_negative_slope)

        dec_chs = list(reversed(chs))
        self.up_blocks = nn.ModuleList()
        in_ch = chs[-1] * 2
        for c in dec_chs:
            self.up_blocks.append(UpBlock(in_ch, c, up_kernel=up_kernel, up_stride=up_stride, conv_kernel=conv_kernel, conv_padding=conv_padding, norm=norm, activation=activation, neg_slope=activation_negative_slope))
            in_ch = c

        self.out_conv = nn.Conv2d(dec_chs[-1], out_channels, kernel_size=1)
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        h = x
        for enc, pool in zip(self.enc_blocks, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h)
        for up in self.up_blocks:
            skip = skips.pop()
            h = up(h, skip)
        seg_logits = self.out_conv(h)
        if self.classifier == "gap":
            logits = F.adaptive_avg_pool2d(seg_logits, (1, 1)).squeeze(-1).squeeze(-1)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier}")
        return logits
