"""
CycleGAN 损失函数

包含 PatchGAN 鉴别器和 LSGAN 损失
"""

import torch
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN 鉴别器（轻量版）

    参考: reference/pytorch-CycleGAN-and-pix2pix-master/models/networks.py

    Args:
        input_nc: 输入通道数
        ndf: 基础特征通道数
        n_layers: 卷积层数
    """
    def __init__(self, input_nc=1, ndf=32, n_layers=2):
        super().__init__()
        kw = 4
        padw = 1

        sequence = [
            nn.Conv2d(input_nc, ndf, kw, 2, padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, padw),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 1, padw),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kw, 1, padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    """LSGAN 损失"""
    def __init__(self, target_real=1.0, target_fake=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real))
        self.register_buffer('fake_label', torch.tensor(target_fake))
        self.loss = nn.MSELoss()

    def __call__(self, prediction, target_is_real):
        target = self.real_label if target_is_real else self.fake_label
        return self.loss(prediction, target.expand_as(prediction))
