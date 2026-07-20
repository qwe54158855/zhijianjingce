"""
193nm 检测分支（通道裁剪版）

基于 FALCO-WAFER 的 C2f_MSD 设计，但通道数裁剪至 57-100%。
F2 层保留全通道（小目标关键），F3/F4 层裁剪至 57%。
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """标准 Conv + BN + SiLU 块"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck_MSD(nn.Module):
    """
    多尺度深度可分离 Bottleneck
    FALCO-WAFER DynamicIncMixerBlock 的简化实现
    """
    def __init__(self, ch, ch_hidden=None):
        super().__init__()
        ch_hidden = ch_hidden or ch // 2

        # 1×1 降维
        self.cv1 = ConvBlock(ch, ch_hidden, k=1, s=1, p=0)
        # 多尺度 3×3 深度可分离
        self.dwconv = nn.Conv2d(ch_hidden, ch_hidden, 3, 1, 1,
                                groups=ch_hidden, bias=False)
        self.bn_dw = nn.BatchNorm2d(ch_hidden)
        # 1×1 升维
        self.cv2 = ConvBlock(ch_hidden, ch, k=1, s=1, p=0)

    def forward(self, x):
        identity = x
        x = self.cv1(x)
        x = self.bn_dw(self.dwconv(x))
        x = self.cv2(x)
        return x + identity


class C2f_MSD_193(nn.Module):
    """
    C2f 结构的 MSD 变体（通道裁剪版）

    输入 ch_in → [split] → ch_hidden (通过 cv1)
    → n 个 Bottleneck_MSD → [concat] → ch_out (通过 cv2)
    """
    def __init__(self, ch_in, ch_out, n=1):
        super().__init__()
        ch_hidden = ch_out // 2
        self.cv1 = ConvBlock(ch_in, ch_hidden * 2, k=1, s=1, p=0)
        self.cv2 = ConvBlock((n + 2) * ch_hidden, ch_out, k=1, s=1, p=0)
        self.m = nn.ModuleList(
            [Bottleneck_MSD(ch_hidden) for _ in range(n)]
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class WaferDetectHead193(nn.Module):
    """
    193nm 检测分支（通道裁剪）

    与 266nm 检测分支结构相同，但 F3/F4 层通道裁剪至 57%。
    F2 层保留完整通道以充分利用 193nm 对小缺陷的 Rayleigh 3.5× 增益。

    Args:
        num_classes: 缺陷类别数
        ch_in: 输入通道（PISM 输出通道=编码器输出通道）
        ch_hidden: C2f_MSD 输出通道（裁剪后）
    """

    def __init__(self,
                 num_classes=4,
                 ch_in=(112, 224, 448),
                 ch_hidden=(56, 64, 128)):
        super().__init__()

        # MSD 特征映射
        self.msd_f2 = C2f_MSD_193(ch_in[0], ch_hidden[0], n=1)
        self.msd_f3 = C2f_MSD_193(ch_in[1], ch_hidden[1], n=1)
        self.msd_f4 = C2f_MSD_193(ch_in[2], ch_hidden[2], n=1)

        # 分类头
        self.cls_f2 = nn.Conv2d(ch_hidden[0], num_classes, 1)
        self.cls_f3 = nn.Conv2d(ch_hidden[1], num_classes, 1)
        self.cls_f4 = nn.Conv2d(ch_hidden[2], num_classes, 1)

        # 回归头: 直接输出 xywh（比 DFL 精简 16 倍）
        self.reg_f2 = nn.Conv2d(ch_hidden[0], 4, 1)
        self.reg_f3 = nn.Conv2d(ch_hidden[1], 4, 1)
        self.reg_f4 = nn.Conv2d(ch_hidden[2], 4, 1)

    def forward(self, feats):
        """
        feats: [F2', F3', F4'] — PISM 输出的虚拟 193nm 特征
        确保 len(feats) == 3
        """
        f2, f3, f4 = feats

        # MSD 特征映射
        x2 = self.msd_f2(f2)
        x3 = self.msd_f3(f3)
        x4 = self.msd_f4(f4)

        # 分类 + 回归预测
        return [
            (self.cls_f2(x2), self.reg_f2(x2)),
            (self.cls_f3(x3), self.reg_f3(x3)),
            (self.cls_f4(x4), self.reg_f4(x4)),
        ]
