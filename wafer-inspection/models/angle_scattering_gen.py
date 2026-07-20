"""
ASG: Angle Scattering Generator 多方位角度散射生成器

在极坐标特征空间中用最低成本生成 15°-75°（每 5°）的方位角特征。

核心物理洞察:
  极坐标中方位角旋转 = 特征图水平平移 (torch.roll 零成本)
  各向异性缺陷的散射强度随角度变化 (learnable modulation)
  遮挡/阴影等效应 (lightweight completion)

设计哲学:
  - 几何平移: 0 参数 (torch.roll)
  - 散射调制: ~50K 参数 (角度条件 + 尺度条件)
  - 生成补全: ~80K 参数 (F1 尺度 only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleScatteringGenerator(nn.Module):
    """
    多方位角度散射生成器

    Args:
        channels_per_stage: 编码器各层通道数
        target_angles: 目标方位角列表 (度)
        polar_width: 极坐标宽度 (角度分辨率)
    """

    def __init__(self,
                 channels_per_stage=(56, 112, 224, 448),
                 target_angles=None,
                 polar_width=512):
        super().__init__()

        if target_angles is None:
            target_angles = list(range(15, 76, 5))  # 15,20,...,75 → 13 个

        self.angles = target_angles
        self.num_angles = len(self.angles)
        self.polar_width = polar_width

        # 预计算每个角度的像素偏移量
        # 极坐标宽度 = polar_width = 360°，每度 = polar_width/360 像素
        pixels_per_degree = polar_width / 360.0
        shifts = [int(round(a * pixels_per_degree)) for a in self.angles]
        self.register_buffer('shifts', torch.tensor(shifts, dtype=torch.long))

        # ===== 子模块 2: 角度依赖散射调制 =====
        # 8 维角度嵌入
        self.angle_embed = nn.Embedding(self.num_angles, 8)

        # 调制网络: 角度嵌入(8) + 散射尺度(1) → 调制增益(1)
        self.scatter_modulator = nn.Sequential(
            nn.Conv2d(9, 16, 1, bias=False),   # 8+1=9
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),
            nn.Tanh()  # 输出 [-1, 1]
        )

        # ===== 子模块 3: 轻量生成补全 =====
        # 仅作用于 F1（最高分辨率，遮挡效果最明显）
        complete_ch = channels_per_stage[0]
        self.completion_net = nn.Sequential(
            nn.Conv2d(complete_ch, complete_ch // 4, 1, bias=False),
            nn.BatchNorm2d(complete_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(complete_ch // 4, complete_ch, 1, bias=False),
        )

        # ===== 角度置信度预测器 =====
        self.confidence_predictor = nn.Sequential(
            nn.Conv2d(channels_per_stage[0], 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1, bias=True),
            nn.Sigmoid()  # [0, 1] 置信度
        )

    def _geometric_shift(self, feature_map, angle_idx):
        """
        子模块 1: 几何重投影（零参数）

        在极坐标中，方位角旋转 = 沿宽度方向循环平移
        使用 torch.roll 实现 360° 周期性边界

        Args:
            feature_map: [B, C, H, W] 极坐标特征
            angle_idx: 目标角度索引

        Returns:
            shifted: [B, C, H, W] 平移后的特征
        """
        shift = int(self.shifts[angle_idx].item())  # 像素偏移量
        if shift == 0:
            return feature_map
        # 沿宽度方向（角度轴）循环平移
        return torch.roll(feature_map, shifts=shift, dims=-1)

    def shift_gain_map(self, gain_map, angle_idx, scale_map=None):
        """
        将增益/尺度图移位，以匹配旋转后的特征（C2 修复）

        ASG 通过 torch.roll 对特征进行循环平移，而 gain_map/scale_map
        来自 PISM 的 0° 坐标。此方法将增益/尺度图按相同偏移量滚动，
        确保与旋转后特征的空间对齐。

        Args:
            gain_map: [B, 1, H, W] 增益图，或 None
            angle_idx: 角度索引
            scale_map: [B, 1, H, W] 尺度图，或 None

        Returns:
            shifted_gain: [B, 1, H, W] 移位后的增益图（或 None）
            shifted_scale: [B, 1, H, W] 移位后的尺度图（或 None）
        """
        shift = int(self.shifts[angle_idx].item())
        shifted_gain = None
        shifted_scale = None
        if gain_map is not None:
            shifted_gain = torch.roll(gain_map, shifts=shift, dims=-1) if shift != 0 else gain_map
        if scale_map is not None:
            shifted_scale = torch.roll(scale_map, shifts=shift, dims=-1) if shift != 0 else scale_map
        return shifted_gain, shifted_scale

    def forward_one_angle(self, feats_266, angle_idx, scale_map_f1):
        """
        生成单个目标角度的 266nm 特征

        Args:
            feats_266: 多尺度特征列表
            angle_idx: 目标角度在 self.angles 中的索引
            scale_map_f1: F1 尺度图 [B, 1, H_F1, W_F1]，PISM 输出
                          （散射类型是逐像素属性，F1 即足够）

        Returns:
            feat_theta: 目标角度的 266nm 特征列表
            conf_map: 该角度的置信度图 [B, 1, H, W]
        """
        # C2 fix: 将 scale_map 按角度偏移，以匹配旋转后的特征坐标
        _, shifted_scale = self.shift_gain_map(None, angle_idx, scale_map_f1)

        angle_feats = []

        for i, feat in enumerate(feats_266):

            # Step 1: 几何重投影（零参数）
            feat_shifted = self._geometric_shift(feat, angle_idx)

            # Step 2: 角度依赖散射调制
            # 角度编码
            angle_idx_tensor = torch.tensor([angle_idx], device=feat.device)
            angle_emb = self.angle_embed(angle_idx_tensor)  # [1, 8]
            angle_emb = angle_emb.view(1, 8, 1, 1)

            # 对齐 F1 尺度图到当前层尺寸（使用已偏移的尺度图）
            scale_map_resized = F.interpolate(
                shifted_scale, size=feat.shape[-2:],
                mode='bilinear', align_corners=False
            )

            # 调制输入: 角度编码 + 散射尺度
            mod_input = torch.cat([
                angle_emb.expand(feat.shape[0], -1, feat.shape[2], feat.shape[3]),
                scale_map_resized
            ], dim=1)  # [B, 9, H, W]

            mod_gain = self.scatter_modulator(mod_input)  # [-1, 1]

            # 应用调制: 输出 = 输入 * (1 + 0.3 * 调制量)
            # 调制范围 [0.7, 1.3]
            feat_modulated = feat_shifted * (1.0 + 0.3 * mod_gain)

            # Step 3: 轻量生成补全（仅 F1）
            if i == 0:
                residual = self.completion_net(feat_modulated)
                feat_completed = feat_modulated + 0.1 * residual
            else:
                feat_completed = feat_modulated

            angle_feats.append(feat_completed)

        # Step 4: 置信度预测（基于 F1）
        conf_map = self.confidence_predictor(angle_feats[0])

        return angle_feats, conf_map

    def forward(self, feats_266, scale_map_f1):
        """
        批量生成所有目标角度的特征

        Args:
            feats_266: 编码器输出的多尺度特征列表
            scale_map_f1: PISM 输出的 F1 尺度图 [B, 1, H_F1, W_F1]
                       （由 PISM 的 scale_estimators 前向得到）

        Returns:
            all_angle_feats: 扁平化列表，每角度每层级一个张量，
                             共 num_angles * len(feats_266) 个
            stacked_conf: [B, 13, H, W] 每个角度的置信度图堆叠
        """
        all_angle_feats = []
        conf_maps = []

        for angle_idx in range(self.num_angles):
            feat_theta, conf = self.forward_one_angle(
                feats_266, angle_idx, scale_map_f1)
            # 扁平化：JIT trace 不支持嵌套 List
            all_angle_feats.extend(feat_theta)
            conf_maps.append(conf)

        # 堆叠置信度图并从 [B, 13, 1, H, W] 压缩为 [B, 13, H, W]
        stacked_conf = torch.stack(conf_maps, dim=1).squeeze(2)

        return all_angle_feats, stacked_conf
