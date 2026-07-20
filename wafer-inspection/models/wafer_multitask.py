"""
WaferMultiTaskModel — 主模型整合

整合所有模块:
  编码器 (RepViT) →
    [enable_physics=True]:
      ├──→ 266nm 检测头
      └──→ ASG(13角度) → 193nm检测头(batch13) → 角度融合 → SGF
    [enable_physics=False]:
      └──→ 检测头(原始单分支)

Key design:
  - ASG produces flattened 52 tensors (13 angles × 4 levels)
  - Grouping: angle i uses indices [i*4 : i*4+4] for [F1_i, F2_i, F3_i, F4_i]
  - 193nm head needs F2/F3/F4 only (skip F1) → feats_i[1:4]
  - Detection outputs from 3 layers merged by interpolating to F2 resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.physics_scattering import ScatteringPhysics
from models.detect_head_193 import WaferDetectHead193, C2f_MSD_193
from models.scattering_fusion import ScatteringGuidedFusion
from models.angle_scattering_gen import AngleScatteringGenerator
from models.angle_attention_fusion import AngleAttentionFusion


class RepViTEncoder(nn.Module):
    """
    RepViT 编码器封装（多尺度特征输出）

    简化版: 真实部署时替换为 reference/RepViT-main/model/repvit.py 的完整实现

    输出 4 层多尺度特征 [F1, F2, F3, F4] 及对应 strides [4, 8, 16, 32]
    """

    def __init__(self, in_channels=1):
        super().__init__()
        self.channels = [56, 112, 224, 448]
        self.strides = [4, 8, 16, 32]

        # Stem: total stride 4 (two stride-2 convs)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 28, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),
            nn.Conv2d(28, 56, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
        )

        # Multi-scale downsampling stages: each stride 2
        self.down2 = nn.Sequential(
            nn.Conv2d(56, 112, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(112),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(112, 224, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(224, 448, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(448),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Return 4-level features [F1, F2, F3, F4]"""
        f1 = self.stem(x)        # [B, 56, H/4, W/4]
        f2 = self.down2(f1)      # [B, 112, H/8, W/8]
        f3 = self.down3(f2)      # [B, 224, H/16, W/16]
        f4 = self.down4(f3)      # [B, 448, H/32, W/32]
        return [f1, f2, f3, f4]


class EnhanceDecoder(nn.Module):
    """增强解码分支 — 从 F2 特征重建增强图像"""

    def __init__(self, in_channels=112, hidden_channels=48):
        super().__init__()
        self.reduction = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(hidden_channels, 1, 3, padding=1, bias=True)
        # Upsample from F2 stride=8 back to input resolution
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)

    def forward(self, feats):
        """
        Args:
            feats: list of 4 feature tensors from encoder
        Returns:
            [B, 1, H, W] enhanced image at input resolution
        """
        x = self.reduction(feats[1])  # F2: [B, 112, H/8, W/8] → [B, 48, H/8, W/8]
        x = self.act(self.bn(x))
        x = self.final_conv(x)        # [B, 1, H/8, W/8]
        x = self.upsample(x)          # [B, 1, H, W]
        return torch.sigmoid(x)


class WaferMultiTaskModel(nn.Module):
    """
    晶圆多任务检测模型（物理增强版）

    Args:
        in_channels: 输入通道数 (1=灰度)
        num_classes: 缺陷类别数
        enable_physics: 是否启用物理增强模块
        enable_polar: 是否启用多角度生成（需 enable_physics=True）
    """

    def __init__(self,
                 in_channels=1,
                 num_classes=4,
                 enable_physics=True,
                 enable_polar=True):
        super().__init__()

        self.num_classes = num_classes
        self.enable_physics = enable_physics
        self.enable_polar = enable_polar and enable_physics

        # === 共享编码器 ===
        self.encoder = RepViTEncoder(in_channels=in_channels)

        # === 增强解码分支 ===
        self.enhance_decoder = EnhanceDecoder()

        if enable_physics:
            # === PISM 物理散射模块 ===
            self.pism = ScatteringPhysics(
                channels_per_stage=self.encoder.channels,
                lambda_in=266.0,
                lambda_out=193.0
            )

            ch = self.encoder.channels

            # === 266nm 检测分支（使用 F2/F3/F4） ===
            self.msd_266_f2 = C2f_MSD_193(ch[1], 56, n=1)
            self.msd_266_f3 = C2f_MSD_193(ch[2], 112, n=1)
            self.msd_266_f4 = C2f_MSD_193(ch[3], 224, n=1)
            self.cls_266_f2 = nn.Conv2d(56, num_classes, 1)
            self.cls_266_f3 = nn.Conv2d(112, num_classes, 1)
            self.cls_266_f4 = nn.Conv2d(224, num_classes, 1)
            self.reg_266_f2 = nn.Conv2d(56, 4, 1)
            self.reg_266_f3 = nn.Conv2d(112, 4, 1)
            self.reg_266_f4 = nn.Conv2d(224, 4, 1)

            # === 193nm 检测分支（通道裁剪） ===
            self.detect_193 = WaferDetectHead193(
                num_classes=num_classes,
                ch_in=(ch[1], ch[2], ch[3]),
                ch_hidden=(56, 64, 128)
            )

            # === SGF 融合层 ===
            self.sgf = ScatteringGuidedFusion(num_classes=num_classes)

            # === ASG 多角度生成器 ===
            if enable_polar:
                self.asg = AngleScatteringGenerator(
                    channels_per_stage=self.encoder.channels
                )
                # === 角度注意力融合 ===
                self.angle_fusion = AngleAttentionFusion(
                    num_classes=num_classes,
                    num_angles=len(self.asg.angles)
                )
        else:
            # === 原始单分支（无物理增强） ===
            ch = self.encoder.channels
            self.detect_msd_f2 = C2f_MSD_193(ch[1], 56, n=1)
            self.detect_msd_f3 = C2f_MSD_193(ch[2], 112, n=1)
            self.detect_msd_f4 = C2f_MSD_193(ch[3], 224, n=1)
            self.detect_cls_f2 = nn.Conv2d(56, num_classes, 1)
            self.detect_cls_f3 = nn.Conv2d(112, num_classes, 1)
            self.detect_cls_f4 = nn.Conv2d(224, num_classes, 1)
            self.detect_reg_f2 = nn.Conv2d(56, 4, 1)
            self.detect_reg_f3 = nn.Conv2d(112, 4, 1)
            self.detect_reg_f4 = nn.Conv2d(224, 4, 1)

    # ── 266nm detection helpers ──────────────────────────────────────

    def _detect_266_forward(self, feats):
        """266nm 检测头前向 — 返回 3 层 (cls, reg)"""
        f2, f3, f4 = feats[1], feats[2], feats[3]

        x2 = self.msd_266_f2(f2)
        x3 = self.msd_266_f3(f3)
        x4 = self.msd_266_f4(f4)

        return [
            (self.cls_266_f2(x2), self.reg_266_f2(x2)),
            (self.cls_266_f3(x3), self.reg_266_f3(x3)),
            (self.cls_266_f4(x4), self.reg_266_f4(x4)),
        ]

    def _detect_266_fused(self, feats_266):
        """
        266nm 分支: 将 3 层 cls/reg 上采样到 F2 分辨率后平均，
        返回 (cls, reg) 张量。
        """
        preds = self._detect_266_forward(feats_266)
        target_size = preds[0][0].shape[-2:]  # F2 resolution

        fused_cls = preds[0][0]  # F2 cls
        fused_reg = preds[0][1]  # F2 reg
        for i in range(1, len(preds)):
            cls_up = F.interpolate(
                preds[i][0], size=target_size,
                mode='bilinear', align_corners=False
            )
            reg_up = F.interpolate(
                preds[i][1], size=target_size,
                mode='bilinear', align_corners=False
            )
            fused_cls = fused_cls + cls_up
            fused_reg = fused_reg + reg_up
        return fused_cls / 3, fused_reg / 3

    # ── 193nm detection helpers ──────────────────────────────────────

    @staticmethod
    def _merge_detect_outputs(outputs_list):
        """
        合并 3 层检测头 (cls, reg) 输出为单个预测张量 [B, C, H, W].

        将 F3/F4 层上采样到 F2 分辨率后平均，cat(reg, cls).

        Args:
            outputs_list: List[Tuple(cls, reg)] from detect head, 3 elements
        Returns:
            [B, 4+num_classes, H_F2, W_F2] merged prediction
        """
        target_size = outputs_list[0][0].shape[-2:]  # F2 resolution
        cls_fused = outputs_list[0][0]
        reg_fused = outputs_list[0][1]
        for i in range(1, len(outputs_list)):
            cls_up = F.interpolate(
                outputs_list[i][0], size=target_size,
                mode='bilinear', align_corners=False
            )
            reg_up = F.interpolate(
                outputs_list[i][1], size=target_size,
                mode='bilinear', align_corners=False
            )
            cls_fused = cls_fused + cls_up
            reg_fused = reg_fused + reg_up
        return torch.cat([reg_fused / 3, cls_fused / 3], dim=1)

    # ── Non-physics detection ────────────────────────────────────────

    def _detect_disabled_forward(self, feats):
        """无物理增强时的检测前向 — 返回 (cls, reg)"""
        f2, f3, f4 = feats[1], feats[2], feats[3]

        x2 = self.detect_msd_f2(f2)
        x3 = self.detect_msd_f3(f3)
        x4 = self.detect_msd_f4(f4)

        cls_f2, reg_f2 = self.detect_cls_f2(x2), self.detect_reg_f2(x2)
        cls_f3, reg_f3 = self.detect_cls_f3(x3), self.detect_reg_f3(x3)
        cls_f4, reg_f4 = self.detect_cls_f4(x4), self.detect_reg_f4(x4)

        target_size = cls_f2.shape[-2:]
        cls = (
            cls_f2
            + F.interpolate(cls_f3, size=target_size,
                            mode='bilinear', align_corners=False)
            + F.interpolate(cls_f4, size=target_size,
                            mode='bilinear', align_corners=False)
        )
        reg = (
            reg_f2
            + F.interpolate(reg_f3, size=target_size,
                            mode='bilinear', align_corners=False)
            + F.interpolate(reg_f4, size=target_size,
                            mode='bilinear', align_corners=False)
        )
        return cls / 3, reg / 3

    # ── Main forward ─────────────────────────────────────────────────

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入图像

        Returns:
            enhanced: [B, 1, H, W] 增强图（与输入同分辨率）
            detections: [B, 4+num_classes, H', W'] 密集检测预测
        """
        # Step 1: 编码器前向
        feats = self.encoder(x)  # 4-level features

        # Step 2: 增强解码（独立于检测分支）
        enhanced = self.enhance_decoder(feats)

        # Step 3: 检测分支
        if not self.enable_physics:
            cls, reg = self._detect_disabled_forward(feats)
            detections = torch.cat([reg, torch.sigmoid(cls)], dim=1)
            return enhanced, detections

        # === 物理增强版 ===

        # Step 3a: PISM 前向 — 266nm → 虚拟 193nm 特征
        feats_193, pism_diag = self.pism(feats)

        # Step 3b: 266nm 检测
        pred_266_cls, pred_266_reg = self._detect_266_fused(feats)
        pred_266 = torch.cat([pred_266_reg, pred_266_cls], dim=1)
        # pred_266: [B, 4+num_classes, H_F2, W_F2]

        # 从 PISM 诊断中提取物理先验
        scale_map = pism_diag['scale_map']  # [B, 1, H_F1, W_F1]
        gain_map = pism_diag['gain_map']    # [B, 1, H_F1, W_F1]

        # Step 3c: 多角度生成与 193nm 检测
        if self.enable_polar:
            # ASG 生成 13 角度特征（扁平化 52 张量）
            all_angle_feats, stacked_conf = self.asg(feats, scale_map)

            # -------------------------------------------------------
            # C1: Batch-13 — 每层级跨13角度拼接，一次前向 (~13x 加速)
            # -------------------------------------------------------
            feats_by_level = [
                torch.cat([all_angle_feats[i * 4 + level_idx] for i in range(13)], dim=0)
                for level_idx in [1, 2, 3]  # F2, F3, F4 (skip F1)
            ]
            batch_outputs = self.detect_193(feats_by_level)  # 3 levels × (cls, reg)

            # 拆回 13 角度并融合层级（上采样 F3/F4 到 F2 分辨率）
            B_single = all_angle_feats[0].shape[0]
            angle_preds = [
                self._merge_detect_outputs([
                    (cls[i * B_single:(i + 1) * B_single],
                     reg[i * B_single:(i + 1) * B_single])
                    for cls, reg in batch_outputs
                ])
                for i in range(13)
            ]

            # 角度注意力融合: 将 conf 从 F1 分辨率插值到 F2 分辨率
            target_hw = angle_preds[0].shape[-2:]
            angle_conf = F.interpolate(
                stacked_conf, size=target_hw,
                mode='bilinear', align_corners=False
            )
            pred_193, angle_diag = self.angle_fusion(angle_preds, angle_conf)

            # -------------------------------------------------------
            # C2: 用融合权重将各角度偏移 gain/scale map 加权聚合
            #     替代全局使用 0° 版本的空间错位问题
            # -------------------------------------------------------
            fusion_weights = angle_diag['weights']  # [B, 13, 1, H, W]

            shifted_gains = []
            shifted_scales = []
            for i in range(13):
                sg, _ = self.asg.shift_gain_map(gain_map, i)
                _, ss = self.asg.shift_gain_map(None, i, scale_map)
                shifted_gains.append(sg)
                shifted_scales.append(ss)

            # 堆叠 [B, 13, 1, H_F1, W_F1] → 插值 → 注意力加权融合
            stacked_gain = torch.stack(shifted_gains, dim=1)
            stacked_scale = torch.stack(shifted_scales, dim=1)

            B = stacked_gain.shape[0]
            stacked_gain = F.interpolate(
                stacked_gain.flatten(0, 1),  # [B*13, 1, H_F1, W_F1]
                size=target_hw, mode='bilinear', align_corners=False
            ).unflatten(0, (B, 13))
            stacked_scale = F.interpolate(
                stacked_scale.flatten(0, 1),
                size=target_hw, mode='bilinear', align_corners=False
            ).unflatten(0, (B, 13))

            fused_gain = (stacked_gain * fusion_weights).sum(dim=1)  # [B, 1, H, W]
            fused_scale = (stacked_scale * fusion_weights).sum(dim=1)
        else:
            # 无 ASG: 直接用 PISM 输出的 193nm 特征（F2/F3/F4）
            theta_outputs = self.detect_193(feats_193[1:])  # skip F1
            pred_193 = self._merge_detect_outputs(theta_outputs)
            fused_gain, fused_scale = gain_map, scale_map

        # pred_193: [B, 4+num_classes, H_F2, W_F2]

        # Step 3d: SGF 融合 266nm 与 193nm 预测
        fused_pred, _ = self.sgf(pred_266, pred_193, fused_gain, fused_scale)

        return enhanced, fused_pred
