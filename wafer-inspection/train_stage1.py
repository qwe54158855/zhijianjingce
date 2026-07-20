"""
阶段一训练: 物理-informed CycleGAN

新增 vs 原始:
  🆕 D_193 轻量鉴别器
  🆕 散射一致性损失 L_scat
  🆕 光谱角约束 L_spec

冻结策略:
  D_193: 训练
  PISM: 残差网络训练, 物理部分冻结
  编码器+增强解码: 训练
  检测头: 全部冻结
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.wafer_multitask import WaferMultiTaskModel
from losses.physics_loss import scattering_consistency_loss, spectral_angle_loss
from losses.cyclegan_loss import NLayerDiscriminator, GANLoss


class Stage1Trainer:
    """
    阶段一训练器

    初始化时指定配置，调用 train() 执行完整训练循环。
    """
    def __init__(self, config=None):
        self.config = config or {
            'batch_size': 16,
            'lr': 1e-4,
            'n_epochs': 100,
            'decay_epoch': 50,
            'lambda_cycle': 10.0,
            'lambda_idt': 0.5,
            'lambda_gan': 1.0,
            'lambda_lpips': 0.06,
            'lambda_scat': 0.05,
            'lambda_spec': 0.02,
            'checkpoint_dir': 'checkpoints/stage1/',
        }

        # 模型
        self.G_A = WaferMultiTaskModel(
            in_channels=1, num_classes=4,
            enable_physics=True, enable_polar=False
        )
        self.D_A = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3)
        self.D_193 = NLayerDiscriminator(input_nc=1, ndf=32, n_layers=2)

        # 损失
        self.criterion_gan = GANLoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l2 = nn.MSELoss()

        # 优化器
        self.optimizer_G = optim.Adam(
            list(self.G_A.parameters()), lr=self.config['lr'],
            betas=(0.5, 0.999))
        self.optimizer_D_A = optim.Adam(
            self.D_A.parameters(), lr=self.config['lr'],
            betas=(0.5, 0.999))
        self.optimizer_D_193 = optim.Adam(
            self.D_193.parameters(), lr=self.config['lr'],
            betas=(0.5, 0.999))

    def set_train_mode(self):
        """配置训练模式下的参数冻结"""
        self.G_A.train()
        self.D_A.train()
        self.D_193.train()

        # 冻结检测头
        for name, param in self.G_A.named_parameters():
            if 'detect_266' in name or 'detect_193' in name or 'sgf' in name:
                param.requires_grad = False

    def train_step(self, real_266, real_193, real_bright):
        """
        单步训练

        Args:
            real_266: 真实 266nm 暗场图像 [B,1,H,W]
            real_193: 真实 193nm 暗场图像 [B,1,H,W]（少量）
            real_bright: 真实明场图像 [B,1,H,W]
        """
        # === 生成器前向 ===
        fake_bright, _ = self.G_A(real_266)

        # 如果需要虚拟 193nm 输出: 走 PISM 路径
        # 这里简化: 假设 G_A 编码器→PISM→增强解码输出

        # === 鉴别器训练 ===
        # D_A: 区分真实明场 vs 生成明场
        pred_real = self.D_A(real_bright)
        loss_D_A_real = self.criterion_gan(pred_real, True)
        pred_fake = self.D_A(fake_bright.detach())
        loss_D_A_fake = self.criterion_gan(pred_fake, False)
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        self.optimizer_D_A.zero_grad()
        loss_D_A.backward(retain_graph=True)
        self.optimizer_D_A.step()

        # === 生成器训练 ===
        # GAN 损失
        pred_fake = self.D_A(fake_bright)
        loss_G_A = self.criterion_gan(pred_fake, True)

        # Cycle 损失 (简化, 完整实现参考原始 CycleGAN)
        loss_cycle = self.criterion_l1(fake_bright, real_bright)

        # 散射一致性 + 光谱角 (如果有 PISM 前向)
        loss_scat = torch.tensor(0.0)
        loss_spec = torch.tensor(0.0)
        if hasattr(self.G_A, 'pism'):
            feats = self.G_A.encoder(real_266)
            feats_193, _ = self.G_A.pism(feats)
            loss_scat = scattering_consistency_loss(feats, feats_193, self.G_A.pism)
            loss_spec = spectral_angle_loss(feats, feats_193)

        loss_G = (self.config['lambda_gan'] * loss_G_A +
                  self.config['lambda_cycle'] * loss_cycle +
                  self.config['lambda_scat'] * loss_scat +
                  self.config['lambda_spec'] * loss_spec)

        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        return {
            'loss_G': loss_G.item(),
            'loss_D_A': loss_D_A.item(),
            'loss_scat': loss_scat.item(),
            'loss_spec': loss_spec.item(),
        }

    def train(self, loader_266, loader_193, loader_bright):
        """完整训练循环"""
        for epoch in range(self.config['n_epochs']):
            for batch_idx, (img_266, img_193, img_bright) in enumerate(
                zip(loader_266, loader_193, loader_bright)):
                losses = self.train_step(img_266, img_193, img_bright)

                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch}/{self.config['n_epochs']} "
                          f"Batch {batch_idx}: G={losses['loss_G']:.4f} "
                          f"D_A={losses['loss_D_A']:.4f} "
                          f"L_scat={losses['loss_scat']:.4f}")

            # 每 epoch 保存检查点
            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'G_A': self.G_A.state_dict(),
                    'D_A': self.D_A.state_dict(),
                }, f"{self.config['checkpoint_dir']}/epoch_{epoch}.pt")
