"""
Tests for stage-1 training: CycleGAN discriminator, GAN loss, and Stage1Trainer
"""

import torch
import pytest
from train_stage1 import NLayerDiscriminator, GANLoss, Stage1Trainer


def test_discriminator_init():
    """鉴别器应能正常初始化并前向"""
    D = NLayerDiscriminator(input_nc=1, ndf=32, n_layers=2)
    x = torch.randn(1, 1, 64, 64)
    y = D(x)
    assert y.shape[-1] > 0, "Discriminator should output valid shape"
    assert y.shape[0] == 1, "Batch dimension preserved"


def test_discriminator_lightweight():
    """D_193 应比 D_A 轻量"""
    D_A = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3)
    D_193 = NLayerDiscriminator(input_nc=1, ndf=32, n_layers=2)
    params_A = sum(p.numel() for p in D_A.parameters())
    params_193 = sum(p.numel() for p in D_193.parameters())
    assert params_193 < params_A, f"D_193 ({params_193}) should be lighter than D_A ({params_A})"


def test_gan_loss():
    """GAN 损失应正常计算真假损失"""
    loss_fn = GANLoss()
    pred = torch.randn(4, 1, 8, 8)
    loss_real = loss_fn(pred, True)
    loss_fake = loss_fn(pred, False)
    assert loss_real > 0
    assert loss_fake > 0
    assert torch.isfinite(loss_real)
    assert torch.isfinite(loss_fake)


def test_gan_loss_target_direction():
    """GAN 损失应对假图给高损失，真图给低损失"""
    loss_fn = GANLoss()
    # Real prediction close to target 1.0 → low loss
    pred_real = torch.ones(1, 1, 4, 4) * 0.99
    loss_real = loss_fn(pred_real, True)
    # Fake prediction close to 1.0 (should be 0.0) → high loss
    pred_fake = torch.ones(1, 1, 4, 4) * 0.99
    loss_fake = loss_fn(pred_fake, False)
    assert loss_real < loss_fake, f"Real loss ({loss_real}) should be < Fake loss ({loss_fake})"


def test_discriminator_multi_channel():
    """鉴别器应支持多通道输入"""
    D = NLayerDiscriminator(input_nc=3, ndf=32, n_layers=2)
    x = torch.randn(2, 3, 64, 64)
    y = D(x)
    assert y.shape[0] == 2


def test_discriminator_varying_sizes():
    """鉴别器应处理不同输入尺寸"""
    D = NLayerDiscriminator(input_nc=1, ndf=32, n_layers=2)
    for size in [32, 64, 128]:
        x = torch.randn(1, 1, size, size)
        y = D(x)
        assert y.shape[-1] > 0, f"Failed at size {size}"


def test_trainer_init():
    """训练器应初始化所有组件"""
    trainer = Stage1Trainer({
        'batch_size': 1, 'n_epochs': 1, 'lr': 1e-4,
        'checkpoint_dir': '/tmp/test_ckpt',
    })
    assert hasattr(trainer, 'G_A')
    assert hasattr(trainer, 'D_A')
    assert hasattr(trainer, 'D_193')
    assert hasattr(trainer, 'optimizer_G')
    assert hasattr(trainer, 'optimizer_D_A')
    assert hasattr(trainer, 'optimizer_D_193')
    # G_A should have PISM enabled (enable_physics=True)
    assert hasattr(trainer.G_A, 'pism'), "G_A should have PISM module"


def test_trainer_set_train_mode():
    """set_train_mode 应正确冻结检测头"""
    trainer = Stage1Trainer({
        'batch_size': 1, 'n_epochs': 1, 'lr': 1e-4,
        'checkpoint_dir': '/tmp/test_ckpt',
    })
    trainer.set_train_mode()

    frozen = []
    trainable = []
    for name, param in trainer.G_A.named_parameters():
        if not param.requires_grad:
            frozen.append(name)
        else:
            trainable.append(name)

    # 检测头应被冻结
    detect_frozen = any('detect_266' in n or 'detect_193' in n or 'sgf' in n for n in frozen)
    assert detect_frozen, "Detection heads should be frozen"

    # 编码器应可训练
    encoder_trainable = any('encoder' in n for n in trainable)
    assert encoder_trainable, "Encoder should remain trainable"


@pytest.mark.slow
def test_trainer_train_step():
    """单步训练应正常执行并返回有限损失"""
    trainer = Stage1Trainer({
        'batch_size': 1, 'n_epochs': 1, 'lr': 1e-4, 'decay_epoch': 50,
        'checkpoint_dir': '/tmp/test_ckpt',
        'lambda_gan': 1.0,
        'lambda_cycle': 1.0,
        'lambda_scat': 0.05,
        'lambda_spec': 0.02,
    })
    trainer.set_train_mode()

    B = 1
    real_266 = torch.randn(B, 1, 64, 64)
    real_193 = torch.randn(B, 1, 64, 64)
    real_bright = torch.randn(B, 1, 64, 64)

    losses = trainer.train_step(real_266, real_193, real_bright)

    for key, val in losses.items():
        assert torch.isfinite(torch.tensor(val)), f"Loss {key}={val} is not finite"
        assert val > 0 or key in ['loss_scat', 'loss_spec'], \
            f"Loss {key}={val} should be > 0"


def test_discriminator_grad_flow():
    """鉴别器应具有梯度流"""
    D = NLayerDiscriminator(input_nc=1, ndf=32, n_layers=2)
    x = torch.randn(1, 1, 64, 64)
    y = D(x)
    loss = y.mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in D.parameters())
    assert has_grad, "No gradient in discriminator"


def test_discriminator_patch_output():
    """鉴别器应输出 patch 而非标量（PatchGAN 特性）"""
    D = NLayerDiscriminator(input_nc=1, ndf=32, n_layers=2)
    x = torch.randn(1, 1, 64, 64)
    y = D(x)
    # PatchGAN: 输出应为 3D 张量 (B, 1, H, W) 而非 1D
    assert y.dim() == 4, f"Expected 4D output (B,1,H,W), got shape {y.shape}"
    # 空间维度应 > 1 (patch 而非 single value)
    assert y.shape[-1] > 1 and y.shape[-2] > 1, \
        f"Expected patch output, got shape {y.shape}"
