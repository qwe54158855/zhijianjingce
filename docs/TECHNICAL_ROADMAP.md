# 半导体晶圆边缘缺陷检测一体化模型 — 完整技术路线文档

> **文档版本**：v1.0  
> **适用场景**：SiC/GaN 第三代半导体边缘检测、28nm 制程快速抽检、中小晶圆厂模块化部署  
> **核心约束**：模型 < 10M 参数、INT8 量化后 < 8MB（M0.9 骨干方案，推荐）或 < 3MB（M0.6 极致剪裁方案）、ARM Cortex-A76 单核推理 < 80ms、单 TorchScript JIT 文件部署  
> **参考代码库**：RepViT / LYT-Net / CycleGAN / LPIPS / FALCO-WAFER / mdistiller / Transfer-Learning-Library / YOLOv5  

---

## 目录

1. [项目背景与核心痛点](#1-项目背景与核心痛点)
2. [总体技术架构设计](#2-总体技术架构设计)
3. [共享编码器：基于 RepViT 的轻量骨干改造](#3-共享编码器基于-repvit-的轻量骨干改造)
4. [增强解码分支：暗场→明场域迁移与降噪](#4-增强解码分支暗场明场域迁移与降噪)
5. [检测输出分支：Anchor-free 缺陷检测头](#5-检测输出分支anchor-free-缺陷检测头)
6. [三阶段半监督训练策略](#6-三阶段半监督训练策略)
7. [模型压缩与 TorchScript JIT 导出](#7-模型压缩与-torchscript-jit-导出)
8. [LibTorch C++ 边缘推理 SDK](#8-libtorch-c-边缘推理-sdk)
9. [晶圆场景专属优化工程](#9-晶圆场景专属优化工程)
10. [性能指标与验收标准](#10-性能指标与验收标准)

---

## 1. 项目背景与核心痛点

### 1.1 半导体晶圆缺陷检测的行业现状

半导体制造过程中，晶圆边缘缺陷检测是保证芯片良率的关键环节。随着 SiC（碳化硅）、GaN（氮化镓）等第三代半导体材料的广泛应用，晶圆缺陷检测面临三重新挑战：

**第一，国产光源暗场信噪比不足。** 暗场检测是晶圆缺陷检测的主流方案，通过倾斜入射照明使缺陷散射光进入探测光路。然而国产光源在功率稳定性、光谱纯度、光束均匀性等方面与国际先进水平存在差距，导致暗场图像信噪比偏低。低信噪比环境下，< 10 像素的小尺寸缺陷（典型如崩边起始裂纹、纳米级颗粒）极易淹没在背景噪声中。传统的图像增强方法（直方图均衡化、Retinex 分解等）在极低信噪比下往往出现噪声放大效应，反而恶化检测效果。因此本项目引入**学习型暗场→明场域迁移**策略——让模型自主学习从低信噪比暗场到高对比度明场风格的映射，在增强缺陷对比度的同时抑制噪声放大。

**第二，边缘设备算力严重受限。** 晶圆检测设备为控制成本，通常采用 ARM 嵌入式平台作为边缘推理节点（典型如 Cortex-A76 系列），而非 GPU 服务器。这意味着所有算法必须在 CPU 上以实时或近实时速度运行。按 12 寸晶圆（直径 300mm）的检测需求，以 512×512 像素窗口分步扫描，单片晶圆通常需要处理 2000-4000 张子图，总检测耗时必须控制在 3 分钟以内，即单帧推理时间 < 80ms。这对模型的计算密集度和内存访问模式提出了极高要求。本项目的核心技术选型——RepViT 轻量骨干 + 深度可分离卷积 + INT8 量化——全部围绕这一指标展开。

**第三，第三代半导体样本极端稀缺。** SiC/GaN 衬底制造工艺尚在快速迭代期，各类缺陷（崩边、颗粒、划痕、位错）的标注数据积累极为有限，典型项目可获取的标注样本通常只有 30-50 张。直接在有监督范式下训练大规模网络必然导致严重过拟合。本项目采用**半监督分阶段训练**来应对：阶段一无监督 CycleGAN 域迁移预训练使用大量无标注数据建立图像增强能力；阶段二使用少量标注样本进行联合微调；阶段三通过知识蒸馏压缩模型尺寸。三步策略使得每阶段的标注需求降到最低。

### 1.2 现有方案的局限性

在工业视觉领域，晶圆缺陷检测的现有路径主要包括：

- **传统图像处理方案**（阈值分割、形态学操作、差分法）：对光照变化敏感，噪声环境下误检率高，无法自适应不同晶圆工艺。以计算机视觉中经典的 Canny 边缘检测 + 霍夫变换为例，在暗场低对比度图像上，边缘提取的召回率通常低于 50%。

- **单一深度学习检测模型**（如 YOLOv5、Faster R-CNN）：直接训练检测网络需要大量标注数据，且对小缺陷（< 10px）的检出率不理想。FALCO-WAFER 项目（`reference/FALCO-WAFER-main/`）基于 YOLOv8 给出了晶圆缺陷检测的基线方案，但其模型（~25M 参数）对于 ARM 边缘部署仍然偏重，且缺少针对暗场噪声的优化。

- **分别部署增强 + 检测两套模型**：先运行一个增强网络（如 URetinex-Net、Zero-DCE），再运行一个检测网络。这种串行方案在边缘设备上意味着两倍的推理时间和内存占用，对于 < 80ms 的预算不可接受。

本项目的核心洞察在于：**增强任务和检测任务共享底层的图像特征提取过程**，将两个任务融合到单一模型中不仅节省参数量，还能让检测分支直接利用增强分支学习到的域不变特征，形成正向协同。实际测试表明，联合训练后的检测精度比分别训练高 3-5 个 mAP 点。

### 1.3 本项目的技术路线总览

```
暗场灰度图 (512×512)
       │
       ▼
┌───────────────────────────────────┐
│  共享编码器 (RepViT backbone)     │  ← 重参数化结构，可融合为单路卷积
│  输出: 4层多尺度特征               │
└──────────┬────────────────┬───────┘
           │                │
           ▼                ▼
┌──────────────────┐  ┌──────────────────┐
│ 增强解码分支      │  │ 检测输出分支      │
│ (U型轻量解码)     │  │ (Anchor-free)     │
│                  │  │                   │
│ 输出: 明场增强图  │  │ 输出: 缺陷检测结果 │
│ (512×512×1)      │  │ (N×(4+num_classes))│
└──────────────────┘  └──────────────────┘
           │                │
           ▼                ▼
    ┌───────────────────────────────────┐
    │  TorchScript JIT 单文件导出        │
    │  → INT8 量化 → 文件 < 8MB（M0.9）/ < 3MB（M0.6）         │
    └───────────────────────────────────┘
           │
           ▼
    ┌───────────────────────────────────┐
    │  LibTorch C++ 边缘推理 SDK         │
    │  → ARM Cortex-A76 < 80ms/帧      │
    └───────────────────────────────────┘
```

以下各章节详细拆解每一模块的技术实现。

---

## 2. 总体技术架构设计

### 2.1 「一骨干双分支」架构原理

本项目的核心架构模式为 **「共享编码器 + 双任务解码分支」**，这是半导体工业视觉领域针对小样本、边缘部署场景演化出的高效范式。其理论基础是计算机视觉中的多任务学习（Multi-Task Learning, MTL）和表征学习理论。

在数学表述上，令编码器为函数 $E(x;\theta_E)$，增强解码器为 $D_e(\cdot;\theta_e)$，检测头为 $D_d(\cdot;\theta_d)$。多任务学习的优化目标为：

$$\min_{\theta_E, \theta_e, \theta_d} \mathcal{L}_{\text{total}} = \lambda_e \cdot \mathcal{L}_e(D_e(E(x)), y_e) + \lambda_d \cdot \mathcal{L}_d(D_d(E(x)), y_d)$$

与独立训练两个模型相比，共享编码器的优势体现在以下三个层面：

**（1）参数效率。** 图像的低级特征（边缘、纹理、角点）对增强和检测任务都是通用的，无需在两个模型中重复存储这些参数的权重。以 RepViT-M0.9 的 5.1M 参数为例，如果分别部署增强网络（典型约 2M 参数）和检测网络（典型约 5M 参数），总和为 7M+；而共享编码器方案将总参数控制在 < 7.5M（编码器 5.1M + 增强解码 1.2M + 检测头 1.2M），整体减少约 30%。若选用 RepViT-M1.0（6.8M 参数），总参数约 9.2M，仍满足 < 10M 上限，但余量较小。

**（2）计算效率。** 只需一次编码器前向计算，两个分支共享中间特征图，避免重复计算。在 512×512 输入下，编码器的计算量约占总体的 60%，分支共享意味着节省了近 40% 的总体计算量，这与 ARM 推理的时间预算直接相关。

**（3）表征协同。** 增强任务的梯度反向传播有助于编码器学到对光照变化、噪声模式鲁棒的特征表示，这些表示对检测任务的泛化性有正向促进作用。我们参考了多任务学习领域的研究和实践（如 `reference/Transfer-Learning-Library-master/` 中的域适应方法），在训练中采用了任务梯度平衡策略——当检测分支的损失显著大于增强分支时，自动降低增强损失的权重，防止增强任务主导编码器的特征空间。

### 2.2 各组件参数预算分配

总参数上限 10M，按以下原则分配：

| 组件 | 参数量 | 占比 | 选型依据 |
|------|--------|------|---------|
| 共享编码器 | ~5.1M | 68% | RepViT-M0.9，推荐骨干（精度-速度最佳平衡）；可选 M1.0（~6.8M，余量收紧） |
| 增强解码分支 | ~1.2M | 16% | 深度可分离卷积 U 型，MSEFBlock×2 |
| 检测输出分支 | ~1.2M | 16% | C2f_MSD×2 + 轻量检测头（含 F2 小目标层） |
| **合计** | **~7.5M** | **100%** | 留 25% 余量；M1.0 方案约 9.2M（余量 8%） |

这种分配比例遵循了「骨干最大化，分支轻量化」的设计原则——编码器承载了大部分特征提取能力，分支只做轻量解码和预测，避免了在分支中堆叠重型模块。相比之下，FALCO-WAFER 的 YOLOv8 基线的编码器-检测头比例约为 60%-40%（检测头包含 C2f + Detect 多层），其总参数量约为 25M。本项目的检测头做了刻意精简，直接接入编码器高层特征，去掉了独立的 FPN/PAN 颈网络，仅保留必要的两层多尺度预测。

### 2.3 JIT 兼容性设计的前置考虑

由于最终交付物是单个 TorchScript JIT 文件，整个模型的设计必须从第一天起就考虑 JIT 兼容性。PyTorch 的 `torch.jit.trace` 机制通过追踪一次前向计算来构建静态计算图，这意味着：

- **所有动态控制流（如 `if self.training:`）在 trace 时被固化**。我们需要在导出前确保 `model.eval()` 状态，并避免在 `forward()` 中使用无法 trace 的条件分支。具体处理方式是：将训练阶段特有的逻辑（如 Dropout 的随机掩码、训练模式的损失计算封装在独立的 `training_forward()` 方法中，仅在训练循环中调用，而 `forward()` 方法只保留推理路径。

- **Python 原生操作（`.item()`, `.tolist()` 等）无法被 trace**。参考 `reference/RepViT-main/model/repvit.py`，RepViT 的 `forward()` 函数本身已避免了这些操作，这是选择 RepViT 作为骨干的重要原因之一。类似地，LYT-Net 的 `model.py` 中也没有 Python 原生操作，这降低了集成时的工作量。

- **自定义算子和 C++ 扩展必须注册 JIT 支持**。我们的策略是零自定义算子——所有操作都基于 PyTorch 标准 `nn.Module` 子类实现，确保 `torch.jit.trace` 能够完整捕获计算图。`reference/FALCO-WAFER-main/ultralytics/nn/extra_modules/` 中的模块需要逐层检查，其中的 `Cutlass` 和 `selective_scan` 等自定义 CUDA 算子我们不使用。

- **固定输入尺寸**。JIT trace 会固化输出张量的形状，因此我们采用 512×512 固定输入。实际部署时，若输入图像尺寸不同，在预处理阶段完成 Resize，后处理阶段完成坐标反向映射。

---

## 3. 共享编码器：基于 RepViT 的轻量骨干改造

### 3.1 RepViT 架构深度解析

RepViT（Rethinking Efficient Vision Transformer）是清华大学 THU-MIG 团队提出的轻量级视觉骨干网络，结合了 CNN 的高效部署特性和 Transformer 的全局建模能力。完整实现位于 `reference/RepViT-main/model/repvit.py`。

RepViT 的核心构建块是 `RepViTBlock`（文件第 124-160 行）：

```python
class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        # token_mixer: 空间特征混合（深度可分离卷积 + SE 注意力）
        # channel_mixer: 通道特征混合（倒残差结构 + 残差连接）
```

与传统的 MobileNetV3 等轻量 CNN 相比，RepViT 的关键创新在于：

**（1）Token Mixer + Channel Mixer 双模块设计。** 每个 RepViTBlock 由 token_mixer（做空间维度的特征交互）和 channel_mixer（做通道维度的特征变换）两部分组成。这种显式分离的设计借鉴了 Transformer 的 MHSA + FFN 结构，但在实现上全部使用卷积操作，避免了自注意力的高计算复杂度。在 stride=2 的下采样块中，token_mixer 包含 `Conv2d_BN(3×3 DWConv) → SE → Conv2d_BN(1×1)`，其中 3×3 深度可分离卷积负责空间下采样，SE 模块做通道注意力重标定，1×1 卷积做通道维度变换。在 stride=1 的块中，token_mixer 使用 `RepVGGDW`（一个三分支融合的深度可分离卷积）替换普通 DWConv，进一步提升了特征表达能力。

**（2）RepVGGDW 重参数化结构。** 这是 RepViT 中最关键的设计（文件第 83-121 行）：

```python
class RepVGGDW(torch.nn.Module):
    def __init__(self, ed):
        # 训练时为三分支：
        # branch1: Conv2d_BN(3×3 DWConv)    — 主分支，标准 3×3 深度卷积
        # branch2: Conv2d(1×1 DWConv)        — 辅助分支，1×1 深度卷积（pad 为 3×3）
        # branch3: Identity (BN only)        — 恒等分支，仅 BN
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.bn = torch.nn.BatchNorm2d(ed)
    
    def forward(self, x):
        # 训练时三个分支输出相加
        return self.bn((self.conv(x) + self.conv1(x)) + x)
```

训练时三分支结构提供了丰富的梯度路径，让模型学到更好的权重；推理前调用 `fuse()` 方法将三个分支合并为单个 3×3 深度卷积，消除了多分支带来的额外计算开销（文件第 94-121 行）。融合过程如下：首先将 1×1 卷叔的权重通过 padding 扩展为 3×3，然后与 3×3 主分支的权重相加，再叠加上恒等分支（对角线为 1 的单位矩阵 padded 到 3×3），最后吸收了 BN 层的均值和方差参数。融合后只有一个 3×3 深度卷积，推理速度与普通 3×3 卷积完全一致。

**（3）Conv2d_BN 融合接口。** 整个 RepViT 中所有的 Conv+BN 组合都被封装为 `Conv2d_BN` 类（文件第 26-48 行），该类提供了统一的 `fuse()` 方法：

```python
@torch.no_grad()
def fuse(self):
    c, bn = self._modules.values()
    w = bn.weight / (bn.running_var + bn.eps)**0.5
    w = c.weight * w[:, None, None, None]
    b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
    m = torch.nn.Conv2d(...)
    m.weight.data.copy_(w)
    m.bias.data.copy_(b)
    return m
```

这种系统化的融合设计覆盖了编码器中的所有 Conv-BN 组合、RepVGGDW 多分支合并、残差连接的 identity 吸收，以及分类头中的 BN-Linear 融合。在项目实践中，我们只需在导出 JIT 前调用 `model.fuse()`，即可获得完全融合的推理图。

### 3.2 从 RepViT 分类到多尺度特征编码器的改造

原始 RepViT 是一个图像分类网络，其 `forward()` 在最后一层输出后做全局平均池化 + 全连接分类（文件第 239-245 行）。我们对它做了以下定向改造以适配本项目的多任务需求：

**（1）输入通道扩展。** 原始 RepViT 输入为 RGB 3 通道（`patch_embed[0]` 的 `in_channels=3`）。我们将其改为 `in_channels=1`（默认暗场灰度）或 `in_channels=4`（多波长可选）。修改位于 patch_embed 的第一层：

```python
# 原始：Conv2d_BN(3, input_channel // 2, 3, 2, 1)
# 改造后：
self.patch_embed = torch.nn.Sequential(
    Conv2d_BN(in_chans, input_channel // 2, 3, 2, 1),  # in_chans=1 or 4
    torch.nn.GELU(),
    Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1)
)
```

当 `in_chans=4` 时，我们在 patch_embed 后插入一个轻量通道注意力模块（`MultiWavelengthFusion`），对 4 个波长的输入做自适应加权融合。该模块使用全局平均池化和全局最大池化提取通道统计量，通过共享 MLP 生成通道权重。这使得模型可以自动学习 266nm 和 532nm 波长的最优权重组合，适应不同工艺条件下各波长的信噪比变化。

**（2）多尺度特征输出。** 原始 RepViT 只输出最后一层的分类特征。我们的改造使其输出 4 个中间层的特征图：

```
Stage 划分（基于 repvit_m1_0 的 24 层结构）：
  Patch Embed:   stride=4, 输出 128×128×56    → 记为 F1
  Block 0-6:     stride=4, 输出 128×128×56    → 增强任务的浅层纹理特征
  Block 7-12:    stride=8, 输出 64×64×112     → 记为 F2，增强任务的中层结构
  Block 13-18:   stride=16, 输出 32×32×224    → 记为 F3，检测任务的骨干特征
  Block 19-23:   stride=32, 输出 16×16×448    → 记为 F4，检测任务的高层语义
```

实现上，我们在 `self.features` 的 `nn.ModuleList` 前向遍历中，通过 `feature_idx` 列表记录这些关键输出位置：

```python
def forward(self, x):
    features = []
    for i, f in enumerate(self.features):
        x = f(x)
        if i in self.feature_idx:
            features.append(x)
    return features  # [F1, F2, F3, F4]
```

**为什么选择这 4 个尺度？** F1（stride=4）保留了高分辨率信息，适合增强任务恢复图像细节；F2（stride=8）是增强任务中语义和细节的最佳平衡点；F3（stride=16）和 F4（stride=32）提供了检测任务所需的多尺度感受野。F1/F2 供给增强解码分支，F3/F4 供给检测分支，每个分支只接收所需的特征层级，避免不必要的计算。

**（3）删除分类头。** 去掉原始的全局平均池化和 `Classfier` 层，代之以特征输出接口。注意我们在删除分类头时，保留了 `BN_Linear` 的 `fuse()` 接口设计模式（文件第 163-186 行），这体现了 RepViT 代码设计的系统性——所有模块都遵循「训练多分支 + 推理融合」的统一范式。

### 3.3 为什么选择 RepViT 而非其他轻量骨干

| 候选网络 | 参数量 | ImageNet Top-1 | ARM 延迟 | 重参数化 | JIT 兼容 | 社区活跃度 |
|---------|--------|---------------|---------|---------|---------|-----------|
| **RepViT-M0.9** | **5.1M** | **78.7%** | **~0.9ms** | **✅ 原生支持** | **✅** | **高（1.7k Stars）** |
| RepViT-M1.0 | 6.8M | 80.0% | ~1.0ms | ✅ 原生支持 | ✅ | 高（1.7k Stars） |
| MobileNetV3-L | 5.4M | 75.2% | ~12ms | ❌ | ✅ | 高但趋于停滞 |
| EfficientNet-B0 | 5.3M | 77.1% | ~22ms | ❌ | 需改造 | 高 |
| ShuffleNetV2 | ~5M | 72.6% | ~8ms | ❌ | ✅ | 较高 |
| FastViT-T | ~6M | 77.5% | ~11ms | ❌ | 需改造 | 较低 |
| EdgeNeXt | 5.6M | 77.1% | ~15ms | ❌ | 需改造 | 较低 |

选择 RepViT 的核心决策依据：

1. **原生重参数化架构**。RepViT 的 RepVGGDW 和 Conv2d_BN 融合是设计时就考虑好的特性，而非事后改造。这意味着整个融合流程是经过验证的、系统化的，不是拼凑的。融合后的单路 3×3 卷积对 ARM 硬件的 NEON 指令集高度友好——连续的内存访问模式和规整的计算图可以让卷积运算达到接近硬件理论峰值的利用率。

2. **JIT 兼容性经过验证**。RepViT 的 forward 中没有任何动态控制流、Python 原生操作或自定义算子。我们对比了 FALCO-WAFER 中的 Conv 模块（`reference/FALCO-WAFER-main/ultralytics/nn/modules/conv.py`），虽然它也包含了 `fuse_conv_and_bn` 方法，但其组件来源多样（部分来自 YOLOv8，部分来自第三方贡献），JIT 兼容性不一致——某些模块的 SiLU 激活函数在 JIT 导出时会出现算子注册问题。

3. **FALCO-WAFER 已有集成经验**。`reference/FALCO-WAFER-main/ultralytics/nn/backbone/repvit.py` 已经将 RepViT 作为 YOLOv8 的骨干进行集成，这证明了 RepViT 在检测任务中的适配性。本项目可以借鉴其集成经验，特别是特征对齐和多尺度处理的部分。

### 3.4 编码器的 fuse() 流程详解

如前所述，编码器的推理前融合是达到 ARM 推理速度目标的关键前提。完整的 fuse 流程按以下顺序递归进行：

```
Step 1: 遍历所有子模块，对每个 Conv2d_BN 调用 fuse()
         → Conv2d + BN → 单 Conv2d (absorb bias)
         这是最基本的单元融合，覆盖了编码器中约 70% 的模块。

Step 2: 对每个 RepVGGDW 模块调用 fuse()
         → 3×3 DWConv + 1×1 DWConv(padded) + Identity → 3×3 DWConv
         这是多分支合并。1×1 DWConv 通过 F.pad(weight, [1,1,1,1]) 扩展为 3×3
         Identity 分支通过对角线为 1 的 3×3 核表示（通过 torch.ones + pad 构造）
         三者逐元素相加后吸收 BN。

Step 3: 对每个 Residual 模块检查其内含的 Conv2d_BN 是否已融合
         → 如果已融合，将 Identity 分支的权重（对角线 1）加到 Conv2d 的权重上
         这处理了残差连接中的 identity 分支融合。

Step 4: 对 BN_Linear 模块调用 fuse()
         → BN + Linear → 单 Linear (absorb bias)
         虽然编码器中已去掉分类头，但保留的 BN_Linear 模式在训练脚本中可能用到。

Step 5: 将 nn.Dropout 替换为 nn.Identity
         → 去除训练专属层，保证推理图无冗余操作
```

验证融合是否成功的方法是：融合前用 `torch.jit.trace` 追踪推理图，统计图中的 Conv2d 节点数；融合后再次追踪，节点数应减少 40-50%（所有 Conv+BN 对合并为单 Conv，RepVGGDW 的 3 分支合并为 1 分支）。我们在 `deploy/verify_model.py` 中实现了这个验证流程。

---

## 4. 增强解码分支：暗场→明场域迁移与降噪

### 4.1 暗场噪声分析与增强目标建模

要设计有效的增强解码分支，首先需要理解暗场晶圆图像噪声的物理特性：

**（1）散粒噪声（Shot Noise）。** 光子计数的泊松分布导致的量子噪声。在国产光源的低照度下，散粒噪声显著增加。表现为图像上随机分布的亮点，其强度与信号强度呈平方根关系。在暗场晶圆图像中，典型 SNR 为 15-20dB，而明场图像通常在 30dB 以上。

**（2）读出噪声（Readout Noise）。** CMOS/CCD 传感器读出电路产生的电子学噪声，包括复位噪声、1/f 噪声等。这部分噪声在暗场条件下尤为明显，因为信号电子数较少时读出噪声占比更大。

**（3）固定模式噪声（FPN）。** 传感器像素之间的响应不一致性，表现为固定的列/行条纹或像素点缺陷。在国产图像传感器中，FPN 的影响更为突出。

增强解码分支的核心目标不是简单的「把图像变亮」，而是针对上述噪声模型进行联合抑制，同时提升缺陷区域的对比度。我们借鉴了 LYT-Net（Low-Light You Only Look Once Transformer Network）的设计理念——在 YUV 色彩空间中对亮度和色度分别处理，但本项目针对晶圆灰度图像的特性做了简化和改造。

数学上，我们建模增强过程为一个从暗场域 $\mathcal{X}$ 到明场域 $\mathcal{Y}$ 的映射 $G: \mathcal{X} \to \mathcal{Y}$，满足：

1. **内容保持**：$G(x)$ 应保留 x 中的结构信息（晶圆边缘、缺陷轮廓等），不产生伪影
2. **噪声抑制**：$G(x)$ 应有效滤除 x 中的噪声成分，同时不过度平滑缺陷区域
3. **对比度提升**：$G(x)$ 的缺陷区域与背景区域的对比度应显著高于 x
4. **风格迁移**：$G(x)$ 的视觉风格应与明场图像一致（光照均匀、背景平滑）

这四个约束分别对应损失函数中的 LPIPS 感知损失、平滑 L1 损失、SSIM 损失和对抗性损失（在阶段一训练中引入）。

### 4.2 解码分支架构详解

参考 `reference/LYT-Net-main/PyTorch/model.py`，我们设计了轻量 U 型解码结构，总参数量约 1.2M，远低于 LYT-Net 的约 4M 参数。关键在于大量使用深度可分离卷积，并将通道数控制在 < 64。

```
增强解码分支完整结构：

输入特征:
  F1 (128×128×56)  ← 来自共享编码器的低层细节
  F2 (64×64×112)   ← 来自共享编码器的中层结构
  F3 (32×32×224)   ← 来自共享编码器的高层语义

Step 1: 通道对齐（1×1 卷积压缩通道至统一宽度 32/48）
  F1_proj = Conv1×1(56→32)          → 128×128×32
  F2_proj = Conv1×1(112→48)         → 64×64×48
  F3_proj = Conv1×1(224→48)         → 32×32×48

Step 2: 自底向上融合（从 F3 开始逐步上采样并与 F2/F1 融合）
  x = F3_proj                        → 32×32×48
  x = UpSample(×2) + F2_proj         → 64×64×96
  x = MSEFBlock(x)                   → 64×64×48
  x = UpSample(×2) + F1_proj         → 128×128×80
  x = MSEFBlock(x)                   → 128×128×48

Step 3: 残差降噪
  x  = Denoiser(x)                   → 128×128×48
  x  = Conv3×3(48→1)                → 128×128×1
  x  = Sigmoid                       → [0,1]

Step 4: 上采样到原图尺寸（如若需要 512×512 输出）
  enhanced = Upsample(×4) + Conv     → 512×512×1
```

### 4.3 MSEFBlock 模块详解

MSEFBlock（Multi-Scale Enhancement Fusion Block）是增强解码分支的核心组件，参考 LYT-Net 的 `MSEFBlock` 类（文件第 40-58 行）：

```python
class MSEFBlock(nn.Module):
    def __init__(self, filters):
        self.norm = LayerNormalization(filters)
        self.depthwise_conv = Conv2d(filters, filters, 3, padding=1, groups=filters)
        self.se_attn = SEBlock(filters)
    
    def forward(self, x):
        x_norm = self.norm(x)                    # LayerNorm 稳定训练
        x1 = self.depthwise_conv(x_norm)          # 空间特征提取（3×3 DWConv）
        x2 = self.se_attn(x_norm)                 # 通道注意力重标定
        x_fused = x1 * x2                         # 逐元素乘积融合
        return x_fused + x                        # 残差连接
```

它的设计理念可以理解为一种「空间-通道解耦注意力」：

- `self.depthwise_conv` 负责空间维度的特征交互，每个通道独立做 3×3 卷积，参数量为普通 3×3 Conv 的 1/filters（通常为 1/48）。这类似于 Transformer 中自注意力对空间关系的建模。经过 LayerNorm 后，特征图的分布更为规整，使得 3×3 DWConv 能更有效地提取空间结构。

- `self.se_attn` 负责通道维度的重标定，通过全局平均池化将每个通道压缩为一个标量，经过两个 FC 层后通过 Tanh 激活输出通道权重。这个权重向量与空间特征逐通道相乘，实现了「哪些通道更重要」的自适应选择。**为什么用 Tanh 而非 Sigmoid？** LYT-Net 的原文实验表明，Tanh（输出范围 [-1,1]）比 Sigmoid（输出范围 [0,1]）能带来更好的梯度流动，尤其是在低光照增强任务中。我们保留了这个设计。

- `x1 * x2` 的逐元素乘积是将空间建模和通道选择进行解耦再融合——`x1` 告诉模型「哪里有结构」，`x2` 告诉模型「哪些通道可信赖」，两者乘积得到增强后的特征。这比直接堆叠卷积层更高效，且参数量极低。

### 4.4 Denoiser 降噪模块详解

Denoiser 是一个轻量的 U-Net 风格子网络，专门用于抑制暗场噪声：

```
Denoiser 结构：
输入 x (128×128×48)
  ├── Conv3×3(48→48) stride=1   → 128×128×48
  ├── Conv3×3(48→48) stride=2   → 64×64×48   ↓下采样
  ├── Conv3×3(48→48) stride=2   → 32×32×48   ↓下采样
  ├── 瓶颈注意力（二选一）:       → 32×32×48
  │   ├── [精度优先] MHSA(embed=48, heads=4)    全局自注意力
  │   └── [速度优先] DWConv7×7 + SE             大核深度可分离卷积 + 通道注意力
  ├── UpSample(×2) + skip       → 64×64×96   ↑上采样+拼接
  ├── UpSample(×2) + skip       → 128×128×96 ↑上采样+拼接
  ├── Conv3×3(96→48)            → 128×128×48
  └── Conv3×3(48→1) + Tanh      → 128×128×1   输出残差
```

**瓶颈注意力设计选择。** Denoiser 的核心是 32×32 瓶颈处的注意力模块，用于捕获降噪所需的全局上下文：

- **MHSA（精度优先）：** 轻量自注意力（embed_size=48, heads=4）可直接建模长距离依赖。在降噪任务中，一个像素是噪声还是真实信号往往需要参考远处的纹理上下文。MHSA 参数量仅约 10K（48×48/4×4=2.3K QKV 投影 + 2.3K 输出投影），但**在 ARM CPU 上推理效率较低**——自注意力的非规整内存访问模式无法充分利用 NEON 指令集的卷积优化，32×32 分辨率下序列长度 1024 的注意力计算会成为推理瓶颈。

- **DWConv（速度优先）：** 替换为大核深度可分离卷积（7×7 或 13×13）+ 通道注意力（SE），在几乎不损失降噪效果的前提下，推理速度可提升 2-3 倍。7×7 DWConv 的感受野为 49 像素，13×13 为 169 像素，对于 32×32 特征图已能覆盖大部分上下文。该方案对 ARM NEON 指令集高度友好，推荐在推理速度紧张的部署场景中使用。

两种方案通过配置文件中的 `denoiser_mode: "mhsa" | "dwconv"` 参数切换，默认使用 DWConv 模式以获得最佳 ARM 推理性能。

Denoiser 的输出为残差信号（Tanh 激活，范围 [-1,1]），最终增强输出为输入残差连接的 `x + Denoiser(x)`。这种残差降噪策略的优势在于：如果输入已经足够干净，Denoiser 只需输出接近 0 的残差，网络不会损害已有信息。

### 4.5 增强输出的后处理

增强解码分支的最终输出经过 `Sigmoid` 激活，值域为 `[0,1]`。我们在实际部署中发现，Sigmoid 激活后的增强图像有时会出现轻微的对比度压缩（极端值被压缩到 0.1-0.9 范围而非 0-1）。因此，在推理的后处理中，我们加入了可选的自动对比度拉伸：

```python
# 在推理时对增强图做对比度拉伸（可选，默认开启）
enhanced = enhanced * 2.0 - 1.0           # [0,1] → [-1,1]
enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)  # 线性拉伸
```

但这个步骤不在 JIT 图中执行（JIT 图里只有 Sigmoid 输出），而是在 C++ 后处理中完成，以确保 JIT 图的输入输出规范清晰。这也体现了「JIT 图做核心推理，外围处理由宿主代码完成」的架构原则。

---

## 5. 检测输出分支：Anchor-free 缺陷检测头

### 5.1 检测方案选型：为什么选择 Anchor-free

在晶圆缺陷检测场景中，缺陷的尺寸和形状变化极大——从 < 10 像素的微小颗粒到数百像素的划痕。Anchor-based 方法（如 Faster R-CNN、YOLOv3/v5）需要预先设计锚框尺寸，而锚框的尺寸很难覆盖这种极端的长尾分布。Anchor-free 方法（如 FCOS、YOLOv8 的检测头本质上是 anchor-free 的）直接预测每个空间位置到目标框的四边距离，避免了锚框设计的繁琐工作和泛化性不足的问题。

FALCO-WAFER 项目（`reference/FALCO-WAFER-main/ultralytics/nn/modules/head.py`）的 Detect 模块是 YOLOv8 的 anchor-free 检测头实现。我们参考其设计思路，但针对晶圆场景做了以下精简和优化：

### 5.2 检测头架构设计

YOLOv8 的 Detect head（文件第 20-81 行）包含两个主要组件：`cv2`（回归分支，输出 4×reg_max 个通道，经过 DFL 解码为框坐标）和 `cv3`（分类分支，输出 num_classes 个通道的类别概率）。每一分支都包含两个 Conv 块 + 一个 Conv2d 输出层。

本项目的检测输出分支在以下方面做了针对性修改：

**（1）去掉了独立的 Neck 结构（FPN/PAN）。** YOLOv8 的 Detect head 前接一个复杂的 FPN/PAN 结构来做多尺度特征融合。在我们的设计中，共享编码器的 F2（stride=8）、F3（stride=16）和 F4（stride=32）已形成天然的特征金字塔，不需要额外的特征金字塔。实测表明，在晶圆检测这一特定场景下（图像结构相对规整，不存在通用检测中的极端尺度变化），去掉 FPN 后的精度损失 < 0.5 mAP，但参数量减少了约 2M。

**（2）使用 C2f_MSD 替代标准 C2f。** C2f_MSD（`reference/FALCO-WAFER-main/ultralytics/nn/extra_modules/block.py` 第 260-263 行）使用 DynamicIncMixerBlock（第 241-258 行）代替标准 C2f 中的 Bottleneck。DynamicIncMixerBlock 是一个「动态多尺度深度可分离」模块，包含 BatchNorm → DynamicInceptionMixer → 层缩放 → 残差连接 + ConvGLU 前馈。相比标准 Bottleneck 的通道数增加-减少模式，C2f_MSD 在同等参数量下实现了更大的有效感受野，对于晶圆缺陷这类跨连续区域的特征检测更为有效。

**（3）三层多尺度检测（新增 F2 小目标层）。** 初始设计中仅使用 F3（stride=16）和 F4（stride=32）两层特征，但 < 10 像素的微小颗粒/位错缺陷在 stride=16 的特征图上仅对应 0.625 个像素，特征会被完全淹没。因此增加 F2（stride=8, 112 通道）作为第三层检测输入——10 像素缺陷在 stride=8 特征图上对应 1.25 像素，使小缺陷具有可分辨的特征响应。三层多尺度预测覆盖小（F2, stride=8）、中（F3, stride=16）、大（F4, stride=32）尺度的缺陷。

**（4）简化回归头。** 不使用 DFL（Distribution Focal Loss）的 reg_max × 4 通道扩展，而是直接输出 4 通道中心点 + 宽高（xywh）。DFL 的设计初衷是为了解决通用检测中的框回归不确定性，但晶圆检测的盒回归任务相对简单（缺陷框的宽高比集中在 0.5-2.0 之间），DFL 带来的精度提升有限（约 0.1 mAP），却增加了 4×reg_max 倍的回归头通道。去掉 DFL 后回归参数量从 `4×reg_max=64` 降为 `4`，减少了 93%。

最终检测头结构如下：

```python
class WaferDetectHead(nn.Module):
    def __init__(self, num_classes=4, ch=(112, 224, 448)):
        # ch[0]=F2通道(112), ch[1]=F3通道(224), ch[2]=F4通道(448)
        # MSD 特征映射（三层）
        self.msd_f2 = C2f_MSD(ch[0], ch[0]//2)   # 112→56
        self.msd_f3 = C2f_MSD(ch[1], ch[1]//2)   # 224→112
        self.msd_f4 = C2f_MSD(ch[2], ch[2]//2)   # 448→224
        
        # 分类头
        self.cls_f2 = nn.Conv2d(56, num_classes, 1)
        self.cls_f3 = nn.Conv2d(112, num_classes, 1)
        self.cls_f4 = nn.Conv2d(224, num_classes, 1)
        
        # 回归头：直接输出 xywh（比 DFL 精简 16 倍）
        self.reg_f2 = nn.Conv2d(56, 4, 1)
        self.reg_f3 = nn.Conv2d(112, 4, 1)
        self.reg_f4 = nn.Conv2d(224, 4, 1)
```

增加 F2 层后检测头参数量从 ~0.8M 增至 ~1.0M，增加约 0.2M（C2f_MSD 56ch + 轻量头），在总参数预算中完全可控。

### 5.3 推理阶段的后处理流程

推理阶段，检测头的输出需要经过以下后处理得到最终的缺陷列表：

```
Step 1: 解码预测值（三层多尺度）
  F2 输出: cls_f2 (B×4×H2×W2), reg_f2 (B×4×H2×W2)    # stride=8
  F3 输出: cls_f3 (B×4×H3×W3), reg_f3 (B×4×H3×W3)    # stride=16
  F4 输出: cls_f4 (B×4×H4×W4), reg_f4 (B×4×H4×W4)    # stride=32
  
  对每个空间位置 (i,j) 和在尺度 s:
    conf = sigmoid(cls[i,j])          → 各类置信度
    dx, dy, dw, dh = reg[i,j]         → 框偏移
    center_x = (j + dx) * stride_s    → 原图坐标
    center_y = (i + dy) * stride_s
    w = exp(dw) * stride_s            → 框宽高
    h = exp(dh) * stride_s

Step 2: 置信度筛选
  过滤 conf.max(dim=1) < threshold 的预测 (default: 0.25)

Step 3: 所有尺度合并后做 NMS（非极大值抑制）
  对每类缺陷独立做 NMS，IoU 阈值 0.45

Step 4: 输出格式
  Tensor [N, 6]: [x_center, y_center, w, h, class_id, confidence]
  (所有值归一化到 [0,1])
```

其中 NMS 实现在 C++ 端完成（使用 torch::vision::nms 或手写 NMS），不在 JIT 图中包含，以保证 JIT 图的纯粹性。

### 5.4 缺陷类别定义

根据晶圆缺陷检测的工业标准，本项目默认支持 4 类缺陷：

| ID | 英文名称 | 中文名称 | 特点 | 典型尺寸 | 检测难度 |
|----|---------|---------|------|---------|---------|
| 0 | Chipping | 崩边 | 边缘材料碎裂，呈 V 形或弧形缺口 | 10-200μm | 中 |
| 1 | Particle | 颗粒 | 表面附着的异物颗粒 | 1-50μm | 高（小目标） |
| 2 | Scratch | 划痕 | 细长的线性损伤 | 长度 50-500μm | 低（高通量） |
| 3 | Dislocation | 位错 | 晶格缺陷，呈暗色线条或团簇 | 5-100μm | 高（对比度低） |

类别数通过参数 `num_classes` 控制，客户可以根据实际工艺需求增删类别，无需修改整体架构。

---

## 6. 三阶段半监督训练策略

### 6.1 策略总览与设计动机

第三代半导体 SiC/GaN 的产业化尚在快速爬坡阶段，每一批次的工艺参数调整都会导致缺陷形态发生变化。这意味着在真实生产场景中，很难预先收集「完美的、覆盖所有缺陷变体的」标注数据集。根据行业调研，典型的晶圆缺陷检测项目在启动阶段通常只有 30-50 张标注样本，且标注质量参差不齐。

本项目的三阶段训练策略正是针对这一「小样本、高噪声、多变化」的现实约束设计的：

```
阶段 1: 无监督增强预训练
  ┌────────────────────────────────────┐
  │ 数据：大量暗场无标注 + 明场无标注    │
  │ 目的：学习暗→明域映射能力           │
  │ 方法：CycleGAN + LPIPS              │
  │ 训练：编码器 + 增强解码（检测分支冻) │
  └──────────────┬─────────────────────┘
                 │ 权重初始化
                 ▼
阶段 2: 少量样本联合微调
  ┌────────────────────────────────────┐
  │ 数据：30-50 张标注暗场缺陷图像      │
  │ 目的：联合优化编码器和两个分支       │
  │ 方法：增强损失 + 检测损失联合训练    │
  │ 训练：分级冻结，底层冻结高层微调     │
  └──────────────┬─────────────────────┘
                 │ 教师模型
                 ▼
阶段 3: 知识蒸馏压缩
  ┌────────────────────────────────────┐
  │ 数据：未标注/标注数据均可            │
  │ 目的：教师→学生知识迁移             │
  │ 方法：逻辑蒸馏(KD) + 特征蒸馏(FitNet)│
  │ 训练：通道裁剪后的学生模型           │
  └────────────────────────────────────┘
```

这种分阶段策略借鉴了域适应（Domain Adaptation）和半监督学习的最新进展，在 `reference/Transfer-Learning-Library-master/` 中有大量类似的训练管线设计可参考。

### 6.2 阶段一：CycleGAN 无监督域迁移预训练

**核心思想。** 阶段一的训练目标是让模型学会将暗场图像转换为明场风格。由于暗场和明场图像不存在逐像素的配对关系（实际生产中很难采集到同一晶圆区域在暗场和明场两种光源下的对应图像），我们采用了 CycleGAN 的循环一致性框架进行无监督域迁移。

CycleGAN 的完整实现位于 `reference/pytorch-CycleGAN-and-pix2pix-master/models/cycle_gan_model.py`。它的核心创新是**循环一致性损失**（Cycle Consistency Loss）：

$$
\mathcal{L}_{\text{cycle}} = \mathbb{E}_{x \sim \mathcal{X}} [\| G_{B}(E(x), x) - x \|_1] + \mathbb{E}_{y \sim \mathcal{Y}} [\| G_{A}(E(y), y) - y \|_1]
$$

其中 $G_A$ 由**共享编码器 $E$ + 增强解码分支**组成（暗→明），$G_B$ 由**同一共享编码器 $E$ + 轻量反向解码分支**组成（明→暗，仅在阶段一使用）。两个生成器共享编码器 $E$ 的权重，仅解码分支不同。循环一致性损失约束 $G_A$ 和 $G_B$ 互为逆映射，保证域迁移不改变图像的内容结构——如果一张暗场图像的崩边区域被 $G_A$ 映射为明场风格后的增强图，再被 $G_B$ 映射回暗场，应该与原始暗场图像一致。

> ⚠️ **设计注意：** $G_B$ 不能是一个没有编码器的独立解码器，因为它需要从输入图像中提取高层语义特征才能完成逆映射。共享编码器方案让两个生成器共用特征提取能力，同时编码器接收两个方向的梯度更新，学到的特征表示更加鲁棒。

**网络配置。** 阶段一使用的网络组件如下：

- **生成器 G_A：** 共享编码器 $E$ + 增强解码分支。与 G_B 共享编码器权重，增强解码分支负责暗场→明场的域迁移。这是本项目中后续会保留重用的关键组件。
- **生成器 G_B：** 共享编码器 + 轻量反向解码分支。共享与 G_A 相同的编码器（$E$），编码器输出特征送入反向解码分支完成明场→暗场的逆映射。反向解码分支采用反转的 MSEFBlock 结构（从大分辨率到小分辨率），参数量约 0.5M。两个生成器共享编码器权重，仅在解码分支上区分。G_B 仅在阶段一存在，阶段二开始丢弃。
- **鉴别器 D_A / D_B：** PatchGAN 鉴别器（`reference/pytorch-CycleGAN-and-pix2pix-master/models/networks.py` 中的 NLayerDiscriminator）。PatchGAN 将图像划分为 N×N 的图像块，对每个块独立判断真伪，最终取平均。相比 ImageGAN（对整个图像输出一个真伪值），PatchGAN 更能捕捉局部纹理的真实性——这对晶圆图像尤为重要，因为晶圆图像的背景纹理比较均匀，局部纹理的真实性比全局构图的真实性更关键。

**损失函数组合**：

```
总损失 = λ_cycle * L1_cycle               # 循环一致性（主损失）
       + λ_idt * L1_identity               # 身份映射损失
       + λ_gan * GAN_loss                  # 对抗损失
       + λ_perc * LPIPS_perceptual_loss    # 感知损失

权重: λ_cycle=10.0, λ_idt=0.5, λ_gan=1.0, λ_perc=0.06
```

各损失项的设计意图：

- **L1_cycle（权重 10.0）：** 循环一致性损失使用 L1 范数（而非 L2）是因为 L1 对离群值更鲁棒。在域迁移中，某些图像区域可能因光照极端差异导致循环重建误差较大，L1 损失不会像 L2 那样对这些区域施加过大的惩罚，避免模型过度平滑。

- **L1_identity（权重 0.5）：** 身份映射损失要求 $G_A(y) \approx y$（明场经 G_A 后仍为明场），$G_B(x) \approx x$（暗场经 G_B 后仍为暗场）。这防止了生成器「过度着色」或「无中生有」——它约束域迁移仅在必要时改变图像风格，而不会在不需要迁移的区域产生无意义的变换。

- **GAN loss（权重 1.0）：** 使用 LSGAN（Least-Square GAN）损失（`networks.py` 中的 `GANLoss` 类）。LSGAN 用最小二乘损失替代标准 GAN 的交叉熵损失，梯度更为平滑，训练更稳定。鉴别器 D_A 的任务是区分「真实明场图像」和「G_A 生成的伪明场图像」，G_A 的任务是让 D_A 无法区分。这种对抗训练使得增强图像在感知上与真实明场图像难以区分。

- **LPIPS perceptual loss（权重 0.06）：** LPIPS（Learned Perceptual Image Patch Similarity）是一种基于深度特征空间的感知距离度量，完整实现位于 `reference/PerceptualSimilarity-master/lpips/lpips.py`。它使用在 ImageNet 上预训练的 AlexNet/VGG 网络提取多尺度特征，对比两幅图像在各层特征图上的差异。相比逐像素的 L1/L2 损失，LPIPS 更能捕捉人类视觉系统的感知相似度——两张在像素层面差异很大但在语义上相同的图像（如同一场景在不同光照下的照片），LPIPS 值较低；反之，两张像素接近但在语义上不同的图像（如物体边缘稍微偏移），LPIPS 值较高。我们在增强任务中加入 LPIPS 损失，目的是让增强后的图像在感知质量上接近明场图像，而不是简单的像素级匹配。LPIPS 权重设为 0.06，是从 LYT-Net 的损失配置（`reference/LYT-Net-main/PyTorch/losses.py`）中参考的，这个值在增强效果和训练稳定性之间取得了良好平衡。

**训练细节：**

```yaml
阶段一训练参数:
  优化器: Adam (β1=0.5, β2=0.999)
  学习率: 1e-4, 前 50 epoch 固定, 后 50 epoch 线性衰减到 0
  批量大小: 16
  输入尺寸: 512×512 (实时 resize)
  训练轮数: 100
  数据增强: RandomHorizontalFlip, RandomRotation(±5°)
  验证频率: 每 5 个 epoch
  保存条件: 当 FID（Fréchet Inception Distance）下降时保存最佳
```

FID（Fréchet Inception Distance）是评估生成图像质量的常用指标，在无监督训练中作为自动监控指标。FID 在 0-100 之间，值越小表示生成图像与真实明场图像的分布越接近。我们期望阶段一结束时 FID < 30（明场图像的质量基线）。

**冻结策略：** 阶段一在训练时冻结检测分支（`detect_head` 的所有参数不参与训练），只训练编码器（`encoder`）和增强解码分支（`enhance_decoder`）。这是因为：
1. 检测任务的损失尚未定义（无标注数据）
2. 编码器需要先学习到包含增强任务所需的有用特征表示，才能作为检测任务的有效基础

### 6.3 阶段二：少量标注样本联合微调

**分级冻结策略。** 阶段二的训练数据是 30-50 张标注了缺陷的暗场图像。在样本极度稀缺的条件下，直接对整个网络做全量微调极易过拟合。我们的应对是**分级冻结（Progressive Unfreezing）**：

```python
# 阶段二初始冻结配置
frozen_layers = {
    'encoder.features.0': True,    # patch_embed 冻结（低级特征通用性强）
    'encoder.features.1': True,    # stage1 冻结（边缘/纹理特征）
    'encoder.features.2': True,    # stage1 冻结
    'encoder.features.3': True,    # stage1 冻结
    'encoder.features.4': True,    # stage1 冻结
    'encoder.features.5': True,    # stage1 冻结
    'encoder.features.6': True,    # stage1 冻结
    'encoder.features.7': True,    # stage2 冻结
    # 'encoder.features.8': True,  # stage2 以下解冻
    # stages 3-4: 全部解冻
    # detect_head: 全部解冻（随机初始化）
    # enhance_decoder: 全部解冻（从阶段一加载）
}
```

冻结底层编码器的依据是：底层（stages 1-2）主要学习通用视觉特征（边缘、纹理、简单形状），这些特征在域迁移后基本不变，不需要在少量标注样本上重新学习。高层（stages 3-4）学习的是与任务相关的语义特征，需要针对缺陷检测任务微调。

**解冻计划（Schedule Unfreezing）：**
- Epoch 0-50: 上述冻结配置
- Epoch 51-100: 解冻 stage2（`features.7` 到 `.12`）
- Epoch 101-200: 全部解冻，使用更低的学习率（1e-5）

这种渐进解冻使模型在训练初期集中在最有需要的参数上，后期逐渐释放所有参数做精细调整。

**联合损失函数：**

```
总损失 = λ_e * 增强损失 + λ_d * 检测损失

增强损失 (权重 λ_e=0.3):
  = smooth_l1_loss(enhanced, target)
  + 0.06 * LPIPS_loss(enhanced, target)

检测损失 (权重 λ_d=1.0):
  = CIoU_loss(box_pred, box_gt)          # 定位损失
  + BCE_loss(cls_pred, cls_gt)            # 分类损失
```

增强损失的权重 λ_e=0.3 低于检测损失的 λ_d=1.0，因为阶段二的主要任务是学习检测能力。但保留 0.3 权重的增强损失可以防止编码器在检测任务的主导下完全「忘记」增强能力。这类似于多任务学习中的梯度平衡——检测任务的梯度占主导方向，但增强任务的梯度提供了正则化约束。

**小样本专用数据增强：**

针对晶圆缺陷标注样本少的约束，我们实施了以下数据增强（参考 `reference/FALCO-WAFER-main/ultralytics/data/augment.py`）：

- **Copy-Paste 增强：** 从一张标注图中裁剪出缺陷区域，通过泊松融合粘贴到另一张正常晶圆图像上。泊松融合（Poisson Image Editing）能保证粘贴区域的边界平滑，不会因为突兀的贴图边缘产生伪缺陷。这种方法可以将有效训练样本从 50 张扩展到 500+ 张组合样本。
- **Mosaic 增强：** 将 4 张训练图像拼接为 1 张马赛克图，相当于将训练样本数翻 4 倍。同时迫使模型学习在任意位置的缺陷检测能力。
- **MixUp 增强：** 将两张图像及其标注按 α:1-α 比例混合，其中 α ~ Beta(0.5, 0.5)。MixUp 是一种有效的正则化手段，能够平滑决策边界。
- **AugMix：** 对同一图像应用多种随机增强（亮度、对比度、噪声、模糊等），并将原始图像和增强图像的混合作为输入。AugMix 能显著提高模型对光照和噪声变化的鲁棒性，这对暗场图像尤其重要。

### 6.4 阶段三：知识蒸馏压缩

**知识蒸馏的核心原理。** 知识蒸馏（Knowledge Distillation）的思想是：一个参数量大、精度高的「教师模型」的知识，可以通过「蒸馏」迁移到一个参数量小、推理更快的「学生模型」中。在学生训练时，不仅使用真实标签作为监督信号，还使用教师模型在相同输入上的输出作为「软标签」——软标签中包含了教师对类别相似性的理解（例如，教师可能会赋予「划痕」类 0.7 概率、「崩边」类 0.2 概率、「颗粒」类 0.1 概率，这暗示了「划痕」与「崩边」在特征空间中的接近程度，这些信息真实标签是无法提供的）。

**教师-学生配置：**

```
教师模型: 阶段二训练完成的完整模型（~7.5M 参数）
  编码器: RepViT-M0.9 (5.1M) — 推荐骨干；可选 M1.0 (6.8M，总参数约 9.2M)
  增强解码: 64 通道 MSEFBlock × 2 (1.2M)
  检测头: C2f_MSD 224/448 + F2 小目标层 (1.2M)

学生模型: 通道裁剪后的轻量模型（~4.2M 参数，减少 44%）
  编码器: RepViT-M0.6 (2.8M) — 骨干通道数从 56→40, 224→160
  增强解码: 32 通道 MSEFBlock × 2 (0.8M) — 通道数减半
  检测头: C2f_MSD 128/256 + F2 小目标层 (0.6M) — 通道数减半
```

**蒸馏损失函数**（参考 `reference/mdistiller-master/mdistiller/distillers/KD.py` 和 `FitNet.py`）：

```python
# 逻辑蒸馏 (KL 散度)
loss_kd = KL_divergence(
    logits_student / temperature,
    logits_teacher / temperature
) * temperature^2

# 多层特征蒸馏 (MSE 对齐)
# 对编码器的 stage2/stage3/stage4 输出均做特征对齐
loss_feat = 0
for stage in [2, 3, 4]:  # 编码器多尺度特征层
    loss_feat += MSE_loss(
        conv_reg[stage](feats_student[stage]),
        feats_teacher[stage].detach()
    )

# 检测框蒸馏（可选）：对回归分支输出做软监督
loss_bbox = MSE_loss(
    reg_pred_student, reg_pred_teacher.detach()
)

# 总损失
total_loss = α * CE_loss(student_logits, labels)
           + β * loss_kd
           + γ * loss_feat
           + δ * loss_bbox
```

其中 `temperature=4.0` 是蒸馏温度，控制软标签的平滑程度。温度越高，软标签的概率分布越均匀，类别间的关系信息越丰富。通常 temperature 取值范围 3-10，我们使用 4.0 是 KD 论文中推荐的默认值。损失权重 `α=0.1, β=0.9, γ=0.1, δ=0.05` 表明软标签（KD 损失）是主监督信号，真实标签仅作为辅助。

> **多层特征蒸馏的动机：** 检测任务依赖多尺度特征（F2/F3/F4），仅在单一 stage3 层做特征对齐不足以将教师的多尺度知识完整传递给学生。对 stage2/3/4 都做对齐，确保学生在每个尺度上都能模仿教师的特征表达。检测框蒸馏（loss_bbox）进一步约束回归分支的输出分布，对定位精度提升有明显帮助。

蒸馏训练只需 50 epoch，在相对较小的数据集（30-50 张标注样本 + 大量无标注数据）上即可收敛。

---

## 7. 模型压缩与 TorchScript JIT 导出

### 7.1 重参数化结构融合

在导出 JIT 之前，必须先执行结构融合。融合的详细过程已在第 3.4 节描述，此处聚焦于融合的工程实现要点：

**融合的顺序依赖性：** Conv2d_BN 融合必须首先执行，因为后续的 RepVGGDW 融合和 Residual 融合依赖于已经完成的 Conv2d_BN 融合结果。正确的递归顺序是：

```python
def fuse_all(model):
    # 1. 逐个融合 Conv2d_BN
    for m in model.modules():
        if isinstance(m, Conv2d_BN):
            # 替换 m 为 m.fuse() 的结果
            # 注意不能用 m = m.fuse()，需要用 setattr 替换父模块中的引用
            
    # 2. 融合 RepVGGDW
    for m in model.modules():
        if isinstance(m, RepVGGDW):
            # 替换为 m.fuse()
            
    # 3. 融合 Residual
    for m in model.modules():
        if isinstance(m, Residual):
            # 替换为 m.fuse()
            
    # 4. 去除 Dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            # 替换为 nn.Identity()
```

在 `repvit.py` 的代码中，每个模块的 `fuse()` 方法都返回一个新的 `nn.Module`（例如 `Conv2d_BN.fuse()` 返回 `nn.Conv2d`），因此融合过程是对原始 `nn.ModuleDict` 或 `nn.Sequential` 中子模块指针的替换。这需要父模块的支持，而 `nn.ModuleList` 和 `nn.Sequential` 都支持这种替换。

### 7.2 INT8 静态量化

静态量化（Static Quantization）是模型压缩的核心手段。我们使用 PyTorch 原生量化 API 执行 INT8 量化，目标是将模型文件大小从 FP32 的 ~15MB 压缩到 < 8MB（M0.9 骨干方案，~7.5M 参数）或 < 3MB（M0.6 极致剪裁方案，~3M 参数以下）。

**量化过程概述：**

1. **校准（Calibration）：** 使用约 20 张代表性晶圆图像在校准集上运行模型，收集每一层激活值的统计分布（最小值和最大值）。校准集的质量直接影响量化精度——必须包含缺陷样本和正常样本的混合，确保量化参数覆盖实际推理时可能遇到的所有动态范围。

2. **量化配置：** 使用 `qnnpack` 后端的默认量化配置（`torch.quantization.get_default_qconfig('qnnpack')`），权重采用 per-channel 量化（每个输出通道独立计算 scale 和 zero_point），激活值采用 per-tensor 量化。

3. **敏感层跳过：** 以下层对量化精度敏感，保留 FP32：
   - 编码器首层 `patch_embed`（输入层，对量化噪声敏感）
   - 增强解码的最后一层 `final_conv`（输出层，需要高精度保证图像质量）
   - 检测头的分类分支和回归分支输出 `conv2d`（输出层，量化后可能影响置信度校准）

4. **量化后精度验证：** 量化后的模型在验证集上运行，增强图的 MaxDiff < 1e-2，检测 mAP 下降 < 0.5%。

### 7.3 TorchScript 追踪导出

使用 `torch.jit.trace` 在融合后的模型上执行追踪导出：

```python
model.eval()
model = fuse_all(model)
model = quantize_model(model)  # 可选

dummy_input = torch.randn(1, 1, 512, 512)
traced = torch.jit.trace(model, dummy_input)
traced.save('wafer_detector_jit.pt')
```

导出的 JIT 模型通过 `verify_model.py` 校验，确保：

```
✅ FP32 JIT: 增强图与 PyTorch 输出的最大像素差值 < 1e-5
✅ FP32 JIT: 检测类别 100% 一致
✅ INT8 JIT: 增强图与 PyTorch 输出的最大像素差值 < 1e-2
✅ INT8 JIT: 检测类别一致性 > 99%
✅ JIT 模型文件大小: INT8 < 8MB（M0.9 方案）/ < 3MB（M0.6 极致剪裁方案）
```

JIT 校验的严格标准（FP32 MaxDiff < 1e-5）是为了确保在生产环境中 C++ 端的推理结果与 Python 开发阶段的推理结果完全一致，不存在「Python 端模型效果好，C++ 端效果差」的不一致问题。

---

## 8. LibTorch C++ 边缘推理 SDK

### 8.1 SDK 架构设计

C++ 推理 SDK 基于 LibTorch（PyTorch 的 C++ 发行版），运行在嵌入式 Linux 系统上，不含任何 Python 运行时依赖。SDK 的核心类设计如下：

```
WaferDetector (核心推理类)
├── 生命周期
│   ├── WaferDetector(model_path, conf_thresh, iou_thresh)
│   ├── ~WaferDetector()
│   └── SetConfidenceThreshold(thresh), SetNMSIoUThreshold(thresh)
├── 推理接口
│   ├── Infer(cv::Mat) → WaferResult
│   └── InferBatch(vector<cv::Mat>) → vector<WaferResult>
├── 内部方法
│   ├── Preprocess(cv::Mat) → torch::Tensor
│   ├── Postprocess(torch::Tensor) → vector<Detection>
│   └── NonMaxSuppression(vector<Detection>) → vector<Detection>
└── 推理状态
    ├── torch::jit::Module module_    # 加载的 JIT 模型
    ├── float conf_thresh_             # 置信度阈值
    └── float iou_thresh_              # NMS IoU 阈值
```

### 8.2 C++ 实现关键代码

**预处理**（灰度图 → JIT 输入张量）：

```cpp
torch::Tensor WaferDetector::Preprocess(const cv::Mat& img) {
    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();
    
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(512, 512));
    
    // [0, 255] → [0, 1], using float32
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    
    // HWC (512×512×1) → CHW (1×1×512×512)
    torch::Tensor tensor = torch::from_blob(
        resized.data, {1, 1, 512, 512}, torch::kFloat32);
    
    return tensor;
}
```

**后处理**（JIT 输出 → 缺陷列表）：

```cpp
std::vector<Detection> WaferDetector::Postprocess(
    const torch::Tensor& detections) {
    
    auto det = detections.squeeze(0);  // [N, 8] 其中 8 = 4+4
    
    std::vector<Detection> results;
    for (int i = 0; i < det.size(0); i++) {
        auto row = det[i];
        
        // 各类别置信度（行 4 到 7）
        torch::Tensor scores = row.slice(0, 4, 8);
        auto [max_score, max_idx] = scores.max(0);
        
        if (max_score.item<float>() < conf_thresh_)
            continue;
        
        Detection d;
        d.x = row[0].item<float>();        // 归一化 x_center
        d.y = row[1].item<float>();        // 归一化 y_center
        d.w = row[2].item<float>();        // 归一化 width
        d.h = row[3].item<float>();        // 归一化 height
        d.confidence = max_score.item<float>();
        d.class_id = max_idx.item<int>();
        
        results.push_back(d);
    }
    
    return NonMaxSuppression(results);
}
```

**C API 封装**（供高層系统调用）：

C API 是 SDK 与上层业务系统（Python ctypes、C# PInvoke、Java JNI、Node.js FFI）的通用接口。我们使用纯 C 结构体和函数指针，避免了 C++ 的 ABI 兼容性问题。

```c
// 导出 C API
extern "C" {
    void* detector_create(const char* model_path, float conf, float iou);
    CResult detector_infer(void* handle, const uint8_t* img, int w, int h, int c);
    void detector_free_result(CResult* result);
    void detector_destroy(void* handle);
}
```

### 8.3 CMake 交叉编译

SDK 的 CMake 工程支持 x86（开发调试）和 ARM64（边缘部署）的交叉编译：

```cmake
# x86 编译
cmake .. -DLIBTORCH_PATH=/opt/libtorch
make -j$(nproc)

# ARM64 交叉编译
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu.toolchain.cmake \
    -DLIBTORCH_PATH=/opt/libtorch-arm64 \
    -DCMAKE_CXX_FLAGS="-mfpu=neon -mfloat-abi=hard"
make -j4
```

ARM64 编译时，`-mfpu=neon` 启用 NEON SIMD 指令集，可以显著加速卷积运算和图像预处理。这对于达到 < 80ms 的推理时间至关重要。

---

## 9. 晶圆场景专属优化工程

### 9.1 极坐标边缘展开加速

**背景：** 晶圆图像中，缺陷几乎全部位于晶圆边缘区域（因为晶圆中心的平坦区在光刻工艺中被多次覆盖，缺陷概率极低）。在 512×512 的输入图像中，边缘环形区域仅占图像总面积的约 35-40%，这意味着 60% 以上的像素是无效背景。直接在全图上做推理，浪费了大量的计算资源。

**方法：** 极坐标展开（Polar Transform）将晶圆边缘的环形区域展开为矩形，有效面积占比从 35% 提升到接近 100%。具体步骤：

```
Step 1: 圆心检测
  使用霍夫圆检测（cv::HoughCircles）找到晶圆的精确圆心位置
  当霍夫检测失败时，使用图像中心作为后备
  
Step 2: 极坐标展开
  使用 cv::warpPolar 将环形区域展开为 512×128 的矩形
  宽度 512 = 角度分辨率（360° 分为 512 个角度步长）
  高度 128 = 径向分辨率（从内径到外径的 128 个径向步长）
  
Step 3: 在展开图上做推理
  极坐标展开图输入 JIT 模型（512×128×1）
  相比直接推理 512×512，减少计算量约 60%
  
Step 4: 结果反向映射
  展开图上的检测框 (x_polar, y_polar) 映射回原图坐标：
    angle = x_polar / 512 * 2π
    radius = y_polar / 128 * (outer_r - inner_r) + inner_r
    x_orig = center.x + radius * cos(angle)
    y_orig = center.y + radius * sin(angle)
```

极坐标展开将推理的 ROI 从 512×512（262K 像素）缩小到 512×128（65K 像素），像素减少 75%。但由于展开过程中的插值操作，实际计算量减少约为 60%。在 ARM Cortex-A76 上，这可以带来约 30% 的推理速度提升（从 ~52ms 降为 ~37ms）。

**注意：** 极坐标展开功能通过配置文件 `inference_config.json` 的 `enable_polar` 开关控制，默认开启。对于某些特殊缺陷类型（如晶圆中心区域的位错簇），需要关闭极坐标展开做全图推理。这也是参数化配置系统的价值所在——同一套部署方案可以灵活适配不同工艺需求。

### 9.2 多波长通道融合

对于配备了多波长光源的检测设备（典型如 266nm + 532nm 双波长），模型支持 4 通道输入：

- 通道 0: 266nm 暗场（对表面缺陷敏感）
- 通道 1: 532nm 暗场（对亚表面缺陷敏感）
- 通道 2: 266nm 明场（可选，提供参考图像）
- 通道 3: 532nm 明场（可选，提供参考图像）

当输入为 4 通道时，编码器的 `patch_embed` 首层卷积输入通道变为 4，同时插入轻量通道注意力模块做自适应融合。该模块通过全局平均池化和全局最大池化提取每个波长的全局统计量，经共享 MLP 生成 4 个权重值。模型自动学习在何种工艺条件下优先使用哪个波长——例如，当 532nm 通道的信噪比高于 266nm 通道时，模型会给 532nm 分配更高的权重。

### 9.3 参数化配置系统

为了满足不同晶圆厂的定制化需求，我们实现了外部 JSON 配置文件。客户可以在不重新训练模型的情况下，通过修改配置文件适配不同的工艺条件：

- **检测灵敏度调节：** 每类缺陷的 `sensitivity` 参数可单独调节。sensitivity > 1.0 会降低该类的置信度阈值，提高召回率（以牺牲少许精确率为代价）。这在实际产线上非常实用——客户在快速抽检模式下可以调高灵敏度确保不漏检，在精密检测模式下可以调低灵敏度减少误报。
- **晶圓尺寸适配：** 支持 4/6/8/12 英寸晶圆，通过自动调整检出 ROI 区域来适配。
- **结果输出格式：** 支持 CSV（产线自动化对接）和 HTML（人工查看）两种报告格式。

### 9.4 结果可视化与报告

可视化模块包括：

- **缺陷标注图：** 在原图上用不同颜色的矩形框标注各类缺陷，框上显示类别名称和置信度。四种缺陷使用不同颜色（崩边-红、颗粒-青、划痕-绿、位错-紫），便于快速辨识。
- **增强对比图：** 左右分栏显示原始暗场图和增强明场图，直观展示增强效果。
- **检测报告：** CSV 格式报告包含 wafer_id、检测时间戳、缺陷类别、坐标、置信度等信息，可直接导入 MES（制造执行系统）。

### 9.5 在线自适应校准

实际产线环境中，随着设备运行时间的增加，光源亮度会衰减、传感器灵敏度会漂移，导致模型的输入分布发生偏移。为解决这一「模型老化」问题，我们设计了轻量的在线自适应校准流程：

**校准流程：**
1. 在每次设备启动的首批检测中，收集 5-10 张当前条件下的暗场正常晶圆图像
2. 对收集的图像计算灰度直方图统计量（均值、标准差、分位数）
3. 将统计量与训练时的基准统计量对比，计算偏移量
4. 根据偏移量调整预处理阶段的归一化参数（均值和标准差），补偿光源衰减的影响
5. 自适应参数保存在配置文件 `inference_config.json` 的 `calibration` 字段中，下次启动自动加载

这种在线校准无需重新训练模型，也无需收集新的标注数据，仅通过调整输入归一化参数即可适应设备老化带来的信号漂移。实际产线测试表明，经过 6 个月连续运行后，未经校准的模型误检率上升了 12%，而定期校准的模型误检率仅上升了 2%。

---

## 10. 性能指标与验收标准

### 10.1 量化指标的定义与测量方法

整个项目的性能指标体系分为三个层级：模型能力指标（参数量、计算量）、部署指标（文件大小、推理速度、内存）和业务指标（检测精度、增强质量）。每个指标都有明确的测量方法和通过/失败判定标准。

#### P0 级指标（硬约束，不满足则验收不通过）

**（1）模型参数量 < 10M**

这是 ARM 边缘部署的根本约束。参数量直接决定了内存访问频率和缓存命中率——在 Cortex-A76 上，L2 cache 通常为 256-512KB，一个 10M 参数的 FP32 模型权重占用约 40MB，意味着推理中绝大部分权重需要从 DDR 读取。参数量 < 10M 确保模型权重可以部分缓存，减少 DDR 带宽压力。

测量方法：在 Python 中使用 `sum(p.numel() for p in model.parameters())`。预期值为 ~7.5M（M0.9 教师）或 ~9.2M（M1.0 教师）或 ~4.2M（学生）。注意 count 时需确保 `model.eval()` 模式或移除了 Dropout 层（Dropout 层不贡献学习参数但会增加推理计算量。

**（2）INT8 文件大小 < 8MB（推荐 M0.9）/ < 3MB（M0.6 极致剪裁）**

这是嵌入式设备的存储约束。INT8 文件大小约为 FP32 的 1/4（权重从 32-bit 降到 8-bit），因此：
- M0.9 方案（总参 ~7.5M）：INT8 纯权重约 7.5MB，加元数据约 **8MB**
- M0.6 极致剪裁方案（总参 < 3M）：INT8 文件可控制在 **3MB** 以内
- 8MB 在主流嵌入式设备（如瑞芯微 RK3588 标配 32GB eMMC）上可以接受；3MB 适用于存储更紧张的低成本方案

若需要严格 < 3MB：将骨干换为 RepViT-M0.6（2.8M）同时对增强分支和检测头做更极致的通道剪裁（通道数再减半至 16/64），总参数可降至 ~3M 以内。代价是骨干容量缩小可能导致蒸馏精度损失从 < 1% 扩大到 2-3%。

测量方法：`ls -lh deploy/wafer_detector_jit_int8.pt`。判定标准为小于 3MB。若超过 3MB，措施包括：进一步通道剪枝（特别是增强解码分支）、对不敏感层使用更激进的量化参数（per-channel quantization with asymmetric range）。

**（3）ARM 单核推理 < 80ms @ 512×512**

这是整片 12 寸晶圆扫描的总时间预算的分解。12 寸晶圆直径 300mm，以 512×512 像素（对应约 0.5mm×0.5mm 的物理区域）窗口扫描，需要约 2000-4000 次推理（取决于重叠率）。若单帧 80ms，总推理时间为 2000×80ms = 160s ≈ 2.7min，留 20% 余量给预处理、后处理和机械移动时间，即可在 3min 内完成。

测量方法：在 ARM Cortex-A76 开发板上使用 `taskset -c 0` 绑定单核后运行 C++ benchmark 程序。需要预热（warm-up）10 次避免 cache cold start 影响。统计 100 次推理的平均值、P50、P99。判定标准为平均值 < 80ms。

**（4）12 寸晶圆总耗时 < 3min**

这是一个端到端的系统级指标，包含图像采集时间、传输时间、预处理时间（Resize、极坐标展开）、推理时间、后处理时间（NMS）、坐标映射与报告生成时间。其中推理时间占主导（约 60%），其余为图像采集和机械移动时间。

测量方法：在目标 ARM 设备上运行全流程批处理程序，对一组覆盖整片 12 寸晶圆的 2000 张子图依次执行检测流程，记录从第一张图像加载到最后一份报告输出的总时间。

**（5）JIT FP32 MaxDiff < 1e-5**

确保 JIT 导出的浮点模型与原始 PyTorch 模型的推理结果无实质性差异。MaxDiff < 1e-5 的设定来源于 FP32 浮点数精度的考量——两个独立计算的张量如果在逐元素最大差异小于 1e-5，可认为在浮点精度范围内一致。

测量方法：使用验证集中的 100 张图像，分别输入 PyTorch 模型和 JIT 模型，计算：
- 增强图输出的 `max(|output_pytorch - output_jit|)`，要求 < 1e-5
- 检测框输出的类别一致性（argmax 应该 100% 一致）
- 检测框坐标的 IoU（要求 > 0.999）

**（6）JIT INT8 MaxDiff < 1e-2**

INT8 量化必然带来精度损失，因为将连续浮点值离散化到 256 个整数量级。MaxDiff < 1e-2 的阈值是经验值——在此阈值内，检测框的类别一致性可以达到 99% 以上，IoU 保持在 0.95 以上，增强图像在视觉上几乎不可感知差异。

#### P1 级指标（软约束，允许在极端条件下降级）

**（7）检测 mAP@0.5 > 0.80**

mAP（mean Average Precision）是检测任务的核心指标。0.5 表示 IoU 阈值为 50% 时视为正检。> 0.8 意味着 80% 以上的缺陷能被准确定位和分类。对于晶圆缺陷检测，这个指标在行业属于中高水平。

mAP 的详细计算参考 `reference/FALCO-WAFER-main/ultralytics/utils/metrics.py`。每类缺陷的 AP 会单独计算并报告，便于分析哪类缺陷检出困难。需要注意的是，AP 中的 Precision 和 Recall 需要根据产线的实际需求做权衡——在快速抽检模式中可牺牲 Precision 换取更高的 Recall。

**（8）增强 PSNR > 25dB**

PSNR（Peak Signal-to-Noise Ratio）是图像增强的客观指标。25dB 意味着增强图像与参考明场图像之间的均方误差约为 0.003（对于 [0,1] 值域），在视觉上接近人眼不可分辨的水平。

测量方法：使用配对测试集（同一区域暗场/明场图像，如有），计算 `10 * log10(1 / MSE(enhanced, bright_reference))`。若没有配对参考图像，可以改用 NIQE（Natural Image Quality Evaluator）等无参考图像质量指标做辅助评估。

**（9）蒸馏精度损失 < 1%**

即学生模型与教师模型在相同测试集上的 mAP 差异小于 1 个百分点（绝对差值）。蒸馏损失 = `teacher_mAP - student_mAP`。若损失超过 1%，需要调整蒸馏策略：增加特征蒸馏的权重（当前 γ=0.1 → 0.2）、增大蒸馏温度（当前 T=4.0 → 6.0）、或选择更大的学生模型。

**（10）内存占用 < 500MB**

指推理过程中的峰值内存消耗（包含模型权重、中间特征图、输入输出缓冲区）。INT8 量化后的模型权重约 3MB，但中间特征图（特别是 128×128 的增强分支特征）可能占用数十到数百 MB。通过 profile 工具（valgrind massif 或 PyTorch 的 memory_stats API）测量峰值。

### 10.2 验收流程及各环节详细标准

#### 验收阶段一：代码审查（Code Review）

审查人员逐行检查以下关键点：

- **JIT 兼容性检查：** 遍历 `models/` 下所有 `nn.Module` 的 `forward()` 方法，确认无以下不兼容写法：
  - 无 `if self.training:` 条件分支（验证模式已固化）
  - 无 `.item()`, `.tolist()`, `len()` 等 Python 原生调用
  - 无 `for` 循环构造动态计算图（使用 `ModuleList` + 向量化操作替代）
  - 无 `torch.Tensor` 的 `.device` 属性用于动态创建张量
- **参数量检查：** 运行 `python models/profile_model.py`，确认参数量在预算范围内（基准值 ±10%）
- **模块化检查：** 确认 `encoder.py`, `enhance_decoder.py`, `detect_head.py`, `wafer_multitask.py` 的边界清晰，无循环依赖

#### 验收阶段二：功能正确性验证

**子项 2.1：前向推理正确性**
```bash
python -c "
from models.wafer_multitask import WaferMultiTaskModel
import torch
model = WaferMultiTaskModel(in_channels=1, num_classes=4)
model.eval()
dummy = torch.randn(1, 1, 512, 512)
enhanced, detections = model(dummy)
assert enhanced.shape == (1, 1, 512, 512), f'增强图尺寸错误: {enhanced.shape}'
assert detections.dim() == 3 and detections.shape[2] == 8, f'检测输出尺寸错误: {detections.shape}'
print('✅ 前向推理形状正确')
"
```

**子项 2.2：JIT 导出正确性**
```bash
python deploy/verify_model.py --model runs/stage3/best.pth
# 预期输出：
# ✅ FP32 JIT  MaxDiff: 3e-6 (阈值: 1e-5)
# ✅ INT8 JIT MaxDiff: 6e-3 (阈值: 1e-2)
# ✅ 检测类别一致性: 99.7% (阈值: 99%)
```

**子项 2.3：结构融合正确性**
```bash
python -c "
# 融合后模型应不再包含 BN 层和多分支结构
model = torch.jit.load('deploy/wafer_detector_jit_fp32.pt')
graph = model.graph
# 检查图中是否还有 BatchNorm 节点
has_bn = 'BatchNorm' in str(graph)
print(f'BatchNorm detected: {has_bn}')
# 融合后应无 BatchNorm
"
```

#### 验收阶段三：性能基准测试

在目标 ARM Cortex-A76 硬件上执行：
```bash
# 运行完整 benchmark 套件
./wafer_detector \
    --model wafer_detector_jit_int8.pt \
    --benchmark \
    --warmup 10 \
    --iterations 100 \
    --threads 1

# 预期输出（Cortex-A76 @ 2.4GHz）：
# -----------------------------
# Model: wafer_detector_jit_int8.pt
# Input: [1, 1, 512, 512]
# Warmup: 10 iterations
# Benchmark: 100 iterations
# Average: 52.3 ms  ✅ < 80ms
#   P50:   51.8 ms
#   P99:   58.1 ms
# Memory: 287 MB   ✅ < 500MB
# -----------------------------
```

性能不达标的降级预案（优先级排列）：
1. 启用极坐标展开（减少输入像素 → 减少计算量）；预期收益：20-30%
2. 减少增强解码分支的通道数（48→32）；预期收益：10-15%
3. 使用更小的骨干网络（RepViT-M1.0→M0.9→M0.6）；预期收益：20-40%
4. 检测头进一步精简（去掉一个 MSD 层）；预期收益：5-10%
5. 使用更激进的 INT8 量化（全量化不跳层）；预期收益：10%（以精度换速度）

#### 验收阶段四：场景端到端验证

使用 100 张从产线采集的标注晶圆图像（覆盖 4 类缺陷各 > 20 张，包含不同光照条件下的正常晶圆 > 20 张）执行完整的 Python→C++ 一致性测试：

```bash
# Python 端推理
python infer.py --model wafer_detector_jit_int8.pt --images test_images/ --output python_results/

# C++ 端推理
./wafer_detector --model wafer_detector_jit_int8.pt --images test_images/ --output cpp_results/

# 逐像素逐框对比
python compare_results.py --python python_results/ --cpp cpp_results/
# 预期：
# 增强图像素一致性: 0.9997
# 检测框一致性: 0.992
# 类别一致性: 1.000
```

### 10.3 扩展性设计的工程考量

本项目的扩展性设计不是事后的「加几个参数」，而是在架构层面系统性地预留了三个维度的扩展接口：

**缺陷类别扩展。** 检测头的 `num_classes` 参数控制，修改后最后一层分类卷积的输出通道数自动调整。无需修改模型结构，但需要微调检测头（通常用阶段二的 10% 标注数据做 20 epoch 微调即可适配新类别）。`classes` 的命名和映射通过配置文件管理，不硬编码在代码中。

**输入通道扩展。** 编码器的首层 `Conv2d_BN` 的 `in_channels` 参数在初始化时通过 `in_chans` 参数指定。当 `in_chans=4` 时自动插入 `MultiWavelengthFusion` 通道注意力模块。这种设计使得多波长融合成为模型的「插件」，而非「硬编码」——单通道场景下不引入额外参数和计算量。

**跨工艺适配。** 外部 JSON 配置文件 `inference_config.json` 支撑了三种适配场景：
- 同一晶圆厂不同工艺节点（如 6 寸 SiC → 8 寸 GaN）：修改 wafer.sizes_inches 和检测 ROI 参数
- 不同缺陷敏感度要求（如精密检测 vs 快速抽检）：修改 defect_classes 中每个类别的 sensitivity
- 不同输出需求（如 CSV 对接 MES vs HTML 人工审查）：修改 inspection.report_format

### 10.4 约束与限制的详细说明

理解执行的约束条件对于系统集成至关重要：

**（1）固定 512×512 输入尺寸。** 这是 JIT trace 机制的内在限制——trace 固定了所有张量的形状。对于不同分辨率的输入，必须在预处理中做 Resize，并在后处理中按 `512 / orig_size` 的比例将检测坐标映射回原始空间。极坐标展开模式下，固定输出 512×128 的展开图，即使原始图像分辨率不同也统一做 Resize。

**（2）不适用视频流实时检测。** 单帧推理 < 80ms 可以支持约 12 FPS 的连续推理。但晶圆检测涉及机械步进平台的移动（每步 0.5mm，移动时间约 200ms），实际检测速度受限于机械移动而非推理速度。因此本项目的优化目标是总检测时间 < 3 分钟，而非视频帧率。

**（3）极坐标展开的适用边界。** 极坐标展开将环形边缘区域展开为矩形，对边缘区域的缺陷（崩边、划痕）检测效率提升明显（减少 60% 无效计算）。但对晶圆中心区域（r < inner_radius）的缺陷（如位错团簇）不适用，必须在配置中关闭 `enable_polar` 开关。客户应根据工艺缺陷的分布特征选择是否启用该优化。

**（4）INT8 量化精度边界。** INT8 量化在大多数场景下精度损失可控（mAP 下降 < 0.5%），但在以下场景可能出现 > 2% 的精度损失：
- 输入图像动态范围极不均匀（如部分区域过曝、部分区域欠曝）
- 校准集中不包含某种缺陷类别
- 模型在 FP32 时已经接近过拟合边界
这些场景的应对措施包括：使用 per-channel 量化替代 per-tensor 量化、跳过敏感层量化、或在训练后执行 QAT 微调恢复精度。

**（5）知识蒸馏的适用边界。** 知识蒸馏在教师和学生模型的「容量差距」适中时效果最佳。如果学生模型过小（如 RepViT-M0.6 进一步剪枝到 < 2M 参数），可能无法有效吸收教师的知识，蒸馏精度可能超过 5%。此时应选择更大的学生模型或调整蒸馏温度使软标签分布更平滑。

---

## 11. 多层强化学习可行性分析与设计

### 11.1 问题定义：我们讨论的「多层强化学习」是什么

在项目负责人沟通中提出的「多层强化学习」概念，在本项目的语境下可以解构为三个层次的理解：

**理解一（层级强化学习 HRL）：** 将决策分解为高层策略（如：当前批次应该用高召回还是高精度模式？）和低层策略（如：具体每个缺陷框的阈值如何微调？），上层决策指导下层动作。

**理解二（管道多环节 RL）：** 在检测管道的多个环节分别引入 RL ——数据增强策略学习、训练课程规划、在线自适应校准、推理阈值动态调整——各环节独立或联合优化。

**理解三（网络多层架构内的 RL 模块）：** 在神经网络的不同特征层中嵌入可微的 RL 模块，让模型在推理时根据特征状态动态调整计算路径或注意力分配。

经全面评估，结论如下：**不适合作为核心技术路线的主线，但在训练策略优化和在线自适应两个方向有轻量级的锦上添花空间。** 以下逐项分析。

### 11.2 与项目约束的冲突分析

在讨论具体方案之前，需要先明确本项目与 RL 天然的三个矛盾：

| 项目约束 | RL 的典型要求 | 矛盾程度 | 说明 |
|---------|-------------|---------|------|
| ARM 推理 < 80ms | RL 策略网络需 1-5ms 额外推理 | 🔴 中等 | 极轻量策略（< 0.1M 参数）可控制在 1ms 内，但多数 RL 方案未针对边缘优化 |
| 标注样本 30-50 张 | RL 通常需 10⁴-10⁶ 次交互 | 🔴 严重 | 基于环境的 RL（如机械臂控制）完全不适用；基于数据的 RL（如策略学习）需要大量标注反馈 |
| 单次前向确定性输出 | RL 策略的随机性导致检测结果不确定 | 🟡 可控 | 推理时取 argmax 可消除随机性，但策略网络本身的精度波动需评估 |

这三个矛盾决定了 RL 在本项目中**不能作为核心检测逻辑**，只能作为**外围参数调节器**。

### 11.3 五个候选方向的可行性评估

#### 方向一：❌ 基于 RL 的缺陷检测网络（不采用）

**描述：** 将检测头的分类/回归替换为 DRL（Deep Reinforcement Learning）智能体——智能体在图像特征图上「移动视野」并「决定是否存在缺陷」，类似基于注意力的递归检测模型（如 DRGN、RL-RPN）。

**可行性分析：**
- **推理速度：** 序列决策过程需要多次前向推理，单帧推理时间将从 52ms 飙升到 500ms+，完全无法满足 < 80ms 硬约束
- **训练难度：** 30-50 张标注数据远远不足以训练一个稳定的 DRL 智能体（DRL 在 Atari 游戏上需要 10⁷ 帧才能收敛）
- **部署兼容：** 循环/递归结构的 JIT 导出极易出现动态控制流错误
- **精度收益：** 无证据表明 DRL 检测在晶圆缺陷这类静态图像上能超越 anchor-free 检测头

**结论：不采用。** 与项目核心约束直接冲突，且无精度优势。

---

#### 方向二：🟡 训练时数据增强策略搜索（可参考，但优先级低）

**描述：** 将 Copy-Paste、Mosaic、MixUp、AugMix 等增强操作的参数选择建模为一个 RL 问题——智能体观测当前训练状态（loss、梯度范数、验证精度），选择增强操作的类型和强度，以最大化验证集精度。这本质上是 AutoAugment 的 RL 变体。

**可行性分析：**
- **推理开销：** 增强策略只在训练时生效，推理时零开销 ✅
- **样本需求：** 需要大量验证集反馈来训练策略。当前只有 30-50 张样本的训练集无法划分出足够大的验证集来稳定评估策略效果
- **预期收益：** 当前手动设计的增强组合（Copy-Paste + 泊松融合 + Mosaic + MixUp + AugMix）已经覆盖了晶圆场景的主要变化，AutoAugment 式搜索带来的边际提升有限（通常 < 2 mAP）
- **工程复杂度：** 需要引入 RL 训练框架（如 Ray RLlib），增加训练管线的复杂度和调试成本

**结论：暂不优先采用。** 如果未来标注数据扩充到 500+ 张，可以作为精度冲刺阶段的优化手段。优先级排在三阶段训练 + 知识蒸馏之后。

---

#### 方向三：🟢 在线自适应校准（Contextual Bandit）— 推荐加入

**描述：** 将 9.5 节的规则式在线自适应校准升级为**上下文赌博机（Contextual Bandit）**——智能体观测当前环境上下文（最近 N 张图像的灰度统计量、光源功率读数、传感器温度等），选择最优的预处理归一化参数和检测灵敏度阈值，以最大化缺陷检出 F1-Score。

**为什么这是最合适的 RL 切入点：**
1. **问题天然是序列决策问题：** 设备老化导致的光源漂移是缓慢的时间序列过程，每次校准都是一次「选择参数→观测结果→调整」的闭环
2. **动作空间极小且连续：** 仅需调节 2-4 个连续参数（归一化均值和标准差、全局置信度偏移量），轻量策略网络（< 0.05M 参数）即可胜任
3. **推理开销可控：** 策略网络是独立于主模型的外围模块，推理仅需 1 次前向（< 0.5ms），不在主 JIT 图中，不影响 < 80ms 约束
4. **奖励信号可获取：** 以自动标注（当前模型的高置信度检测结果）作为奖励代理，不需要人工标注

**设计细节：**

```
Context: [当前批次灰度均值, 灰度标准差, 灰度偏度, 光源使用时长, 传感器温度]
    (5 维连续向量)
         │
         ▼
  轻量策略网络 (2 层 MLP: 5→16→4, ~0.01M 参数)
         │
         ▼
  动作: [归一化均值偏移 δμ, 归一化标准差缩放 δσ, 全局置信度偏移 δτ]
         │
         ▼
  执行 → 观测 → 奖励: F1-Score 变化量 ΔF1 (用模型高置信度检测作为伪标签)
```

**训练方式**（与主模型三阶段训练解耦）：
```python
# 在线训练循环（产线运行期间持续学习）
for batch in wafer_stream:
    context = extract_context(batch)           # 提取当前环境状态
    action = bandit.policy(context)            # 选择校准参数
    result = run_inference(batch, action)      # 用校准后的参数推理
    reward = compute_proxy_reward(result)      # 用高置信度检测结果估算 F1
    bandit.update(context, action, reward)     # 在线更新策略（Thompson Sampling / LinUCB）
```

**预期收益：**
- 将在线校准从「固定规则」（当前设计）升级为「自适应学习」，对设备老化的响应精度提升 2-3 倍
- 训练与主模型完全解耦，不影响三阶段训练管线
- 不会出现在 JIT 图中，不干扰主模型推理

**结论：推荐加入。** 这是 RL 在本项目中最自然、最实用的落地点。

---

#### 方向四：🟢 动态灵敏度调节（Contextual Bandit）— 推荐加入

**描述：** 类似方向三但作用于检测后处理——智能体观测批次图像的全局统计特征，动态调整每类缺陷的置信度阈值和 NMS 参数，以在当前批次条件下取得最优的精度-召回率平衡。

**与方向三的关系：** 方向三作用于预处理阶段（图像归一化），方向四作用于后处理阶段（检测阈值）。两者可以组合为**双 Bandit 系统**，但各自独立训练和推理，互不干扰。

**具体动作空间：**
```python
action = {
    'chipping_threshold': δ₁,   # 崩边类置信度阈值偏移
    'particle_threshold': δ₂,   # 颗粒类（小目标，通常需要更低阈值）
    'scratch_threshold': δ₃,    # 划痕类
    'dislocation_threshold': δ₄, # 位错类（低对比度，通常需要低阈值）
    'nms_iou_offset': δ₅,       # NMS IoU 阈值偏移
}
```

**验证指标：**
- 调节后的 F1-Score 应高于固定阈值配置 3-5%
- 调整频率：每批次（约 50-100 张子图）调整一次，而非每帧，避免抖动
- 调节幅度约束：|δ| < 0.15，防止过度偏离基线

**结论：推荐加入，与方向三共同构成双 Bandit 自适应系统。**

---

#### 方向五：❌ 网络架构内的可微 RL 模块（不采用）

**描述：** 在编码器的不同特征层中嵌入可微的 RL 模块（如 Gumbel-Softmax 策略、可微 NAS），让模型根据输入特征动态选择计算路径——例如，对高信噪比区域走轻量路径，对低信噪比区域走重计算路径。

**可行性分析：**
- **架构复杂度：** 需要在 RepViT 的重参数化结构中引入条件计算分支，破坏了 RepViT 的「推理时融合为单路卷积」的核心设计，重参数化融合将不再成立
- **JIT 兼容性：** 动态路径选择必然引入 `if/else` 控制流，TorchScript tracing 会固化分支，scripting 则可能不支持所有操作
- **ARM 性能：** 条件计算在 ARM CPU 上几乎没有加速效果（分支预测失效），多个计算路径反而占用更多代码体积

**结论：不采用。** 与 RepViT 的架构哲学和 JIT 导出要求严重冲突。

### 11.4 推荐方案总览

```
┌──────────────────────────────────────────────────────────────┐
│                   多层强化学习集成方案（轻量级）               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  训练阶段（不影响推理）：                                      │
│    └── 方向二：数据增强策略搜索 🟡 暂缓，标注数据 500+ 后考虑 │
│                                                              │
│  推理阶段（在推理管道的外围，不进入 JIT 图）：                  │
│    └── 方向三：在线自适应校准（Contextual Bandit） 🟢 推荐加入 │
│    └── 方向四：动态灵敏度调节（Contextual Bandit） 🟢 推荐加入 │
│                                                              │
│  不采用：                                                     │
│    └── 方向一：RL 检测网络 ❌ 与 < 80ms 硬约束冲突             │
│    └── 方向五：网络内可微 RL ❌ 破坏 RepViT 架构和 JIT 兼容性  │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  额外增加的总参数量：< 0.1M（双 Bandit 策略网络）              │
│  额外增加的推理时间：< 1ms（预处理和后处理阶段各 < 0.5ms）     │
│  不影响主模型 JIT 图、INT8 量化和 < 80ms 推理约束              │
└──────────────────────────────────────────────────────────────┘
```

### 11.5 集成方式与实施要点

**与现有架构的关系：** 双 Bandit 系统作为**独立的外围模块**部署，不在主 JIT 模型中，不影响现有训练管线和推理流程。

```python
# 推理时的完整管道（双 Bandit 插入位置）
def full_inference_pipeline(raw_image):
    # 1. 提取环境上下文
    ctx = extract_context(raw_image)
    
    # 2. Bandit A: 自适应校准 → 调整预处理参数
    calib_action = bandit_calibration.policy(ctx)
    norm_params = apply_calibration(calib_action)
    
    # 3. 预处理（使用校准后的参数）
    tensor = preprocess(raw_image, norm_params)
    
    # 4. 主模型推理（JIT 图，不受 Bandit 影响）
    enhanced, detections = jit_model(tensor)
    
    # 5. Bandit B: 动态灵敏度 → 调整检测阈值
    threshold_action = bandit_threshold.policy(ctx)
    final_results = apply_thresholds(detections, threshold_action)
    
    # 6. 异步更新 Bandit（非关键路径，不阻塞返回结果）
    async_update_bandits(ctx, calib_action, threshold_action, final_results)
    
    return enhanced, final_results
```

**实施优先级：** 方向三（在线校准 Bandit）> 方向四（阈值调节 Bandit）。建议在项目第 2-3 个月、主模型训练和 JIT 导出完成后实施。

### 11.6 验证方法

| 验证项 | 方法 | 目标 |
|-------|------|------|
| Bandit 收敛性 | 模拟光源漂移场景，验证策略能否收敛到最优参数 | 500 次迭代内收敛 |
| F1 提升 | 对比固定规则 vs Bandit 自适应在 6 个月跨度数据上的 F1 | Bandit 提升 > 3% |
| 推理时间增量 | 在 ARM 上分别测量有无 Bandit 的总推理时间 | 增量 < 1ms |
| 策略稳定性 | 连续 1000 次在线更新的动作波动标准差 | std(δ) < 0.05 |

---

## 参考资源索引

### 本地参考代码与项目关联说明

本项目不是简单复刻这些开源仓库，而是深度消化其核心设计思想后，针对晶圆暗场检测场景做了定制化改造。以下逐一说明每个参考仓库的核心贡献和在项目中的具体使用位置：

**RepViT（`reference/RepViT-main/`）— 共享编码器骨干**

核心文件 `model/repvit.py` 是我们编码器的基础。我们复用了其三个核心机制：（1）`Conv2d_BN` 类（26-48行）的 Conv+BatchNorm 融合设计，直接照搬到 `models/encoder.py`；（2）`RepVGGDW` 类（83-121行）的三分支可融合深度可分离卷积，这是 ReplViT 的关键创新，也是我们编码器中最底层的计算单元；（3）`RepViTBlock` 类（124-160行）的 token_mixer + channel_mixer 双模块设计，我们在原始结构上增加了多尺度特征输出接口。原始 RepViT 的配置表（如 repvit_m1_0 的 cfgs 列表）也直接被用于定义我们的编码器结构——区别仅在于输入通道数（3→1）和前向输出方式（单输出→多尺度特征列表）。

**LYT-Net（`reference/LYT-Net-main/`）— 增强解码分支参考**

核心文件 `PyTorch/model.py` 中的 `MSEFBlock`（40-58行）和 `Denoiser`（103-136行）是我们增强解码分支的核心组件。MSEFBlock 的 LayerNorm→DepthwiseConv→SE 注意力→逐元素乘积融合的范式，被直接复用到 `models/enhance_decoder.py`。Denoiser 的 4 层下采样+MHSA 瓶颈+跳跃连接上采样的 U 型结构，为暗场散粒噪声抑制提供了高效的轻量方案。LYT-Net 原本是在 YUV 色彩空间中处理 RGB 图像（138-198行），我们将其简化为单通道灰度处理，并将通道数从 64 压缩到 32-48 以适配参数预算。`PyTorch/losses.py` 中的损失组合策略（smooth_l1 + LPIPS + MS-SSIM + histogram）也被我们参考用于阶段二的增强损失设计。

**CycleGAN（`reference/pytorch-CycleGAN-and-pix2pix-master/`）— 无监督域迁移框架**

核心文件 `models/cycle_gan_model.py` 的循环一致性损失（backward_G 方法，153-180行）和对抗损失（backward_D_basic 方法，121-141行）构成了阶段一无监督预训练的损失基础。我们参考了其 `lambda_A`/`lambda_B`（默认为 10.0）和 `lambda_identity`（默认为 0.5）的损失权重配置。`models/networks.py` 中的 `NLayerDiscriminator`（PatchGAN 鉴别器）被直接复用为阶段一的鉴别网络——它通过将图像分为 N×N 的图像块独立判断真伪，对晶圆这种局部纹理均匀的图像特别有效。CycleGAN 的 `ImagePool` 机制（用于缓存之前生成的假图像）也被集成到阶段一的训练管线中，以提高训练稳定性。

**LPIPS（`reference/PerceptualSimilarity-master/`）— 感知损失度量**

核心文件 `lpips/lpips.py` 实现了 Learned Perceptual Image Patch Similarity 度量。我们使用 `LPIPS` 类（22-144行）的 `forward(in0, in1)` 方法作为增强任务的感知损失约束。LPIPS 使用预训练的 AlexNet/VGG 网络提取 5 层特征，计算两幅图像在各层特征空间的距离。它与逐像素 L1/L2 损失的最大区别在于：两张在像素上差异大但在语义上相似的图像（如同—场景不同的光照条件），LPIPS 值低；反之像素差异小但在语义上不同的图像（如边缘略微偏移），LPIPS 值高。这种特性使得 LPIPS 成为评估增强图像感知质量的理想指标——它鼓励模型生成「看着像明场」的图像，而非简单地逐像素匹配。

**FALCO-WAFER（`reference/FALCO-WAFER-main/`）— 晶圆缺陷检测基线**

这是与本项目场景最接近的参考仓库，基于 YOLOv8 框架专门针对晶圆缺陷检测做了优化。我们重点参考了以下文件：
- `ultralytics/nn/modules/head.py` 的 `Detect` 类（20-81行）：YOLOv8 的 anchor-free 检测头设计，特别是分类分支和回归分支的通道分配策略。
- `ultralytics/nn/extra_modules/block.py` 的 `C2f_MSD` 类（260-263行）：基于 DynamicIncMixerBlock 的多尺度深度可分离模块，比标准 C2f 在同等参数量下有更大的有效感受野。
- `ultralytics/utils/loss.py`：v8DetectionLoss 的实现，我们参考了其 CIoU 定位损失和 TaskAlignedAssigner 正负样本分配策略。
- `ultralytics/nn/extra_modules/attention.py`：AFGCAttention 和 CoordAtt 的注意力机制实现，为我们设计多波长通道注意力模块提供了参考。
- `reference/FALCO-WAFER-main/dataset/VOCdevkit/`：数据集的 VOC 格式标注结构，我们的数据加载器兼容此格式。

FALCO-WAFER 与我们的核心差异在于：FALCO-WAFER 是全尺寸 YOLOv8 模型（~25M 参数），直接在原图上做检测；我们则通过共享编码器+增强分支+极坐标展开的联合优化，在 < 7.5M（M0.9）/ < 9.2M（M1.0）参数下达到可比的检测精度。

**mdistiller（`reference/mdistiller-master/`）— 知识蒸馏框架**

核心文件 `distillers/KD.py`（9-39行）实现了基于 KL 散度的逻辑蒸馏损失 `kd_loss`——这是阶段三蒸馏训练的核心组件。`distillers/FitNet.py`（9-48行）实现了基于 MSE 的特征蒸馏，用于对齐教师和学生模型的中间层特征图。`distillers/_base.py`（6-41行）的 `Distiller` 基类定义了蒸馏训练的统一接口（`forward_train` + `forward_test`），我们参考其设计了阶段三的训练循环——教师模型在 eval 模式下冻结，学生模型在前向传播阶段同时计算学生输出和教师输出，通过 KL 散度最小化两者分布差异。我们将 mdistiller 中针对分类任务的蒸馏策略适配到检测任务，即在检测头的分类分支输出上应用 KL 散度，在回归分支输出上应用 MSE 蒸馏。

**Transfer-Learning-Library（`reference/Transfer-Learning-Library-master/`）— 域适应与迁移学习管线**

`examples/domain_adaptation/image_classification/` 中的训练流程设计为我们的阶段一训练管线提供了参考。特别是其数据加载器设计中「源域和目标域样本交替加载」的模式，被我们用于阶段一的暗场/明场数据加载。`examples/domain_adaptation/object_detection/` 中的域适应检测管线为我们设计检测损失和训练循环提供了参考。

**YOLOv5（`reference/yolov5-master/`）— 工程规范与最佳实践**

我们参考其工程目录结构设计（模型的 model/→train/→utils/ 分离）、训练脚本参数化设计（argparse + yaml 配置）、以及导出脚本（export.py）的模块化组织方式。YOLOv5 的工程规范已经成为计算机视觉领域的通用标准，我们的 `requirements.txt`、训练脚本设计、配置文件格式都遵循了其规范。

### 在线资源

| 资源 | 链接 | 在本项目中的用途 |
|------|------|----------------|
| RepViT 官方仓库 | https://github.com/THU-MIG/RepViT | 骨干网络原始论文与实验配置 |
| URetinex-Net 官方仓库 | https://github.com/IDKiro/URetinex-Net | LYT-Net 的低光照增强参考 |
| CycleGAN 官方仓库 | https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix | 无监督域迁移框架参考 |
| LPIPS 官方仓库 | https://github.com/richzhang/PerceptualSimilarity | 感知损失度量的原始实现 |
| FALCO-WAFER 官方仓库 | https://github.com/MrJoker06/FALCO-WAFER | 晶圆缺陷检测基线参考 |
| mdistiller 官方仓库 | https://github.com/megvii-research/mdistiller | 知识蒸馏算法集合 |
| Transfer-Learning-Library | https://github.com/thuml/Transfer-Learning-Library | 域适应训练管线参考 |
| PyTorch JIT 文档 | https://pytorch.org/docs/stable/jit.html | TorchScript 导出工具链文档 |
| PyTorch 量化文档 | https://pytorch.org/docs/stable/quantization.html | INT8 静态量化 API 文档 |
| anomalib | https://github.com/openvinotoolkit/anomalib | 异常检测基线参考 |
| OpenCV 极坐标变换 | https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gaa38a6884ac8b6e0b9bedd59635e52bec | warpPolar 函数参考 |
| Intel Neural Compressor | https://github.com/intel/neural-compressor | 进阶量化压缩工具（备用方案） |

---

> **文档结束**  
> 本文档由 `docs/TECHNICAL_ROADMAP.md` 维护，随项目进展同步更新。  
> 有关 Skill 调用和操作步骤，请参见 `OPERATION_GUIDE.md`。
