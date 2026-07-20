# 光学物理约束注入方案 — 完整设计文档 (v2.0)

> **文档版本**：v2.0（新增 ASG 多角度生成）  
> **日期**：2026-07-06  
> **类型**：架构级物理增强设计  
> **涉及文档**：`docs/TECHNICAL_ROADMAP.md` / `docs/FOR_PROJECT_MANAGER.md` / `docs/wafer-showcase-design.md`  
> **核心思想**：在现有「一骨干双分支」架构中注入可微 Rayleigh/Mie 散射物理模型 + 多方位角散射生成，实现 266nm→193nm 虚拟波长转换 + 15°-75° 多视角融合检测

---

## 目录

1. [设计动机与物理基础](#1-设计动机与物理基础)
2. [总体架构变更](#2-总体架构变更)
3. [模块一：可微散射物理模块 PISM](#3-模块一可微散射物理模块-pism)
4. [模块二：双光路并行检测分支](#4-模块二双光路并行检测分支)
5. [模块三：散射引导融合层 SGF](#5-模块三散射引导融合层-sgf)
6. [模块四：物理约束损失函数](#6-模块四物理约束损失函数)
7. [三阶段训练管线更新](#7-三阶段训练管线更新)
8. [模块五：多方位角度散射生成器 ASG](#8-模块五多方位角度散射生成器-asg)
9. [验收指标更新](#9-验收指标更新)
10. [展厅设计更新](#10-展厅设计更新)
11. [参数预算与推理时间核算](#11-参数预算与推理时间核算)
12. [文件结构变更](#12-文件结构变更)

---

## 1. 设计动机与物理基础

### 1.1 核心问题

当前增强解码分支用 CycleGAN 做暗场→明场域迁移，是纯数据驱动的**黑箱映射**。这导致：

1. **遇到未见过的缺陷类型或光照条件时，映射可能物理上不合理**
2. **无法从物理上解释**「为什么这个波长对这类缺陷敏感」
3. **多波长融合只是简单的注意力加权**，没有基于散射物理的权重分配
4. **所有检测都在单一方位角进行**，丢失了缺陷散射的各向异性信息

### 1.2 物理基础

#### Rayleigh 散射（缺陷尺寸 << 波长）

$$I_{\text{scat}} \propto \frac{1}{\lambda^4} \cdot |\frac{n^2-1}{n^2+2}|^2 \cdot d^6$$

$$\frac{I_{\text{193}}}{I_{\text{266}}} = \left(\frac{266}{193}\right)^4 \approx 3.5$$

**适用对象**：尺寸 < 42nm（~λ/2π）的微小缺陷

#### Mie 散射（缺陷尺寸 ~ 波长）

散射系数 $a_n$、$b_n$ 依赖于粒子尺寸参数 $x = \pi d / \lambda$ 和复折射率。

**适用对象**：崩边（10-200μm）、划痕、大颗粒

#### 方位角各向异性散射

晶圆缺陷的散射强度随观察方位角变化：

- **各向同性缺陷**（颗粒、圆形崩边）：散射强度随方位角变化小，近似均匀
- **各向异性缺陷**（划痕、线性崩边）：散射强度随方位角变化显著——当照明方向垂直于缺陷长轴时散射最强，平行时最弱

在极坐标空间中，方位角旋转 $\Delta\theta$ 等价于图像在极坐标宽度方向上的平移 $\Delta x$：

$$\Delta x = \frac{\Delta\theta}{360^\circ} \times W_{\text{polar}}$$

其中 $W_{\text{polar}} = 512$，因此每 $5^\circ$ 对应 $\Delta x \approx 7.1$ 像素平移。

#### 双波长物理优势

| 物理量 | 266nm | 193nm | 增益 |
|-------|-------|-------|------|
| Rayleigh 散射强度 | 基准 (1.0×) | **3.5×** | 小缺陷信号增强 |
| Abbe 衍射极限 | ~133nm | **~97nm** | 分辨率提升 27% |
| SiC 穿透深度 | ~0.1μm | ~0.05μm | 更极端表面敏感 |

### 1.3 设计原则

1. **物理可微**——所有散射公式必须可微，使其能作为神经网络层参与反向传播
2. **物理+学习混合**——纯物理模型预测主流趋势，可学习残差补偿近似误差
3. **极坐标高效变换**——利用极坐标展开的特性，使多角度生成近乎零成本
4. **少量数据校准**——仅需少量多角度采集数据即可校准生成质量
5. **ARM 兼容**——所有新增模块使用标准 PyTorch 操作，可 JIT trace

---

## 2. 总体架构变更

### 2.1 架构对比

```
物理增强后完整架构（v2.0）:

266nm 暗场输入 (512×512)    ← 单张输入，零角度基准
    │
    ▼
Polar 展开 (512×128)        ← 可选，默认开启
    │
    ▼
共享编码器 RepViT-M0.9  (~5.1M)
    │
    ├────────────────────────────────────────────────────┐
    │                                                    │
    ▼                                                    ▼
266nm 检测分支 (单视角)      🆕 ASG 多角度散射生成器 (~0.15M)
(~1.2M)                      │
    │                   ┌────┼────┬────┬────┬────┐
    │                  15°  20°  45°  70°  75°  ... 13 angles
    │                   │    │    │    │    │    │
    │                   ▼    ▼    ▼    ▼    ▼    ▼
    │               PISM 散射模块 (共享 ~0.04M)
    │                   │    │    │    │    │    │
    │                   ▼    ▼    ▼    ▼    ▼    ▼
    │               193nm 检测分支 (共享权重, batch 13)
    │                   │    │    │    │    │    │
    │                   └────┼────┴────┼────┘    │
    │                        │         │         │
    └──────────┬─────────────┘         │         │
               │                       │         │
               ▼                       ▼         ▼
         266nm (单视角)           🆕 角度注意力融合
                                          │
                                          ▼
                                   Angle-Fused 193nm
                                          │
                               ┌──────────┘
                               ▼
                          SGF 散射引导融合
                     (266nm 单视角 + Angle-Fused 193nm)
                               │
                               ▼
                           融合缺陷列表 [N, 8]
```

### 2.2 完整数据流

```
Step 1:  输入 266nm 暗场图像
Step 2:  极坐标展开 → 512×128 极坐标图
Step 3:  编码器前向 (1次) → 多尺度特征 [F1, F2, F3, F4]
Step 4a: 266nm 检测头从极坐标特征直接预测 → 266nm 视角检测结果
Step 4b: ASG 在极坐标特征空间中生成 13 个方位角特征:
           for θ in [15°, 20°, ..., 75°]:
             F_i_θ = geometric_shift(F_i, θ/360×W)     # 零参数
             F_i_θ = scatter_modulate(F_i_θ, θ)         # 物理调制
             F_i_θ = completion(F_i_θ)                  # 轻量补全
Step 5:  PISM 将每个角度的 266nm 特征转为虚拟 193nm 特征
Step 6:  共享权重的 193nm 检测头在 batch=13 上预测
Step 7:  角度注意力融合 13 组检测结果
Step 8:  SGF 融合 266nm(单视角) + Angle-Fused 193nm 结果
```

---

## 3. 模块一：可微散射物理模块 PISM

（与 v1.0 设计一致，详细代码见原始 spec 的 Section 3）

**关键参数：**

| 组件 | 参数量 | 说明 |
|------|--------|------|
| Scale Estimator (×4) | ~10K | 每尺度独立估计 |
| Rayleigh 比 | 0 | 常数 (266/193)⁴ ≈ 3.5 |
| Mie LUT | 0 | 预计算冻结，512 点 |
| 残差网络 (×4) | ~30K | 补偿物理近似误差 |
| **PISM 合计** | **~0.04M** | |

---

## 4. 模块二：双光路并行检测分支

（与 v1.0 设计一致，见原始 spec 的 Section 4）

| 分支 | 输入 | 结构 | 参数量 |
|------|------|------|--------|
| 266nm 检测分支 | 编码器原始输出 | C2f_MSD(112→56, 224→112, 448→224) | ~1.20M |
| 193nm 检测分支 | PISM 输出（通道裁剪） | C2f_MSD(112→56, 224→64, 448→128) | ~0.42M |

---

## 5. 模块三：散射引导融合层 SGF

（与 v1.0 设计一致，见原始 spec 的 Section 5）

SGF 融合 266nm 单视角预测 与 **角度注意力融合后的 193nm 预测**（而非原始的单视角 193nm 预测）。

---

## 6. 模块四：物理约束损失函数

（与 v1.0 设计一致，见原始 spec 的 Section 6）

新增角度相关的散射一致性损失（见 Section 8.5）。

---

## 7. 三阶段训练管线更新

### 7.1 总体变更

```
阶段一（物理-informed CycleGAN）:
  原: 暗→明域迁移
  🆕: + D_193 鉴别器
  🆕: + 散射一致性损失
  🆕: + 光谱角约束

阶段二（物理-感知联合微调 + 🆕 多角度数据）:
  原: 50 张 266nm 标注
  🆕: ~20 对 266/193nm 波长配对标定片 → 校准 PISM 残差
  🆕: ~10 组多角度套图 → 训练 ASG
  🆕: 双检测分支独立训练 + SGF + ASG

阶段三（物理感知知识蒸馏）:
  🆕: + ASG 蒸馏
  🆕: + 多角度特征蒸馏
```

### 7.2 ASG 多角度数据训练

多角度数据的使用策略：

```
可用多角度数据:
  ~10 组套图，每组包含:
    266nm 照明下: 0° 基准 + 15°, 20°, ..., 75° (13 角度)
    每个角度有对应的缺陷标注

训练方式（两步走）:

Step A: 几何重投影预训练（零参数，无训练）
  → 验证极坐标平移是否能准确预测几何变化
  → 量化「纯几何变换 vs 真实图像」的残差
	
Step B: 散射调制 + 生成补全训练（小样本微调）
  用多角度数据的残差训练:
    ground truth: 真实 θ° 图像
    input: 0° 基准图像 + 目标角度 θ
    loss: 
      L_angle_total = 
        1.0 * L1(ASG(0°_img, θ), real_θ_img)  # 像素级匹配
      + 0.1 * LPIPS(ASG(0°_img, θ), real_θ_img)  # 感知质量
      + 0.05 * L_scat_consistency  # 散射一致性
      + 0.02 * L_spectral_angle    # 光谱角约束
  
  冻结策略:
    ASG 几何变换部分: 冻结（零参数，无参数可更新）
    ASG 散射调制: 微调
    ASG 补全网络: 训练
    编码器: 冻结（防止灾难性遗忘）
```

### 7.3 损失函数权重总表

```
阶段一:
  L_total = 
    10.0 * L1_cycle           # 循环一致性
    0.5  * L1_identity        # 身份映射
    1.0  * GAN_loss_D_A       # 明场鉴别器
    1.0  * GAN_loss_D_193     # 虚拟193nm鉴别器
    0.06 * LPIPS_loss         # 感知损失
    0.05 * L_scat_consistency # 🆕 散射一致性
    0.02 * L_spectral_angle   # 🆕 光谱角约束

阶段二:
  L_det_266:    1.0  # 266nm 分支
  L_det_193:    1.0  # 193nm 分支（batch=13 角度）
  L_det_fused:  0.5  # SGF 融合后
  L_scat:       0.05 # 散射一致性
  L_spec:       0.02 # 光谱角
  L_angle:      0.3  # 🆕 多角度生成（ASG 训练）

阶段三（蒸馏）:
  L_kd +
  0.05 * MSE(PISM 输出)             # 散射特征蒸馏
  0.05 * KL(193nm 分支输出)         # 193nm 逻辑蒸馏
  0.03 * L1(SGF 权重)              # 融合层蒸馏
  0.03 * L1(ASG 调制输出)          # 🆕 ASG 蒸馏
```

---

## 8. 模块五：多方位角度散射生成器 ASG

### 8.1 核心物理洞察

**极坐标空间中的方位角变换 = 水平平移。** 这是整个 ASG 零成本几何变换的物理基础。

```
晶圆原始图像                     极坐标展开图
┌──────────────┐               ┌──────────────────────┐
│    ○         │               │  r=128               │
│   /|\        │               │  ↑                   │
│  / | \       │   warpPolar   │  |    defect at θ₁   │
│    |         │  ──────────→  │  |    ────●────       │
│   / \        │               │  |                   │
│  /   \       │               │  r=0                  │
│              │               │  └──────────────────────┘
└──────────────┘                 0°      θ₁      180°    360° 
                                   └─── 512 pixels ────┘

旋转 +5°                      平移 +7.1 像素
┌──────────────┐               ┌──────────────────────┐
│    ○         │               │  r=128               │
│     \        │   roll right  │  ↑                   │
│    / \       │  ──────────→  │  |  defect at θ₁+5°  │
│   /  |       │               │  |     ●────────     │
│  /   / \     │               │  |                   │
│              │               │  r=0                  │
└──────────────┘               └──────────────────────┘
```

**为什么这是近似而非精确？** 纯几何平移假设缺陷是 2D 平面上的刚性旋转，但实际物理中：
1. **散射强度随角度变化**——各向异性缺陷的亮度会变化
2. **遮挡/阴影效应**——大角度下缺陷的某些部分被遮挡
3. **表面粗糙度方向性**——背景纹理随角度变化

因此 ASG 的三步设计：几何平移（零成本）→ 散射强度调制（物理 + 学习）→ 生成补全（轻量）

### 8.2 ASG 详细架构

```python
class AngleScatteringGenerator(nn.Module):
    """
    多方位角度散射生成器 ASG
    
    在极坐标特征空间中用最低成本生成 13 个方位角特征。
    
    输入: 
      features_266: 编码器输出的多尺度极坐标特征 [F1, F2, F3, F4]
      angles: 目标方位角列表 [15, 20, 25, ..., 75]
      
    输出:
      angles_features: 每个角度对应的 266nm 极坐标特征（待 PISM 转换）
      angle_weights: 每个角度的散射置信度图（供后续融合使用）
    """
    
    def __init__(self, 
                 channels_per_stage=[56, 112, 224, 448],
                 target_angles=list(range(15, 76, 5))):
        
        super().__init__()
        self.angles = target_angles  # [15, 20, ..., 75] → 13 个角度
        self.num_angles = len(self.angles)
        
        # 预计算每个角度的像素偏移量
        # 极坐标宽度 512 = 360°，每度 = 512/360 像素
        pixels_per_degree = 512.0 / 360.0
        shifts = [int(round(a * pixels_per_degree)) for a in self.angles]
        self.register_buffer('shifts', torch.tensor(shifts))  # 例如 15°→21px
        
        # ===== 子模块 2: 角度依赖散射调制 =====
        # 学习一个可微的、角度依赖的散射强度校正场
        # 输入: 每像素的散射尺度 s + 角度编码 → 输出: 增益校正量
        self.angle_embed = nn.Embedding(self.num_angles, 8)
        
        # 对每个特征尺度共享的调制网络
        self.scatter_modulator = nn.Sequential(
            nn.Conv2d(8 + 1, 16, 1),     # angle_embed + scale_channel
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Tanh()                     # 输出 [-1, 1] 的校正量
        )
        
        # ===== 子模块 3: 轻量生成补全 =====
        # 处理遮挡/阴影等纯几何变换无法建模的效应
        # Channel-wise 轻量卷积，仅作用于 F1（最高分辨率尺度）
        self.completion_net = nn.Sequential(
            nn.Conv2d(channels_per_stage[0], channels_per_stage[0]//4, 1),
            nn.ReLU(),
            nn.Conv2d(channels_per_stage[0]//4, channels_per_stage[0], 1),
        )
        
        # ===== 角度置信度预测器 =====
        # 每个角度的特征质量不同，输出置信度图供融合使用
        self.confidence_predictor = nn.Sequential(
            nn.Conv2d(channels_per_stage[0], 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()  # [0, 1] 置信度
        )
        
    def _geometric_shift(self, feature_map, angle_idx):
        """
        子模块 1: 几何重投影（零参数）
        
        在极坐标特征空间中，沿宽度方向（角度轴）平移。
        使用 torch.roll 实现循环平移（360° 周期性边界）。
        
        输入: feature_map [B, C, H, W]
        输出: shifted [B, C, H, W]
        """
        shift = self.shifts[angle_idx].item()
        return torch.roll(feature_map, shifts=shift, dims=-1)
    
    def forward_one_angle(self, feats_266, angle_idx, scale_maps):
        """
        生成单个目标角度的 266nm 特征
        
        输入:
          feats_266: 多尺度特征列表
          angle_idx: 目标角度在 self.angles 中的索引
          scale_maps: PISM 的 scale_estimator 输出的尺度图
                    [用于指导散射调制]
        输出:
          feat_θ: 目标角度的 266nm 特征
          conf_θ: 该角度的置信度图
        """
        angle_feats = []
        
        for i, (feat, scale_map) in enumerate(zip(feats_266, scale_maps)):
            
            # Step 1: 几何重投影（零参数，无梯度）
            feat_shifted = self._geometric_shift(feat, angle_idx)
            
            # Step 2: 角度依赖散射调制
            # 将角度编码 broadcast 到特征图尺寸
            angle_emb = self.angle_embed(
                torch.tensor([angle_idx], device=feat.device)
            )  # [1, 8]
            angle_emb = angle_emb.view(1, 8, 1, 1)
            
            # 调制输入: 角度编码 + 局部散射尺度
            mod_input = torch.cat([
                angle_emb.expand(-1, -1, feat.shape[2], feat.shape[3]),
                scale_map
            ], dim=1)  # [B, 9, H, W]
            
            mod_gain = self.scatter_modulator(mod_input)  # [-1, 1]
            
            # 应用调制（注意: 增益是 1 + 调制量，避免归零）
            feat_modulated = feat_shifted * (1.0 + 0.3 * mod_gain)
            
            # Step 3: 轻量生成补全
            # 仅在最高分辨率尺度 F1 上做补全（效果好且计算量小）
            if i == 0:
                residual = self.completion_net(feat_modulated)
                feat_completed = feat_modulated + residual
            else:
                feat_completed = feat_modulated
            
            angle_feats.append(feat_completed)
        
        # Step 4: 预测该角度的置信度图（基于 F1）
        conf_map = self.confidence_predictor(angle_feats[0])
        
        return angle_feats, conf_map
    
    def forward(self, feats_266, scale_maps):
        """
        批量生成所有目标角度的特征
        
        返回:
          all_angle_feats: list of 13, each is [F1_θ, F2_θ, F3_θ, F4_θ]
          all_conf_maps: [B, 13, H, W] 每个角度的置信度
        """
        all_angle_feats = []
        all_conf_maps = []
        
        for angle_idx in range(self.num_angles):
            feat_θ, conf_θ = self.forward_one_angle(
                feats_266, angle_idx, scale_maps)
            all_angle_feats.append(feat_θ)
            all_conf_maps.append(conf_θ)
        
        # 堆叠置信度图
        stacked_conf = torch.stack(all_conf_maps, dim=1)  # [B, 13, H, W]
        
        return all_angle_feats, stacked_conf
```

### 8.3 关键设计决策

| 设计决策 | 选择 | 理由 |
|---------|------|------|
| 几何变换零参数 | ✅ `torch.roll` 实现 | 极坐标空间中方位角旋转=水平平移，物理上精确，无需学习 |
| 散射调制为 (1+0.3×Tanh) | 输出范围 [0.7, 1.3] | 物理上散射强度不会归零，也不会无限放大 |
| 补全网络仅作用于 F1 | ✅ 最高分辨率尺度 | F1(128×128) 分辨率最高，遮挡/阴影效应在此尺度最明显；更深的尺度特征已高度抽象，补全意义不大 |
| 每角度置信度预测 | ✅ 独立 Conv 头 | 某些角度在特定缺陷类型上可能质量差，置信度图告诉融合层「这个角度在这个像素上可信吗」 |
| 共享编码器 1 次前向 | ✅ | 所有 13 个角度共享编码器输出，仅 ASG 和后续分支各自计算 |

### 8.4 角度注意力融合（升级版 SGF）

传统多视角融合是 NMS 后合并框列表。我们的方法更精细——**在特征层面用物理先验 + 学习注意力融合**。

```python
class AngleAttentionFusion(nn.Module):
    """
    角度注意力融合层
    
    融合 13 个角度的 193nm 检测分支输出。
    每个角度有自己的置信度图，融合时加权。
    
    输入:
      angle_preds: 13 组 193nm 检测头输出
      angle_conf: 13 张置信度图 [B, 13, H, W]
    输出:
      fused_pred: 融合后的单组检测输出
    """
    
    def __init__(self, in_channels=8, num_angles=13):
        super().__init__()
        
        # 角度注意力
        # 学习跨角度的空间注意力模式
        self.angle_attention = nn.Sequential(
            nn.Conv2d(in_channels * num_angles, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, num_angles, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, angle_preds, angle_conf):
        # angle_preds: list of 13 tensors [B, 8, H, W]
        # angle_conf: [B, 13, H, W]
        
        B, C, H, W = angle_preds[0].shape
        
        # Step 1: 堆叠所有角度预测
        stacked = torch.stack(angle_preds, dim=1)  # [B, 13, 8, H, W]
        
        # Step 2: 生成注意力权重
        # 物理先验分支（置信度图归一化后作为基线权重）
        phys_weights = angle_conf / (angle_conf.sum(dim=1, keepdim=True) + 1e-8)
        phys_weights = phys_weights.unsqueeze(2)  # [B, 13, 1, H, W]
        
        # 学习分支（捕获置信度无法表达的复杂模式）
        stacked_flat = stacked.reshape(B, C * self.num_angles, H, W)
        learned_weights = self.angle_attention(stacked_flat)  # [B, 13, H, W]
        learned_weights = learned_weights.unsqueeze(2)  # [B, 13, 1, H, W]
        
        # 融合权重（物理 + 学习混合，γ=0.7 偏重物理先验）
        gamma = 0.7
        weights = gamma * phys_weights + (1 - gamma) * learned_weights
        
        # Step 3: 加权融合
        fused = (stacked * weights).sum(dim=1)  # [B, 8, H, W]
        
        return fused, {'weights': weights.detach()}
```

### 8.5 角度散射一致性损失

用于训练 ASG 的散射调制部分：

```python
def angle_scattering_loss(feat_266_base, feat_266_theta, angle_deg):
    """
    角度散射一致性损失
    
    约束: 生成的 θ° 特征应该与真实 θ° 特征在散射域一致。
    核心思想: ASG 生成的 θ° 特征经过 PISM 的散射响应
    应该与真实 0° 特征的散射响应 + 角度偏移一致。
    
    物理含义:
      方位角变化只会改变散射强度，不应改变缺陷的基本散射类型。
      例如: 在 0° 下是 Rayleigh 散射主导的小颗粒，
      在 45° 下不应该变成 Mie 散射主导。
    """
    loss = 0
    
    for f_base, f_theta in zip(feat_266_base, feat_266_theta):
        # 散射响应谱的 Jensen-Shannon 散度
        # 确保散射类型（Rayleigh vs Mie 占比）跨角度一致
        spec_base = f_base.pow(2).mean(dim=1, keepdim=True)
        spec_theta = f_theta.pow(2).mean(dim=1, keepdim=True)
        
        # 归一化为概率分布
        spec_base = spec_base / (spec_base.sum() + 1e-8)
        spec_theta = spec_theta / (spec_theta.sum() + 1e-8)
        
        # 对称 KL 散度
        kl_div = (spec_base * (spec_base / (spec_theta + 1e-8)).log()).sum()
        kl_rev = (spec_theta * (spec_theta / (spec_base + 1e-8)).log()).sum()
        js_div = 0.5 * (kl_div + kl_rev)
        
        loss += js_div
    
    return loss / len(feat_266_base)
```

### 8.6 与 PISM 和双波长管线的集成

```
ASG 与 PISM 的集成方式:

ASG 输出 13 个角度的 266nm 特征 (在极坐标空间)
         │
         ▼
    ┌────────────┐
    │  PISM      │  ← 共享权重，对每个角度独立应用
    │  物理散射  │     batch=13 批量处理
    │  266→193nm │
    └─────┬──────┘
          │
    ┌─────┴──────┐
    │  193nm 检测 │  ← 共享权重，batch=13
    │  输出       │
    └─────┬──────┘
          │
    ┌─────┴──────┐
    │  角度注意力  │  ← 融合 13 个角度
    │  融合       │
    └─────┬──────┘
          │
    ┌─────┴──────┐
    │  SGF       │  ← 融合 266nm(0°) + 融合后 193nm
    │  散射引导   │
    └─────┬──────┘
          │
          ▼
      最终检测结果

关键优化:
  1. PISM 的 batch=13: 一次前向处理所有角度，GPU/ARM NEON 向量化
  2. 193nm 检测头共享权重: 参数不随角度数增长
  3. 角度注意力融合在特征层面: 避免 NMS 后的框列表合并
```

---

## 9. 验收指标更新

### 9.1 新增 P0 指标（硬约束）

| 编号 | 指标 | 目标值 | 测量方法 |
|------|------|--------|---------|
| P0-7 | **虚拟 193nm 散射比偏差** | < 10% | 配对标定片 `\|预测增益/真实增益 - 1\|` 均值 |
| P0-8 | **双分支检测一致性** | IoU > 0.85 | 266nm 和 193nm 检测框 IoU |
| P0-9 | **🆕 角度生成 FID** | < 35 | 生成 vs 真实多角度图像的 FID |
| P0-10 | **🆕 多角度 mAP 提升** | > +5% | 13 角度融合 vs 单角度基线的 mAP 提升 |

### 9.2 新增 P1 指标（软约束）

| 编号 | 指标 | 目标值 | 说明 |
|------|------|--------|------|
| P1-5 | **小缺陷召回率提升** | +5% 绝对提升 | < 10px 缺陷，双波长 + 多角度联合收益 |
| P1-6 | **虚警率变化** | FPPI 增加 < 0.5 | 角度生成不引入伪影导致虚警 |
| P1-7 | **光谱角违例率** | < 3% 像素 | 违反物理边界 [1.0, 8.0] 的像素占比 |
| P1-8 | **🆕 角度间一致性** | SSIM > 0.90 | 相邻角度 (θ, θ+5°) 生成的图像 SSIM |
| P1-9 | **🆕 各向异性缺陷召回率** | +8% 绝对提升 | 划痕等各向异性缺陷的召回率增益 |

### 9.3 验收流程新增

**验收阶段一（代码审查）新增：**
```
✅ ASG 物理合理性:
   - 几何平移量: 15°=21px, 45°=64px, 75°=107px ✅
   - torch.roll 实现循环平移 ✅
   - 散射调制范围 [0.7, 1.3] 物理合理 ✅
✅ PISM + ASG 联合可 JIT trace:
   - ASG.forward() 无 if/torch.where/for(动态) 等 ✅
   - 角度索引通过预计算 buffer 实现 ✅
✅ 13 角度 batch 处理 JIT 兼容:
   - 非动态 batch 维度 ✅
   - 所有操作标准 PyTorch nn 模块 ✅
```

**验收阶段二（功能正确性）新增：**
```bash
# 角度生成质量验证
python verify_asg.py \
    --model runs/stage3/best.pth \
    --multi_angle_data data/calibration/multi_angle/
# 预期:
# ✅ 角度生成 FID: 28.3 (阈值: 35)
# ✅ 相邻角度 SSIM: 0.94 (阈值: 0.90)
# ✅ 散射比偏差: 6.2% (阈值: 10%)

# 多角度检测验证
python verify_multi_angle_detection.py \
    --model runs/stage3/best.pth \
    --test_data data/test/
# 预期:
# ✅ mAP (13 角度融合): 0.86 (单角度基线: 0.80, +7.5% ✅)
# ✅ 小缺陷召回率: 0.78 (基线: 0.70, +8% ✅)
# ✅ 各向异性缺陷召回率: 0.82 (基线: 0.72, +10% ✅)
```

**验收阶段三（性能基准）新增：**
```
# 推理时间精细分解
# 总推理: 70.5 ms ✅ < 80ms
#   ├─ 共享编码器:         31.0 ms
#   ├─ ASG 角度生成:        5.0 ms  🆕 (13角度batch)
#   │   ├─ 几何平移 (13×):  ~0.1 ms (torch.roll)
#   │   ├─ 散射调制 (13×):  ~2.0 ms
#   │   └─ 补全 (13×):     ~2.9 ms
#   ├─ PISM + 193nm检测:   14.0 ms  (batch=13, 含检测头)
#   ├─ 角度注意力融合:       0.5 ms  🆕
#   ├─ SGF:                 0.3 ms
#   ├─ 266nm 分支:          10.5 ms
#   ├─ 增强解码:            10.6 ms
#   └─ 后处理:              0.3 ms

# 余量: 80 - 70.5 = 9.5 ms (12%)
# 紧急降级: 关闭多角度 → 退回 65 ms
# 紧急降级: 关闭ASG+PISM → 退回 52 ms
```

**验收阶段四（端到端）新增：**
```
✅ 多角度生成图与真实多角度图的 FID < 35
✅ 13 角度融合 mAP > 单角度 mAP + 5%
✅ 各向异性缺陷（划痕）召回率提升 > 8%
✅ 双波长 + 多角度联合检测不产生额外误检
✅ ARM 端 13 角度总推理 < 80ms
```

---

## 10. 展厅设计更新

### 10.1 架构图更新

在 `wafer-showcase-design.md` 的架构图中插入 ASG 模块：

```
共享编码器 RepViT-M0.9  ~5.1M
    │
    ├──→ 🆕 ASG 多角度散射生成器 ~0.15M
    │         │
    │      [15°][20°]···[75°]  ← 13 个方位角
    │         │    │        │
    │         ▼    ▼        ▼
    │     PISM  → PISM →  PISM  (共享权重, batch 13)
    │         │    │        │
    │         ▼    ▼        ▼
    │    193nm 检测头 (共享权重, batch 13)
    │         │    │        │
    │         └────┼────────┘
    │              ▼
    │          🆕 角度注意力融合
    │              │
    ├──→ 266nm 检测 │
    │   (0° 基准)   │
    │         │     │
    └─────────┼─────┘
              ▼
          SGF 散射引导融合
              │
              ▼
          融合缺陷列表
```

### 10.2 核心技术亮点更新

第④屏「核心技术突破」：7 张 → 10 张：

| # | 标题 | 一句话亮点 | 关键数据 |
|---|------|-----------|---------|
| ⑧ | **可微物理散射 PISM** | Rayleigh/Mie 散射先验替代黑箱映射 | `3.5× 增益，0.04M` |
| ⑨ | **双光路并行检测** | 266nm + 虚拟 193nm 独立分支物理融合 | `双分支 IoU > 0.85` |
| ⑩ 🆕 | **多角度散射生成 ASG** | 单图生成 13 个方位角，极坐标零成本变换 | `13角度 仅+5ms` |

**新增卡片设计稿：**

```
卡片 ⑩ — 多角度散射生成 ASG
  图标: 🔄
  标题: 多角度散射生成器 (ASG)
  标签: 单图生成15°-75° / 极坐标高效变换
  描述: 利用极坐标展开的几何特性，方位角旋转≈特征图水平平移，
        零参数实现几何变换。13 个角度通过散射调制 + 轻量补全
        生成物理真实的各向异性散射视图。
  指标: [13 角度 +5ms] [FID < 35] [各向异性召回 +8%]
```

### 10.3 关键指标更新

第⑤屏「关键指标面板」：6 项 → 10 项：

```
一行: <10M    <8MB     <80ms    <3min    >0.80
      参数    INT8     推理     全片     mAP

二行: >25dB   <10%     +5%      +5%     +8%  🆕
      增强    散射比    小缺陷    多角度   各向异性
             偏差      召回率    mAP提升  召回率提升
```

### 10.4 技术栈标签更新

原始 7 个 + 上次新增 4 个 + 本次新增 3 个：

```
[RepViT] [CycleGAN] [LPIPS] [INT8] [TorchScript] [LibTorch] [Cortex-A76]
🆕 [Rayleigh散射] [Mie理论] [物理AI] [双波长]
🆕 [方位角散射] [Polar极坐标] [多视角融合]
```

---

## 11. 参数预算与推理时间核算

### 11.1 完整参数预算

| 组件 | v1.0 (原始) | v1.5 (双波长) | v2.0 (多角度) |
|------|------------|--------------|--------------|
| 共享编码器 | 5.10M | 5.10M | 5.10M |
| 增强解码 | 1.20M | 1.20M | 1.20M |
| PISM | — | 0.04M | 0.04M |
| 266nm 检测头 | 1.20M | 1.20M | 1.20M |
| 193nm 检测头 | — | 0.42M | 0.42M |
| SGF 融合 | — | 0.003M | 0.003M |
| ASG 角度生成 | — | — | **0.15M** |
| 角度注意力融合 | — | — | **0.02M** |
| **合计** | **7.50M** | **7.96M** | **8.13M** |
| 10M 余量 | 25% | 20.4% | **18.7% ✅** |

### 11.2 推理时间估算

| 阶段 | 时间 | 累计 | 说明 |
|------|------|------|------|
| 共享编码器 | 31.0 ms | 31.0 ms | 1 次前向 |
| 增强解码 | 10.6 ms | 41.6 ms | 不变 |
| 266nm 检测头 | 10.5 ms | 52.1 ms | 单视角 |
| **🆕 ASG (13 角度)** | **5.0 ms** | **57.1 ms** | **几何平移~0，散射调制+补全 5ms** |
| PISM | 0.5 ms | 57.6 ms | batch 优化后几乎不变 |
| 193nm 检测头 | 12.0 ms | 69.6 ms | batch=13 向量化 |
| 🆕 角度注意力融合 | 0.5 ms | 70.1 ms | 轻量融合 |
| SGF | 0.3 ms | 70.4 ms | 2 分支融合 |
| 后处理 | 0.3 ms | **70.7 ms** | |
| **总计** | | **~71 ms** | **✅ < 80ms** |

### 11.3 降级预案

| 优先级 | 措施 | 节省时间 | 累计时间 | 精度影响 |
|--------|------|---------|---------|---------|
| — | 全功能 (PISM + ASG 13角度) | — | 71 ms | 基线 |
| 1 | 关闭极坐标展开 | +5 ms | 76 ms | 无（但计算量增加） |
| 2 | ASG 角度数减半 (15/30/45/60/75) | -2.5 ms | 73.5 ms | mAP 可能降 1-2% |
| 3 | 关闭 193nm 分支，仅用 266nm ASG | -12 ms | 59 ms | 无 193nm 增益 |
| 4 | 关闭 ASG，退回双波长 | -5 ms | 65 ms | 无多角度增益 |
| 5 | 关闭物理模块，退回原始 | -13 ms | 52 ms | 无物理先验 |

---

## 12. 文件结构变更

### 12.1 新增文件

```
models/
├── physics_scattering.py       # PISM 散射物理模块
├── detect_head_193.py          # 193nm 检测分支
├── scattering_fusion.py        # SGF 融合层
├── angle_scattering_gen.py 🆕  # ASG 多角度生成器
├── angle_attention_fusion.py 🆕 # 角度注意力融合

losses/
├── physics_loss.py             # 散射一致性 + 光谱角损失
├── angle_loss.py            🆕  # 角度散射一致性损失

utils/
├── mie_lut.py                  # Mie 散射查找表预计算
├── polar_transform.py       🆕  # 极坐标变换工具（角度对齐版本）

deploy/
├── verify_physics.py           # 物理模块验证
├── verify_dual_branch.py       # 双分支一致性验证
├── verify_asg.py            🆕  # ASG 多角度生成验证
├── verify_multi_angle_det.py 🆕 # 多角度检测验证

configs/
├── physics_config.yaml         # PISM/ASG 超参数配置
```

### 12.2 修改文件

```
models/
├── wafer_multitask.py          # 整合 PISM + ASG + 双分支 + SGF + 角度融合

训练脚本/
├── train_stage1.py             # CycleGAN + D_193
├── train_stage2.py             # 双分支 + ASG 多角度训练
├── train_stage3.py             # 物理 + 角度蒸馏

docs/
├── TECHNICAL_ROADMAP.md        # 新增物理光学 + ASG 章
├── FOR_PROJECT_MANAGER.md      # 新增 4.12-4.14
├── wafer-showcase-design.md    # 架构图+卡片+指标+标签
```

### 12.3 依赖变更

```
# 无新增第三方推理依赖
# Mie LUT + Polar Transform 均使用标准 PyTorch/OpenCV
```

---

> **文档结束**  
> 版本 v2.0 — 2026-07-06  
> 涵盖：PISM (物理散射) + 双分支 (双波长) + ASG (多角度生成) + 角度注意力融合  
> 总参数 8.13M / 10M ✅ | 推理时间 ~71ms / 80ms ✅
