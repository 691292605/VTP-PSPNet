# VTP + PSPNet 语义分割

基于生成式视觉分词大模型 VTP-Large 与 PSPNet 的语义分割实现，并在 PASCAL VOC 2012 上与判别式模型 DINOv3 进行了对比分析。

## 结果

| 方法 | mIoU | PA | 备注 |
|------|------|-----|------|
| VTP-Large + PSPNet (Baseline) | 80.7% | — | 冻结 Backbone |
| VTP-Large + PSPNet (+ DenseCRF) | — | — | 边缘质量显著改善 |
| VTP-Small + PSPNet (1024×1024, 解冻8层) | 74.7% | — | 边缘更锐利但全局 mIoU 下降 |

### 与其他方法对比

| 方法 | mIoU |
|------|------|
| FCN | 62.2% |
| DeepLabV1 | 71.6% |
| DPN | 74.1% |
| PSPNet (原论文) | 82.6% |
| **DINOv3 + PSPNet (本项目)** | **85.15%** |
| **VTP-Large + PSPNet (本项目)** | **80.7%** |

> 注：FCN、DeepLab、DPN、PSPNet 数据引自 PSPNet 原论文在 VOC 2012 测试集上的结果；本项目结果为 VOC 2012 验证集结果。

## 判别式 vs 生成式对比

本项目对比了两种视觉预训练范式对下游语义分割任务的影响：

| 特性 | DINOv3 (判别式) | VTP (生成式) |
|------|-----------------|--------------|
| 预训练方式 | 自监督自蒸馏 | CLIP + SSL + 像素重建联合优化 |
| 特征特点 | 前景与背景清晰剥离，边界清晰 | 保留纹理细节与空间高频信息 |
| 分割边界 | 锐利 | 稍模糊（可通过 DenseCRF 优化） |
| 遮挡/复杂结构 | 一般 | 更鲁棒 |
| mIoU | 85.15% | 80.7% |

## 架构概述

```
Input Image (512×512)
    │
    ▼
┌──────────────────────┐
│  VTP-Large Backbone  │  生成式预训练 (完全冻结)
│  → 4 层中间特征       │
│  → 均值融合           │
└─────────┬────────────┘
          │
          ▼
    ┌─────────────┐
    │ FeatureAdapter │  可选 (1×1 Conv + 3×3 Conv)
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   PPM 模块   │  金字塔池化 (1,2,3,6)
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   分类头      │  Conv + BN + ReLU + Dropout + Conv
    └──────┬──────┘
           │
           ▼
    分割掩码 (512×512)
```

## 消融实验

### DenseCRF 后处理

引入 DenseCRF 后处理可在零额外训练成本下优化生成式模型的边缘模糊问题，利用原图空间色彩先验抑制局部分类噪声。

### VTP-Small 高分辨率微调

将 Backbone 替换为 VTP-Small，输入分辨率提升至 1024×1024，解冻最后 8 层进行微调。视觉上边界更锐利，但由于 VTP-Small 感受野有限，全局 mIoU 反而下降。

### PCA 特征可视化

通过 PCA 降维可视化发现，VTP 的特征呈现出与 DINOv3 截然不同的分布：DINOv3 将前景与背景清晰剥离，而 VTP 保留了更丰富的纹理与几何细节。

## 项目结构

```
├── model.py              # VTP Backbone + PSPNet 模型定义
├── data.py               # PASCAL VOC 数据集加载
├── train.py              # 训练入口 (冻结策略 + 梯度累积)
├── loss.py               # CE+Dice / Focal+Dice 损失函数
├── predict.py            # 推理与可视化
├── predict_compare.py    # 多模型对比推理
├── miou.py               # mIoU 评估
├── PCA.py                # PCA 特征可视化
├── CRF_compare.py        # DenseCRF 消融对比
└── VTP_main/             # VTP 模型源码
```

## 依赖

- Python 3.x
- PyTorch
- torchvision
- NumPy
- PIL (Pillow)
- tqdm
- safetensors
- pydensecrf (DenseCRF 后处理)

## 使用方法

### 数据准备

下载 PASCAL VOC 2012 数据集，放置于 `./data/VOCdevkit/VOC2012/` 目录下。

### 训练

```bash
python train.py
```

### 推理与对比

```bash
python predict.py              # 单模型推理
python predict_compare.py     # 多模型对比
```

### 消融实验

```bash
python CRF_compare.py         # DenseCRF 对比
python PCA.py                 # PCA 特征可视化
```

## 参考

- [VTP: Towards Scalable Pre-training of Visual Tokenizers for Generation](https://arxiv.org/abs/2512.13687)
- [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
- [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

