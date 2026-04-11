# 心脏超声切面分类模型 - 项目文档

---

## 1 项目概述

### 1.1 项目原理与背景

心脏超声（超声心动图）是评估心脏功能的重要影像学检查手段，不同的扫描切面（view）提供不同角度的心脏Structural信息。进行Automatic分类对于超声质量控制、工作流效率和标准化数据收集具有重要价值。本项目基于USF-MAE（Ultrasound-Specific Feature Masked Autoencoder）预训练模型，实现了一个Automatic的心脏超声切面分类系统。

传统的监督学习方法需要大量标注数据，成本较高。USF-MAE是一种专门针对超声图像设计的自监督预训练模型，通过在43个超声数据集上进行masked autoencoder预训练，学习通用的心脏超声特征表示。预训练完成后，使用下游任务的少量标注数据进行微调，即可获得良好的分类性能。本项目基于USF-MAE官方提供的预训练数据，进一步进行微调，完成了一个心脏超声切面分类模型。

### 1.2 任务目标

自动识别心脏超声图像的切面类型，帮助医生快速分类和归档超声检查结果。

### 1.3 支持的类别

#### 多分类任务（7类）

| 编号 | 类别 | 英文名称 | 说明 |
|------|------|----------|------|
| 0 | A4C | Apical Four-Chamber | 心尖四腔心切面 |
| 1 | PL | Parasternal Long-axis | 胸骨旁长轴切面 |
| 2 | PSAV | Parasternal Short-axis Aortic Valve | 胸骨旁短轴-主动脉瓣切面 |
| 3 | PSMV | Parasternal Short-axis Mitral Valve | 胸骨旁短轴-二尖瓣切面 |
| 4 | Random | Random | 随机图像（非标准切面） |
| 5 | SC | Subcostal | 剑突下四腔心切面 |
| 6 | A2C | Apical Two-Chamber | 心尖两腔心切面（来自CAMUS） |

#### 二分类任务（A2C vs A4C）

| 编号 | 类别 | 英文名称 | 说明 |
|------|------|----------|------|
| 0 | A2C | Apical Two-Chamber | 心尖两腔心切面 |
| 1 | A4C | Apical Four-Chamber | 心尖四腔心切面 |

> **注意**：二分类任务仅使用 CAMUS 数据集，区分 A2C（心尖两腔心）和 A4C（心尖四腔心）两种切面。

### 1.4 技术规格

| 项目 | 规格 |
|------|------|
| 预训练模型 | USF-MAE (ViT-Base) |
| 输入尺寸 | 224 × 224 |
| 骨干网络 | Vision Transformer (ViT-Base) |
| 隐藏维度 | 768 |
| Transformer层数 | 12 |
| 注意力头数 | 12 |
| Patch Size | 16 × 16 |
| 分类头 | 全局平均池化 + Linear(768→num_classes) |
| 支持任务 | 多分类(7类) / 二分类(A2C vs A4C) |

---

## 2 数据集说明

### 2.1 CACTUS 数据集

CACTUS（Classification of Acquisitions in Transthoracic UltraSound）是一个公开的心脏超声切面分类数据集。

**数据集结构：**

```
CACTUS/Images Dataset/
├── A4C/      (7,422 张)   ← 心尖四腔心
├── PL/       (6,102 张)   ← 胸骨旁长轴
├── PSAV/     (5,832 张)   ← 胸骨旁短轴-主动脉瓣
├── PSMV/     (6,014 张)   ← 胸骨旁短轴-二尖瓣
├── Random/   (6,021 张)   ← 随机图像
└── SC/       (6,345 张)   ← 剑突下四腔心
```

| 项目 | 说明 |
|------|------|
| 图像格式 | JPEG (.jpg) |
| 文件命名 | `{患者ID}_frame_{帧号}_v2.jpg` |
| 总图像数 | 37,726 张 |
| 类别数 | 6 类 |

### 2.2 CAMUS 数据集

CAMUS 数据集是一个公开的心脏超声心动图分割数据集，包含500名患者的超声序列数据。

**数据集结构：**

```
CAMUS/
└── database_nifti/
    ├── patient0001/
    │   ├── patient0001_2CH_sequence.nii.gz
    │   ├── patient0001_2CH_half_sequence.nii.gz
    │   ├── patient0001_4CH_sequence.nii.gz
    │   └── patient0001_4CH_half_sequence.nii.gz
    ├── patient0002/
    └── ...
```

| 项目 | 说明 |
|------|------|
| 数据格式 | NIfTI (.nii.gz) |
| 患者数 | 500 名 |
| 序列类型 | 2CH (两腔心) + 4CH (四腔心) |
| 每患者帧数 | 约 20 帧 |
| 总帧数 | 约 10,000 帧 |

**使用说明：**
- CAMUS 数据主要用于二分类任务（A2C vs A4C）
- 2CH 序列对应 A2C（心尖两腔心）
- 4CH 序列对应 A4C（心尖四腔心）
- 首次使用时会自动生成缓存（`_camus_cache/images/`）

### 2.3 数据分布

#### CACTUS 数据集（多分类任务）

| 类别 | 样本数 | 占比 |
|------|--------|------|
| A4C | 7,422 | 19.67% |
| PL | 6,102 | 16.17% |
| PSAV | 5,832 | 15.46% |
| PSMV | 6,014 | 15.93% |
| Random | 6,021 | 15.96% |
| SC | 6,345 | 16.82% |
| **总计** | **37,726** | **100%** |

#### CAMUS 数据集（二分类任务）

| 类别 | 说明 |
|------|------|
| A2C | 心尖两腔心（2CH 序列） |
| A4C | 心尖四腔心（4CH 序列） |

> 二分类任务的样本数取决于 CAMUS 数据的实际帧数（每个患者约 20 帧 × 500 患者 ≈ 10,000 帧）

---

## 3 模型架构

### 3.1 整体架构

```
输入图像 (224×224×3)
    ↓
USF-MAE Encoder (ViT-Base)
    ├── Patch Embedding: Conv2d(3→768, 16×16)
    ├── 位置编码: 可学习位置编码 (197×768)
    ├── Transformer Blocks × 12
    │   └── Multi-Head Self Attention
    │   └── Feed Forward Network
    └── LayerNorm
    ↓
全局平均池化 (Global Average Pooling)
    ↓
分类头 (Dropout + Linear: 768→num_classes)
    ↓
Softmax → 类别概率
```

其中 `num_classes` 根据任务类型自动调整：
- 多分类任务：7
- 二分类任务（A2C vs A4C）：2

### 3.2 USF-MAE 预训练

**MAE（Masked Autoencoder）核心设计：**

```
原始图像 → 分割为 14×14 = 196 个 patches
    ↓
随机遮盖 75% 的 patches (仅保留 25%)
    ↓
编码器处理可见的 25% patches → 潜在表示
    ��
解码器重建被遮盖的 75% patches
    ↓
学习图像的通用语义表示
```

**MAE 特性：**
- 非对称编码器-解码器架构
- 高遮盖率（75%）促使模型学习更丰富的语义特征
- 预训练数据：43 个心脏超声数据集
- 预训练轮次：100 epochs

### 3.3 预训练权重

| 文件 | 说明 |
|------|------|
| `USF-MAE pretrained/USF-MAE_full_pretrain_43dataset_100epochs.pt` | 预训练权重（约 428MB） |

---

## 4 环境配置

### 4.1 依赖列表

```
torch>=1.8.0
torchvision>=0.9.0
Pillow>=8.0.0
numpy>=1.19.0
scikit-learn>=0.24.0
nibabel>=3.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
PyYAML>=5.4.0
tqdm>=4.60.0
```

### 4.2 安装步骤

```bash
pip install -r requirements.txt
```

### 4.3 项目结构

```
HeartEcho_Classification_Model/
├── data/                          # 数据加载模块
│   ├── __init__.py
│   ├── dataset.py                 # 统一数据管道
│   ├── cactus_loader.py           # CACTUS 数据加载
│   └── camus_loader.py           # CAMUS 数据加载
├── models/                       # 模型定义模块
│   ├── __init__.py
│   ├── backbone.py             # USF-MAE Backbone
│   └── classifier.py           # 分类模型
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── trainer.py             # 训练逻辑
│   ├── evaluate.py           # 评估指标
│   └── logger.py             # 日志记录
├── inference/                   # 推理模块
│   ├── __init__.py
│   ├── cli.py
│   ├── predict.py
│   └── classifier.py
├── configs/                    # 配置文件
│   ├── train_config.yaml
│   └── eval_config.yaml
├── CACTUS/                    # CACTUS 数据集
│   └── Images Dataset/
├── USF-MAE pretrained/         # 预训练权重
│   └── USF-MAE_full_pretrain_43dataset_100epochs.pt
├── train.py                     # 训练入口
├── eval.py                     # 评估入口
└── requirements.txt            # 依赖列表
```

---

## 5 训练流程

### 5.1 数据预处理

**训练时数据增强：**

```python
transforms.Compose([
    RandomResizedCrop(224, scale=(0.5, 2.0)),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation(degrees=(0, 90)),
    ColorJitter(brightness=0.2, contrast=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**验证/测试时：**

```python
transforms.Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 5.2 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task_type` | multi_class | 任务类型：multi_class(7类) / binary(二分类) |
| `--camus_data` | (见下方) | CAMUS数据集路径（二分类任务必需） |
| `--batch_size` | 32 | 批大小 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--weight_decay` | 0.01 | 权重衰减 |
| `--val_split` | 0.15 | 验证集比例 |
| `--test_split` | 0.15 | 测试集比例 |
| `--kfold` | 0 | K折交叉验证（0=不使用，5=5折） |
| `--freeze_backbone` | false | 是否冻结 backbone |
| `--scheduler` | cosine | 学习率调度器 |
| `--use_amp` | true | 混合精度训练 |
| `--gradient_clip` | 1.0 | 梯度裁剪阈值 |
| `--early_stopping_patience` | 10 | 早停耐心值 |
| `--dropout` | 0.1 | Dropout 概率 |

> **CAMUS 数据集默认路径**：`D:/SRTP_Project__DeepLearning/project/Resources/database_nifti`

### 5.3 训练命令

#### 多分类任务（7类）

**基本训练：**

```bash
python train.py --cactus_data "CACTUS/Images Dataset" --batch_size 32 --epochs 50
```

**使用预训练权重：**

```bash
python train.py \
    --cactus_data "CACTUS/Images Dataset" \
    --pretrained "USF-MAE pretrained/USF-MAE_full_pretrain_43dataset_100epochs.pt" \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4
```

**冻结 Backbone 训练：**

```bash
python train.py \
    --cactus_data "CACTUS/Images Dataset" \
    --pretrained "USF-MAE pretrained/USF-MAE_full_pretrain_43dataset_100epochs.pt" \
    --freeze_backbone \
    --lr 1e-3
```

**使用配置文件：**

```bash
python train.py --config configs/train_config.yaml
```

#### 二分类任务（A2C vs A4C）

**基本训练：**

```bash
python train.py \
    --task_type binary \
    --camus_data "D:/SRTP_Project__DeepLearning/project/Resources/database_nifti" \
    --epochs 50
```

**使用 K-Fold 交叉验证：**

```bash
python train.py \
    --task_type binary \
    --camus_data "D:/SRTP_Project__DeepLearning/project/Resources/database_nifti" \
    --kfold 5 \
    --epochs 30
```

**冻结 Backbone 训练：**

```bash
python train.py \
    --task_type binary \
    --camus_data "D:/SRTP_Project__DeepLearning/project/Resources/database_nifti" \
    --freeze_backbone \
    --lr 1e-3
```

### 5.4 输出文件

#### 多分类任务

**模型保存：**

```
checkpoints/
├── best_model.pth           # 最佳模型（验证集准确率最高）
├── last_model.pth           # 最后模型
└── checkpoint_epoch_*.pth   # 定期保存
```

**日志保存：**

```
logs/{experiment_name}/
├── training.log         # 训练日志
├── config.json        # 配置记录
└── history.json      # 训练历史
```

#### 二分类任务

**模型保存：**

```
checkpoints/binary/
├── best_model.pth           # 最佳模型
├── last_model.pth           # 最后模型
└── kfold/                   # K-Fold 交叉验证结果
    ├── fold_1.pth
    ├── fold_2.pth
    ├── fold_5.pth
    ├── best.pth             # 最佳折模型
    └── kfold_results.json   # K-Fold 结果汇总
```

**日志保存：**

```
logs/binary/{experiment_name}/
├── training.log         # 训练日志
├── config.json        # 配置记录
└── history.json      # 训练历史
```

---

## 6 评估流程

### 6.1 评估命令

**基本评估：**

```bash
python eval.py --checkpoint checkpoints/best_model.pth --data "CACTUS/Images Dataset"
```

**生成可视化：**

```bash
python eval.py \
    --checkpoint checkpoints/best_model.pth \
    --plot_cm \
    --plot_roc
```

**仅测试集评估：**

```bash
python eval.py --checkpoint checkpoints/best_model.pth --test_only
```

**全数据集评估（K-Fold 模型）：**

```bash
python eval.py --checkpoint checkpoints/kfold/best.pth --full_data --plot_cm --plot_roc
```

#### 二分类任务（A2C vs A4C）评估

**基本评估：**

```bash
python eval.py \
    --task_type binary \
    --checkpoint checkpoints/binary/best_model.pth \
    --camus_data "D:/SRTP_Project__DeepLearning/project/Resources/database_nifti"
```

**生成可视化：**

```bash
python eval.py \
    --task_type binary \
    --checkpoint checkpoints/binary/best_model.pth \
    --camus_data "D:/SRTP_Project__DeepLearning/project/Resources/database_nifti" \
    --plot_cm \
    --plot_roc
```

**K-Fold 模型评估：**

```bash
python eval.py \
    --task_type binary \
    --checkpoint checkpoints/binary/kfold/best.pth \
    --camus_data "D:/SRTP_Project__DeepLearning/project/Resources/database_nifti" \
    --full_data
```

### 6.2 评估指标

| 指标 | 说明 |
|------|------|
| Accuracy | 总体准确率 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1-score | F1 分数 |
| AUC | ROC 曲线下面积 |

### 6.3 评估输出

```
results/
├── test_metrics.json           # 测试集指标
├── val_metrics.json            # 验证集指标
├── test_confusion_matrix.png  # 测试集混淆矩阵
├── test_roc_curves.png        # 测试集 ROC 曲线
├── val_confusion_matrix.png  # 验证集混淆矩阵
└── val_roc_curves.png        # 验证集 ROC 曲线
```

---

## 7 训练结果

### 7.1 评估指标

模型在 CACTUS 完整数据集上的评估结果：

| 指标 | 值 |
|------|------|
| Accuracy | 99.26% |
| Precision | 99.27% |
| Recall | 99.26% |
| F1-score | 99.26% |
| Precision (Macro) | 99.21% |
| Recall (Macro) | 99.18% |
| F1-score (Macro) | 99.19% |
| AUC | 0.9999 |

### 7.2 混淆矩阵

| 预测→ | A4C | PL | PSAV | PSMV | Random | SC | A2C |
|-------|-----|-----|------|------|--------|-----|-----|
| A4C | 7416 | 0 | 0 | 0 | 6 | 0 | 0 |
| PL | 0 | 6082 | 0 | 0 | 20 | 0 | 0 |
| PSAV | 0 | 0 | 5826 | 0 | 6 | 0 | 0 |
| PSMV | 8 | 0 | 2 | 5830 | 174 | 0 | 0 |
| Random | 43 | 3 | 8 | 25 | 5932 | 10 | 0 |
| SC | 11 | 0 | 0 | 0 | 34 | 6300 | 0 |
| A2C | 0 | 0 | 0 | 0 | 0 | 0 | 9268 |

### 7.3 各类别性能分析

| 类别 | 精确率 | 召回率 | F1-score | 样本数 |
|------|--------|--------|----------|--------|
| A4C | 99.92% | 99.92% | 99.92% | 7,422 |
| PL | 99.95% | 99.67% | 99.81% | 6,102 |
| PSAV | 99.83% | 99.90% | 99.86% | 5,832 |
| PSMV | 99.57% | 96.97% | 98.25% | 6,014 |
| Random | 96.10% | 98.52% | 97.29% | 6,021 |
| SC | 99.84% | 99.29% | 99.56% | 6,345 |
| A2C | 100.00% | 100.00% | 100.00% | 9,268 |

**分析：**
- A2C（心尖两腔心）分类效果最佳，达到 100% 准确率
- PSMV 和 Random 类别存在一定混淆，主要因为这两类图像特征较为相似
- 其他类别分类准确率均超过 99%

### 7.4 可视化结果

模型生成了以下可视化文件：

```
results/
├── full_metrics.json           # 完整评估指标
├── full_confusion_matrix.png    # 混淆矩阵可视化
└── full_roc_curves.png         # ROC 曲线可视化
```

---

## 8 K-Fold 交叉验证

### 8.1 使用方法

#### 多分类任务

**5 折交叉验证：**

```bash
python train.py \
    --cactus_data "CACTUS/Images Dataset" \
    --kfold 5 \
    --epochs 30
```

#### 二分类任务

**5 折交叉验证：**

```bash
python train.py \
    --task_type binary \
    --camus_data "D:/SRTP_Project__DeepLearning/project/Resources/database_nifti" \
    --kfold 5 \
    --epochs 30
```

### 8.2 输出

#### 多分类任务

```
checkpoints/kfold/
├── fold_1.pth          # Fold 1 模型
├── fold_2.pth          # Fold 2 模型
├── fold_5.pth          # Fold 5 模型
├── best.pth             # 最佳模型（验证准确率最高）
└── kfold_results.json  # 交叉验证结果
```

#### 二分类任务

```
checkpoints/binary/kfold/
├── fold_1.pth          # Fold 1 模型
├── fold_2.pth          # Fold 2 模型
├── fold_5.pth          # Fold 5 模型
├── best.pth             # 最佳模型（验证准确率最高）
└── kfold_results.json  # 交叉验证结果
```

### 8.3 评估 K-Fold 模型

```bash
python eval.py \
    --checkpoint checkpoints/kfold/best.pth \
    --full_data \
    --plot_cm \
    --plot_roc
```

---

## 9 推理预测

### 9.1 命令行推理

#### 多分类推理（7类）

```bash
python -m inference \
    --checkpoint checkpoints/best_model.pth \
    --input path/to/image.jpg
```

#### 二分类推理（A2C vs A4C）

```bash
python -m inference \
    --checkpoint checkpoints/binary/best_model.pth \
    --task_type binary \
    --input path/to/image.jpg
```

### 9.2 Python API

```python
from inference.predict import load_model, predict_single
from inference.transforms import get_val_transforms
from inference.constants import get_class_names

# 多分类推理
model = load_model('checkpoints/best_model.pth', task_type='multi_class', device='cuda')
transform = get_val_transforms(224)
result = predict_single(model, 'path/to/image.jpg', transform=transform, device='cuda', preprocessing_mode='auto')
print(result['predicted_class'])

# 二分类推理
model = load_model('checkpoints/binary/best_model.pth', task_type='binary', device='cuda')
result = predict_single(model, 'path/to/image.jpg', transform=transform, device='cuda', preprocessing_mode='auto')
print(result['predicted_class'])  # 自动使用 A2C/A4C 类别名称
```

### 9.3 推理模块详情

详细使用说明请参考 [inference/INFERENCE.md](inference/INFERENCE.md)

---

## 10 参考资料

- **USF-MAE 论文：** Chen L, Zeng Y, Chen S, et al. Benchmarking Self-Supervised Models for Cardiac Ultrasound View Classification[J]. arXiv:2602.15339, 2024.
- **USF-MAE 代码库：** https://github.com/Yusufii9/USF-MAE
- **CACTUS 数据集：** https://www.creatis.insa-lyon.fr/Challenge/camus

---

*文档版本：v2.0*  
*最后更新：2026.4.5*