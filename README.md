# 心脏超声切面分类模型 - 使用指南

## 项目概述

本项目实现了一个基于 USF-MAE 预训练模型的心脏超声切面分类系统，支持 7 类心脏切面分类任务。

### 支持的类别

| 类别 | 英文名称 | 说明 |
|------|----------|------|
| A4C | Apical Four-Chamber | 心尖四腔心 |
| PL | Parasternal Long-axis | 胸骨旁长轴 |
| PSAV | Parasternal Short-axis Aortic Valve | 胸骨旁短轴-主动脉瓣 |
| PSMV | Parasternal Short-axis Mitral Valve | 胸骨旁短轴-二尖瓣 |
| Random | Random | 随机图像 |
| SC | Subcostal | 剑突下四腔心 |
| A2C | Apical Two-Chamber | 心尖两腔心 (需CAMUS数据) |

---

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 依赖列表

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

---

## 数据准备

### 数据集结构

```
project/
├── CACTUS/
│   └── Images Dataset/
│       ├── A4C/      (7422 张)
│       ├── PL/       (6102 张)
│       ├── PSAV/     (5832 张)
│       ├── PSMV/     (6014 张)
│       ├── Random/   (6021 张)
│       └── SC/       (6345 张)
├── CAMUS/            (NIfTI格式，~10,000帧)
│   └── database_nifti/
│       ├── patient0001/
│       │   └── patient0001_2CH_half_sequence.nii.gz
│       ├── patient0002/
│       └── ...
└── USF-MAE pretrained/
    └── USF-MAE_full_pretrain_43dataset_100epochs.pt
```

### 您的 CAMUS 数据路径

您的 CAMUS 数据位于: `D:/SRTP_Project__DeepLearning/project/Resources/database_nifti/`

- 500 个患者文件夹 (patient0001 ~ patient0500)
- 每个患者的 `*_2CH_half_sequence.nii.gz` 包含约 15-25 帧
- 总计约 10,000 张 A2C 图像
- 图像尺寸不统一，需 resize 到 224×224

---

## 快速开始

### 训练模型

#### 基本训练

```bash
python train.py --cactus_data "CACTUS/Images Dataset" --batch_size 32 --epochs 50
```

#### 使用预训练权重

```bash
python train.py \
    --cactus_data "CACTUS/Images Dataset" \
    --pretrained "USF-MAE pretrained/USF-MAE_full_pretrain_43dataset_100epochs.pt" \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4
```

#### 冻结 Backbone 训练

```bash
python train.py \
    --cactus_data "CACTUS/Images Dataset" \
    --pretrained "USF-MAE pretrained/USF-MAE_full_pretrain_43dataset_100epochs.pt" \
    --freeze_backbone \
    --lr 1e-3
```

#### 使用配置文件

```bash
python train.py --config configs/config.yaml
```

### 评估模型

#### 基本评估

```bash
python eval.py --checkpoint checkpoints/best_model.pth --data "CACTUS/Images Dataset"
```

#### 生成可视化

```bash
python eval.py \
    --checkpoint checkpoints/best_model.pth \
    --plot_cm \
    --plot_roc
```

#### 仅测试集评估

```bash
python eval.py --checkpoint checkpoints/best_model.pth --test_only
```

---

## 命令行参数详解

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cactus_data` | `CACTUS/Images Dataset` | CACTUS数据集路径 |
| `--camus_data` | `CAMUS` | CAMUS数据集路径 |
| `--batch_size` | `32` | 批大小 |
| `--num_workers` | `4` | 数据加载线程 |
| `--val_split` | `0.15` | 验证集比例 |
| `--test_split` | `0.15` | 测试集比例 |
| `--img_size` | `224` | 输入图像尺寸 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pretrained` | 预训练权重路径 | 预训练模型路径 |
| `--num_classes` | `7` | 分类类别数 |
| `--freeze_backbone` | `False` | 是否冻结backbone |
| `--dropout` | `0.1` | Dropout概率 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | `50` | 训练轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--weight_decay` | `0.01` | 权重衰减 |
| `--scheduler` | `cosine` | 学习率调度器 |
| `--use_amp` | `True` | 混合精度训练 |
| `--gradient_clip` | `1.0` | 梯度裁剪 |
| `--early_stopping_patience` | `10` | 早停耐心值 |

### 输出参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint_dir` | `checkpoints` | 模型保存目录 |
| `--log_dir` | `logs` | 日志保存目录 |
| `--experiment_name` | 自动生成 | 实验名称 |
| `--seed` | `42` | 随机种子 |

---

## 输出说明

### 训练输出

```
checkpoints/
├── best_model.pth           # 最佳模型
├── last_model.pth           # 最后模型
└── checkpoint_epoch_*.pht   # 定期保存

logs/{experiment_name}/
├── training.log             # 训练日志
├── config.json              # 配置记录
└── history.json            # 训练历史
```

### 评估输出

```
results/
├── test_metrics.json        # 测试指标
├── val_metrics.json         # 验证指标
├── test_confusion_matrix.png
└── test_roc_curves.png
```

---

## 训练技巧

### 1. 数据不平衡处理

项目已自动计算类别权重处理 A2C 类样本较少的问题。

### 2. 训练策略选择

| 场景 | 建议配置 |
|------|----------|
| 小数据集 (< 10k) | `--freeze_backbone --lr 1e-3` |
| 中等数据集 | `--lr 1e-4 --epochs 50` |
| 大数据集 | `--lr 5e-5 --epochs 100` |

### 3. 混合精度训练

默认启用 AMP 混合精度训练，减少显存占用并加速训练。

### 4. 早停机制

当验证集准确率连续 10 个 epoch 未提升时自动停止训练。

---

## 常见问题

### Q1: 预训练权重加载失败

确保权重文件路径正确，文件完整性未损坏。

### Q2: CUDA 内存不足

减小 `--batch_size`，或使用 `--use_amp false` 关闭混合精度。

### Q3: CAMUS 数据未找到

CAMUS 数据需从官网单独下载，若无此数据，模型将自动使用 CACTUS 的 6 类数据进行训练。

### Q4: 数据加载速度慢

增加 `--num_workers` 参数，或将数据放到 SSD 硬盘。

---

## 项目结构

```
HeartEcho_Classification_Model/
├── data/                    # 数据加载模块
│   ├── cactus_loader.py     # CACTUS数据加载
│   ├── camus_loader.py      # CAMUS数据加载
│   └── dataset.py           # 统一数据管道
├── models/                  # 模型定义模块
│   ├── backbone.py          # USF-MAE Backbone
│   └── classifier.py        # 分类模型
├── utils/                   # 工具模块
│   ├── trainer.py           # 训练逻辑
│   ├── evaluate.py          # 评估指标
│   └── logger.py            # 日志记录
├── configs/                 # 配置文件
│   └── config.yaml
├── train.py                 # 训练入口
├── eval.py                  # 评估入口
└── requirements.txt         # 依赖列表
```

---

## 参考资料

- USF-MAE 论文: [arXiv:2602.15339](https://arxiv.org/abs/2602.15339)
- CACTUS 数据集: https://www.creatis.insa-lyon.fr/Challenge/camus
- CAMUS 数据集: https://humanheart-project.creatis.insa-lyon.fr