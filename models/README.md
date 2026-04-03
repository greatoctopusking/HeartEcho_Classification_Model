# Models 模块说明文档

本模块负责心脏超声切面分类的模型定义，包括 USF-MAE Backbone 加载和分类器设计。

## 目录结构

```
models/
├── __init__.py          # 模块导出
├── backbone.py          # USF-MAE Backbone 编码器
├── classifier.py        # 分类模型定义
└── README.md            # 本说明文档
```

## 模型架构

```
输入图像 (224×224×3)
    ↓
USF-MAE Backbone (ViT-Base, 768维输出)
    ↓
特征提取 (CLS token 或 Global Avg Pool)
    ↓
分类头 (Linear: 768 → 7)
    ↓
Softmax → 7类输出
```

## 快速使用

### 1. 加载预训练模型

```python
from models import load_pretrained_usfmae, CardiacClassifier

# 方法1: 加载预训练的 Backbone
backbone = load_pretrained_usfmae(
    checkpoint_path='USF-MAE pretrained/USF-MAE_full_pretrain_43dataset_100epochs.pt',
    device='cuda'
)

# 方法2: 创建分类模型（推荐）
model = load_model_with_pretrained(
    pretrained_path='USF-MAE pretrained/USF-MAE_full_pretrain_43dataset_100epochs.pt',
    num_classes=7,
    freeze_backbone=False,  # 是否冻结 backbone
    device='cuda'
)
```

### 2. 创建模型（不使用预训练）

```python
from models import create_usfmae_backbone, CardiacClassifier

# 创建 backbone（随机初始化）
backbone = create_usfmae_backbone(
    pretrained_path=None,  # 无预训练权重
    freeze=False,
    device='cuda'
)

# 创建分类模型
model = CardiacClassifier(
    backbone=backbone,
    num_classes=7,
    use_cls_token=True,
    dropout=0.1
)
```

### 3. 使用不同的分类头

```python
from models import create_usfmae_backbone, create_classifier

backbone = create_usfmae_backbone()

# 线性分类器
model = create_classifier(backbone, num_classes=7, classifier_type='linear')

# MLP 分类器
model = create_classifier(
    backbone, 
    num_classes=7, 
    classifier_type='mlp',
    hidden_dim=512,
    num_layers=2,
    dropout=0.3
)
```

## 模型组件

### USFMAEEncoder (backbone.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| img_size | 224 | 输入图像尺寸 |
| patch_size | 16 | Patch 大小 |
| embed_dim | 768 | 嵌入维度 (ViT-Base) |
| depth | 12 | Transformer 层数 |
| num_heads | 12 | 注意力头数 |
| use_cls_token | True | 是否使用 [CLS] token |

### CardiacClassifier (classifier.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| backbone | 必填 | 骨干网络 |
| num_classes | 7 | 分类类别数 |
| use_cls_token | True | 使用 CLS token 特征 |
| use_global_avg_pool | True | 使用全局平均池化 |
| dropout | 0.0 | Dropout 概率 |

## 模型参数量

| 模型 | 参数量 |
|------|--------|
| USFMAEEncoder (Backbone) | ~86M |
| CardiacClassifier (分类头) | ~5.4K |
| **总计** | **~86M** |

## 训练策略

### 策略1: 冻结 Backbone

适用于小数据集，训练速度快：

```python
model = load_model_with_pretrained(
    pretrained_path='path/to/pretrained.pt',
    num_classes=7,
    freeze_backbone=True  # 冻结 backbone
)

# 只训练分类头
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
```

### 策略2: 完全微调

适用于中等规模数据，可能获得更高精度：

```python
model = load_model_with_pretrained(
    pretrained_path='path/to/pretrained.pt',
    num_classes=7,
    freeze_backbone=False  # 微调整个模型
)

# 训练所有参数
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

### 策略3: 分层微调

先训练分类头，再逐层解冻：

```python
# 阶段1: 只训练分类头
for param in model.backbone.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

# 阶段2: 解冻部分层
for param in model.backbone.blocks[-3:].parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

## 特征提取

```python
# 获取特征嵌入（用于特征可视化或迁移学习）
model.eval()
with torch.no_grad():
    embedding = model.get_embedding(x)
    # shape: (batch_size, 768)
```

## 依赖

- Python 3.8+
- PyTorch 1.8+
- NumPy 1.19+

## 注意事项

1. **预训练权重**: 确保预训练权重文件路径正确
2. **设备**: 使用 CUDA 加速训练，检查 GPU 可用性
3. **特征维度**: 确认 backbone 输出维度为 768（ViT-Base）
4. **分类头**: 根据任务调整 num_classes 参数