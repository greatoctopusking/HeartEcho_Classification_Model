# USF-MAE 预训练模型用于心脏超声切面分类的下游任务实施计划

## 1. 数据格式说明

### 1.1 当前 CACTUS 数据集

```
CACTUS/Images Dataset/
├── A4C/      (7422 张)   ← 心尖四腔心
├── PL/       (6102 张)   ← 胸骨旁长轴
├── PSAV/     (5832 张)   ← 胸骨旁短轴-主动脉瓣
├── PSMV/     (6014 张)   ← 胸骨旁短轴-二尖瓣
├── Random/   (6021 张)   ← 随机图像
└── SC/       (6345 张)   ← 剑突下四腔心
```

| 项目 | 格式 |
|------|------|
| 图像格式 | JPEG (.jpg) |
| 文件命名 | `{患者ID}_frame_{帧号}_v2.jpg` |
| 标签方式 | 文件夹名称 = 类别标签 |
| 总图像数 | 37,726 张 |

### 1.2 缺少的 CAMUS A2C 数据

- 需要从 CAMUS 官网申请下载 A2C (心尖两腔心) ~500 张
- 来源: https://humanheart-project.creatis.insa-lyon.fr

### 1.3 数据组织建议

**方案: 按文件夹分类（与CACTUS一致）**
```
data/
├── A4C/      (7422 张)
├── PL/       (6102 张)
├── PSAV/     (5832 张)
├── PSMV/     (6014 张)
├── Random/   (6021 张)
├── SC/       (6345 张)
└── A2C/      (~500 张, 需下载)
```

---

## 2. 模型架构详解

### 2.1 整体架构流程

```
输入图像 (224×224)
    ↓
USF-MAE Backbone (ViT-Base, 预训练)
    ↓
全局平均池化 (Global Average Pooling)
    ↓
分类头 (Linear: 768 → 7)
    ↓
Softmax → 7类输出
```

### 2.2 USF-MAE 核心参数

| 参数 | 值 |
|------|-----|
| Backbone | ViT-Base (Vision Transformer) |
| 预训练任务 | Masked Autoencoder (MAE) |
| 隐藏维度 (embed_dim) | 768 |
| 注意力头数 (num_heads) | 12 |
| Transformer层数 (depth) | 12 |
| Patch Size | 16×16 |
| 输入分辨率 | 224×224 |
| 预训练数据 | 43个心脏超声数据集 |
| 预训练轮次 | 100 epochs |

### 2.3 MAE 预训练机制

```
原始图像 → 分割为 14×14 = 196 个 patches
    ↓
随机遮盖 75% 的 patches (仅保留 25%)
    ↓
编码器处理可见的 25% patches
    ↓
解码器重建被遮盖的 75% patches
    ↓
学习图像的通用表示
```

**MAE 核心设计**:
- 非对称编码器-解码器架构
- 高遮盖率(75%)促使模型学习更丰富的语义特征

### 2.4 预训练权重

- 文件: `USF-MAE pretrained/USF-MAE_full_pretrain_43dataset_100epochs.pt`
- 大小: 428MB

---

## 3. 三种训练策略对比

| 策略 | 方法 | 适用场景 | 优缺点 |
|------|------|----------|--------|
| 策略A | 冻结 Backbone，仅训练分类头 | 小数据集 | 训练快，泛化好，但可能欠拟合 |
| 策略B | 完全微调 (Fine-tuning) | 中等规模数据 | 可能获得更高精度，需更多计算 |
| 策略C | 分层微调 (Gradual Unfreezing) | 平衡场景 | 先训练分类头，再逐层解冻 |

**推荐**: 策略B，完全微调整个模型（与原论文一致）

---

## 4. 数据预处理

### 4.1 ImageNet 标准化参数

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### 4.2 训练时数据增强

```python
transforms.Compose([
    RandomResizedCrop(224, scale=(0.5, 2.0)),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),       # 心脏超声常需要
    RandomRotation(degrees=(0, 90)),  # 心脏切面方向多样
    ColorJitter(brightness=0.2, contrast=0.2),
    ToTensor(),
    Normalize(mean, std)
])
```

### 4.3 验证/测试时

```python
transforms.Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean, std)
])
```

---

## 5. 训练配置

| 参数 | 推荐值 |
|------|--------|
| 输入尺寸 | 224×224 |
| Batch Size | 32-64 (根据显存调整) |
| 优化器 | AdamW (lr=1e-4, weight_decay=0.01) |
| 损失函数 | CrossEntropyLoss + ClassWeight |
| 学习率调度 | CosineAnnealing |
| Epoch | 30-50 (早停 patience=10) |
| 验证集比例 | 15% |

---

## 6. 类别权重计算

由于 CAMUS A2C 只有 ~500 张，需启用类别权重来处理数据不平衡：

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
# 用于 CrossEntropyLoss(weight=class_weights)
```

---

## 7. 项目结构建议

```
HeartEcho_Classification/
├── data/
│   ├── __init__.py
│   ├── cactus_loader.py      # CACTUS数据加载
│   ├── camus_loader.py      # CAMUS数据加载
│   └── dataset.py           # 统一数据管道
├── models/
│   ├── __init__.py
│   ├── backbone.py          # USF-MAE backbone加载
│   └── classifier.py        # 分类模型定义
├── utils/
│   ├── __init__.py
│   ├── trainer.py           # 训练逻辑
│   ├── evaluate.py          # 评估指标
│   └── logger.py            # 日志记录
├── configs/
│   └── config.yaml          # 配置文件
├── train.py                 # 训练入口
├── eval.py                  # 评估入口
└── requirements.txt         # 依赖
```

---

## 8. 关键代码片段

### 8.1 加载预训练模型

```python
import torch
from vit_mae import ViTMAE

# 加载预训练权重
checkpoint = torch.load('USF-MAE_full_pretrain_43dataset_100epochs.pt', map_location='cpu')

# 提取 backbone (encoder) 部分
model = ViTMAE(patch_size=16, embed_dim=768, depth=12, num_heads=12)
model.load_state_dict(checkpoint['model'], strict=False)
# 移除 decoder，仅保留 encoder 作为 backbone
```

### 8.2 构建分类模型

```python
class CardiacClassifier(nn.Module):
    def __init__(self, pretrained_backbone, num_classes=7):
        super().__init__()
        self.encoder = pretrained_backbone
        self.head = nn.Linear(768, num_classes)
    
    def forward(self, x):
        features = self.encoder(x, mask=None)  # 获取全局特征
        return self.head(features)
```

### 8.3 冻结/解冻策略

```python
# 方案A: 冻结Backbone + 线性探针
for param in model.encoder.parameters():
    param.requires_grad = False

# 方案B: 完全微调 (推荐)
for param in model.parameters():
    param.requires_grad = True
```

---

## 9. 评估指标

- **分类指标**: Accuracy, Precision, Recall, F1-score
- **可视化**: 混淆矩阵, ROC曲线
- **每类分析**: 各类别的precision/recall

---

## 10. 待完成事项

- [ ] 获取 CAMUS A2C 数据 (~500张)
- [ ] 整理数据目录结构
- [ ] 编写数据加载代码 (cactus_loader.py, camus_loader.py)
- [ ] 编写模型加载代码 (backbone.py, classifier.py)
- [ ] 编写训练代码 (train.py)
- [ ] 编写评估代码 (eval.py)
- [ ] 配置超参数文件 (config.yaml)
- [ ] 开始训练并监控指标

---

## 参考资料

- USF-MAE: https://github.com/Yusufii9/USF-MAE
- CACTUS数据集: https://www.creatis.insa-lyon.fr/Challenge/camus
- CAMUS数据集: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8
- 原论文: arXiv:2602.15339 (Benchmarking Self-Supervised Models for Cardiac Ultrasound View Classification)