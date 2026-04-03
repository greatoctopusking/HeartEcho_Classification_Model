# 心脏超声切面分类项目计划

## 1. 项目概述

- **任务**: 7类心脏切面分类
- **数据集**: CACTUS + CAMUS
- **框架**: PyTorch + USF-MAE预训练模型

## 2. 数据方案

### 2.1 数据来源

| 数据集 | 类别 | 数量 |
|--------|------|------|
| CACTUS | A4C (心尖四腔心) | 7,422 |
| CACTUS | SC (剑突下四腔心) | 6,345 |
| CACTUS | PL (胸骨旁长轴) | 6,102 |
| CACTUS | PSAV (胸骨旁短轴-主动脉瓣) | 5,832 |
| CACTUS | PSMV (胸骨旁短轴-二尖瓣) | 6,014 |
| CACTUS | Random (随机图像) | 6,021 |
| CAMUS | A2C (心尖两腔心) | ~500 |
| **总计** | **7类** | **~38,200+** |

### 2.2 数据不平衡处理

- CAMUS A2C仅500张，数量远少于其他类别
- 解决方案：
  - 强数据增强（几何变换 + 颜色抖动 + MixUp/CutMix）
  - 类别加权损失函数

## 3. 技术方案

### 3.1 模型架构

```
输入图像 (224x224)
    ↓
USF-MAE Backbone (ViT-Base, 预训练)
    ↓
全局平均池化
    ↓
分类头 (7类输出)
    ↓
Softmax → 预测类别
```

### 3.2 训练配置

| 参数 | 值 |
|------|-----|
| Backbone | USF-MAE (ViT-B) 预训练权重 |
| 输入尺寸 | 224×224 |
| Batch Size | 32-64 |
| 优化器 | AdamW (lr=1e-4, weight_decay=1e-4) |
| 损失函数 | CrossEntropyLoss + ClassWeight |
| 学习率调度 | CosineAnnealing |
| Epoch | 30-50 (早停patience=10) |
| 验证集比例 | 15% |

### 3.3 数据增强策略

**训练时**:
- RandomResizedCrop (224, scale=(0.5, 2.0))
- RandomHorizontalFlip (p=0.5)
- RandomVerticalFlip (p=0.5)
- RandomRotation (degrees=(0, 90))
- ColorJitter (brightness=0.2, contrast=0.2)
- ImageNet标准化

**验证/测试时**:
- Resize (224, 224)
- ImageNet标准化

## 4. 项目结构

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

## 5. 评估指标

- **分类指标**: Accuracy, Precision, Recall, F1-score
- **可视化**: 混淆矩阵, ROC曲线
- **每类分析**: 各类别的precision/recall

## 6. 实施步骤

### Phase 1: 数据准备
1. 整理CACTUS 6类数据到统一目录
2. 处理CAMUS A2C数据
3. 生成数据索引CSV文件

### Phase 2: 模型构建
1. 加载USF-MAE预训练权重
2. 修改分类头为7类
3. 实现数据加载管道

### Phase 3: 训练
1. 划分训练/验证/测试集
2. 设置类别权重
3. 开始训练 + 日志记录

### Phase 4: 评估
1. 在测试集上评估
2. 生成混淆矩阵
3. 分析各类别表现

## 7. 参考资料

- USF-MAE: https://github.com/Yusufii9/USF-MAE
- CACTUS数据集: https://www.creatis.insa-lyon.fr/Challenge/camus
- CAMUS数据集: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8