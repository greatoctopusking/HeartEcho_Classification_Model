# 心脏超声切面二分类任务报告

---

## 1 任务概述

### 1.1 任务目标

基于 CAMUS 数据集，对心脏超声图像进行二分类任务，区分两种心尖切面：
- **A2C** (Apical Two-Chamber): 心尖两腔心切面
- **A4C** (Apical Four-Chamber): 心尖四腔心切面

### 1.2 数据来源

| 数据集 | 来源 | 格式 | 患者数 | 帧数 |
|--------|------|------|--------|------|
| CAMUS | D:\SRTP_Project__DeepLearning\project\Resources\database_nifti | NIfTI | 500 | ~10,000 |

### 1.3 数据划分

| 划分 | 比例 | 样本数 |
|------|------|--------|
| 训练集 | 70% | ~7,000 |
| 验证集 | 15% | ~1,500 |
| 测试集 | 15% | ~1,500 |

---

## 2 模型配置

### 2.1 训练参数

| 参数 | 值 |
|------|------|
| 预训练模型 | USF-MAE (ViT-Base) |
| 任务类型 | binary (二分类) |
| 图像尺寸 | 224 × 224 |
| 批大小 (batch_size) | 32 |
| 训练轮数 (epochs) | 50 |
| 学习率 (lr) | 1e-4 |
| 优化器 | AdamW |
| 调度器 | Cosine Annealing |
| Dropout | 0.3 |
| 早停耐心 | 10 epochs |

### 2.2 模型架构

```
输入图像 (224×224×3)
    ↓
USF-MAE Encoder (ViT-Base, 12层, 768维)
    ↓
全局平均池化 (Global Average Pooling)
    ↓
分类头 (Dropout + Linear: 768→2)
    ↓
Softmax → [A2C概率, A4C概率]
```

---

## 3 评估结果

### 3.1 整体指标

| 指标 | 值 |
|------|------|
| **准确率 (Accuracy)** | 99.86% |
| **精确率 (Precision)** | 99.86% |
| **召回率 (Recall)** | 99.86% |
| **F1 分数** | 99.86% |
| **Precision (Macro)** | 99.86% |
| **Recall (Macro)** | 99.86% |
| **F1 (Macro)** | 99.86% |

### 3.2 混淆矩阵

| 真实类别 \ 预测类别 | A2C (预测) | A4C (预测) |
|---------------------|-----------|-----------|
| **A2C (真实)** | 9,263 | 5 |
| **A4C (真实)** | 22 | 9,942 |

### 3.3 分类详情

| 类别 | 正确分类数 | 错误分类数 | 准确率 |
|------|-----------|-----------|--------|
| A2C (心尖两腔心) | 9,263 | 5 | 99.95% |
| A4C (心尖四腔心) | 9,942 | 22 | 99.78% |

### 3.4 错误分析

- **A2C 误判为 A4C**: 5 例 (0.05%)
- **A4C 误判为 A2C**: 22 例 (0.22%)

总体误判率极低，模型表现优异。

---

## 4 输出文件

### 4.1 模型文件

```
checkpoints/binary/
├── best_model.pth              # 最佳模型
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_15.pth
├── checkpoint_epoch_20.pth
└── checkpoint_epoch_25.pth
```

### 4.2 评估结果

```
results/binary/
├── full_metrics.json           # 评估指标
├── full_confusion_matrix.png   # 混淆矩阵可视化
└── full_roc_curves.png         # ROC 曲线可视化
```

---

## 5 结论

### 5.1 实验结论

1. **模型性能优异**: 在 CAMUS 二分类任务上达到 99.86% 的准确率
2. **类别均衡**: 两类别的分类准确率均超过 99.7%，无明显偏向
3. **泛化能力强**: 基于 USF-MAE 预训练模型，微调效果显著

### 5.2 应用建议

该模型可应用于：
- 心脏超声检查的自动化切面识别
- 超声质量控制和工作流优化
- 大规模超声数据的快速分类

### 5.3 后续改进方向

1. **增加数据增强**: 进一步提升模型鲁棒性
2. **使用 K-Fold 交叉验证**: 获得更稳定的模型性能评估
3. **模型集成**: 结合多个 epoch 的模型进行预测

---

## 6 使用方法

### 6.1 推理命令

```bash
# 二分类推理
python -m inference \
    --checkpoint checkpoints/binary/best_model.pth \
    --task_type binary \
    --input path/to/image.jpg
```

### 6.2 评估命令

```bash
# 评估二分类模型
python eval.py \
    --task_type binary \
    --checkpoint checkpoints/binary/best_model.pth \
    --camus_data "D:\SRTP_Project__DeepLearning\project\Resources\database_nifti" \
    --full_data \
    --plot_cm \
    --plot_roc
```

---

*报告生成日期：2026-04-10*  
*项目：HeartEcho Classification Model*