# Data 模块说明文档

本模块负责心脏超声切面分类数据集的加载、预处理和数据增强。

## 目录结构

```
data/
├── __init__.py          # 模块导出
├── cactus_loader.py     # CACTUS 数据集加载器
├── camus_loader.py      # CAMUS 数据集加载器
├── dataset.py           # 统一数据集类和数据加载器
└── README.md            # 本说明文档
```

## 数据集支持

### 1. CACTUS 数据集 (6类)

| 类别 | 英文名称 | 图像数量 | 说明 |
|------|----------|----------|------|
| A4C | Apical Four-Chamber | 7,422 | 心尖四腔心 |
| PL | Parasternal Long-axis | 6,102 | 胸骨旁长轴 |
| PSAV | Parasternal Short-axis Aortic Valve | 5,832 | 胸骨旁短轴-主动脉瓣 |
| PSMV | Parasternal Short-axis Mitral Valve | 6,014 | 胸骨旁短轴-二尖瓣 |
| Random | Random | 6,021 | 随机图像 |
| SC | Subcostal | 6,345 | 剑突下四腔心 |

**数据格式**:
- 位置: `CACTUS/Images Dataset/{类别名}/`
- 格式: JPEG (.jpg)
- 文件命名: `{患者ID}_frame_{帧号}_v2.jpg`

### 2. CAMUS 数据集 (1类)

| 类别 | 英文名称 | 图像数量 | 说明 |
|------|----------|----------|------|
| A2C | Apical Two-Chamber | ~10,000 | 心尖两腔心 |

**注意**: CAMUS 数据为 NIfTI 格式，每个患者的 `*_2CH_half_sequence.nii.gz` 文件包含约 15-25 帧

**您的数据路径**: `D:/SRTP_Project__DeepLearning/project/Resources/database_nifti/`

**数据格式**:
- 格式: NIfTI (.nii.gz)
- 结构: `patient0001/patient0001_2CH_half_sequence.nii.gz`
- 帧数: 每患者约 15-25 帧
- 图像尺寸: 不统一 (约 355×357 ~ 708×584)，需 resize 到 224×224

## 类别标签映射

```python
CLASS_TO_IDX = {
    'A4C': 0,    # 心尖四腔心
    'PL': 1,     # 胸骨旁长轴
    'PSAV': 2,   # 胸骨旁短轴-主动脉瓣
    'PSMV': 3,   # 胸骨旁短轴-二尖瓣
    'Random': 4, # 随机图像
    'SC': 5,     # 剑突下四腔心
    'A2C': 6     # 心尖两腔心
}
```

## 快速使用

### 方法1: 使用数据加载器

```python
from data import get_data_loaders

# 获取训练、验证、测试数据加载器
train_loader, val_loader, test_loader = get_data_loaders(
    cactus_data_root='CACTUS/Images Dataset',
    camus_data_root='CAMUS',  # 可选，若无CAMUS数据则设为None
    batch_size=32,
    num_workers=4,
    val_split=0.15,
    test_split=0.15
)

# 遍历数据
for images, labels in train_loader:
    print(images.shape)  # (batch_size, 3, 224, 224)
    print(labels.shape)  # (batch_size,)
    break
```

### 方法2: 使用数据集类

```python
from data.dataset import CardiacDataset, get_train_transforms, get_val_transforms
from data.cactus_loader import get_cactus_data_info

# 获取数据路径和标签
image_paths, labels = get_cactus_data_info('CACTUS/Images Dataset')

# 创建数据集
train_dataset = CardiacDataset(
    image_paths=image_paths,
    labels=labels,
    transform=get_train_transforms()
)

# 获取类别权重（用于处理数据不平衡）
class_weights = train_dataset.get_class_weights()
print(class_weights)
```

### 方法3: 手动划分数据集

```python
from data.dataset import CardiacDataset, combine_datasets

# 合并并划分数据集
train_dataset, val_dataset, test_dataset = combine_datasets(
    cactus_data_root='CACTUS/Images Dataset',
    camus_data_root='CAMUS',
    val_split=0.15,
    test_split=0.15,
    random_seed=42
)
```

## 数据增强

### 训练时增强

| 变换 | 参数 | 说明 |
|------|------|------|
| RandomResizedCrop | scale=(0.5, 2.0) | 随机裁剪并缩放 |
| RandomHorizontalFlip | p=0.5 | 水平翻转 |
| RandomVerticalFlip | p=0.5 | 垂直翻转 |
| RandomRotation | degrees=(0, 90) | 随机旋转 |
| ColorJitter | brightness=0.2, contrast=0.2 | 颜色抖动 |
| Normalize | ImageNet统计量 | 标准化 |

### 验证/测试时变换

| 变换 | 参数 | 说明 |
|------|------|------|
| Resize | (224, 224) | 调整大小 |
| Normalize | ImageNet统计量 | 标准化 |

## 数据不平衡处理

由于 CAMUS A2C 类只有约 500 张图像，与其他类别 (~6000+) 差距较大，数据集类提供了类别权重计算功能：

```python
# 获取类别权重
class_weights = dataset.get_class_weights()

# 在训练中使用
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
```

## 工具函数

### 检查数据集完整性

```python
from data.cactus_loader import verify_cactus_data, get_class_counts

# 验证 CACTUS 数据集
if verify_cactus_data('CACTUS/Images Dataset'):
    print("数据集完整")
    
# 获取各类别数量
counts = get_class_counts('CACTUS/Images Dataset')
print(counts)
```

### 按患者获取 CAMUS 数据

```python
from data.camus_loader import get_camus_data_by_patient

patient_data = get_camus_data_by_patient('CAMUS')
for patient_id, images in patient_data.items():
    print(f"{patient_id}: {len(images)} 张图像")
```

## 依赖

- Python 3.8+
- PyTorch 1.8+
- torchvision 0.9+
- Pillow 8.0+
- NumPy 1.19+

## 注意事项

1. **CAMUS 数据**: 需要从官网申请下载，放置在与 CACTUS 同级的目录
2. **内存**: 大量图像可能占用较多内存，建议 num_workers=4 并启用 pin_memory=True
3. **数据增强**: 训练时启用强增强，验证/测试时仅使用基础变换