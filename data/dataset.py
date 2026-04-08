import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from .cactus_loader import get_cactus_data_info, CLASS_NAMES as CACTUS_CLASSES
from .camus_loader import get_camus_data_info, CLASS_NAME as CAMUS_CLASS, CLASS_IDX as CAMUS_IDX


ALL_CLASS_NAMES = ['A4C', 'PL', 'PSAV', 'PSMV', 'Random', 'SC', 'A2C']

CLASS_TO_IDX = {name: idx for idx, name in enumerate(ALL_CLASS_NAMES)}

IDX_TO_CLASS = {idx: name for idx, name in enumerate(ALL_CLASS_NAMES)}


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    获取训练时的数据增强 transforms
    
    Args:
        image_size: 目标图像尺寸
    
    Returns:
        训练数据增强的 transforms 组合
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.333)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    获取验证/测试时的 transforms (无数据增强)
    
    Args:
        image_size: 目标图像尺寸
    
    Returns:
        验证数据的 transforms 组合
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class CardiacDataset(Dataset):
    """
    心脏超声切面分类数据集
    
    支持加载 CACTUS 数据集 (6类) 和 CAMUS 数据集 (A2C类)
    自动合并为统一的 7 类分类数据集
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        return_path: bool = False
    ):
        """
        初始化数据集
        
        Args:
            image_paths: 图像文件路径列表
            labels: 对应的标签索引列表
            transform: 数据变换函数
            return_path: 是否返回图像路径（用于调试）
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.return_path = return_path
        
        assert len(image_paths) == len(labels), \
            f"图像路径数量 ({len(image_paths)}) 与标签数量 ({len(labels)}) 不匹配"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        获取单个数据样本
        
        Returns:
            如果 return_path=False:
                (image_tensor, label)
            如果 return_path=True:
                (image_tensor, label, image_path)
        """
        img_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.return_path:
            return image, label, img_path
        else:
            return image, label
    
    def get_class_counts(self) -> np.ndarray:
        """
        获取每个类别的样本数量
        
        Returns:
            类别数量的 numpy 数组
        """
        counts = np.zeros(len(ALL_CLASS_NAMES), dtype=int)
        for label in self.labels:
            counts[label] += 1
        return counts
    
    def get_class_weights(self) -> torch.Tensor:
        """
        计算类别权重（用于处理数据不平衡）
        
        使用 sklearn 标准方法计算平衡权重
        
        Returns:
            类别权重张量
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        labels_array = np.array(self.labels)
        unique_classes = np.arange(len(ALL_CLASS_NAMES))
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=labels_array
        )
        
        return torch.FloatTensor(class_weights)


def combine_datasets(
    cactus_data_root: str,
    camus_data_root: Optional[str] = None,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    stratified: bool = True
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    合并 CACTUS 和 CAMUS 数据集，并划分为训练/验证/测试集
    
    Args:
        cactus_data_root: CACTUS 数据集根目录
        camus_data_root: CAMUS 数据集根目录（可选）
        val_split: 验证集比例
        test_split: 测试集比例
        random_seed: 随机种子
        stratified: 是否使用分层采样
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    all_paths = []
    all_labels = []
    
    cactus_paths, cactus_labels = get_cactus_data_info(cactus_data_root)
    all_paths.extend(cactus_paths)
    all_labels.extend(cactus_labels)
    
    if camus_data_root and os.path.exists(camus_data_root):
        camus_paths, camus_labels = get_camus_data_info(camus_data_root)
        all_paths.extend(camus_paths)
        all_labels.extend(camus_labels)
    else:
        print("警告: CAMUS 数据集未找到，仅使用 CACTUS 数据集 (6类)")
    
    total_samples = len(all_paths)
    print(f"合并后总样本数: {total_samples}")
    
    all_labels = np.array(all_labels)
    
    test_size = int(total_samples * test_split)
    val_size = int(total_samples * val_split)
    
    if stratified:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=random_seed)
        train_idx, test_val_idx = next(sss.split(all_paths, all_labels))
        
        train_paths = [all_paths[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        
        remaining_paths = [all_paths[i] for i in test_val_idx]
        remaining_labels = [all_labels[i] for i in test_val_idx]
        
        val_size_adjusted = int(len(remaining_paths) * (val_size / (test_size + val_size)))
        
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_seed)
        val_idx_final, test_idx_final = next(sss_val.split(remaining_paths, remaining_labels))
        
        val_paths = [remaining_paths[i] for i in val_idx_final]
        val_labels = [remaining_labels[i] for i in val_idx_final]
        test_paths = [remaining_paths[i] for i in test_idx_final]
        test_labels = [remaining_labels[i] for i in test_idx_final]
    else:
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size + val_size]
        train_indices = indices[test_size + val_size:]
        
        train_paths = [all_paths[i] for i in train_indices]
        train_labels = [all_labels[i] for i in train_indices]
        val_paths = [all_paths[i] for i in val_indices]
        val_labels = [all_labels[i] for i in val_indices]
        test_paths = [all_paths[i] for i in test_indices]
        test_labels = [all_labels[i] for i in test_indices]
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_dataset = CardiacDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=train_transform
    )
    
    val_dataset = CardiacDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=val_transform
    )
    
    test_dataset = CardiacDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=val_transform
    )
    
    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    print("\n各类别分布:")
    for dataset, name in [(train_dataset, "训练"), (val_dataset, "验证"), (test_dataset, "测试")]:
        counts = dataset.get_class_counts()
        print(f"  {name}集:")
        for i, class_name in enumerate(ALL_CLASS_NAMES):
            print(f"    {class_name}: {counts[i]}")
    
    return train_dataset, val_dataset, test_dataset


def get_data_loaders(
    cactus_data_root: str,
    camus_data_root: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取训练、验证、测试数据加载器
    
    Args:
        cactus_data_root: CACTUS 数据集根目录
        camus_data_root: CAMUS 数据集根目录（可选）
        batch_size: 批大小
        num_workers: 数据加载线程数
        val_split: 验证集比例
        test_split: 测试集比例
        random_seed: 随机种子
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset, val_dataset, test_dataset = combine_datasets(
        cactus_data_root=cactus_data_root,
        camus_data_root=camus_data_root,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_subsample_dataset(
    dataset: Dataset,
    samples_per_class: int,
    random_seed: int = 42
) -> Dataset:
    """
    创建每个类别的子采样数据集（用于快速测试）
    
    Args:
        dataset: 原始数据集
        samples_per_class: 每个类别采样的数量
        random_seed: 随机种子
    
    Returns:
        子采样后的数据集
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    class_indices = {i: [] for i in range(len(ALL_CLASS_NAMES))}
    
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    selected_indices = []
    for class_idx, indices in class_indices.items():
        if len(indices) >= samples_per_class:
            selected = random.sample(indices, samples_per_class)
        else:
            selected = indices
        selected_indices.extend(selected)
    
    return Subset(dataset, selected_indices)


def create_full_data_loader(
    cactus_data_root: str,
    camus_data_root: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224
) -> Tuple[DataLoader, Dict]:
    """
    创建包含全部数据的DataLoader（无划分），用于评估kfold模型
    
    Args:
        cactus_data_root: CACTUS 数据集根目录
        camus_data_root: CAMUS 数据集根目录（可选）
        batch_size: 批大小
        num_workers: 数据加载线程数
        img_size: 图像尺寸
    
    Returns:
        (data_loader, class_counts_dict)
    """
    all_paths = []
    all_labels = []
    
    cactus_paths, cactus_labels = get_cactus_data_info(cactus_data_root)
    all_paths.extend(cactus_paths)
    all_labels.extend(cactus_labels)
    
    if camus_data_root and os.path.exists(camus_data_root):
        camus_paths, camus_labels = get_camus_data_info(camus_data_root)
        all_paths.extend(camus_paths)
        all_labels.extend(camus_labels)
    else:
        print("警告: CAMUS 数据集未找到，仅使用 CACTUS 数据集")
    
    total_samples = len(all_paths)
    print(f"全数据集总样本数: {total_samples}")
    
    transform = get_val_transforms(img_size)
    full_dataset = CardiacDataset(
        image_paths=all_paths,
        labels=all_labels,
        transform=transform
    )
    
    class_counts = full_dataset.get_class_counts()
    class_counts_dict = {ALL_CLASS_NAMES[i]: int(class_counts[i]) for i in range(len(ALL_CLASS_NAMES))}
    
    data_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return data_loader, class_counts_dict


if __name__ == '__main__':
    cactus_root = 'CACTUS/Images Dataset'
    camus_root = 'CAMUS'
    
    print("测试数据加载器...")
    print("=" * 60)
    
    train_loader, val_loader, test_loader = get_data_loaders(
        cactus_data_root=cactus_root,
        camus_data_root=camus_root,
        batch_size=32,
        num_workers=0
    )
    
    print("\n测试数据加载:")
    batch_images, batch_labels = next(iter(train_loader))
    print(f"Batch 图像形状: {batch_images.shape}")
    print(f"Batch 标签形状: {batch_labels.shape}")
    print(f"Batch 标签内容: {batch_labels[:10].tolist()}")
    
    print("\n类别权重:")
    print(train_loader.dataset.get_class_weights())