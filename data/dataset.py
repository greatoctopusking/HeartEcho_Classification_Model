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
from .camus_loader import get_camus_data_info, get_camus_binary_data_info


ALL_CLASS_NAMES = ['A4C', 'PL', 'PSAV', 'PSMV', 'Random', 'SC', 'A2C']

CLASS_TO_IDX = {name: idx for idx, name in enumerate(ALL_CLASS_NAMES)}

IDX_TO_CLASS = {idx: name for idx, name in enumerate(ALL_CLASS_NAMES)}

BINARY_CLASS_NAMES = ['A2C', 'A4C']
BINARY_CLASS_TO_IDX = {'A2C': 0, 'A4C': 1}
BINARY_IDX_TO_CLASS = {0: 'A2C', 1: 'A4C'}


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GRAY_MEAN = [0.485]
GRAY_STD = [0.229]


class PreprocessCactusTransform:
    """CACTUS图像预处理 - 可pickle的自定义transform类"""
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        return preprocess_cactus_image(img, self.target_size)


class PreprocessImageTransform:
    """统一图像预处理 - 可pickle的自定义transform类"""
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        return preprocess_image(img, "", self.target_size)


def preprocess_cactus_image(img: Image.Image, target_size: int = 224) -> Image.Image:
    """
    CACTUS 图像预处理:
    1. 裁剪掉设备信息边框 (左15%, 右35%, 下25%)
    2. 转灰度
    3. Resize (保持宽高比)
    4. Padding 到目标尺寸
    
    Args:
        img: 输入的 PIL Image
        target_size: 目标尺寸
    
    Returns:
        处理后的 PIL Image (灰度, 224x224)
    """
    orig_w, orig_h = img.size
    
    left = int(orig_w * 0.15)
    right = orig_w - int(orig_w * 0.35)
    top = 0
    bottom = orig_h - int(orig_h * 0.25)
    crop_box = (left, top, right, bottom)
    img = img.crop(crop_box)
    crop_w, crop_h = img.size
    
    img = img.convert('L')
    
    new_w = target_size
    new_h = int(crop_h * target_size / crop_w)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    result = Image.new('L', (target_size, target_size), 0)
    paste_y = (target_size - new_h) // 2
    result.paste(img, (0, paste_y))
    
    return result.convert('RGB')


def preprocess_image(img: Image.Image, img_path: str, target_size: int = 224) -> Image.Image:
    """
    统一的图像预处理:
    - CACTUS: 裁剪 + 转灰度 + Resize + Padding
    - CAMUS: 已经是处理好的灰度图，只需 Resize + Padding
    
    Args:
        img: 输入的 PIL Image
        img_path: 图像路径，用于判断来源
        target_size: 目标尺寸
    
    Returns:
        处理后的 PIL Image (灰度, 224x224)
    """
    if '_camus_cache' in img_path or 'database_nifti' in img_path:
        img = img.convert('L')
        orig_w, orig_h = img.size
        aspect_ratio = orig_w / orig_h
        new_w = target_size
        new_h = int(new_w / aspect_ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        result = Image.new('L', (target_size, target_size), 0)
        paste_y = (target_size - new_h) // 2
        result.paste(img, (0, paste_y))
        return result.convert('RGB')
    else:
        return preprocess_cactus_image(img, target_size)


def get_train_transforms(image_size: int = 224, use_gray: bool = True) -> transforms.Compose:
    """
    获取训练时的数据增强 transforms
    
    Args:
        image_size: 目标图像尺寸
        use_gray: 是否使用灰度图 (CACTUS和CAMUS统一为灰度)
    
    Returns:
        训练数据增强的 transforms 组合
    """
    if use_gray:
        return transforms.Compose([
            PreprocessImageTransform(image_size),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.333)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
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


def get_val_transforms(image_size: int = 224, use_gray: bool = True) -> transforms.Compose:
    """
    获取验证/测试时的 transforms (无数据增强)
    
    Args:
        image_size: 目标图像尺寸
        use_gray: 是否使用灰度图
    
    Returns:
        验证数据的 transforms 组合
    """
    if use_gray:
        return transforms.Compose([
            PreprocessImageTransform(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
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
            image = Image.open(img_path)
            image = preprocess_image(image, img_path, target_size=224)
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}: {e}")
            image = Image.new('L', (224, 224), color=0)
        
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
        unique_labels = set(self.labels)
        if not unique_labels:
            return np.zeros(1, dtype=int)
        num_classes = max(unique_labels) + 1
        counts = np.zeros(num_classes, dtype=int)
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
        unique_classes = np.unique(labels_array)
        
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
    stratified: bool = True,
    holdout_split: float = 0.0
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
        holdout_split: 保留测试集比例（用于最终验证，不参与训练）
    
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
    all_paths = np.array(all_paths, dtype=object)
    
    if holdout_split > 0:
        holdout_size = int(total_samples * holdout_split)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=holdout_size, random_state=random_seed)
        train_val_idx, holdout_idx = next(sss.split(all_paths, all_labels))
        
        train_val_paths = all_paths[train_val_idx].tolist()
        train_val_labels = all_labels[train_val_idx].tolist()
        holdout_paths = all_paths[holdout_idx].tolist()
        holdout_labels = all_labels[holdout_idx].tolist()
        
        print(f"Hold-out 测试集: {len(holdout_paths)} 样本")
        
        train_val_labels = np.array(train_val_labels)
        test_size = int(len(train_val_paths) * test_split)
        val_size = int(len(train_val_paths) * val_split)
        
        if stratified:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=random_seed)
            train_idx, test_val_idx = next(sss.split(train_val_paths, train_val_labels))
            
            train_paths = [train_val_paths[i] for i in train_idx]
            train_labels = [train_val_labels[i] for i in train_idx]
            
            remaining_paths = [train_val_paths[i] for i in test_val_idx]
            remaining_labels = [train_val_labels[i] for i in test_val_idx]
            
            val_size_adjusted = int(len(remaining_paths) * (val_size / (test_size + val_size)))
            
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_seed)
            val_idx_final, test_idx_final = next(sss_val.split(remaining_paths, remaining_labels))
            
            val_paths = [remaining_paths[i] for i in val_idx_final]
            val_labels = [remaining_labels[i] for i in val_idx_final]
            test_paths = [remaining_paths[i] for i in test_idx_final]
            test_labels = [remaining_labels[i] for i in test_idx_final]
        else:
            indices = list(range(len(train_val_paths)))
            random.shuffle(indices)
            
            test_indices = indices[:test_size]
            val_indices = indices[test_size:test_size + val_size]
            train_indices = indices[test_size + val_size:]
            
            train_paths = [train_val_paths[i] for i in train_indices]
            train_labels = [train_val_labels[i] for i in train_indices]
            val_paths = [train_val_paths[i] for i in val_indices]
            val_labels = [train_val_labels[i] for i in val_indices]
            test_paths = [train_val_paths[i] for i in test_indices]
            test_labels = [train_val_labels[i] for i in test_indices]
    else:
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
        
        holdout_paths = []
        holdout_labels = []
    
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
    
    holdout_dataset = None
    if holdout_paths:
        holdout_dataset = CardiacDataset(
            image_paths=holdout_paths,
            labels=holdout_labels,
            transform=val_transform
        )
        print(f"  Hold-out集: {len(holdout_dataset)} 样本")
    
    return train_dataset, val_dataset, test_dataset, holdout_dataset


def get_data_loaders(
    cactus_data_root: str,
    camus_data_root: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    holdout_split: float = 0.0
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DataLoader]]:
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
        holdout_split: 保留测试集比例（用于最终验证，不参与训练）
    
    Returns:
        (train_loader, val_loader, test_loader, holdout_loader)
    """
    train_dataset, val_dataset, test_dataset, holdout_dataset = combine_datasets(
        cactus_data_root=cactus_data_root,
        camus_data_root=camus_data_root,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed,
        holdout_split=holdout_split
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
    
    holdout_loader = None
    if holdout_dataset is not None:
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader, holdout_loader


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


def get_binary_data_loaders(
    camus_data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    holdout_split: float = 0.0
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DataLoader]]:
    """
    获取二分类任务 (A2C vs A4C) 的数据加载器
    
    Args:
        camus_data_root: CAMUS数据集根目录
        batch_size: 批大小
        num_workers: 数据加载线程数
        val_split: 验证集比例
        test_split: 测试集比例
        random_seed: 随机种子
        holdout_split: 保留测试集比例（用于最终验证，不参与训练）
    
    Returns:
        (train_loader, val_loader, test_loader, holdout_loader)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    image_paths, labels = get_camus_binary_data_info(camus_data_root)
    
    if not image_paths:
        raise ValueError("未能在CAMUS数据集中加载二分类数据 (A2C和A4C)")
    
    total_samples = len(image_paths)
    print(f"二分类数据集总样本数: {total_samples}")
    
    all_paths = np.array(image_paths, dtype=object)
    all_labels = np.array(labels)
    
    if holdout_split > 0:
        holdout_size = int(total_samples * holdout_split)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=holdout_size, random_state=random_seed)
        train_val_idx, holdout_idx = next(sss.split(all_paths, all_labels))
        
        train_val_paths = all_paths[train_val_idx].tolist()
        train_val_labels = all_labels[train_val_idx].tolist()
        holdout_paths = all_paths[holdout_idx].tolist()
        holdout_labels = all_labels[holdout_idx].tolist()
        
        print(f"Hold-out 测试集: {len(holdout_paths)} 样本")
        
        train_val_labels = np.array(train_val_labels)
        test_size = int(len(train_val_paths) * test_split)
        val_size = int(len(train_val_paths) * val_split)
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=random_seed)
        train_idx, test_val_idx = next(sss.split(train_val_paths, train_val_labels))
        
        train_paths = [train_val_paths[i] for i in train_idx]
        train_labels = [train_val_labels[i] for i in train_idx]
        
        remaining_paths = [train_val_paths[i] for i in test_val_idx]
        remaining_labels = [train_val_labels[i] for i in test_val_idx]
        
        val_size_adjusted = int(len(remaining_paths) * (val_size / (test_size + val_size)))
        
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_seed)
        val_idx_final, test_idx_final = next(sss_val.split(remaining_paths, remaining_labels))
        
        val_paths = [remaining_paths[i] for i in val_idx_final]
        val_labels = [remaining_labels[i] for i in val_idx_final]
        test_paths = [remaining_paths[i] for i in test_idx_final]
        test_labels = [remaining_labels[i] for i in test_idx_final]
    else:
        test_size = int(total_samples * test_split)
        val_size = int(total_samples * val_split)
        
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
        
        holdout_paths = []
        holdout_labels = []
    
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
        for i, class_name in enumerate(BINARY_CLASS_NAMES):
            print(f"    {class_name}: {counts[i]}")
    
    holdout_dataset = None
    if holdout_paths:
        holdout_dataset = CardiacDataset(
            image_paths=holdout_paths,
            labels=holdout_labels,
            transform=val_transform
        )
        print(f"  Hold-out集: {len(holdout_dataset)} 样本")
    
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
    
    holdout_loader = None
    if holdout_dataset is not None:
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader, holdout_loader


def create_binary_full_data_loader(
    camus_data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224
) -> Tuple[DataLoader, Dict]:
    """
    创建包含全部二分类数据的DataLoader（无划分），用于评估kfold模型
    
    Args:
        camus_data_root: CAMUS数据集根目录
        batch_size: 批大小
        num_workers: 数据加载线程数
        img_size: 图像尺寸
    
    Returns:
        (data_loader, class_counts_dict)
    """
    image_paths, labels = get_camus_binary_data_info(camus_data_root)
    
    if not image_paths:
        raise ValueError("未能在CAMUS数据集中加载二分类数据")
    
    total_samples = len(image_paths)
    print(f"二分类全数据集总样本数: {total_samples}")
    
    transform = get_val_transforms(img_size)
    full_dataset = CardiacDataset(
        image_paths=image_paths,
        labels=labels,
        transform=transform
    )
    
    class_counts = full_dataset.get_class_counts()
    class_counts_dict = {BINARY_CLASS_NAMES[i]: int(class_counts[i]) for i in range(len(BINARY_CLASS_NAMES))}
    
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