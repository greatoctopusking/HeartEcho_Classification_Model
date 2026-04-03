import os
from pathlib import Path
from typing import List, Tuple, Dict


CLASS_NAMES = ['A4C', 'PL', 'PSAV', 'PSMV', 'Random', 'SC']

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def get_cactus_data_info(data_root: str) -> Tuple[List[str], List[int]]:
    """
    获取CACTUS数据集的文件路径和标签信息
    
    Args:
        data_root: CACTUS数据集根目录，例如: 'CACTUS/Images Dataset'
    
    Returns:
        image_paths: 图像文件路径列表
        labels: 对应的标签索引列表
    """
    image_paths = []
    labels = []
    
    data_path = Path(data_root)
    
    for class_name in CLASS_NAMES:
        class_dir = data_path / class_name
        if not class_dir.exists():
            print(f"警告: 类别目录不存在 - {class_dir}")
            continue
        
        for img_file in class_dir.glob('*.jpg'):
            image_paths.append(str(img_file))
            labels.append(CLASS_TO_IDX[class_name])
        for img_file in class_dir.glob('*.png'):
            image_paths.append(str(img_file))
            labels.append(CLASS_TO_IDX[class_name])
        for img_file in class_dir.glob('*.jpeg'):
            image_paths.append(str(img_file))
            labels.append(CLASS_TO_IDX[class_name])
    
    print(f"CACTUS数据集加载完成: {len(image_paths)} 张图像")
    for class_name in CLASS_NAMES:
        count = labels.count(CLASS_TO_IDX[class_name])
        print(f"  - {class_name}: {count} 张")
    
    return image_paths, labels


def get_class_counts(data_root: str) -> Dict[str, int]:
    """
    获取每个类别的图像数量
    
    Args:
        data_root: CACTUS数据集根目录
    
    Returns:
        类别名称到数量的字典
    """
    counts = {}
    data_path = Path(data_root)
    
    for class_name in CLASS_NAMES:
        class_dir = data_path / class_name
        if class_dir.exists():
            jpg_count = len(list(class_dir.glob('*.jpg')))
            png_count = len(list(class_dir.glob('*.png')))
            jpeg_count = len(list(class_dir.glob('*.jpeg')))
            counts[class_name] = jpg_count + png_count + jpeg_count
    
    return counts


def verify_cactus_data(data_root: str) -> bool:
    """
    验证CACTUS数据集是否完整
    
    Args:
        data_root: CACTUS数据集根目录
    
    Returns:
        是否验证通过
    """
    data_path = Path(data_root)
    all_exist = True
    
    for class_name in CLASS_NAMES:
        class_dir = data_path / class_name
        if not class_dir.exists():
            print(f"错误: 缺少类别目录 - {class_name}")
            all_exist = False
    
    return all_exist


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = 'CACTUS/Images Dataset'
    
    print(f"数据集路径: {data_root}")
    print("=" * 50)
    
    if verify_cactus_data(data_root):
        print("数据集结构验证通过")
    else:
        print("数据集结构验证失败")
    
    print("=" * 50)
    get_class_counts(data_root)