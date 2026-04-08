import os
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np


CLASS_NAME_2CH = 'A2C'
CLASS_IDX_2CH = 6

CLASS_NAME_4CH = 'A4C'
CLASS_IDX_4CH = 0

__all__ = ['CLASS_NAME_2CH', 'CLASS_IDX_2CH', 'CLASS_NAME_4CH', 'CLASS_IDX_4CH']

CACHE_DIR_NAME = "_camus_cache"
CACHE_METADATA_FILE = "cache_metadata.json"


def get_cache_dir(data_root: str) -> Path:
    """获取缓存目录路径"""
    return Path(data_root) / CACHE_DIR_NAME


def get_cache_hash(data_root: str) -> str:
    """计算数据目录的hash值，用于检测数据是否变化"""
    data_path = Path(data_root)
    patient_dirs = sorted(data_path.glob("patient*"))
    
    hasher = hashlib.md5()
    for patient_dir in patient_dirs[:10]:
        hasher.update(patient_dir.name.encode())
    
    return hasher.hexdigest()


def check_cache_valid(data_root: str) -> bool:
    """
    检查缓存是否有效
    
    Args:
        data_root: CAMUS数据集根目录
    
    Returns:
        缓存是否有效
    """
    cache_dir = get_cache_dir(data_root)
    metadata_file = cache_dir / CACHE_METADATA_FILE
    
    if not cache_dir.exists() or not metadata_file.exists():
        return False
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        current_hash = get_cache_hash(data_root)
        cached_hash = metadata.get('data_hash', '')
        
        if current_hash != cached_hash:
            print(f"缓存过期: 数据目录已更改")
            return False
        
        cache_images_dir = cache_dir / "images"
        if not cache_images_dir.exists():
            return False
        
        num_cached = len(list(cache_images_dir.glob("*.png")))
        expected_num = metadata.get('num_frames', 0)
        
        if num_cached < expected_num * 0.9:
            print(f"缓存不完整: {num_cached} < {expected_num}")
            return False
        
        print(f"缓存有效: {num_cached} 张图像")
        return True
        
    except Exception as e:
        print(f"检查缓存失败: {e}")
        return False


def load_camus_from_cache(data_root: str) -> Tuple[List[str], List[int]]:
    """
    从缓存加载CAMUS数据
    
    Args:
        data_root: CAMUS数据集根目录
    
    Returns:
        (image_paths, labels)
    """
    cache_dir = get_cache_dir(data_root)
    cache_images_dir = cache_dir / "images"
    
    image_paths = []
    labels = []
    
    for img_file in sorted(cache_images_dir.glob("*.png")):
        filename = img_file.name
        if '_2CH_' in filename:
            labels.append(CLASS_IDX_2CH)
        elif '_4CH_' in filename:
            labels.append(CLASS_IDX_4CH)
        else:
            continue
        image_paths.append(str(img_file))
    
    return image_paths, labels


def generate_cache(data_root: str, force: bool = False) -> bool:
    """
    生成CAMUS数据缓存 (包括 2CH 和 4CH)
    
    Args:
        data_root: CAMUS数据集根目录
        force: 是否强制重新生成
    
    Returns:
        是否成功生成缓存
    """
    if not force and check_cache_valid(data_root):
        print("缓存已有效，跳过生成")
        return True
    
    try:
        import nibabel as nib
    except ImportError:
        print("错误: 需要安装 nibabel 库来读取 NIfTI 文件")
        print("请运行: pip install nibabel")
        return False
    
    data_path = Path(data_root)
    cache_dir = get_cache_dir(data_root)
    cache_images_dir = cache_dir / "images"
    
    if cache_images_dir.exists():
        import shutil
        shutil.rmtree(cache_images_dir)
    cache_images_dir.mkdir(parents=True, exist_ok=True)
    
    patient_dirs = sorted(data_path.glob("patient*"))
    print(f"找到 {len(patient_dirs)} 个患者目录")
    print(f"缓存目录: {cache_images_dir}")
    
    frame_count_2ch = 0
    frame_count_4ch = 0
    patient_count = 0
    failed_patients = []
    
    for patient_dir in patient_dirs:
        if not patient_dir.is_dir():
            continue
        
        patient_id = patient_dir.name
        
        half_sequence_2ch = patient_dir / f"{patient_id}_2CH_half_sequence.nii.gz"
        half_sequence_4ch = patient_dir / f"{patient_id}_4CH_half_sequence.nii.gz"
        
        has_2ch = half_sequence_2ch.exists()
        has_4ch = half_sequence_4ch.exists()
        
        if not has_2ch and not has_4ch:
            continue
        
        patient_processed = False
        
        for view_type in ['2CH', '4CH']:
            half_sequence_file = patient_dir / f"{patient_id}_{view_type}_half_sequence.nii.gz"
            
            if not half_sequence_file.exists():
                continue
            
            try:
                img = nib.load(str(half_sequence_file))
                data = img.get_fdata()
                
                if len(data.shape) == 3:
                    num_frames = data.shape[2]
                elif len(data.shape) == 4:
                    num_frames = data.shape[3]
                else:
                    continue
                
                for frame_idx in range(num_frames):
                    if len(data.shape) == 3:
                        frame = data[:, :, frame_idx]
                    else:
                        frame = data[:, :, 0, frame_idx]
                    
                    frame_normalized = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)
                    
                    from PIL import Image
                    frame_image = Image.fromarray(frame_normalized)
                    
                    frame_image = frame_image.rotate(-90)
                    
                    output_path = cache_images_dir / f"{patient_id}_{view_type}_frame_{frame_idx:03d}.png"
                    frame_image.save(output_path)
                    
                    if view_type == '2CH':
                        frame_count_2ch += 1
                    else:
                        frame_count_4ch += 1
                
                if not patient_processed:
                    patient_count += 1
                    patient_processed = True
                
                if patient_count % 100 == 0:
                    print(f"已处理 {patient_count} 个患者, 2CH:{frame_count_2ch} 帧, 4CH:{frame_count_4ch} 帧")
                
            except Exception as e:
                failed_patients.append(patient_id)
                print(f"警告: 处理 {patient_id} 时出错: {e}")
                continue
    
    if failed_patients:
        print(f"失败的患者数: {len(failed_patients)}")
    
    metadata = {
        'data_hash': get_cache_hash(data_root),
        'num_frames_2ch': frame_count_2ch,
        'num_frames_4ch': frame_count_4ch,
        'num_patients': patient_count,
        'num_failed': len(failed_patients),
        'failed_patients': failed_patients,
        'cache_created': str(Path(__file__).stat().st_mtime)
    }
    
    metadata_file = cache_dir / CACHE_METADATA_FILE
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"CAMUS缓存生成完成:")
    print(f"  患者数: {patient_count}")
    print(f"  2CH帧数: {frame_count_2ch}")
    print(f"  4CH帧数: {frame_count_4ch}")
    print(f"  缓存目录: {cache_images_dir}")
    
    return frame_count_2ch > 0 or frame_count_4ch > 0


def clear_cache(data_root: str) -> bool:
    """
    清理CAMUS数据缓存
    
    Args:
        data_root: CAMUS数据集根目录
    
    Returns:
        是否成功清理
    """
    cache_dir = get_cache_dir(data_root)
    
    if not cache_dir.exists():
        print("缓存目录不存在，无需清理")
        return True
    
    try:
        import shutil
        shutil.rmtree(cache_dir)
        print(f"已清理缓存: {cache_dir}")
        return True
    except Exception as e:
        print(f"清理缓存失败: {e}")
        return False


def get_cache_status(data_root: str) -> Dict:
    """
    获取缓存状态信息
    
    Args:
        data_root: CAMUS数据集根目录
    
    Returns:
        缓存状态字典
    """
    cache_dir = get_cache_dir(data_root)
    metadata_file = cache_dir / CACHE_METADATA_FILE
    
    status = {
        'exists': False,
        'valid': False,
        'num_frames': 0,
        'num_patients': 0
    }
    
    if not cache_dir.exists():
        return status
    
    cache_images_dir = cache_dir / "images"
    if cache_images_dir.exists():
        status['num_frames'] = len(list(cache_images_dir.glob("*.png")))
        status['exists'] = True
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            status['num_patients'] = metadata.get('num_patients', 0)
            status['valid'] = check_cache_valid(data_root)
        except Exception:
            pass
    
    return status


def get_camus_data_info(
    data_root: str, 
    use_cache: bool = True,
    force_cache: bool = False
) -> Tuple[List[str], List[int]]:
    """
    获取CAMUS数据集的2CH视图帧数据
    
    CAMUS数据集结构 (您的实际数据):
        database_nifti/
        ├── patient0001/
        │   ├── patient0001_2CH_half_sequence.nii.gz
        │   └── ...
        ├── patient0002/
        │   └── ...
        └── ...
    
    每个 half_sequence 文件包含 15-25 帧超声图像，
    我们提取所有帧作为 A2C 类别的训练数据。
    
    Args:
        data_root: CAMUS数据集根目录
        use_cache: 是否使用缓存（默认启用）
        force_cache: 是否强制重新生成缓存
    
    Returns:
        image_paths: 图像文件路径列表
        labels: 对应的标签索引列表 (CAMUS A2C类，标签为6)
    """
    data_path = Path(data_root)
    
    if not data_path.exists():
        print(f"警告: CAMUS数据集目录不存在 - {data_path}")
        print("请检查数据路径是否正确")
        return [], []
    
    if use_cache:
        if force_cache:
            print("强制重新生成缓存...")
            if generate_cache(data_root, force=True):
                return load_camus_from_cache(data_root)
            return [], []
        
        if check_cache_valid(data_root):
            print("从缓存加载CAMUS数据...")
            return load_camus_from_cache(data_root)
        
        print("缓存无效或不存在，正在生成缓存...")
        if generate_cache(data_root):
            return load_camus_from_cache(data_root)
        return [], []
    
    return extract_camus_frames(data_root)


def extract_camus_frames(data_root: str) -> Tuple[List[str], List[int]]:
    """
    直接从NIfTI提取帧（不使用缓存）
    
    Args:
        data_root: CAMUS数据集根目录
    
    Returns:
        (image_paths, labels)
    """
    try:
        import nibabel as nib
    except ImportError:
        print("错误: 需要安装 nibabel 库来读取 NIfTI 文件")
        print("请运行: pip install nibabel")
        return [], []
    
    image_paths = []
    labels = []
    
    data_path = Path(data_root)
    patient_dirs = sorted(data_path.glob("patient*"))
    
    print(f"找到 {len(patient_dirs)} 个患者目录")
    print("警告: 每次运行都会重新提取帧，建议使用缓存")
    
    frame_count = 0
    patient_count = 0
    
    for patient_dir in patient_dirs:
        if not patient_dir.is_dir():
            continue
        
        patient_id = patient_dir.name
        half_sequence_file = patient_dir / f"{patient_id}_2CH_half_sequence.nii.gz"
        
        if not half_sequence_file.exists():
            continue
        
        try:
            img = nib.load(str(half_sequence_file))
            data = img.get_fdata()
            
            if len(data.shape) == 3:
                num_frames = data.shape[2]
            elif len(data.shape) == 4:
                num_frames = data.shape[3]
            else:
                continue
            
            for frame_idx in range(num_frames):
                if len(data.shape) == 3:
                    frame = data[:, :, frame_idx]
                else:
                    frame = data[:, :, 0, frame_idx]
                
                frame_normalized = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)
                
                from PIL import Image
                frame_image = Image.fromarray(frame_normalized)
                
                output_path = data_path / f"_temp_{patient_id}_frame_{frame_idx:03d}.png"
                frame_image.save(output_path)
                
                image_paths.append(str(output_path))
                labels.append(CLASS_IDX)
                frame_count += 1
            
            patient_count += 1
            
        except Exception as e:
            print(f"警告: 处理 {patient_id} 时出错: {e}")
            continue
    
    print(f"CAMUS数据集加载完成:")
    print(f"  患者数: {patient_count}")
    print(f"  2CH帧数: {frame_count}")
    print(f"  类别: A2C (索引: {CLASS_IDX})")
    
    return image_paths, labels


def get_camus_data_info_patient_level(data_root: str) -> Dict[str, Dict]:
    """
    按患者获取CAMUS数据信息（不提取帧）
    
    Args:
        data_root: CAMUS数据集根目录
    
    Returns:
        患者信息字典，包含每个患者的帧数等信息
    """
    try:
        import nibabel as nib
    except ImportError:
        print("错误: 需要安装 nibabel 库")
        return {}
    
    patient_info = {}
    data_path = Path(data_root)
    
    if not data_path.exists():
        return {}
    
    patient_dirs = sorted(data_path.glob("patient*"))
    
    for patient_dir in patient_dirs:
        if not patient_dir.is_dir():
            continue
        
        patient_id = patient_dir.name
        half_sequence_file = patient_dir / f"{patient_id}_2CH_half_sequence.nii.gz"
        
        if not half_sequence_file.exists():
            continue
        
        try:
            img = nib.load(str(half_sequence_file))
            data = img.get_fdata()
            
            if len(data.shape) == 3:
                num_frames = data.shape[2]
                width, height = data.shape[0], data.shape[1]
            elif len(data.shape) == 4:
                num_frames = data.shape[3]
                width, height = data.shape[0], data.shape[1]
            else:
                continue
            
            patient_info[patient_id] = {
                'num_frames': num_frames,
                'width': width,
                'height': height,
                'file': str(half_sequence_file)
            }
            
        except Exception as e:
            continue
    
    return patient_info


def get_camus_statistics(data_root: str) -> Dict:
    """
    获取CAMUS数据集的统计信息
    
    Args:
        data_root: CAMUS数据集根目录
    
    Returns:
        统计信息字典
    """
    cache_status = get_cache_status(data_root)
    
    if cache_status.get('num_frames', 0) > 0:
        return {
            'num_patients': cache_status.get('num_patients', 0),
            'total_frames': cache_status.get('num_frames', 0),
            'cache_valid': cache_status.get('valid', False),
            'source': 'cache'
        }
    
    patient_info = get_camus_data_info_patient_level(data_root)
    
    if not patient_info:
        return {}
    
    num_frames_list = [info['num_frames'] for info in patient_info.values()]
    width_list = [info['width'] for info in patient_info.values()]
    height_list = [info['height'] for info in patient_info.values()]
    
    stats = {
        'num_patients': len(patient_info),
        'total_frames': sum(num_frames_list),
        'avg_frames': np.mean(num_frames_list),
        'min_frames': min(num_frames_list),
        'max_frames': max(num_frames_list),
        'avg_width': np.mean(width_list),
        'avg_height': np.mean(height_list),
        'min_width': min(width_list),
        'max_width': max(width_list),
        'min_height': min(height_list),
        'max_height': max(height_list),
        'source': 'raw_nifti'
    }
    
    return stats


def verify_camus_data(data_root: str) -> bool:
    """
    验证CAMUS数据集是否存在
    
    Args:
        data_root: CAMUS数据集根目录
    
    Returns:
        是否存在
    """
    data_path = Path(data_root)
    exists = data_path.exists() and any(data_path.glob("patient*"))
    
    if not exists:
        print(f"错误: CAMUS数据集目录不存在或格式不正确 - {data_root}")
        print("期望目录结构: database_nifti/patient0001/...")
    
    return exists


def download_camus_instructions() -> str:
    """
    获取CAMUS数据集下载说明
    
    Returns:
        下载说明文本
    """
    return """
CAMUS 数据集说明
================

您的数据位于: D:/SRTP_Project__DeepLearning/project/Resources/database_nifti/

数据格式说明:
- 每个患者有一个文件夹 (patient0001 ~ patient0500)
- 每个文件夹包含多个 .nii.gz 文件
- 我们只使用 *2CH_half_sequence.nii.gz 文件

数据量估算:
- 500 个患者
- 每个患者约 15-25 帧
- 总计约 10,000 张 A2C (心尖两腔心) 图像

使用说明:
- 2CH_half_sequence 包含心脏跳动的半周期超声视频帧
- 每帧需要从 NIfTI 格式中提取并转换为图像
- 图像尺寸不统一，需要 resize 到 224x224

缓存说明:
- 首次运行时会自动生成缓存到 _camus_cache/images/
- 缓存生成后可重复使用，无需再次解压NIfTI
- 可使用 clear_cache() 清理缓存
- 可通过 get_cache_status() 查看缓存状态
"""


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = 'D:/SRTP_Project__DeepLearning/project/Resources/database_nifti'
    
    print(f"CAMUS数据集路径: {data_root}")
    print("=" * 50)
    
    if verify_camus_data(data_root):
        print("\n缓存状态:")
        status = get_cache_status(data_root)
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print("\n数据集统计:")
        stats = get_camus_statistics(data_root)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 50)
        print("测试加载数据...")
        paths, labels = get_camus_data_info(data_root, use_cache=True)
        print(f"加载了 {len(paths)} 张图像")
    else:
        print(download_camus_instructions())
