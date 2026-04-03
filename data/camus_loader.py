import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np


CLASS_NAME = 'A2C'
CLASS_IDX = 6


def get_camus_data_info(data_root: str) -> Tuple[List[str], List[int]]:
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
    
    Returns:
        image_paths: 图像文件路径列表（实际为临时文件路径）
        labels: 对应的标签索引列表 (CAMUS A2C类，标签为6)
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
    
    if not data_path.exists():
        print(f"警告: CAMUS数据集目录不存在 - {data_root}")
        print("请检查数据路径是否正确")
        return image_paths, labels
    
    patient_dirs = sorted(data_path.glob("patient*"))
    
    print(f"找到 {len(patient_dirs)} 个患者目录")
    
    temp_dir = Path(data_root) / "_temp_frames"
    temp_dir.mkdir(exist_ok=True)
    
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
                
                output_path = temp_dir / f"{patient_id}_frame_{frame_idx:03d}.png"
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
        'max_height': max(height_list)
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
=================

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
        stats = get_camus_statistics(data_root)
        print("数据集统计:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    else:
        print(download_camus_instructions())