"""
调试脚本：从 CACTUS 和 CAMUS 提取数据并预处理
"""
import os
import numpy as np
from PIL import Image
import nibabel as nib
import glob
from pathlib import Path

# 配置路径
CACTUS_ROOT = "D:/GithubRepositories/HeartEcho_Classification_Model/CACTUS/Images Dataset"
CAMUS_ROOT = "D:/SRTP_Project__DeepLearning/project/Resources/database_nifti"
OUTPUT_DIR = "D:/GithubRepositories/HeartEcho_Classification_Model/debug/preprocessed"
TARGET_SIZE = 224

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_cactus(img_path, target_size=TARGET_SIZE):
    """
    CACTUS 预处理:
    1. 裁剪掉设备信息边框 (左1/10, 右1/4, 下1/4)
    2. 转灰度
    3. Resize (保持宽高比)
    4. Padding 到目标尺寸
    """
    img = Image.open(img_path)
    orig_w, orig_h = img.size
    
    # 1. 裁剪 (左1/10, 右1/4, 下1/4)
    left = int(orig_w * 0.15)       # 左侧10%
    right = orig_w - int(orig_w * 0.35)  # 右侧25%
    top = 0
    bottom = orig_h - int(orig_h * 0.25)  # 下方25%
    crop_box = (left, top, right, bottom)
    img = img.crop(crop_box)
    crop_w, crop_h = img.size
    
    # 2. 转灰度
    img = img.convert('L')
    
    # 3. Resize: 保持16:9比例，宽度224，高度按比例
    new_w = target_size
    new_h = int(crop_h * target_size / crop_w)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # 4. Padding 到正方形
    result = Image.new('L', (target_size, target_size), 0)
    paste_y = (target_size - new_h) // 2
    result.paste(img, (0, paste_y))
    
    return result


def preprocess_camus(nifti_path, frame_idx=0, target_size=TARGET_SIZE):
    """
    CAMUS 预处理:
    1. 从NIfTI提取帧
    2. 顺时针旋转90度
    3. 转灰度
    4. Resize (保持宽高比)
    5. Padding 到目标尺寸
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # 提取单帧 (取中间帧)
    if len(data.shape) == 3:
        mid_frame = data.shape[2] // 2
        frame = data[:, :, mid_frame]
    elif len(data.shape) == 4:
        mid_frame = data.shape[3] // 2
        frame = data[:, :, 0, mid_frame]
    else:
        frame = data
    
    # 归一化到 0-255
    frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)
    
    # 转为 PIL Image
    img = Image.fromarray(frame, mode='L')
    
    # 顺时针旋转90度
    img = img.rotate(-90)  # PIL的rotate正角度是逆时针，所以用-90顺时针
    
    orig_w, orig_h = img.size
    
    # 计算宽高比 (旋转后变成横向的)
    aspect_ratio = orig_w / orig_h
    
    # 4. Resize: 宽度224，高度按比例
    new_w = target_size
    new_h = int(new_w / aspect_ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # 5. Padding 到正方形
    result = Image.new('L', (target_size, target_size), 0)
    paste_y = (target_size - new_h) // 2
    result.paste(img, (0, paste_y))
    
    return result


def main():
    print("=" * 60)
    print("数据预处理调试脚本")
    print("=" * 60)
    
    # ========== CACTUS ==========
    print("\n[1/2] 处理 CACTUS 数据...")
    cactus_categories = ['A4C', 'PL', 'PSAV', 'PSMV', 'Random', 'SC']
    cactus_output = os.path.join(OUTPUT_DIR, 'cactus')
    os.makedirs(cactus_output, exist_ok=True)
    
    cactus_count = 0
    for cat in cactus_categories:
        cat_dir = os.path.join(CACTUS_ROOT, cat)
        files = sorted(glob.glob(cat_dir + '/*.jpg'))[:2]  # 每类取2张
        for f in files:
            try:
                img = preprocess_cactus(f)
                out_path = os.path.join(cactus_output, f'{cat}_{cactus_count:02d}.png')
                img.save(out_path)
                print(f"  Saved: {out_path}")
                cactus_count += 1
                if cactus_count >= 10:
                    break
            except Exception as e:
                print(f"  Error processing {f}: {e}")
        if cactus_count >= 10:
            break
    
    print(f"  CACTUS 完成: {cactus_count} 张图像")
    
    # ========== CAMUS ==========
    print("\n[2/2] 处理 CAMUS 数据...")
    camus_output = os.path.join(OUTPUT_DIR, 'camus')
    os.makedirs(camus_output, exist_ok=True)
    
    # 获取所有2CH half_sequence文件
    camus_files = sorted(glob.glob(CAMUS_ROOT + '/patient*/patient*_2CH_half_sequence.nii.gz'))[:10]
    
    camus_count = 0
    for f in camus_files:
        try:
            # 取中间帧
            img = preprocess_camus(f)
            patient_id = Path(f).stem.replace('_2CH_half_sequence', '')
            out_path = os.path.join(camus_output, f'{patient_id}_{camus_count:02d}.png')
            img.save(out_path)
            print(f"  Saved: {out_path}")
            camus_count += 1
            if camus_count >= 10:
                break
        except Exception as e:
            print(f"  Error processing {f}: {e}")
    
    print(f"  CAMUS 完成: {camus_count} 张图像")
    
    # ========== 验证 ==========
    print("\n[3/2] 验证输出...")
    print(f"\n输出目录: {OUTPUT_DIR}")
    
    # 检查输出
    for subdir in ['cactus', 'camus']:
        subdir_path = os.path.join(OUTPUT_DIR, subdir)
        files = sorted(glob.glob(subdir_path + '/*.png'))
        print(f"\n{subdir}: {len(files)} files")
        
        # 检查第一张图片
        if files:
            img = Image.open(files[0])
            arr = np.array(img)
            print(f"  Sample: {files[0].split('/')[-1]}")
            print(f"    Size: {img.size}")
            print(f"    Mode: {img.mode}")
            print(f"    Pixel range: {arr.min()} - {arr.max()}")
            print(f"    Shape: {arr.shape}")
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
