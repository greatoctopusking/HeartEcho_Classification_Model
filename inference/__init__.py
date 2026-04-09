"""推理模块 - 独立于训练脚本的心脏超声切面分类推理功能"""

from .classifier import CardiacClassifier, load_model
from .predict import predict_single, predict_batch, predict_directory, predict_from_path
from .transforms import get_val_transforms, preprocess_image, load_image, load_nifti_image
from .constants import (
    ALL_CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, 
    NUM_CLASSES, DEFAULT_IMAGE_SIZE,
    BINARY_CLASS_NAMES, BINARY_CLASS_TO_IDX, BINARY_IDX_TO_CLASS,
    get_class_names, get_class_to_idx, get_idx_to_class
)

__all__ = [
    'CardiacClassifier',
    'load_model',
    'predict_single',
    'predict_batch',
    'predict_directory',
    'predict_from_path',
    'get_val_transforms',
    'preprocess_image',
    'load_image',
    'load_nifti_image',
    'ALL_CLASS_NAMES',
    'CLASS_TO_IDX',
    'IDX_TO_CLASS',
    'NUM_CLASSES',
    'DEFAULT_IMAGE_SIZE',
    'BINARY_CLASS_NAMES',
    'BINARY_CLASS_TO_IDX',
    'BINARY_IDX_TO_CLASS',
    'get_class_names',
    'get_class_to_idx',
    'get_idx_to_class',
]

__version__ = '1.1.0'