#!/usr/bin/env python
"""
心脏超声切面分类模型训练入口

使用方法:
    python train.py --cactus_data "CACTUS/Images Dataset" --batch_size 32 --epochs 50
    python train.py --config configs/train_config.yaml
    python train.py --cactus_data "data/CACTUS" --pretrained "weights/usfmae.pt" --lr 1e-4
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path

from data import get_data_loaders
from models import load_model_with_pretrained, create_usfmae_backbone, CardiacClassifier
from utils import Trainer, Logger, create_optimizer, create_scheduler, log_system_info


CLASS_NAMES = ['A4C', 'PL', 'PSAV', 'PSMV', 'Random', 'SC', 'A2C']


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='心脏超声切面分类模型训练',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--cactus_data', 
        type=str, 
        default='CACTUS/Images Dataset',
        help='CACTUS数据集路径'
    )
    
    parser.add_argument(
        '--camus_data', 
        type=str, 
        default='CAMUS',
        help='CAMUS数据集路径 (可选)'
    )
    
    parser.add_argument(
        '--pretrained', 
        type=str, 
        default='USF-MAE pretrained/USF-MAE_full_pretrain_43dataset_100epochs.pt',
        help='预训练权重路径'
    )
    
    parser.add_argument(
        '--num_classes', 
        type=int, 
        default=7,
        help='分类类别数'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='批大小'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='训练轮数'
    )
    
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-4,
        help='学习率'
    )
    
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=0.01,
        help='权重衰减'
    )
    
    parser.add_argument(
        '--img_size', 
        type=int, 
        default=224,
        help='输入图像尺寸'
    )
    
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=4,
        help='数据加载线程数'
    )
    
    parser.add_argument(
        '--val_split', 
        type=float, 
        default=0.15,
        help='验证集比例'
    )
    
    parser.add_argument(
        '--test_split', 
        type=float, 
        default=0.15,
        help='测试集比例'
    )
    
    parser.add_argument(
        '--freeze_backbone', 
        action='store_true',
        help='冻结backbone参数'
    )
    
    parser.add_argument(
        '--dropout', 
        type=float, 
        default=0.1,
        help='Dropout概率'
    )
    
    parser.add_argument(
        '--use_amp', 
        action='store_true',
        default=True,
        help='使用混合精度训练'
    )
    
    parser.add_argument(
        '--gradient_clip', 
        type=float, 
        default=1.0,
        help='梯度裁剪阈值'
    )
    
    parser.add_argument(
        '--early_stopping_patience', 
        type=int, 
        default=10,
        help='早停耐心值'
    )
    
    parser.add_argument(
        '--scheduler', 
        type=str, 
        default='cosine',
        choices=['cosine', 'step', 'plateau', 'none'],
        help='学习率调度器类型'
    )
    
    parser.add_argument(
        '--checkpoint_dir', 
        type=str, 
        default='checkpoints',
        help='模型保存目录'
    )
    
    parser.add_argument(
        '--log_dir', 
        type=str, 
        default='logs',
        help='日志保存目录'
    )
    
    parser.add_argument(
        '--experiment_name', 
        type=str, 
        default=None,
        help='实验名称'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='随机种子'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        help='训练设备'
    )
    
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='从checkpoint恢复训练'
    )
    
    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> dict:
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    
    if args.config is not None:
        config = load_config_from_yaml(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if hasattr(args, sub_key):
                            setattr(args, sub_key, sub_value)
                else:
                    setattr(args, key, value)
    
    set_seed(args.seed)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if args.experiment_name is None:
        args.experiment_name = f"cardiac_classifier_{args.num_classes}classes"
    
    logger = Logger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        use_tensorboard=False
    )
    
    logger.info("=" * 60)
    logger.info("心脏超声切面分类模型训练")
    logger.info("=" * 60)
    
    log_system_info(logger)
    
    logger.info("\n数据配置:")
    logger.info(f"  CACTUS数据: {args.cactus_data}")
    logger.info(f"  CAMUS数据: {args.camus_data}")
    logger.info(f"  验证集比例: {args.val_split}")
    logger.info(f"  测试集比例: {args.test_split}")
    logger.info(f"  图像尺寸: {args.img_size}")
    logger.info(f"  批大小: {args.batch_size}")
    logger.info(f"  Workers: {args.num_workers}")
    
    logger.info("\n模型配置:")
    logger.info(f"  预训练权重: {args.pretrained}")
    logger.info(f"  类别数: {args.num_classes}")
    logger.info(f"  冻结backbone: {args.freeze_backbone}")
    logger.info(f"  Dropout: {args.dropout}")
    
    logger.info("\n训练配置:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  学习率: {args.lr}")
    logger.info(f"  权重衰减: {args.weight_decay}")
    logger.info(f"  调度器: {args.scheduler}")
    logger.info(f"  混合精度: {args.use_amp}")
    logger.info(f"  梯度裁剪: {args.gradient_clip}")
    logger.info(f"  早停耐心: {args.early_stopping_patience}")
    
    logger.log_config(vars(args))
    
    print("\n验证数据集...")
    from data.cactus_loader import verify_cactus_data
    from data.camus_loader import verify_camus_data
    
    cactus_valid = verify_cactus_data(args.cactus_data)
    if not cactus_valid:
        logger.error("CACTUS 数据集验证失败，请检查数据路径")
        return
    
    camus_valid = True
    if args.camus_data and os.path.exists(args.camus_data):
        camus_valid = verify_camus_data(args.camus_data)
        if not camus_valid:
            logger.error("CAMUS 数据集验证失败，请检查数据路径")
            return
    else:
        logger.info("未找到 CAMUS 数据集，将仅使用 CACTUS 数据 (6类)")
        args.camus_data = None
    
    logger.info("数据集验证通过")
    
    print("\n加载数据集...")
    try:
        train_loader, val_loader, test_loader = get_data_loaders(
            cactus_data_root=args.cactus_data,
            camus_data_root=args.camus_data if os.path.exists(args.camus_data) else None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            test_split=args.test_split,
            random_seed=args.seed
        )
        
        class_weights = train_loader.dataset.get_class_weights()
        logger.info(f"\n类别权重: {class_weights.numpy()}")
        
        data_info = {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'num_classes': args.num_classes,
            'class_names': CLASS_NAMES
        }
        logger.log_dataset_info(data_info)
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return
    
    print("创建模型...")
    try:
        if os.path.exists(args.pretrained):
            model = load_model_with_pretrained(
                pretrained_path=args.pretrained,
                num_classes=args.num_classes,
                freeze_backbone=args.freeze_backbone,
                device=device
            )
            logger.info(f"成功加载预训练权重: {args.pretrained}")
        else:
            logger.warning(f"预训练权重不存在: {args.pretrained}，使用随机初始化")
            backbone = create_usfmae_backbone(
                pretrained_path=None,
                freeze=args.freeze_backbone,
                device=device
            )
            model = CardiacClassifier(
                backbone=backbone,
                num_classes=args.num_classes,
                dropout=args.dropout
            )
            model = model.to(device)
        
        logger.log_model_info(model)
        
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        return
    
    optimizer_config = {
        'type': 'adamw',
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }
    optimizer = create_optimizer(model, optimizer_config)
    
    scheduler_config = {
        'type': args.scheduler,
        'T_max': args.epochs,
        'eta_min': args.lr * 0.01
    }
    scheduler = create_scheduler(optimizer, scheduler_config)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        logger=logger,
        use_amp=args.use_amp,
        gradient_clip=args.gradient_clip if args.gradient_clip > 0 else None,
        early_stopping_patience=args.early_stopping_patience
    )
    
    if args.resume is not None and os.path.exists(args.resume):
        logger.info(f"从checkpoint恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    print("\n开始训练...")
    history = trainer.train(
        num_epochs=args.epochs,
        save_best=True,
        save_last=True,
        save_frequency=5
    )
    
    logger.save_history(history)
    
    logger.info("\n训练完成!")
    logger.info(f"最佳验证准确率: {trainer.best_val_acc:.2f}%")
    logger.info(f"模型保存目录: {args.checkpoint_dir}")
    logger.close()
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证准确率: {trainer.best_val_acc:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()