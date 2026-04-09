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
        default='D:/SRTP_Project__DeepLearning/project/Resources/database_nifti',
        help='CAMUS数据集路径 (二分类任务必需)'
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
        default=0.3,
        help='Dropout概率'
    )
    
    parser.add_argument(
        '--label_smoothing',
        type=float,
        default=0.1,
        help='标签平滑系数'
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
    
    parser.add_argument(
        '--kfold', 
        type=int, 
        default=0,
        help='K折交叉验证 (0=不使用kfold，5=5折交叉验证)'
    )
    
    parser.add_argument(
        '--kfold_save_all', 
        action='store_true',
        default=True,
        help='保存所有kfold模型的权重'
    )
    
    parser.add_argument(
        '--stratified', 
        action='store_true',
        default=True,
        help='分层采样，保持类别比例'
    )
    
    parser.add_argument(
        '--no_kfold',
        action='store_true',
        help='显式使用简单划分模式（不使用kfold）'
    )
    
    parser.add_argument(
        '--holdout_split', 
        type=float, 
        default=0.0,
        help='保留测试集比例（用于最终验证，不参与训练）'
    )
    
    parser.add_argument(
        '--task_type', 
        type=str, 
        default='multi_class',
        choices=['multi_class', 'binary'],
        help='任务类型: multi_class(7类) 或 binary(二分类 A2C vs A4C)'
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
    logger.info(f"  Holdout比例: {args.holdout_split}")
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
    
    # 根据任务类型调整checkpoint目录和类别
    if args.task_type == 'binary':
        BINARY_CLASS_NAMES = ['A2C', 'A4C']
        args.num_classes = 2
        checkpoint_dir_base = args.checkpoint_dir
        args.checkpoint_dir = os.path.join(checkpoint_dir_base, 'binary')
        log_dir_base = args.log_dir
        args.log_dir = os.path.join(log_dir_base, 'binary')
        if args.experiment_name is None:
            args.experiment_name = 'binary_classifier'
        print(f"\n任务类型: 二分类 (A2C vs A4C)")
        print(f"结果目录: {args.checkpoint_dir}")
    else:
        BINARY_CLASS_NAMES = None
        print(f"\n任务类型: 多分类 (7类)")
    
    print("\n验证数据集...")
    from data.cactus_loader import verify_cactus_data
    from data.camus_loader import verify_camus_data
    
    if args.task_type == 'binary':
        if not os.path.exists(args.camus_data):
            logger.error(f"CAMUS 数据集不存在: {args.camus_data}")
            return
        camus_valid = verify_camus_data(args.camus_data)
        if not camus_valid:
            logger.error("CAMUS 数据集验证失败，请检查数据路径")
            return
        logger.info("CAMUS 数据集验证通过")
    else:
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
    
    if args.no_kfold:
        args.kfold = 0
    
    print("\n加载数据集...")
    try:
        if args.task_type == 'binary':
            from data import get_binary_data_loaders
            train_loader, val_loader, test_loader, holdout_loader = get_binary_data_loaders(
                camus_data_root=args.camus_data,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                val_split=args.val_split,
                test_split=args.test_split,
                random_seed=args.seed,
                holdout_split=args.holdout_split
            )
            class_names = BINARY_CLASS_NAMES
        else:
            from data import get_data_loaders
            train_loader, val_loader, test_loader, holdout_loader = get_data_loaders(
                cactus_data_root=args.cactus_data,
                camus_data_root=args.camus_data if os.path.exists(args.camus_data) else None,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                val_split=args.val_split,
                test_split=args.test_split,
                random_seed=args.seed,
                holdout_split=args.holdout_split
            )
            class_names = CLASS_NAMES
        
        class_weights = train_loader.dataset.get_class_weights()
        logger.info(f"\n类别权重: {class_weights.numpy()}")
        
        data_info = {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'holdout_samples': len(holdout_loader.dataset) if holdout_loader else 0,
            'num_classes': args.num_classes,
            'class_names': class_names
        }
        logger.log_dataset_info(data_info)
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return
    
    logger.info("\n训练配置:")
    if args.kfold > 0:
        logger.info(f"  训练模式: {args.kfold}折交叉验证")
        logger.info(f"  分层采样: {args.stratified}")
        logger.info(f"  保存所有折: {args.kfold_save_all}")
        kfold_dir = os.path.join(args.checkpoint_dir, 'kfold')
        os.makedirs(kfold_dir, exist_ok=True)
        logger.info(f"  KFold模型目录: {kfold_dir}")
    else:
        logger.info(f"  训练模式: 简单划分")
    
    if args.kfold > 0:
        run_kfold_training(args, logger, class_weights, device, train_loader, val_loader, data_info, class_names)
    else:
        run_simple_training(args, logger, class_weights, device, train_loader, val_loader, test_loader, holdout_loader)
    
    logger.info("\n训练完成!")
    logger.close()


def run_simple_training(args, logger, class_weights, device, train_loader, val_loader, test_loader, holdout_loader=None):
    """简单划分模式训练"""
    from models import load_model_with_pretrained, create_usfmae_backbone, CardiacClassifier
    from utils import create_optimizer, create_scheduler
    
    print("创建模型...")
    if os.path.exists(args.pretrained):
        model = load_model_with_pretrained(
            pretrained_path=args.pretrained,
            num_classes=args.num_classes,
            freeze_backbone=args.freeze_backbone,
            device=device
        )
    else:
        backbone = create_usfmae_backbone(pretrained_path=None, freeze=args.freeze_backbone, device=device)
        model = CardiacClassifier(backbone=backbone, num_classes=args.num_classes, dropout=args.dropout).to(device)
    
    optimizer_config = {'type': 'adamw', 'lr': args.lr, 'weight_decay': args.weight_decay}
    optimizer = create_optimizer(model, optimizer_config)
    scheduler = create_scheduler(optimizer, {'type': args.scheduler, 'T_max': args.epochs, 'eta_min': args.lr * 0.01})
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, checkpoint_dir=args.checkpoint_dir, logger=logger,
        use_amp=args.use_amp, gradient_clip=args.gradient_clip if args.gradient_clip > 0 else None,
        early_stopping_patience=args.early_stopping_patience
    )
    
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
    
    print("\n开始训练...")
    history = trainer.train(num_epochs=args.epochs, save_best=True, save_last=True, save_frequency=5)
    
    logger.save_history(history)
    logger.info(f"\n训练完成! 最佳验证准确率: {trainer.best_val_acc:.2f}%")
    
    if holdout_loader is not None:
        logger.info("\n在Holdout集上进行最终验证...")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in holdout_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        holdout_acc = 100 * correct / total
        logger.info(f"Holdout集准确率: {holdout_acc:.2f}%")


def run_kfold_training(args, logger, class_weights, device, train_loader, val_loader, data_info, class_names, holdout_loader=None):
    """K折交叉验证训练"""
    import json
    from sklearn.model_selection import KFold, StratifiedKFold
    from data.dataset import CardiacDataset, get_train_transforms, get_val_transforms
    from models import load_model_with_pretrained, create_usfmae_backbone, CardiacClassifier
    from utils import create_optimizer, create_scheduler
    
    kfold_dir = os.path.join(args.checkpoint_dir, 'kfold')
    
    all_paths = []
    all_labels = []
    
    if args.task_type == 'binary':
        from data.camus_loader import get_camus_binary_data_info
        camus_paths, camus_labels = get_camus_binary_data_info(args.camus_data)
        all_paths.extend(camus_paths)
        all_labels.extend(camus_labels)
    else:
        from data.cactus_loader import get_cactus_data_info
        from data.camus_loader import get_camus_data_info
        cactus_paths, cactus_labels = get_cactus_data_info(args.cactus_data)
        all_paths.extend(cactus_paths)
        all_labels.extend(cactus_labels)
        
        if args.camus_data and os.path.exists(args.camus_data):
            camus_paths, camus_labels = get_camus_data_info(args.camus_data)
            all_paths.extend(camus_paths)
            all_labels.extend(camus_labels)
    
    all_labels = np.array(all_labels)
    n_samples = len(all_paths)
    
    print(f"\n开始{args.kfold}折交叉验证...")
    logger.info(f"\n开始{args.kfold}折交叉验证，共{n_samples}个样本")
    
    if args.stratified:
        kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    else:
        kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    
    fold_results = []
    best_fold = 0
    best_acc = 0.0
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_paths, all_labels)):
        fold_num = fold + 1
        print(f"\n{'='*60}")
        print(f"{args.kfold}折交叉验证 - Fold {fold_num}/{args.kfold}")
        print(f"{'='*60}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{args.kfold}折交叉验证 - Fold {fold_num}/{args.kfold}")
        logger.info(f"{'='*60}")
        
        train_paths = [all_paths[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_paths = [all_paths[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        
        logger.info(f"训练集: {len(train_paths)}样本, 验证集: {len(val_paths)}样本")
        
        train_transform = get_train_transforms(args.img_size)
        val_transform = get_val_transforms(args.img_size)
        
        train_dataset = CardiacDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = CardiacDataset(val_paths, val_labels, transform=val_transform)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        print("创建模型...")
        if os.path.exists(args.pretrained):
            model = load_model_with_pretrained(
                pretrained_path=args.pretrained,
                num_classes=args.num_classes,
                freeze_backbone=args.freeze_backbone,
                device=device
            )
        else:
            backbone = create_usfmae_backbone(pretrained_path=None, freeze=args.freeze_backbone, device=device)
            model = CardiacClassifier(backbone=backbone, num_classes=args.num_classes, dropout=args.dropout).to(device)
        
        optimizer_config = {'type': 'adamw', 'lr': args.lr, 'weight_decay': args.weight_decay}
        optimizer = create_optimizer(model, optimizer_config)
        scheduler = create_scheduler(optimizer, {'type': args.scheduler, 'T_max': args.epochs, 'eta_min': args.lr * 0.01})
        
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=args.label_smoothing)
        
        fold_checkpoint_dir = os.path.join(kfold_dir, f'fold_{fold_num}')
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        
        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            device=device, checkpoint_dir=fold_checkpoint_dir, logger=logger,
            use_amp=args.use_amp, gradient_clip=args.gradient_clip if args.gradient_clip > 0 else None,
            early_stopping_patience=args.early_stopping_patience
        )
        
        history = trainer.train(num_epochs=args.epochs, save_best=True, save_last=False, save_frequency=0)
    
    val_acc = trainer.best_val_acc
    fold_results.append({
        'fold': fold_num,
        'val_acc': val_acc,
        'train_samples': len(train_paths),
        'val_samples': len(val_paths)
    })
    
    if args.kfold_save_all:
        best_model_path = os.path.join(fold_checkpoint_dir, 'best_model.pth')
        new_model_path = os.path.join(kfold_dir, f'fold_{fold_num}.pth')
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy(best_model_path, new_model_path)
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_fold = fold_num
    
    logger.info(f"Fold {fold_num}完成! Val Acc: {val_acc:.2f}%")
    print(f"Fold {fold_num}完成! Val Acc: {val_acc:.2f}%")

    # 复制最佳模型到 best.pth (在所有折训练完成后)
    import shutil
    best_model_source = os.path.join(kfold_dir, f'fold_{best_fold}.pth')
    best_model_dest = os.path.join(kfold_dir, 'best.pth')
    if os.path.exists(best_model_source):
        shutil.copy(best_model_source, best_model_dest)
        print(f"✓ 最佳模型已保存: best.pth")
    
    accs = [r['val_acc'] for r in fold_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    print(f"\n{'='*60}")
    print(f"{args.kfold}折交叉验证结果汇总")
    print(f"{'='*60}")
    print(f"| Fold | Val Acc |")
    print(f"|------|---------|")
    for r in fold_results:
        marker = " ★" if r['fold'] == best_fold else ""
        print(f"|  {r['fold']}   | {r['val_acc']:.2f}%  |{marker}")
    print(f"{'-'*60}")
    print(f"平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"最佳折: Fold {best_fold} ({best_acc:.2f}%)")
    print(f"{'='*60}")
    print(f"✓ 最佳模型: fold_{best_fold}.pth (Val Acc: {best_acc:.2f}%)")
    print(f"✓ 已复制到: best.pth")
    
    # 复制最佳模型到 best.pth
    import shutil
    best_model_source = os.path.join(kfold_dir, f'fold_{best_fold}.pth')
    best_model_dest = os.path.join(kfold_dir, 'best.pth')
    if os.path.exists(best_model_source):
        shutil.copy(best_model_source, best_model_dest)
        print(f"✓ 最佳模型已保存: best.pth")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{args.kfold}折交叉验证结果汇总")
    logger.info(f"{'='*60}")
    for r in fold_results:
        marker = " ★" if r['fold'] == best_fold else ""
        logger.info(f"Fold {r['fold']}: Val Acc = {r['val_acc']:.2f}%{marker}")
    logger.info(f"平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"最佳折: Fold {best_fold} ({best_acc:.2f}%)")
    logger.info(f"✓ 最佳模型: kfold/fold_{best_fold}.pth")
    logger.info(f"✓ 已复制到: kfold/best.pth")
    
    kfold_results = {
        'fold_results': fold_results,
        'mean_acc': float(mean_acc),
        'std_acc': float(std_acc),
        'best_fold': best_fold,
        'best_acc': float(best_acc),
        'kfold': args.kfold,
        'stratified': args.stratified
    }
    
    results_path = os.path.join(kfold_dir, 'kfold_results.json')
    with open(results_path, 'w') as f:
        json.dump(kfold_results, f, indent=2)
    logger.info(f"结果已保存: {results_path}")
    
    if holdout_loader is not None:
        logger.info("\n在Holdout集上进行最终验证...")
        best_model_path = os.path.join(kfold_dir, 'best.pth')
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in holdout_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        holdout_acc = 100 * correct / total
        logger.info(f"Holdout集准确率: {holdout_acc:.2f}%")
        
        kfold_results['holdout_acc'] = float(holdout_acc)
        with open(results_path, 'w') as f:
            json.dump(kfold_results, f, indent=2)
    
    logger.close()


if __name__ == '__main__':
    main()