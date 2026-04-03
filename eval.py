#!/usr/bin/env python
"""
心脏超声切面分类模型评估入口

使用方法:
    python eval.py --checkpoint checkpoints/best_model.pth --data "CACTUS/Images Dataset"
    python eval.py --checkpoint checkpoints/best_model.pth --test_only
    python eval.py --checkpoint checkpoints/best_model.pth --plot_cm --plot_roc
"""

import os
import sys
import argparse
import torch
import yaml
from pathlib import Path

from data import get_data_loaders
from models import load_model_with_pretrained
from utils import Evaluator, compute_metrics, print_metrics, save_metrics


CLASS_NAMES = ['A4C', 'PL', 'PSAV', 'PSMV', 'Random', 'SC', 'A2C']


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='心脏超声切面分类模型评估',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='模型检查点路径'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='CACTUS/Images Dataset',
        help='数据集路径'
    )
    
    parser.add_argument(
        '--camus_data', 
        type=str, 
        default='CAMUS',
        help='CAMUS数据集路径'
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
        '--test_only', 
        action='store_true',
        help='仅在测试集上评估'
    )
    
    parser.add_argument(
        '--val_only', 
        action='store_true',
        help='仅在验证集上评估'
    )
    
    parser.add_argument(
        '--plot_cm', 
        action='store_true',
        help='绘制并保存混淆矩阵'
    )
    
    parser.add_argument(
        '--plot_roc', 
        action='store_true',
        help='绘制并保存ROC曲线'
    )
    
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='results',
        help='结果保存目录'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        help='评估设备'
    )
    
    parser.add_argument(
        '--fold', 
        type=int, 
        default=0,
        help='评估指定折的模型 (0=默认best模型)'
    )
    
    return parser.parse_args()


def load_config_from_yaml(config_path: str, args) -> None:
    """从YAML文件加载配置到args"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理顶层配置
    for key, value in config.items():
        if hasattr(args, key):
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if hasattr(args, sub_key):
                        setattr(args, sub_key, sub_value)
            else:
                setattr(args, key, value)


def main():
    args = parse_args()
    
    if args.config is not None:
        load_config_from_yaml(args.config, args)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if args.fold > 0:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.checkpoint = os.path.join(checkpoint_dir, f'fold_{args.fold}.pth')
        print(f"评估Fold {args.fold}模型: {args.checkpoint}")
    elif args.checkpoint.endswith('best.pth') or 'best_model' in args.checkpoint:
        print("评估最佳模型")
    else:
        print(f"评估模型: {args.checkpoint}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("加载数据集...")
    train_loader, val_loader, test_loader = get_data_loaders(
        cactus_data_root=args.data,
        camus_data_root=args.camus_data if os.path.exists(args.camus_data) else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=42
    )
    
    print(f"数据集加载完成:")
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  验证集: {len(val_loader.dataset)} 样本")
    print(f"  测试集: {len(test_loader.dataset)} 样本")
    
    print(f"\n加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model = load_model_with_pretrained(
            pretrained_path=None,
            num_classes=args.num_classes,
            freeze_backbone=False,
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = load_model_with_pretrained(
            pretrained_path=None,
            num_classes=args.num_classes,
            freeze_backbone=False,
            device=device
        )
        model.load_state_dict(checkpoint)
    
    print("模型加载成功")
    
    evaluator = Evaluator(
        model=model,
        device=device,
        class_names=CLASS_NAMES
    )
    
    if args.test_only:
        print("\n在测试集上评估...")
        test_metrics = evaluator.evaluate(test_loader)
        print_metrics(test_metrics)
        save_metrics(test_metrics, os.path.join(args.save_dir, 'test_metrics.json'))
        
        if args.plot_cm:
            evaluator.plot_confusion_matrix(
                test_loader,
                save_path=os.path.join(args.save_dir, 'test_confusion_matrix.png')
            )
        
        if args.plot_roc:
            evaluator.plot_roc_curves(
                test_loader,
                save_path=os.path.join(args.save_dir, 'test_roc_curves.png')
            )
    
    elif args.val_only:
        print("\n在验证集上评估...")
        val_metrics = evaluator.evaluate(val_loader)
        print_metrics(val_metrics)
        save_metrics(val_metrics, os.path.join(args.save_dir, 'val_metrics.json'))
        
        if args.plot_cm:
            evaluator.plot_confusion_matrix(
                val_loader,
                save_path=os.path.join(args.save_dir, 'val_confusion_matrix.png')
            )
        
        if args.plot_roc:
            evaluator.plot_roc_curves(
                val_loader,
                save_path=os.path.join(args.save_dir, 'val_roc_curves.png')
            )
    
    else:
        print("\n在测试集上评估...")
        test_metrics = evaluator.evaluate(test_loader)
        print_metrics(test_metrics)
        save_metrics(test_metrics, os.path.join(args.save_dir, 'test_metrics.json'))
        
        print("\n在验证集上评估...")
        val_metrics = evaluator.evaluate(val_loader)
        print_metrics(val_metrics)
        save_metrics(val_metrics, os.path.join(args.save_dir, 'val_metrics.json'))
        
        if args.plot_cm:
            evaluator.plot_confusion_matrix(
                test_loader,
                save_path=os.path.join(args.save_dir, 'test_confusion_matrix.png')
            )
            evaluator.plot_confusion_matrix(
                val_loader,
                save_path=os.path.join(args.save_dir, 'val_confusion_matrix.png')
            )
        
        if args.plot_roc:
            evaluator.plot_roc_curves(
                test_loader,
                save_path=os.path.join(args.save_dir, 'test_roc_curves.png')
            )
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print(f"结果保存目录: {args.save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()