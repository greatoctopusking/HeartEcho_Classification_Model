import os
import sys
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
import torch


class Logger:
    """
    日志记录器，用于记录训练过程和实验结果
    """
    
    def __init__(
        self,
        log_dir: str = 'logs',
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = False
    ):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称
            use_tensorboard: 是否使用 TensorBoard
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.use_tensorboard = use_tensorboard
        
        self.exp_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.logger = self._setup_logger()
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.exp_dir)
            except ImportError:
                print("警告: TensorBoard 不可用，已禁用")
                self.writer = None
                self.use_tensorboard = False
        else:
            self.writer = None
        
        self.logger.info(f"实验目录: {self.exp_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置 logger"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        
        logger.handlers = []
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler(
            os.path.join(self.exp_dir, 'training.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str):
        """记录信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误"""
        self.logger.error(message)
    
    def log_config(self, config: Dict[str, Any]):
        """记录配置信息"""
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        self.logger.info(f"配置已保存至: {config_path}")
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = 'train'):
        """
        记录指标
        
        Args:
            epoch: 当前 epoch
            metrics: 指标字典
            phase: 阶段 ('train', 'val', 'test')
        """
        metric_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} [{phase}] - {metric_str}")
        
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f'{phase}/{key}', value, epoch)
    
    def log_learning_rate(self, epoch: int, lr: float):
        """记录学习率"""
        self.logger.info(f"Epoch {epoch} - Learning Rate: {lr:.6f}")
        if self.writer is not None:
            self.writer.add_scalar('learning_rate', lr, epoch)
    
    def log_model_info(self, model: torch.nn.Module):
        """记录模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = f"模型参数量: {total_params:,} (可训练: {trainable_params:,})"
        self.logger.info(info)
        
        if self.writer is not None:
            self.writer.add_text('model_info', info)
    
    def log_dataset_info(self, data_info: Dict[str, Any]):
        """记录数据集信息"""
        info = "数据集信息:\n"
        for key, value in data_info.items():
            info += f"  {key}: {value}\n"
        self.logger.info(info)
    
    def save_history(self, history: Dict[str, list]):
        """保存训练历史"""
        history_path = os.path.join(self.exp_dir, 'history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        self.logger.info(f"训练历史已保存至: {history_path}")
    
    def close(self):
        """关闭日志记录器"""
        if self.writer is not None:
            self.writer.close()
        self.logger.info(f"实验完成，日志保存于: {self.exp_dir}")


def setup_logging(
    log_dir: str = 'logs',
    experiment_name: Optional[str] = None,
    level: int = logging.INFO
) -> Logger:
    """
    设置日志系统
    
    Args:
        log_dir: 日志目录
        experiment_name: 实验名称
        level: 日志级别
    
    Returns:
        Logger 实例
    """
    logger_obj = Logger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=False
    )
    
    return logger_obj


def log_system_info(logger: Optional[Logger] = None):
    """
    记录系统信息
    
    Args:
        logger: Logger 实例
    """
    import platform
    
    info = {
        'Python Version': platform.python_version(),
        'PyTorch Version': torch.__version__,
        'CUDA Available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['CUDA Version'] = torch.version.cuda
        info['GPU Device'] = torch.cuda.get_device_name(0)
        info['GPU Count'] = torch.cuda.device_count()
    
    if logger:
        logger.info("系统信息:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
    else:
        print("系统信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")


def save_checkpoint_info(checkpoint_path: str, save_path: str):
    """
    保存检查点信息
    
    Args:
        checkpoint_path: 检查点路径
        save_path: 保存路径
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'best_val_acc': checkpoint.get('best_val_acc', 0.0),
        'history_keys': list(checkpoint.get('history', {}).keys())
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    
    print(f"检查点信息已保存至: {save_path}")


if __name__ == '__main__':
    print("Logger 模块测试...")
    
    logger = Logger(
        log_dir='logs',
        experiment_name='test_experiment'
    )
    
    logger.info("测试信息")
    logger.log_config({'batch_size': 32, 'lr': 1e-4})
    logger.log_metrics(1, {'loss': 0.5, 'acc': 80.0})
    logger.log_model_info(torch.nn.Linear(10, 2))
    
    logger.close()
    
    print("Logger 测试完成")