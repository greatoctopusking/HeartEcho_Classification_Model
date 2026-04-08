import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
import time
from tqdm import tqdm
import numpy as np


class Trainer:
    """
    训练器类，负责模型训练和验证
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        logger: Optional[Callable] = None,
        use_amp: bool = False,
        gradient_clip: Optional[float] = None,
        early_stopping_patience: int = 10
    ):
        """
        初始化训练器
        
        Args:
            model: 待训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 训练设备
            checkpoint_dir: 模型保存目录
            logger: 日志记录器
            use_amp: 是否使用混合精度训练
            gradient_clip: 梯度裁剪阈值
            early_stopping_patience: 早停耐心值
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(model.parameters())
        self.scheduler = scheduler
        
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            epoch: 当前 epoch 编号
        
        Returns:
            训练指标字典
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss / (batch_idx + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'acc': epoch_acc}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            epoch: 当前 epoch 编号
        
        Returns:
            验证指标字典
        """
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0}
        
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss / (pbar.n + 1):.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return {'loss': val_loss, 'acc': val_acc}
    
    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        save_last: bool = True,
        save_frequency: int = 5
    ) -> Dict[str, list]:
        """
        执行完整的训练过程
        
        Args:
            num_epochs: 训练轮数
            save_best: 是否保存最佳模型
            save_last: 是否保存最后一个模型
            save_frequency: 保存频率（每N个epoch保存一次）
        
        Returns:
            训练历史记录
        """
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"开始训练，共 {num_epochs} 个 epoch")
        print(f"设备: {self.device}")
        print(f"训练集: {len(self.train_loader)} batches")
        if self.val_loader:
            print(f"验证集: {len(self.val_loader)} batches")
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            
            if self.val_loader:
                val_metrics = self.validate(epoch)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['acc'])
                
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch}/{num_epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.2f}% - "
                      f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.2f}% - "
                      f"Time: {epoch_time:.1f}s")
                
                if save_best and val_metrics['acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['acc']
                    self.epochs_without_improvement = 0
                    self.save_checkpoint('best_model.pth', epoch, val_metrics)
                    print(f"  → 保存最佳模型 (Val Acc: {val_metrics['acc']:.2f}%)")
                else:
                    self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n早停触发！连续 {self.early_stopping_patience} 个 epoch 未提升")
                    break
            else:
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch}/{num_epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.2f}% - "
                      f"Time: {epoch_time:.1f}s")
            
            if save_last and epoch == num_epochs:
                self.save_checkpoint('last_model.pth', epoch)
            
            if save_frequency > 0 and epoch % save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
        
        print(f"\n训练完成！最佳验证准确率: {self.best_val_acc:.2f}%")
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Optional[Dict] = None):
        """
        保存模型检查点
        
        Args:
            filename: 保存文件名
            epoch: 当前 epoch
            metrics: 当前指标
        """
        import os
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"从 checkpoint 恢复: Epoch {checkpoint['epoch']}, Best Val Acc: {checkpoint.get('best_val_acc', 0.0):.2f}%")
        
        return checkpoint.get('epoch', 0)


class EarlyStopping:
    """
    早停机制
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: 耐心值
            min_delta: 最小变化量
            mode: 'max' 或 'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        config: 优化器配置
    
    Returns:
        优化器实例
    """
    optimizer_type = config.get('type', 'adamw')
    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    
    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"未知的优化器类型: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler._LRScheduler:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 调度器配置
    
    Returns:
        调度器实例
    """
    scheduler_type = config.get('type', 'cosine')
    
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 50),
            eta_min=config.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 10),
            gamma=config.get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'max'),
            factor=config.get('factor', 0.1),
            patience=config.get('patience', 5)
        )
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"未知的调度器类型: {scheduler_type}")


if __name__ == '__main__':
    print("Trainer 模块测试...")
    
    model = torch.nn.Linear(768, 7)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(100, 3, 224, 224),
            torch.randint(0, 7, (100,))
        ),
        batch_size=16
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device='cpu'
    )
    
    history = trainer.train(num_epochs=2)
    print("训练完成")
    print(f"历史记录: {history}")