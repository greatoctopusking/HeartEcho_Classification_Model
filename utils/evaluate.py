import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class Evaluator:
    """
    评估器类，负责模型评估和指标计算
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        class_names: Optional[List[str]] = None
    ):
        """
        初始化评估器
        
        Args:
            model: 待评估的模型
            device: 评估设备
            class_names: 类别名称列表
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names or [f'Class_{i}' for i in range(7)]
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对数据进行预测
        
        Args:
            data_loader: 数据加载器
        
        Returns:
            (predictions, labels, probabilities)
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc='Evaluating'):
                images = images.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return (
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities)
        )
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """
        评估模型并返回所有指标
        
        Args:
            data_loader: 数据加载器
        
        Returns:
            评估指标字典
        """
        predictions, labels, probabilities = self.predict(data_loader)
        
        metrics = compute_metrics(
            y_true=labels,
            y_pred=predictions,
            y_prob=probabilities,
            class_names=self.class_names
        )
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        data_loader: DataLoader,
        save_path: str = 'confusion_matrix.png',
        normalize: bool = False
    ):
        """
        绘制并保存混淆矩阵
        
        Args:
            data_loader: 数据加载器
            save_path: 保存路径
            normalize: 是否归一化
        """
        predictions, labels, _ = self.predict(data_loader)
        
        cm = confusion_matrix(labels, predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"混淆矩阵已保存至: {save_path}")
    
    def plot_roc_curves(
        self,
        data_loader: DataLoader,
        save_path: str = 'roc_curves.png'
    ):
        """
        绘制并保存 ROC 曲线
        
        Args:
            data_loader: 数据加载器
            save_path: 保存路径
        """
        _, labels, probabilities = self.predict(data_loader)
        
        n_classes = len(self.class_names)
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(
                (labels == i).astype(int),
                probabilities[:, i]
            )
            auc = roc_auc_score(
                (labels == i).astype(int),
                probabilities[:, i]
            )
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"ROC 曲线已保存至: {save_path}")
    
    def per_class_analysis(self, data_loader: DataLoader) -> Dict:
        """
        每个类别的详细分析
        
        Args:
            data_loader: 数据加载器
        
        Returns:
            每类别的指标
        """
        predictions, labels, probabilities = self.predict(data_loader)
        
        results = {}
        
        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            
            if mask.sum() == 0:
                continue
            
            true_positive = ((predictions == i) & (labels == i)).sum()
            false_positive = ((predictions == i) & (labels != i)).sum()
            false_negative = ((predictions != i) & (labels == i)).sum()
            true_negative = ((predictions != i) & (labels != i)).sum()
            
            precision = true_positive / (true_positive + false_positive + 1e-10)
            recall = true_positive / (true_positive + false_negative + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            results[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': mask.sum()
            }
        
        return results


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    计算分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率
        class_names: 类别名称
    
    Returns:
        指标字典
    """
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    metrics['precision'] = precision_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['recall'] = recall_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['f1_score'] = f1_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics['precision_macro'] = precision_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['recall_macro'] = recall_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['f1_macro'] = f1_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='weighted'
            )
        except ValueError:
            metrics['auc'] = 0.0
    
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    if class_names is not None:
        metrics['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0
        )
    
    return metrics


def print_metrics(metrics: Dict):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
    """
    print("\n" + "=" * 50)
    print("模型评估结果")
    print("=" * 50)
    
    print(f"\n整体指标:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    if 'auc' in metrics:
        print(f"  AUC:       {metrics['auc']:.4f}")
    
    print(f"\nMacro 平均:")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  F1 Score:  {metrics['f1_macro']:.4f}")
    
    if 'classification_report' in metrics:
        print("\n" + "-" * 50)
        print("分类报告:")
        print("-" * 50)
        print(metrics['classification_report'])
    
    print("=" * 50 + "\n")


def save_metrics(metrics: Dict, save_path: str):
    """
    保存指标到文件
    
    Args:
        metrics: 指标字典
        save_path: 保存路径
    """
    import json
    
    metrics_to_save = metrics.copy()
    
    if 'classification_report' in metrics_to_save:
        del metrics_to_save['classification_report']
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"指标已保存至: {save_path}")


if __name__ == '__main__':
    print("Evaluator 模块测试...")
    
    model = torch.nn.Linear(768, 7)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(50, 3, 224, 224),
            torch.randint(0, 7, (50,))
        ),
        batch_size=16
    )
    
    evaluator = Evaluator(
        model=model,
        device='cpu',
        class_names=['A4C', 'PL', 'PSAV', 'PSMV', 'Random', 'SC', 'A2C']
    )
    
    metrics = evaluator.evaluate(test_loader)
    print_metrics(metrics)
    
    print("评估完成")