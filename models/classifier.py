import torch
import torch.nn as nn
from typing import Optional, List
import torch.nn.functional as F


class CardiacClassifier(nn.Module):
    """
    心脏超声切面分类模型
    
    基于 USF-MAE Backbone + 分类头
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 7,
        use_cls_token: bool = True,
        use_global_avg_pool: bool = True,
        dropout: float = 0.0
    ):
        """
        初始化分类模型
        
        Args:
            backbone: 骨干网络 (如 USFMAEEncoder)
            num_classes: 分类类别数
            use_cls_token: 是否使用 [CLS] token 的特征
            use_global_avg_pool: 是否使用全局平均池化
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.use_global_avg_pool = use_global_avg_pool
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        feature_dim = backbone.get_feature_dim()
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化分类头权重"""
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 tensor, shape (B, C, H, W)
        
        Returns:
            分类 logits, shape (B, num_classes)
        """
        features = self.backbone(x)
        
        if self.use_cls_token:
            cls_features = features[:, 0]
        elif self.use_global_avg_pool:
            cls_features = features[:, 1:].mean(dim=1)
        else:
            cls_features = features.mean(dim=1)
        
        if self.dropout is not None:
            cls_features = self.dropout(cls_features)
        
        logits = self.classifier(cls_features)
        
        return logits
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征嵌入（用于特征可视化或迁移学习）
        
        Args:
            x: 输入图像 tensor
        
        Returns:
            特征嵌入向量
        """
        features = self.backbone(x)
        
        if self.use_cls_token:
            embedding = features[:, 0]
        else:
            embedding = features[:, 1:].mean(dim=1)
        
        return embedding


class MultiHeadClassifier(nn.Module):
    """
    多头分类器（支持多任务学习）
    
    可以同时训练多个分类任务
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes_list: List[int],
        use_cls_token: bool = True,
        dropout: float = 0.0
    ):
        """
        初始化多头分类器
        
        Args:
            backbone: 骨干网络
            num_classes_list: 每个任务的类别数列表
            use_cls_token: 是否使用 [CLS] token
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.backbone = backbone
        self.use_cls_token = use_cls_token
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        feature_dim = backbone.get_feature_dim()
        
        self.heads = nn.ModuleList([
            nn.Linear(feature_dim, num_classes)
            for num_classes in num_classes_list
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 tensor
        
        Returns:
            每个任务的 logits 列表
        """
        features = self.backbone(x)
        
        if self.use_cls_token:
            cls_features = features[:, 0]
        else:
            cls_features = features[:, 1:].mean(dim=1)
        
        if self.dropout is not None:
            cls_features = self.dropout(cls_features)
        
        outputs = [head(cls_features) for head in self.heads]
        
        return outputs


def create_classifier(
    backbone: nn.Module,
    num_classes: int = 7,
    classifier_type: str = 'linear',
    **kwargs
) -> nn.Module:
    """
    创建分类器
    
    Args:
        backbone: 骨干网络
        num_classes: 分类类别数
        classifier_type: 分类器类型 ('linear', 'mlp', 'multi_head')
        **kwargs: 额外参数
    
    Returns:
        分类模型
    """
    if classifier_type == 'linear':
        return CardiacClassifier(
            backbone=backbone,
            num_classes=num_classes,
            **kwargs
        )
    elif classifier_type == 'mlp':
        return MLPClassifier(
            backbone=backbone,
            num_classes=num_classes,
            **kwargs
        )
    elif classifier_type == 'multi_head':
        return MultiHeadClassifier(
            backbone=backbone,
            num_classes_list=kwargs.get('num_classes_list', [num_classes]),
            **kwargs
        )
    else:
        raise ValueError(f"未知的分类器类型: {classifier_type}")


class MLPClassifier(nn.Module):
    """
    MLP 分类器（使用多层感知机作为分类头）
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 7,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_cls_token: bool = True
    ):
        super().__init__()
        
        self.backbone = backbone
        self.use_cls_token = use_cls_token
        
        feature_dim = backbone.get_feature_dim()
        
        layers = []
        in_dim = feature_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        
        if self.use_cls_token:
            cls_features = features[:, 0]
        else:
            cls_features = features[:, 1:].mean(dim=1)
        
        return self.classifier(cls_features)


def load_model_with_pretrained(
    pretrained_path: str,
    num_classes: int = 7,
    freeze_backbone: bool = False,
    device: str = 'cuda'
) -> CardiacClassifier:
    """
    加载带预训练权重的分类模型
    
    Args:
        pretrained_path: 预训练权重路径
        num_classes: 分类类别数
        freeze_backbone: 是否冻结 backbone
        device: 设备
    
    Returns:
        加载好的分类模型
    """
    from .backbone import create_usfmae_backbone
    
    backbone = create_usfmae_backbone(
        pretrained_path=pretrained_path,
        freeze=freeze_backbone,
        device=device
    )
    
    model = CardiacClassifier(
        backbone=backbone,
        num_classes=num_classes,
        use_cls_token=True,
        use_global_avg_pool=True
    )
    
    return model.to(device)


if __name__ == '__main__':
    from .backbone import USFMAEEncoder
    
    print("测试分类模型...")
    
    backbone = USFMAEEncoder()
    print(f"Backbone 参数量: {sum(p.numel() for p in backbone.parameters()) / 1e6:.2f}M")
    
    model = CardiacClassifier(
        backbone=backbone,
        num_classes=7,
        dropout=0.1
    )
    print(f"分类模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出 logits 形状: {logits.shape}")
    print(f"预测类别: {logits.argmax(dim=1)}")