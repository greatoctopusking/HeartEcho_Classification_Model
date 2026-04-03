import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class USFMAEEncoder(nn.Module):
    """
    USF-MAE Encoder (ViT-Base Backbone)
    
    这是 USF-MAE 预训练模型的编码器部分，用于提取心脏超声图像特征。
    原始 MAE 模型包含 encoder 和 decoder，在分类任务中我们只使用 encoder。
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_cls_token: bool = True
    ):
        """
        初始化 USF-MAE Encoder
        
        Args:
            img_size: 输入图像尺寸
            patch_size: 每个 patch 的尺寸
            in_chans: 输入通道数
            embed_dim: 嵌入维度 (ViT-Base 为 768)
            depth: Transformer 层数 (ViT-Base 为 12)
            num_heads: 注意力头数 (ViT-Base 为 12)
            mlp_ratio: MLP 隐藏层维度与 embedding 维度的比例
            drop_rate: Dropout 概率
            attn_drop_rate: 注意力 Dropout 概率
            drop_path_rate: Drop path 概率
            use_cls_token: 是否使用 [CLS] token 进行分类
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        self.num_tokens = 1 if use_cls_token else 0
        
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        """初始化模块权重"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 tensor, shape (B, C, H, W)
        
        Returns:
            特征 tensor, shape (B, embed_dim) 或 (B, num_patches+1, embed_dim)
        """
        x = self.patch_embed(x)
        
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.embed_dim


class PatchEmbed(nn.Module):
    """图像到 Patch 嵌入"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    """多头注意力"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP 模块"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        drop: float = 0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


def load_pretrained_usfmae(
    checkpoint_path: str,
    device: str = 'cuda'
) -> USFMAEEncoder:
    """
    加载 USF-MAE 预训练权重
    
    Args:
        checkpoint_path: 预训练权重文件路径
        device: 设备 ('cuda' 或 'cpu')
    
    Returns:
        加载好的 USFMAEEncoder 模型
    """
    model = USFMAEEncoder(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'decoder' not in key:
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
    
    missing_keys, unexpected_keys = model.load_state_dict(
        new_state_dict, strict=False
    )
    
    if missing_keys:
        print(f"警告: 缺少以下键: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"警告: 意外键: {unexpected_keys[:5]}...")
    
    model = model.to(device)
    model.eval()
    
    return model


def create_usfmae_backbone(
    pretrained_path: Optional[str] = None,
    freeze: bool = False,
    device: str = 'cuda'
) -> USFMAEEncoder:
    """
    创建 USF-MAE Backbone
    
    Args:
        pretrained_path: 预训练权重路径，若为 None 则使用随机初始化
        freeze: 是否冻结参数
        device: 设备
    
    Returns:
        USFMAEEncoder 模型
    """
    model = USFMAEEncoder(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    
    if pretrained_path is not None:
        model = load_pretrained_usfmae(pretrained_path, device)
    
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    
    return model


if __name__ == '__main__':
    import sys
    
    print("测试 USF-MAE Backbone...")
    
    model = USFMAEEncoder(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    x = torch.randn(2, 3, 224, 224)
    features = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {features.shape}")
    
    if len(sys.argv) > 1:
        pretrained_path = sys.argv[1]
        print(f"\n加载预训练权重: {pretrained_path}")
        model = load_pretrained_usfmae(pretrained_path)
        print("加载成功!")