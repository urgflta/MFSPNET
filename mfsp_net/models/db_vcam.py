"""
DB-VCAM: Dual-Branch Vision Cross-Attention Module

核心思想：
1. 单向交叉注意力：语义特征(Q) 主动检索 细节特征(K,V) 中的有效信息
2. 非对称设计：利用语义特征的高可靠性引导细节特征的筛选
3. 门控残差：在噪声严重区域回退到可靠的语义特征

解决问题：
- 多源特征可靠性差异导致的融合困难
- 简单concat/add无法区分有效信息和噪声
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    """
    交叉注意力模块
    
    Q 来自语义分支（高可靠性）
    K, V 来自细节分支（高空间精度但含噪）
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim: 特征维度
            num_heads: 注意力头数
            qkv_bias: 是否使用偏置
            attn_drop: 注意力dropout
            proj_drop: 输出dropout
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q 投影（来自语义特征）
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # K, V 投影（来自细节特征）
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, q_feat, kv_feat):
        """
        Args:
            q_feat: Query特征 (B, N, C) - 来自语义分支
            kv_feat: Key/Value特征 (B, N, C) - 来自细节分支
        Returns:
            out: 注意力输出 (B, N, C)
        """
        B, N, C = q_feat.shape
        
        # 计算 Q, K, V
        q = self.q_proj(q_feat).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_feat).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_feat).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        
        return out


class GatedFusion(nn.Module):
    """
    门控融合模块
    
    自适应地在注意力增强特征和原始语义特征之间进行加权
    在噪声严重区域，门控值趋向0，回退到可靠的语义特征
    """
    
    def __init__(self, dim):
        super().__init__()
        
        # 门控生成网络
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, attn_feat, semantic_feat):
        """
        Args:
            attn_feat: 注意力增强特征 (B, C, H, W)
            semantic_feat: 原始语义特征 (B, C, H, W)
        Returns:
            fused: 门控融合结果 (B, C, H, W)
        """
        # 拼接计算门控
        concat = torch.cat([attn_feat, semantic_feat], dim=1)
        gate = self.gate_conv(concat)
        
        # 门控融合
        fused = gate * attn_feat + (1 - gate) * semantic_feat
        
        return fused


class DBVCAM(nn.Module):
    """
    DB-VCAM: Dual-Branch Vision Cross-Attention Module
    
    完整的跨分支融合模块
    """
    
    def __init__(
        self, 
        detail_channels,      # 细节分支通道数 (ResNet输出)
        semantic_channels,    # 语义分支通道数 (SAM输出)
        hidden_dim=256,       # 注意力隐藏维度
        num_heads=8,
        dropout=0.1
    ):
        """
        Args:
            detail_channels: 细节特征通道数 (来自ResNet+BCW-DSHR)
            semantic_channels: 语义特征通道数 (来自SAM/dummy)
            hidden_dim: 交叉注意力的隐藏维度
            num_heads: 注意力头数
            dropout: dropout率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 特征投影到统一维度
        self.detail_proj = nn.Sequential(
            nn.Conv2d(detail_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 交叉注意力
        self.cross_attn = CrossAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout
        )
        
        # Layer Norm
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 门控融合
        self.gated_fusion = GatedFusion(hidden_dim)
        
        # 输出投影（恢复到原始通道数）
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, detail_channels, 1, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, detail_feat, semantic_feat):
        """
        Args:
            detail_feat: 细节特征 (B, C_d, H, W) - 来自ResNet+BCW-DSHR
            semantic_feat: 语义特征 (B, C_s, H, W) - 来自SAM/dummy
        Returns:
            fused: 融合特征 (B, C_d, H, W)
        """
        B, _, H, W = detail_feat.shape
        
        # 确保尺寸一致
        if semantic_feat.shape[2:] != detail_feat.shape[2:]:
            semantic_feat = F.interpolate(
                semantic_feat, 
                size=detail_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 投影到统一维度
        detail_proj = self.detail_proj(detail_feat)      # (B, hidden_dim, H, W)
        semantic_proj = self.semantic_proj(semantic_feat)  # (B, hidden_dim, H, W)
        
        # 保存用于门控融合
        semantic_for_gate = semantic_proj
        
        # 转换为序列格式 (B, H*W, C)
        detail_seq = detail_proj.flatten(2).permute(0, 2, 1)    # (B, N, C)
        semantic_seq = semantic_proj.flatten(2).permute(0, 2, 1)  # (B, N, C)
        
        # Layer Norm
        q = self.norm_q(semantic_seq)   # 语义特征作为 Q
        kv = self.norm_kv(detail_seq)   # 细节特征作为 K, V
        
        # 交叉注意力
        attn_out = self.cross_attn(q, kv)
        
        # 残差连接
        attn_out = semantic_seq + attn_out
        
        # FFN
        attn_out = attn_out + self.ffn(self.norm_out(attn_out))
        
        # 转回空间格式 (B, C, H, W)
        attn_feat = attn_out.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W)
        
        # 门控融合
        fused = self.gated_fusion(attn_feat, semantic_for_gate)
        
        # 输出投影
        fused = self.out_proj(fused)
        
        # 残差连接（与原始细节特征）
        fused = fused + detail_feat
        
        return fused


class DummySemanticBranch(nn.Module):
    """
    Dummy 语义分支
    
    用于在 SAM 接入之前验证 DB-VCAM 的有效性
    使用一个简单的编码器模拟 SAM 的输出
    """
    
    def __init__(self, out_channels=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 模拟 SAM 的多尺度特征提取
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Block 1
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 3 (输出 H/16)
            nn.Conv2d(256, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, 3, H, W)
        Returns:
            feat: 语义特征 (B, out_channels, H/16, W/16)
        """
        return self.encoder(x)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("DB-VCAM 模块测试")
    print("=" * 60)
    
    # 测试交叉注意力
    print("\n1. 测试 CrossAttention")
    cross_attn = CrossAttention(dim=256, num_heads=8)
    q = torch.randn(2, 900, 256)  # 30x30
    kv = torch.randn(2, 900, 256)
    out = cross_attn(q, kv)
    print(f"   Q shape: {q.shape}")
    print(f"   KV shape: {kv.shape}")
    print(f"   Output shape: {out.shape}")
    
    # 测试门控融合
    print("\n2. 测试 GatedFusion")
    gated = GatedFusion(dim=256)
    attn_feat = torch.randn(2, 256, 30, 30)
    sem_feat = torch.randn(2, 256, 30, 30)
    fused = gated(attn_feat, sem_feat)
    print(f"   Attn feat shape: {attn_feat.shape}")
    print(f"   Semantic feat shape: {sem_feat.shape}")
    print(f"   Fused shape: {fused.shape}")
    
    # 测试完整 DB-VCAM
    print("\n3. 测试 DBVCAM")
    db_vcam = DBVCAM(
        detail_channels=2048,
        semantic_channels=256,
        hidden_dim=256,
        num_heads=8
    )
    detail = torch.randn(2, 2048, 30, 30)
    semantic = torch.randn(2, 256, 30, 30)
    fused = db_vcam(detail, semantic)
    print(f"   Detail feat shape: {detail.shape}")
    print(f"   Semantic feat shape: {semantic.shape}")
    print(f"   Fused shape: {fused.shape}")
    
    params = sum(p.numel() for p in db_vcam.parameters())
    print(f"   参数量: {params / 1e6:.2f}M")
    
    # 测试 Dummy 语义分支
    print("\n4. 测试 DummySemanticBranch")
    dummy = DummySemanticBranch(out_channels=256)
    x = torch.randn(2, 3, 480, 480)
    feat = dummy(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {feat.shape}")
    
    params = sum(p.numel() for p in dummy.parameters())
    print(f"   参数量: {params / 1e6:.2f}M")
    
    print("\n" + "=" * 60)
    print("DB-VCAM 模块测试完成!")
    print("=" * 60)
