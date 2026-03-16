"""
BCW-DSHR: Backbone-Conditioned Wavelet Downsampling with High-frequency Residual

核心思想：
1. 使用离散小波变换(DWT)替代传统池化下采样
2. 通过主干网络特征条件门控，选择性保留有意义的高频信息
3. 高频残差注入，保留边缘细节

解决问题：
- 传统下采样(maxpool/avgpool)丢失高频细节
- 水下图像中边缘与噪声在高频域混叠
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DWT2d(nn.Module):
    """
    2D 离散小波变换 (Discrete Wavelet Transform)
    
    将输入分解为4个子带：
    - LL: 低频近似 (approximation)
    - LH: 水平细节 (horizontal detail)
    - HL: 垂直细节 (vertical detail)  
    - HH: 对角细节 (diagonal detail)
    
    输出尺寸：每个子带为 (B, C, H/2, W/2)
    """
    
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        
        # Haar小波滤波器 (最简单且有效)
        if wavelet == 'haar':
            # 低通滤波器
            self.low = torch.tensor([1.0, 1.0]) / np.sqrt(2)
            # 高通滤波器
            self.high = torch.tensor([1.0, -1.0]) / np.sqrt(2)
        elif wavelet == 'db2':
            # Daubechies-2 小波
            self.low = torch.tensor([0.4830, 0.8365, 0.2241, -0.1294])
            self.high = torch.tensor([-0.1294, -0.2241, 0.8365, -0.4830])
        else:
            raise ValueError(f"Unsupported wavelet: {wavelet}")
        
        # 注册为buffer（不参与训练，但会随模型保存/移动设备）
        self.register_buffer('low_filter', self.low)
        self.register_buffer('high_filter', self.high)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            LL, LH, HL, HH: 各为 (B, C, H/2, W/2)
        """
        B, C, H, W = x.shape
        
        # 构建2D滤波器
        # LL = low × low^T, LH = low × high^T, HL = high × low^T, HH = high × high^T
        low = self.low_filter.view(1, 1, -1, 1)   # (1, 1, K, 1)
        high = self.high_filter.view(1, 1, -1, 1)  # (1, 1, K, 1)
        
        low_t = self.low_filter.view(1, 1, 1, -1)   # (1, 1, 1, K)
        high_t = self.high_filter.view(1, 1, 1, -1)  # (1, 1, 1, K)
        
        # Padding
        pad_h = (self.low_filter.size(0) - 1) // 2
        pad_w = (self.low_filter.size(0) - 1) // 2
        
        # 对每个通道独立处理
        # 先对行做变换，再对列做变换
        
        # Step 1: 行方向滤波 + 下采样
        x_pad = F.pad(x, (pad_w, pad_w, 0, 0), mode='reflect')
        
        # 分组卷积实现通道独立处理
        low_row = F.conv2d(x_pad, low_t.expand(C, 1, 1, -1), groups=C, stride=(1, 2))
        high_row = F.conv2d(x_pad, high_t.expand(C, 1, 1, -1), groups=C, stride=(1, 2))
        
        # Step 2: 列方向滤波 + 下采样
        low_row_pad = F.pad(low_row, (0, 0, pad_h, pad_h), mode='reflect')
        high_row_pad = F.pad(high_row, (0, 0, pad_h, pad_h), mode='reflect')
        
        LL = F.conv2d(low_row_pad, low.expand(C, 1, -1, 1), groups=C, stride=(2, 1))
        LH = F.conv2d(low_row_pad, high.expand(C, 1, -1, 1), groups=C, stride=(2, 1))
        HL = F.conv2d(high_row_pad, low.expand(C, 1, -1, 1), groups=C, stride=(2, 1))
        HH = F.conv2d(high_row_pad, high.expand(C, 1, -1, 1), groups=C, stride=(2, 1))
        
        return LL, LH, HL, HH


class BackboneConditionedGating(nn.Module):
    """
    主干条件门控模块
    
    利用主干网络的语义特征生成门控信号，
    选择性保留有意义的高频信息（边缘），抑制噪声
    """
    
    def __init__(self, in_channels, backbone_channels, reduction=4):
        """
        Args:
            in_channels: 高频特征通道数 (来自DWT)
            backbone_channels: 主干网络条件特征通道数
            reduction: 通道压缩比例
        """
        super().__init__()
        
        hidden_channels = max(in_channels // reduction, 32)
        
        # 条件特征处理
        self.condition_net = nn.Sequential(
            nn.Conv2d(backbone_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        # 门控生成网络
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels + hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1),
            nn.Sigmoid()  # 输出 [0, 1] 的门控值
        )
    
    def forward(self, hf_feat, backbone_feat):
        """
        Args:
            hf_feat: 高频特征 (B, C, H, W)
            backbone_feat: 主干条件特征 (B, C', H', W')
        Returns:
            gated_hf: 门控后的高频特征 (B, C, H, W)
        """
        # 调整backbone特征尺寸以匹配高频特征
        if backbone_feat.shape[2:] != hf_feat.shape[2:]:
            backbone_feat = F.interpolate(
                backbone_feat, 
                size=hf_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 条件特征编码
        cond = self.condition_net(backbone_feat)
        
        # 拼接高频特征和条件特征，生成门控
        concat = torch.cat([hf_feat, cond], dim=1)
        gate = self.gate_net(concat)
        
        # 门控高频特征
        gated_hf = gate * hf_feat
        
        return gated_hf


class BCWDSHR(nn.Module):
    """
    BCW-DSHR: Backbone-Conditioned Wavelet Downsampling with High-frequency Residual
    
    完整模块，用于替代ResNet中的下采样操作
    """
    
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        backbone_channels=None,
        wavelet='haar',
        use_hh=False,  # 是否使用对角高频分量 (通常包含更多噪声)
        learnable_gamma=True
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            backbone_channels: 条件特征通道数，None则使用in_channels
            wavelet: 小波类型 ('haar' 或 'db2')
            use_hh: 是否使用HH分量
            learnable_gamma: 高频注入强度是否可学习
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_hh = use_hh
        
        if backbone_channels is None:
            backbone_channels = in_channels
        
        # 小波变换
        self.dwt = DWT2d(wavelet=wavelet)
        
        # 高频融合 (LH + HL，可选HH)
        self.hf_fusion = nn.Sequential(
            nn.Conv2d(in_channels * (3 if use_hh else 2), in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 主干条件门控
        self.gating = BackboneConditionedGating(
            in_channels=in_channels,
            backbone_channels=backbone_channels
        )
        
        # 可学习的高频注入强度
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.ones(1) * 0.1)  # 初始化为较小值
        else:
            self.register_buffer('gamma', torch.ones(1) * 0.1)
        
        # 通道调整 (如果in_channels != out_channels)
        if in_channels != out_channels:
            self.channel_adjust = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.channel_adjust = nn.Identity()
        
        # 输出激活
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, backbone_feat=None):
        """
        Args:
            x: 输入特征 (B, C_in, H, W)
            backbone_feat: 主干条件特征 (B, C_backbone, H', W')
                          如果为None，则使用x自身作为条件
        Returns:
            out: 下采样后的特征 (B, C_out, H/2, W/2)
        """
        # 如果没有提供backbone特征，使用自身
        if backbone_feat is None:
            backbone_feat = x
        
        # Step 1: DWT分解
        LL, LH, HL, HH = self.dwt(x)
        
        # Step 2: 高频融合
        if self.use_hh:
            hf_concat = torch.cat([LH, HL, HH], dim=1)
        else:
            hf_concat = torch.cat([LH, HL], dim=1)
        hf_fused = self.hf_fusion(hf_concat)
        
        # Step 3: 主干条件门控
        # 需要将backbone_feat下采样到与LL相同尺寸
        if backbone_feat.shape[2:] != LL.shape[2:]:
            backbone_feat_down = F.adaptive_avg_pool2d(backbone_feat, LL.shape[2:])
        else:
            backbone_feat_down = backbone_feat
        
        hf_gated = self.gating(hf_fused, backbone_feat_down)
        
        # Step 4: 高频残差注入
        out = LL + self.gamma * hf_gated
        
        # Step 5: 通道调整
        out = self.channel_adjust(out)
        out = self.relu(out)
        
        return out


class BCWDSHRResNetBlock(nn.Module):
    """
    用于替换ResNet下采样的封装模块
    
    ResNet的下采样发生在每个stage的第一个block中：
    - 通过stride=2的3x3卷积实现
    - 同时有一个downsample分支处理shortcut
    
    这个模块用BCW-DSHR替换stride=2卷积
    """
    
    def __init__(self, in_channels, out_channels, backbone_channels=None):
        super().__init__()
        
        self.bcw_dshr = BCWDSHR(
            in_channels=in_channels,
            out_channels=out_channels,
            backbone_channels=backbone_channels,
            wavelet='haar',
            use_hh=False,
            learnable_gamma=True
        )
    
    def forward(self, x, backbone_feat=None):
        return self.bcw_dshr(x, backbone_feat)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("BCW-DSHR 模块测试")
    print("=" * 60)
    
    # 测试DWT
    print("\n1. 测试 DWT2d")
    dwt = DWT2d(wavelet='haar')
    x = torch.randn(2, 64, 128, 128)
    LL, LH, HL, HH = dwt(x)
    print(f"   输入: {x.shape}")
    print(f"   LL: {LL.shape}, LH: {LH.shape}, HL: {HL.shape}, HH: {HH.shape}")
    
    # 测试门控模块
    print("\n2. 测试 BackboneConditionedGating")
    gating = BackboneConditionedGating(in_channels=64, backbone_channels=128)
    hf = torch.randn(2, 64, 64, 64)
    backbone = torch.randn(2, 128, 64, 64)
    gated = gating(hf, backbone)
    print(f"   高频输入: {hf.shape}")
    print(f"   条件输入: {backbone.shape}")
    print(f"   门控输出: {gated.shape}")
    
    # 测试完整BCW-DSHR
    print("\n3. 测试 BCWDSHR")
    bcw = BCWDSHR(in_channels=256, out_channels=512, backbone_channels=256)
    x = torch.randn(2, 256, 64, 64)
    backbone = torch.randn(2, 256, 64, 64)
    out = bcw(x, backbone)
    print(f"   输入: {x.shape}")
    print(f"   输出: {out.shape}")
    print(f"   gamma: {bcw.gamma.item():.4f}")
    
    # 参数量统计
    params = sum(p.numel() for p in bcw.parameters())
    print(f"   参数量: {params / 1e3:.2f}K")
    
    print("\n" + "=" * 60)
    print("BCW-DSHR 模块测试完成!")
    print("=" * 60)
