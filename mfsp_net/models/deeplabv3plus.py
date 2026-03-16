"""
DeepLabV3+ 模型实现
支持 BCW-DSHR 模块的版本

两种模式：
1. baseline: 标准 DeepLabV3+ with ResNet-50
2. with_bcw_dshr: 集成 BCW-DSHR 的增强版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import Bottleneck

from .bcw_dshr import BCWDSHR
from .db_vcam import DBVCAM, DummySemanticBranch
from .a2_lora_sam import A2LoRASAM


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    多尺度空洞卷积特征提取
    """
    
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()
        
        # 1x1 卷积分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 空洞卷积分支
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # 全局平均池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合卷积
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(rates)), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # 各分支特征
        feat1 = self.conv1x1(x)
        feat_atrous = [conv(x) for conv in self.atrous_convs]
        feat_global = self.global_pool(x)
        feat_global = F.interpolate(feat_global, size=size, mode='bilinear', align_corners=False)
        
        # 拼接所有特征
        feat = torch.cat([feat1] + feat_atrous + [feat_global], dim=1)
        
        return self.project(feat)


class DeepLabV3PlusDecoder(nn.Module):
    """
    DeepLabV3+ 解码器
    融合低层特征 + 上采样
    """
    
    def __init__(self, low_level_channels, aspp_channels=256, num_classes=8):
        super().__init__()
        
        # 低层特征处理 (来自 ResNet Stage1, 通常是256通道)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 融合后的卷积
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # 分类头
        self.classifier = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, aspp_feat, low_level_feat, target_size):
        """
        Args:
            aspp_feat: ASPP输出特征, shape (B, 256, H/16, W/16)
            low_level_feat: 低层特征, shape (B, 256, H/4, W/4)
            target_size: 输出目标尺寸 (H, W)
        """
        # 处理低层特征
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # 上采样 ASPP 特征到低层特征尺寸
        aspp_feat = F.interpolate(
            aspp_feat, 
            size=low_level_feat.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 拼接
        fused = torch.cat([aspp_feat, low_level_feat], dim=1)
        fused = self.fuse_conv(fused)
        
        # 分类 + 上采样到原图尺寸
        out = self.classifier(fused)
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        
        return out


class ResNetEncoder(nn.Module):
    """
    ResNet-50 编码器 (Baseline版本)
    
    输出多尺度特征:
    - low_level_feat: Stage1 输出, H/4, 256通道 (用于Decoder)
    - high_level_feat: Stage4 输出, H/16, 2048通道 (用于ASPP)
    """
    
    def __init__(self, pretrained=True, output_stride=16):
        super().__init__()
        
        # 加载预训练 ResNet-50
        if pretrained:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = resnet50(weights=None)
        
        # Stage 0: conv1 + bn1 + relu + maxpool -> H/4
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # Stage 1: layer1 -> H/4, 256通道
        self.layer1 = resnet.layer1
        
        # Stage 2: layer2 -> H/8, 512通道
        self.layer2 = resnet.layer2
        
        # Stage 3: layer3 -> H/16, 1024通道
        self.layer3 = resnet.layer3
        
        # Stage 4: layer4 -> H/16 (使用空洞卷积保持分辨率), 2048通道
        if output_stride == 16:
            self.layer4 = resnet.layer4
            self._modify_layer4_dilation()
        else:
            self.layer4 = resnet.layer4
    
    def _modify_layer4_dilation(self):
        """修改 layer4 使用空洞卷积"""
        for module in self.layer4.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                if module.kernel_size == (3, 3):
                    module.dilation = (2, 2)
                    module.padding = (2, 2)
    
    def forward(self, x):
        """
        Args:
            x: 输入图像, shape (B, 3, H, W)
        Returns:
            low_level_feat: Stage1 特征, shape (B, 256, H/4, W/4)
            high_level_feat: Stage4 特征, shape (B, 2048, H/16, W/16)
        """
        # Stem: H/4
        x = self.stem(x)
        
        # Stage 1: H/4, 256通道
        x = self.layer1(x)
        low_level_feat = x  # 保存低层特征
        
        # Stage 2: H/8, 512通道
        x = self.layer2(x)
        
        # Stage 3: H/16, 1024通道
        x = self.layer3(x)
        
        # Stage 4: H/16, 2048通道
        x = self.layer4(x)
        high_level_feat = x
        
        return low_level_feat, high_level_feat


class ResNetEncoderWithBCWDSHR(nn.Module):
    """
    集成 BCW-DSHR 的 ResNet-50 编码器
    
    BCW-DSHR 模块插入位置：
    - 在 layer2 输出后增强 (H/8 特征)
    - 在 layer3 输出后增强 (H/16 特征)
    - 在 layer4 输出后增强 (H/16 特征)
    
    BCW-DSHR 使用上一阶段特征作为条件信号
    """
    
    def __init__(self, pretrained=True, output_stride=16):
        super().__init__()
        
        # 加载预训练 ResNet-50
        if pretrained:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = resnet50(weights=None)
        
        # Stage 0: conv1 + bn1 + relu + maxpool -> H/4
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # Stage 1: layer1 -> H/4, 256通道
        self.layer1 = resnet.layer1
        
        # Stage 2: layer2 -> H/8, 512通道
        self.layer2 = resnet.layer2
        
        # Stage 3: layer3 -> H/16, 1024通道
        self.layer3 = resnet.layer3
        
        # Stage 4: layer4
        if output_stride == 16:
            self.layer4 = resnet.layer4
            self._modify_layer4_dilation()
        else:
            self.layer4 = resnet.layer4
        
        # ==================== BCW-DSHR 模块 ====================
        # BCW-DSHR #1: 增强 layer2 输出 (512通道)
        # 使用 layer1 特征(256通道)作为条件
        self.bcw_dshr_1 = BCWDSHR(
            in_channels=512,
            out_channels=512,
            backbone_channels=256,  # layer1 输出通道
            wavelet='haar',
            use_hh=False,
            learnable_gamma=True
        )
        
        # BCW-DSHR #2: 增强 layer3 输出 (1024通道)
        # 使用 layer2 增强后特征(512通道)作为条件
        self.bcw_dshr_2 = BCWDSHR(
            in_channels=1024,
            out_channels=1024,
            backbone_channels=512,  # layer2 输出通道
            wavelet='haar',
            use_hh=False,
            learnable_gamma=True
        )
        
        # BCW-DSHR #3: 增强 layer4 输出 (2048通道)
        # 使用 layer3 增强后特征(1024通道)作为条件
        self.bcw_dshr_3 = BCWDSHR(
            in_channels=2048,
            out_channels=2048,
            backbone_channels=1024,  # layer3 输出通道
            wavelet='haar',
            use_hh=False,
            learnable_gamma=True
        )
        
        # 上采样模块：将BCW-DSHR输出恢复到原尺寸
        # (因为DWT会将尺寸减半，需要恢复)
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.upsample_3 = nn.Sequential(
            nn.Conv2d(2048, 2048, 3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
    
    def _modify_layer4_dilation(self):
        """修改 layer4 使用空洞卷积"""
        for module in self.layer4.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                if module.kernel_size == (3, 3):
                    module.dilation = (2, 2)
                    module.padding = (2, 2)
    
    def forward(self, x):
        """
        Args:
            x: 输入图像, shape (B, 3, H, W)
        Returns:
            low_level_feat: Stage1 特征, shape (B, 256, H/4, W/4)
            high_level_feat: 增强后的Stage4特征, shape (B, 2048, H/16, W/16)
        """
        # Stem: H/4
        x = self.stem(x)
        
        # Stage 1: H/4, 256通道
        feat1 = self.layer1(x)
        low_level_feat = feat1  # 保存低层特征
        
        # Stage 2: H/8, 512通道
        feat2 = self.layer2(feat1)
        # BCW-DSHR #1 增强，使用feat1作为条件
        feat2_bcw = self.bcw_dshr_1(feat2, feat1)
        # 上采样恢复尺寸
        feat2_bcw = F.interpolate(feat2_bcw, size=feat2.shape[2:], mode='bilinear', align_corners=False)
        feat2_enhanced = self.upsample_1(feat2_bcw) + feat2  # 残差连接
        
        # Stage 3: H/16, 1024通道
        feat3 = self.layer3(feat2_enhanced)
        # BCW-DSHR #2 增强，使用feat2_enhanced作为条件
        feat3_bcw = self.bcw_dshr_2(feat3, feat2_enhanced)
        # 上采样恢复尺寸
        feat3_bcw = F.interpolate(feat3_bcw, size=feat3.shape[2:], mode='bilinear', align_corners=False)
        feat3_enhanced = self.upsample_2(feat3_bcw) + feat3  # 残差连接
        
        # Stage 4: H/16, 2048通道
        feat4 = self.layer4(feat3_enhanced)
        # BCW-DSHR #3 增强，使用feat3_enhanced作为条件
        feat4_bcw = self.bcw_dshr_3(feat4, feat3_enhanced)
        # 上采样恢复尺寸
        feat4_bcw = F.interpolate(feat4_bcw, size=feat4.shape[2:], mode='bilinear', align_corners=False)
        feat4_enhanced = self.upsample_3(feat4_bcw) + feat4  # 残差连接
        
        high_level_feat = feat4_enhanced
        
        return low_level_feat, high_level_feat


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ 完整模型
    
    支持多种模式：
    - use_bcw_dshr=False, use_db_vcam=False: 标准baseline
    - use_bcw_dshr=True: 仅BCW-DSHR
    - use_db_vcam=True: DB-VCAM (可选 Dummy 或 SAM 分支)
    - use_sam=True: 使用 A²-LoRA-SAM 替代 Dummy 分支
    """
    
    def __init__(self, num_classes=8, pretrained=True, output_stride=16, 
                 use_bcw_dshr=False, use_db_vcam=False, use_sam=False,
                 sam_checkpoint=None, lora_rank=4, sam_input_size=512):
        super().__init__()
        
        self.use_bcw_dshr = use_bcw_dshr
        self.use_db_vcam = use_db_vcam
        self.use_sam = use_sam
        
        # 编码器
        if use_bcw_dshr:
            self.encoder = ResNetEncoderWithBCWDSHR(pretrained=pretrained, output_stride=output_stride)
        else:
            self.encoder = ResNetEncoder(pretrained=pretrained, output_stride=output_stride)
        
        # DB-VCAM 相关组件
        if use_db_vcam:
            # 语义分支选择
            if use_sam:
                # A²-LoRA-SAM 分支
                self.semantic_branch = A2LoRASAM(
                    sam_checkpoint=sam_checkpoint,
                    out_channels=256,
                    lora_rank=lora_rank,
                    alpha_min=0.1,
                    alpha_max=1.0,
                    use_simple_encoder=(sam_checkpoint is None),
                    input_size=480,
                    sam_input_size=sam_input_size,  # 可调节
                    efficient_mode=True,  # 使用高效模式
                )
            else:
                # Dummy 分支 (用于对照实验)
                self.semantic_branch = DummySemanticBranch(out_channels=256)
            
            # DB-VCAM 融合模块
            self.db_vcam = DBVCAM(
                detail_channels=2048,    # ResNet Stage4 输出
                semantic_channels=256,   # SAM/Dummy 输出
                hidden_dim=256,
                num_heads=8,
                dropout=0.1
            )
        
        # ASPP
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        
        # 解码器
        self.decoder = DeepLabV3PlusDecoder(
            low_level_channels=256,  # ResNet Stage1 输出通道数
            aspp_channels=256,
            num_classes=num_classes
        )
    
    def forward(self, x, return_edge=False):
        """
        Args:
            x: 输入图像, shape (B, 3, H, W)
            return_edge: 是否返回边界预测（训练时使用）
        Returns:
            out: 分割预测, shape (B, num_classes, H, W)
            edge_pred: 边界预测（可选）
        """
        input_size = x.shape[2:]
        
        # 编码器提取特征
        low_level_feat, high_level_feat = self.encoder(x)
        
        # DB-VCAM 融合
        edge_pred = None
        if self.use_db_vcam:
            # 语义分支
            if return_edge and self.use_sam and self.training:
                semantic_feat, edge_pred = self.semantic_branch(x, return_edge=True)
            else:
                semantic_feat = self.semantic_branch(x)
            
            # 跨分支融合
            high_level_feat = self.db_vcam(high_level_feat, semantic_feat)
        
        # ASPP 多尺度特征
        aspp_feat = self.aspp(high_level_feat)
        
        # 解码器
        out = self.decoder(aspp_feat, low_level_feat, input_size)
        
        if return_edge and edge_pred is not None:
            return out, edge_pred
        
        return out
    
    def get_bcw_dshr_params(self):
        """获取BCW-DSHR模块的参数"""
        if not self.use_bcw_dshr:
            return []
        
        params = []
        params.extend(self.encoder.bcw_dshr_1.parameters())
        params.extend(self.encoder.bcw_dshr_2.parameters())
        params.extend(self.encoder.bcw_dshr_3.parameters())
        params.extend(self.encoder.upsample_1.parameters())
        params.extend(self.encoder.upsample_2.parameters())
        params.extend(self.encoder.upsample_3.parameters())
        return params
    
    def get_db_vcam_params(self):
        """获取DB-VCAM模块的参数（不含语义分支）"""
        if not self.use_db_vcam:
            return []
        return list(self.db_vcam.parameters())
    
    def get_sam_params(self):
        """获取 SAM 相关参数（用于分组学习率）"""
        if not (self.use_db_vcam and self.use_sam):
            return {'lora': [], 'estimator': [], 'alignment': []}
        
        return {
            'lora': self.semantic_branch.get_lora_params(),
            'estimator': self.semantic_branch.get_estimator_params(),
            'alignment': self.semantic_branch.get_alignment_params(),
        }
    
    def get_semantic_branch_params(self):
        """获取语义分支全部参数"""
        if not self.use_db_vcam:
            return []
        return list(self.semantic_branch.parameters())
    
    def get_backbone_params(self):
        """获取backbone参数"""
        params = []
        params.extend(self.encoder.stem.parameters())
        params.extend(self.encoder.layer1.parameters())
        params.extend(self.encoder.layer2.parameters())
        params.extend(self.encoder.layer3.parameters())
        params.extend(self.encoder.layer4.parameters())
        return params
    
    def get_last_alpha(self):
        """获取最近一次前向传播的 alpha 值（调试用）"""
        if self.use_db_vcam and self.use_sam:
            return self.semantic_branch.last_alpha
        return None


def get_model(cfg, use_bcw_dshr=False, use_db_vcam=False, use_sam=False, 
              sam_checkpoint=None, lora_rank=4, sam_input_size=512):
    """根据配置创建模型"""
    model = DeepLabV3Plus(
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        output_stride=cfg.output_stride,
        use_bcw_dshr=use_bcw_dshr,
        use_db_vcam=use_db_vcam,
        use_sam=use_sam,
        sam_checkpoint=sam_checkpoint,
        lora_rank=lora_rank,
        sam_input_size=sam_input_size
    )
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("DeepLabV3+ 模型测试")
    print("=" * 60)
    
    x = torch.randn(2, 3, 480, 480)
    
    # 测试 baseline
    print("\n1. Baseline 模型")
    model_baseline = DeepLabV3Plus(num_classes=8, pretrained=False)
    model_baseline.eval()
    
    with torch.no_grad():
        out = model_baseline(x)
    
    print(f"   输入: {x.shape}")
    print(f"   输出: {out.shape}")
    
    params_baseline = sum(p.numel() for p in model_baseline.parameters())
    print(f"   参数量: {params_baseline / 1e6:.2f}M")
    
    # 测试 BCW-DSHR 版本
    print("\n2. +BCW-DSHR")
    model_bcw = DeepLabV3Plus(num_classes=8, pretrained=False, use_bcw_dshr=True)
    model_bcw.eval()
    
    with torch.no_grad():
        out = model_bcw(x)
    
    params_bcw = sum(p.numel() for p in model_bcw.parameters())
    print(f"   参数量: {params_bcw / 1e6:.2f}M (+{(params_bcw - params_baseline) / 1e6:.2f}M)")
    
    # 测试 DB-VCAM (Dummy) 版本
    print("\n3. +DB-VCAM (Dummy分支)")
    model_dummy = DeepLabV3Plus(num_classes=8, pretrained=False, use_db_vcam=True, use_sam=False)
    model_dummy.eval()
    
    with torch.no_grad():
        out = model_dummy(x)
    
    params_dummy = sum(p.numel() for p in model_dummy.parameters())
    print(f"   参数量: {params_dummy / 1e6:.2f}M (+{(params_dummy - params_baseline) / 1e6:.2f}M)")
    
    # 测试 DB-VCAM (SAM) 版本
    print("\n4. +DB-VCAM (A²-LoRA-SAM分支，简化版编码器)")
    model_sam = DeepLabV3Plus(
        num_classes=8, pretrained=False, 
        use_db_vcam=True, use_sam=True,
        sam_checkpoint=None,  # 使用简化版
        lora_rank=4
    )
    model_sam.eval()
    
    with torch.no_grad():
        out = model_sam(x)
    
    print(f"   输入: {x.shape}")
    print(f"   输出: {out.shape}")
    
    params_sam = sum(p.numel() for p in model_sam.parameters())
    sam_params = model_sam.get_sam_params()
    lora_params = sum(p.numel() for p in sam_params['lora'])
    
    print(f"   总参数量: {params_sam / 1e6:.2f}M (+{(params_sam - params_baseline) / 1e6:.2f}M)")
    print(f"   LoRA参数量: {lora_params / 1e3:.2f}K")
    
    if model_sam.get_last_alpha() is not None:
        print(f"   Alpha值: {model_sam.get_last_alpha().squeeze().tolist()}")
    
    # 测试完整版本
    print("\n5. 完整 MFSP-Net (BCW-DSHR + A²-LoRA-SAM + DB-VCAM)")
    model_full = DeepLabV3Plus(
        num_classes=8, pretrained=False,
        use_bcw_dshr=True, use_db_vcam=True, use_sam=True,
        sam_checkpoint=None, lora_rank=4
    )
    model_full.eval()
    
    with torch.no_grad():
        out = model_full(x)
    
    params_full = sum(p.numel() for p in model_full.parameters())
    print(f"   总参数量: {params_full / 1e6:.2f}M (+{(params_full - params_baseline) / 1e6:.2f}M)")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
