"""
A²-LoRA-SAM: Adaptive² LoRA-enhanced SAM Encoder

核心思想：
1. 利用 SAM 预训练模型的强大语义先验
2. 通过 LoRA 进行参数高效的域适配
3. 自适应 α(I) 根据图像退化程度动态调控 LoRA 强度
4. 特征对齐层确保与主干网络的兼容性

三阶段训练策略（解决梯度耦合问题）：
┌─────────────────────────────────────────────────────────────┐
│ 阶段一：LoRA预热（Epoch 1 ~ N₁）                              │
│ • 冻结：条件分支 g(I)                                        │
│ • 固定：α = 0.5（中间值）                                    │
│ • 训练：LoRA(A, B), 对齐适配器 φ(·)                          │
│ • 目的：让LoRA先学习基本的水下域适配能力                      │
├─────────────────────────────────────────────────────────────┤
│ 阶段二：条件分支引入（Epoch N₁+1 ~ N₁+N₂）                    │
│ • 解冻：条件分支 g(I)                                        │
│ • 动态：α(I) ∈ [0.1, 1.0]                                   │
│ • 训练：全部可训练参数                                        │
│ • 目的：学习图像条件到适配强度的映射                          │
├─────────────────────────────────────────────────────────────┤
│ 阶段三：联合微调（Epoch N₁+N₂+1 ~ N_total）                   │
│ • 降低学习率（×0.1）                                         │
│ • 继续联合优化所有可训练参数                                  │
│ • 目的：精细调整，稳定收敛                                    │
└─────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== 退化程度估计网络 ====================

class DegradationEstimator(nn.Module):
    """
    退化程度估计网络 g(I)
    
    轻量级 CNN，估计输入图像的退化程度
    输出 α(I) ∈ [α_min, α_max]
    
    改进：
    1. 更深的特征提取
    2. 多尺度信息融合
    3. 更好的初始化确保输出多样性
    """
    
    def __init__(self, alpha_min=0.1, alpha_max=1.0, init_bias=0.0):
        super().__init__()
        
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # 更强的特征提取
        self.features = nn.Sequential(
            # Stage 1: 提取低级特征（颜色、亮度）
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Stage 2: 提取中级特征（纹理、对比度）
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 3: 提取高级特征（退化模式）
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 全局池化
            nn.AdaptiveAvgPool2d(1),
        )
        
        # 预测头：更大的容量
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 轻微 dropout 增加鲁棒性
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        
        # 初始化：让最后一层有更大的初始方差
        # 这样不同图像会产生不同的初始 α
        nn.init.xavier_normal_(self.head[-1].weight, gain=2.0)
        nn.init.constant_(self.head[-1].bias, init_bias)
    
    def forward(self, x):
        feat = self.features(x)
        logit = self.head(feat)
        # Sigmoid 映射到 [alpha_min, alpha_max]
        alpha = torch.sigmoid(logit) * (self.alpha_max - self.alpha_min) + self.alpha_min
        return alpha


# ==================== LoRA 模块 ====================

class LoRALinear(nn.Module):
    """LoRA 增强的线性层"""
    
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        
        # LoRA 矩阵: W = W0 + α * B @ A
        # A: (rank, in_features) - 降维
        # B: (out_features, rank) - 升维
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # A 使用 kaiming 初始化，B 使用小随机值（不是零，避免梯度消失）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.normal_(self.lora_B, mean=0, std=0.01)  # 小随机值而非零
    
    def forward(self, x, alpha=1.0):
        # LoRA 输出: α * (x @ A^T @ B^T) / rank
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return lora_out * (alpha / self.rank)


# ==================== 轻量级语义编码器 ====================

class LightweightSemanticEncoder(nn.Module):
    """
    轻量级语义编码器
    
    当 SAM 无法使用或显存不足时的替代方案
    结构更简单但保留了 LoRA 风格的可调节性
    """
    
    def __init__(self, out_channels=256, lora_rank=4):
        super().__init__()
        
        # 主干网络：高效的多尺度特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        
        self.layer1 = self._make_layer(64, 128, stride=2)
        self.layer2 = self._make_layer(128, 256, stride=2)
        self.layer3 = self._make_layer(256, 512, stride=2)
        
        # 输出投影（冻结部分）
        self.out_proj = nn.Sequential(
            nn.Conv2d(512, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # LoRA 风格的可调节增强模块
        # 这部分模拟 LoRA 的行为：基础输出 + α * 增量
        self.lora_down = nn.Conv2d(out_channels, lora_rank * 4, 1, bias=False)
        self.lora_up = nn.Conv2d(lora_rank * 4, out_channels, 1, bias=False)
        
        # 初始化：lora_up 使用小随机值
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.normal_(self.lora_up.weight, mean=0, std=0.01)
        
        self.lora_layers = nn.ModuleList([self.lora_down, self.lora_up])
    
    def _make_layer(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, alpha=1.0):
        # 主干特征提取
        x = self.stem(x)      # H/4
        x = self.layer1(x)    # H/8
        x = self.layer2(x)    # H/16
        x = self.layer3(x)    # H/32
        
        # 基础输出
        base_out = self.out_proj(x)
        
        # LoRA 增量: α * lora_up(relu(lora_down(x)))
        lora_delta = self.lora_down(base_out)
        lora_delta = F.relu(lora_delta)
        lora_delta = self.lora_up(lora_delta)
        
        # 输出 = 基础 + α * 增量
        out = base_out + alpha * lora_delta
        
        return out


# ==================== 高效 SAM 编码器 ====================

class EfficientSAMEncoder(nn.Module):
    """
    高效 SAM 编码器
    
    使用官方 SAM 但通过以下方式节省显存：
    1. 使用更小的输入尺寸 (512 而非 1024)
    2. 只在最后几层添加 LoRA
    3. 冻结部分使用 torch.no_grad()
    """
    
    def __init__(self, sam_checkpoint, rank=4, input_size=512, num_lora_layers=4):
        super().__init__()
        
        self.input_size = input_size
        self.rank = rank
        
        from segment_anything import sam_model_registry
        
        print(f"[EfficientSAMEncoder] 加载 SAM: {sam_checkpoint}")
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.image_encoder = sam.image_encoder
        del sam
        
        # 冻结所有原始参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # 获取维度信息
        embed_dim = self.image_encoder.patch_embed.proj.out_channels
        
        # 只在最后几层添加 LoRA
        num_blocks = len(self.image_encoder.blocks)
        self.lora_block_indices = list(range(num_blocks - num_lora_layers, num_blocks))
        
        self.lora_layers = nn.ModuleList()
        for _ in self.lora_block_indices:
            lora = LoRALinear(embed_dim, embed_dim, rank)
            self.lora_layers.append(lora)
        
        print(f"[EfficientSAMEncoder] 在最后 {len(self.lora_block_indices)} 层添加 LoRA")
        print(f"[EfficientSAMEncoder] LoRA rank={rank}, embed_dim={embed_dim}")
    
    def forward(self, x, alpha=1.0):
        B = x.shape[0]
        
        # Resize 到目标尺寸
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        # Patch embedding
        with torch.no_grad():
            x = self.image_encoder.patch_embed(x)
            if self.image_encoder.pos_embed is not None:
                pos_embed = self.image_encoder.pos_embed
                if pos_embed.shape[1:3] != x.shape[1:3]:
                    pos_embed = F.interpolate(
                        pos_embed.permute(0, 3, 1, 2),
                        size=x.shape[1:3],
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                x = x + pos_embed
        
        # Transformer blocks
        lora_idx = 0
        for block_idx, block in enumerate(self.image_encoder.blocks):
            if block_idx in self.lora_block_indices:
                # 带 LoRA 的块
                shortcut = x
                x = block.norm1(x)
                
                # 原始注意力
                B_size, H, W, C = x.shape
                with torch.no_grad():
                    attn_out = block.attn(x)
                
                # LoRA 增量（应用到注意力输出）
                x_flat = x.reshape(B_size, H * W, C)
                lora_delta = self.lora_layers[lora_idx](x_flat, alpha)
                lora_delta = lora_delta.reshape(B_size, H, W, C)
                
                x = shortcut + attn_out + lora_delta
                
                # FFN
                with torch.no_grad():
                    x = x + block.mlp(block.norm2(x))
                
                lora_idx += 1
            else:
                # 完全冻结的块
                with torch.no_grad():
                    shortcut = x
                    x = block.norm1(x)
                    x = block.attn(x)
                    x = shortcut + x
                    x = x + block.mlp(block.norm2(x))
        
        # Neck
        with torch.no_grad():
            x = self.image_encoder.neck(x.permute(0, 3, 1, 2))
        
        return x


# ==================== 边界预测头 ====================

class EdgePredictionHead(nn.Module):
    """
    轻量化边界预测头
    
    仅在训练阶段使用，用于增强语义分支对边界的响应能力
    推理阶段移除，不增加计算开销
    """
    
    def __init__(self, in_channels=256, mid_channels=64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, 1, 1)  # 二分类：边界/非边界
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


# ==================== 完整的 A²-LoRA-SAM ====================

class A2LoRASAM(nn.Module):
    """
    A²-LoRA-SAM: Adaptive² LoRA-enhanced SAM
    
    支持三阶段训练策略：
    - 阶段 1：LoRA 预热，冻结条件分支，固定 α=0.5
    - 阶段 2：条件分支引入，解冻全部，动态 α(I)
    - 阶段 3：联合微调，降低学习率
    """
    
    # 训练阶段常量
    STAGE_LORA_WARMUP = 1      # LoRA 预热
    STAGE_CONDITION_INTRO = 2  # 条件分支引入
    STAGE_JOINT_FINETUNE = 3   # 联合微调
    
    def __init__(
        self,
        sam_checkpoint=None,
        out_channels=256,
        lora_rank=4,
        alpha_min=0.1,
        alpha_max=1.0,
        use_simple_encoder=False,
        input_size=480,
        sam_input_size=512,
        efficient_mode=True,
        use_edge_head=True,  # 是否使用边界预测头
    ):
        super().__init__()
        
        self.input_size = input_size
        self.out_channels = out_channels
        self.use_official_sam = False
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.use_edge_head = use_edge_head
        
        # 当前训练阶段（默认阶段1）
        self._current_stage = self.STAGE_LORA_WARMUP
        self._fixed_alpha = 0.5  # 阶段1使用的固定 α 值
        
        # 退化程度估计网络 g(I)
        self.degradation_estimator = DegradationEstimator(
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            init_bias=0.0  # 初始输出在中间
        )
        
        # 选择编码器
        if use_simple_encoder or sam_checkpoint is None:
            print("[A²-LoRA-SAM] 使用轻量级语义编码器")
            self.encoder = LightweightSemanticEncoder(out_channels=256, lora_rank=lora_rank)
            self.sam_input_size = None
        else:
            print(f"[A²-LoRA-SAM] 尝试加载 SAM: {sam_checkpoint}")
            try:
                self.encoder = EfficientSAMEncoder(
                    sam_checkpoint, 
                    rank=lora_rank,
                    input_size=sam_input_size,
                    num_lora_layers=4
                )
                self.sam_input_size = sam_input_size
                self.use_official_sam = True
                print("[A²-LoRA-SAM] SAM 加载成功")
            except Exception as e:
                print(f"[A²-LoRA-SAM] SAM 加载失败: {e}")
                print("[A²-LoRA-SAM] 回退到轻量级编码器")
                self.encoder = LightweightSemanticEncoder(out_channels=256, lora_rank=lora_rank)
                self.sam_input_size = None
        
        # 特征对齐层 φ(·)
        self.alignment = nn.Sequential(
            nn.Conv2d(256, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # 边界预测头（仅训练时使用）
        if use_edge_head:
            self.edge_head = EdgePredictionHead(in_channels=out_channels, mid_channels=64)
            print("[A²-LoRA-SAM] 启用边界辅助损失")
        else:
            self.edge_head = None
        
        # 记录值（用于监控和调试）
        self.last_alpha = None
        self.last_edge_pred = None
        
        # 初始化时设置阶段1的参数状态
        self._apply_stage_config()
    
    def set_stage(self, stage):
        """
        设置训练阶段
        
        Args:
            stage: 1, 2, 或 3
        """
        assert stage in [1, 2, 3], f"Invalid stage: {stage}"
        
        if stage != self._current_stage:
            print(f"[A²-LoRA-SAM] 切换到阶段 {stage}")
            self._current_stage = stage
            self._apply_stage_config()
    
    def _apply_stage_config(self):
        """根据当前阶段配置参数的 requires_grad"""
        
        if self._current_stage == self.STAGE_LORA_WARMUP:
            # 阶段1：冻结条件分支，只训练 LoRA 和对齐层
            print("[A²-LoRA-SAM] 阶段1配置: 冻结条件分支 g(I), 固定α=0.5")
            
            # 冻结条件分支
            for param in self.degradation_estimator.parameters():
                param.requires_grad = False
            
            # 训练 LoRA 层
            for param in self.get_lora_params():
                param.requires_grad = True
            
            # 训练对齐层
            for param in self.alignment.parameters():
                param.requires_grad = True
            
            # 训练边界预测头
            if self.edge_head is not None:
                for param in self.edge_head.parameters():
                    param.requires_grad = True
                
        elif self._current_stage == self.STAGE_CONDITION_INTRO:
            # 阶段2：解冻条件分支
            print("[A²-LoRA-SAM] 阶段2配置: 解冻条件分支 g(I), 动态α(I)")
            
            # 解冻条件分支
            for param in self.degradation_estimator.parameters():
                param.requires_grad = True
            
            # 继续训练 LoRA 和对齐层
            for param in self.get_lora_params():
                param.requires_grad = True
            for param in self.alignment.parameters():
                param.requires_grad = True
            
            # 训练边界预测头
            if self.edge_head is not None:
                for param in self.edge_head.parameters():
                    param.requires_grad = True
                
        elif self._current_stage == self.STAGE_JOINT_FINETUNE:
            # 阶段3：继续联合训练（学习率由外部控制）
            print("[A²-LoRA-SAM] 阶段3配置: 联合微调")
            
            for param in self.degradation_estimator.parameters():
                param.requires_grad = True
            for param in self.get_lora_params():
                param.requires_grad = True
            for param in self.alignment.parameters():
                param.requires_grad = True
            if self.edge_head is not None:
                for param in self.edge_head.parameters():
                    param.requires_grad = True
    
    def get_alpha(self, x):
        """
        获取 α 值（根据当前阶段决定是固定值还是动态值）
        
        Args:
            x: 输入图像 (B, 3, H, W)
        Returns:
            alpha: (B, 1) 或标量
        """
        if self._current_stage == self.STAGE_LORA_WARMUP:
            # 阶段1：使用固定 α
            B = x.shape[0]
            alpha = torch.full((B, 1), self._fixed_alpha, device=x.device, dtype=x.dtype)
        else:
            # 阶段2/3：使用条件分支计算 α(I)
            alpha = self.degradation_estimator(x)
        
        return alpha
    
    def forward(self, x, return_edge=False):
        """
        Args:
            x: 输入图像 (B, 3, H, W)
            return_edge: 是否返回边界预测（训练时使用）
        Returns:
            feat: 语义特征 (B, out_channels, H/16, W/16)
            edge_pred: 边界预测 (B, 1, H/16, W/16)，仅当 return_edge=True 且训练模式
        """
        B, C, H, W = x.shape
        target_size = (H // 16, W // 16)
        
        # 获取 α（根据阶段可能是固定值或动态值）
        alpha = self.get_alpha(x)
        self.last_alpha = alpha.detach()
        alpha_mean = alpha.mean()
        
        # 编码
        feat = self.encoder(x, alpha=alpha_mean)
        
        # 对齐
        feat = self.alignment(feat)
        
        # 调整到目标尺寸
        if feat.shape[2:] != target_size:
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
        
        # 边界预测（训练时）
        if return_edge and self.edge_head is not None and self.training:
            edge_pred = self.edge_head(feat)
            self.last_edge_pred = edge_pred.detach()
            return feat, edge_pred
        
        return feat
    
    def get_lora_params(self):
        """获取 LoRA 参数"""
        params = []
        if hasattr(self.encoder, 'lora_layers'):
            for module in self.encoder.lora_layers:
                params.extend(module.parameters())
        return params
    
    def get_estimator_params(self):
        """获取退化估计网络参数"""
        return list(self.degradation_estimator.parameters())
    
    def get_alignment_params(self):
        """获取对齐层参数"""
        return list(self.alignment.parameters())
    
    def get_edge_head_params(self):
        """获取边界预测头参数"""
        if self.edge_head is not None:
            return list(self.edge_head.parameters())
        return []
    
    def get_stage_info(self):
        """获取当前阶段信息"""
        stage_names = {
            1: "LoRA预热 (α=0.5固定)",
            2: "条件分支引入 (α动态)",
            3: "联合微调"
        }
        return {
            'stage': self._current_stage,
            'name': stage_names[self._current_stage],
            'estimator_trainable': any(p.requires_grad for p in self.degradation_estimator.parameters()),
            'alpha_mode': 'fixed' if self._current_stage == 1 else 'dynamic'
        }


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("A²-LoRA-SAM 三阶段训练测试")
    print("=" * 60)
    
    # 创建模型
    model = A2LoRASAM(
        sam_checkpoint=None,
        out_channels=256,
        lora_rank=4,
        use_simple_encoder=True
    )
    
    x = torch.randn(2, 3, 480, 480)
    
    # 测试阶段1
    print("\n--- 阶段1: LoRA预热 ---")
    model.set_stage(1)
    info = model.get_stage_info()
    print(f"阶段: {info['name']}")
    print(f"条件分支可训练: {info['estimator_trainable']}")
    print(f"α模式: {info['alpha_mode']}")
    
    with torch.no_grad():
        feat = model(x)
    print(f"输出形状: {feat.shape}")
    print(f"α值: {model.last_alpha.squeeze().tolist()}")
    
    # 测试阶段2
    print("\n--- 阶段2: 条件分支引入 ---")
    model.set_stage(2)
    info = model.get_stage_info()
    print(f"阶段: {info['name']}")
    print(f"条件分支可训练: {info['estimator_trainable']}")
    print(f"α模式: {info['alpha_mode']}")
    
    with torch.no_grad():
        feat = model(x)
    print(f"输出形状: {feat.shape}")
    print(f"α值: {model.last_alpha.squeeze().tolist()}")
    
    # 测试阶段3
    print("\n--- 阶段3: 联合微调 ---")
    model.set_stage(3)
    info = model.get_stage_info()
    print(f"阶段: {info['name']}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("A²-LoRA-SAM 模块测试")
    print("=" * 60)
    
    # 测试轻量级编码器
    print("\n1. 测试 LightweightSemanticEncoder")
    encoder = LightweightSemanticEncoder(out_channels=256)
    x = torch.randn(2, 3, 480, 480)
    with torch.no_grad():
        feat = encoder(x, alpha=0.5)
    print(f"   输入: {x.shape}")
    print(f"   输出: {feat.shape}")
    params = sum(p.numel() for p in encoder.parameters())
    print(f"   参数量: {params / 1e6:.2f}M")
    
    # 测试完整模块（轻量版）
    print("\n2. 测试 A²-LoRA-SAM (轻量级编码器)")
    a2_sam = A2LoRASAM(
        sam_checkpoint=None,
        out_channels=256,
        lora_rank=4,
        use_simple_encoder=True
    )
    
    with torch.no_grad():
        feat = a2_sam(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {feat.shape}")
    print(f"   Alpha: {a2_sam.last_alpha.squeeze().tolist()}")
    
    params = sum(p.numel() for p in a2_sam.parameters())
    print(f"   总参数量: {params / 1e6:.2f}M")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
