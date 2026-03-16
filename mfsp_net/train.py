"""
训练脚本
DeepLabV3+ Baseline for SUIM Dataset

支持 A²-LoRA-SAM 三阶段训练策略：
- 阶段1 (1~N1): LoRA预热，α=0.5固定
- 阶段2 (N1+1~N1+N2): 条件分支引入，α动态
- 阶段3 (N1+N2+1~Total): 联合微调，学习率×0.1
"""

import os
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import cfg
from datasets.suim_dataset import get_dataloaders
from models.deeplabv3plus import get_model
from utils.losses import (
    get_loss_function, get_scheduler, SegmentationMetrics,
    CombinedLossWithEdge, compute_edge_from_mask, alpha_diversity_loss
)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, cfg, 
                    use_sam=False, edge_weight=0.5, alpha_div_weight=0.1, current_stage=1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_seg_loss = 0
    total_edge_loss = 0
    total_alpha_div_loss = 0
    num_batches = len(train_loader)
    
    # 获取 SAM 阶段信息（如果使用）
    stage_info = ""
    use_edge = False
    if use_sam and hasattr(model, 'semantic_branch'):
        if hasattr(model.semantic_branch, 'get_stage_info'):
            info = model.semantic_branch.get_stage_info()
            stage_info = f" [Stage{info['stage']}]"
        # 检查是否有边界预测头
        if hasattr(model.semantic_branch, 'edge_head') and model.semantic_branch.edge_head is not None:
            use_edge = True
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}{stage_info}")
    
    # 记录 alpha 统计
    alpha_values = []
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with autocast():
            if use_edge:
                # 使用边界辅助损失
                outputs, edge_pred = model(images, return_edge=True)
                
                # 计算边界标签
                edge_target = compute_edge_from_mask(masks, kernel_size=3)
                # 调整到与edge_pred相同尺寸
                if edge_target.shape[2:] != edge_pred.shape[2:]:
                    edge_target = F.interpolate(
                        edge_target, size=edge_pred.shape[2:], mode='nearest'
                    )
                
                # 分割损失
                seg_loss = criterion(outputs, masks)
                
                # 边界损失
                pos_weight = torch.tensor([5.0], device=device)
                edge_loss = F.binary_cross_entropy_with_logits(
                    edge_pred, edge_target, pos_weight=pos_weight
                )
                
                # 总损失
                loss = seg_loss + edge_weight * edge_loss
                
                total_seg_loss += seg_loss.item()
                total_edge_loss += edge_loss.item()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_seg_loss += loss.item()
            
            # α 多样性正则化（仅 Stage2/3）
            if use_sam and current_stage >= 2 and hasattr(model, 'get_last_alpha'):
                alpha = model.get_last_alpha()
                if alpha is not None:
                    alpha_div_loss = alpha_diversity_loss(alpha, target_std=0.15)
                    loss = loss + alpha_div_weight * alpha_div_loss
                    total_alpha_div_loss += alpha_div_loss.item()
                    alpha_values.extend(alpha.view(-1).tolist())
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # 更新进度条
        if (batch_idx + 1) % cfg.log_interval == 0 or batch_idx == num_batches - 1:
            avg_loss = total_loss / (batch_idx + 1)
            postfix = {
                'loss': f'{avg_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            }
            
            # 显示 alpha 值（均值和标准差）
            if use_sam and hasattr(model, 'get_last_alpha'):
                alpha = model.get_last_alpha()
                if alpha is not None:
                    postfix['α'] = f'{alpha.mean().item():.2f}'
                    if len(alpha_values) > 1:
                        import numpy as np
                        postfix['α_std'] = f'{np.std(alpha_values):.3f}'
            
            # 显示边界损失
            if use_edge and total_edge_loss > 0:
                avg_edge = total_edge_loss / (batch_idx + 1)
                postfix['edge'] = f'{avg_edge:.3f}'
            
            pbar.set_postfix(postfix)
    
    # 打印 epoch 结束时的 α 统计
    if use_sam and len(alpha_values) > 0:
        import numpy as np
        alpha_arr = np.array(alpha_values)
        print(f"  α 统计: mean={alpha_arr.mean():.3f}, std={alpha_arr.std():.3f}, "
              f"min={alpha_arr.min():.3f}, max={alpha_arr.max():.3f}")
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, test_loader, criterion, device, cfg):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    metrics = SegmentationMetrics(
        num_classes=cfg.num_classes,
        class_names=cfg.class_names
    )
    
    pbar = tqdm(test_loader, desc="Evaluating")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        total_loss += loss.item()
        
        # 获取预测结果
        preds = outputs.argmax(dim=1)
        metrics.update(preds, masks)
    
    avg_loss = total_loss / len(test_loader)
    results = metrics.get_metrics()
    
    return avg_loss, results


def save_checkpoint(model, optimizer, scheduler, epoch, best_miou, save_path, model_config=None):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_miou': best_miou,
        'model_config': model_config,  # 保存模型配置
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def get_current_stage(epoch, stage1_epochs, stage2_epochs):
    """根据当前 epoch 确定训练阶段"""
    if epoch <= stage1_epochs:
        return 1
    elif epoch <= stage1_epochs + stage2_epochs:
        return 2
    else:
        return 3


def adjust_learning_rate_for_stage(optimizer, base_lr, stage, stage3_lr_decay=0.1):
    """根据阶段调整学习率（保持各组的相对比例）"""
    decay = stage3_lr_decay if stage == 3 else 1.0
    
    for param_group in optimizer.param_groups:
        # 保持各组的相对学习率比例
        if param_group.get('name') == 'estimator':
            # 条件分支保持 10× 的比例
            param_group['lr'] = base_lr * 10 * decay
        else:
            param_group['lr'] = base_lr * decay
    
    return base_lr * decay


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train DeepLabV3+ on SUIM')
    parser.add_argument('--exp_name', type=str, default=cfg.exp_name, help='实验名称')
    parser.add_argument('--epochs', type=int, default=cfg.epochs, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size, help='批次大小')
    parser.add_argument('--lr', type=float, default=cfg.lr, help='学习率')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--use_bcw_dshr', action='store_true', help='使用BCW-DSHR模块')
    parser.add_argument('--use_db_vcam', action='store_true', help='使用DB-VCAM模块')
    parser.add_argument('--use_sam', action='store_true', help='使用A²-LoRA-SAM替代Dummy分支')
    parser.add_argument('--sam_checkpoint', type=str, default=None, help='SAM预训练权重路径')
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA的秩')
    parser.add_argument('--sam_input_size', type=int, default=512, help='SAM输入尺寸(降低可节省显存)')
    
    # 三阶段训练参数
    parser.add_argument('--stage1_epochs', type=int, default=30, help='阶段1(LoRA预热)的epoch数')
    parser.add_argument('--stage2_epochs', type=int, default=30, help='阶段2(条件分支引入)的epoch数')
    parser.add_argument('--stage3_lr_decay', type=float, default=0.1, help='阶段3学习率衰减因子')
    parser.add_argument('--edge_weight', type=float, default=0.5, help='边界辅助损失权重 λ')
    
    args = parser.parse_args()
    
    # 更新配置
    cfg.exp_name = args.exp_name
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    
    # 根据使用的模块调整实验名
    if args.use_bcw_dshr and 'bcw' not in cfg.exp_name.lower():
        cfg.exp_name = cfg.exp_name + "_bcw_dshr"
    if args.use_db_vcam and 'vcam' not in cfg.exp_name.lower():
        cfg.exp_name = cfg.exp_name + "_db_vcam"
    if args.use_sam and 'sam' not in cfg.exp_name.lower():
        cfg.exp_name = cfg.exp_name + "_sam"
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.output_dir, f"{cfg.exp_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MFSP-Net Training")
    print("=" * 60)
    print(f"实验名称: {cfg.exp_name}")
    print(f"输出目录: {output_dir}")
    print(f"训练轮数: {cfg.epochs}")
    print(f"批次大小: {cfg.batch_size}")
    print(f"学习率: {cfg.lr}")
    print(f"使用BCW-DSHR: {args.use_bcw_dshr}")
    print(f"使用DB-VCAM: {args.use_db_vcam}")
    print(f"使用A²-LoRA-SAM: {args.use_sam}")
    if args.use_sam:
        print(f"SAM权重: {args.sam_checkpoint or '轻量级编码器'}")
        print(f"SAM输入尺寸: {args.sam_input_size}")
        print(f"LoRA Rank: {args.lora_rank}")
        print("-" * 40)
        print("三阶段训练配置:")
        print(f"  阶段1 (LoRA预热):    Epoch 1-{args.stage1_epochs}")
        print(f"  阶段2 (条件分支引入): Epoch {args.stage1_epochs+1}-{args.stage1_epochs+args.stage2_epochs}")
        print(f"  阶段3 (联合微调):    Epoch {args.stage1_epochs+args.stage2_epochs+1}-{args.epochs} (lr×{args.stage3_lr_decay})")
        print(f"  边界辅助损失权重 λ:  {args.edge_weight}")
    print("=" * 60)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 数据加载
    print("\n加载数据集...")
    train_loader, test_loader = get_dataloaders(cfg)
    print(f"训练集: {len(train_loader.dataset)} 张图像")
    print(f"测试集: {len(test_loader.dataset)} 张图像")
    
    # 创建模型
    print("\n创建模型...")
    model = get_model(
        cfg, 
        use_bcw_dshr=args.use_bcw_dshr, 
        use_db_vcam=args.use_db_vcam,
        use_sam=args.use_sam,
        sam_checkpoint=args.sam_checkpoint,
        lora_rank=args.lora_rank,
        sam_input_size=args.sam_input_size
    )
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    if args.use_bcw_dshr:
        bcw_params = sum(p.numel() for p in model.get_bcw_dshr_params())
        print(f"BCW-DSHR参数量: {bcw_params / 1e6:.2f}M")
    
    if args.use_db_vcam:
        dbvcam_params = sum(p.numel() for p in model.get_db_vcam_params())
        print(f"DB-VCAM参数量: {dbvcam_params / 1e6:.2f}M")
    
    if args.use_sam:
        sam_params = model.get_sam_params()
        lora_params = sum(p.numel() for p in sam_params['lora'])
        print(f"SAM LoRA参数量: {lora_params / 1e3:.2f}K")
    
    # 损失函数
    criterion = get_loss_function(cfg)
    
    # 优化器（分组学习率）
    if args.use_sam:
        # 条件分支使用更大的学习率
        sam_params = model.get_sam_params()
        estimator_params = sam_params['estimator']
        estimator_param_ids = set(id(p) for p in estimator_params)
        
        # 其他参数
        other_params = [p for p in model.parameters() 
                       if p.requires_grad and id(p) not in estimator_param_ids]
        
        param_groups = [
            {'params': other_params, 'lr': cfg.lr},
            {'params': estimator_params, 'lr': cfg.lr * 10, 'name': 'estimator'},  # 10× 学习率
        ]
        
        print(f"\n分组学习率:")
        print(f"  - 主干网络: {cfg.lr:.2e}")
        print(f"  - 条件分支 g(I): {cfg.lr * 10:.2e} (10×)")
        
        if cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        # 标准优化器
        if cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=0.9,
                weight_decay=cfg.weight_decay
            )
    
    # 学习率调度器（阶段3会额外降低学习率）
    scheduler = get_scheduler(optimizer, cfg)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 恢复训练
    start_epoch = 1
    best_miou = 0
    
    if args.resume:
        print(f"\n恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint['best_miou']
        print(f"从 epoch {start_epoch} 继续训练, best mIoU: {best_miou:.2f}%")
    
    # 训练日志
    log_file = os.path.join(output_dir, 'training_log.txt')
    
    print("\n开始训练...")
    print("=" * 60)
    
    # 当前阶段跟踪（用于检测阶段切换）
    current_stage = 0
    
    for epoch in range(start_epoch, cfg.epochs + 1):
        # ===== 三阶段训练逻辑 =====
        if args.use_sam:
            new_stage = get_current_stage(epoch, args.stage1_epochs, args.stage2_epochs)
            
            # 检测阶段切换
            if new_stage != current_stage:
                print(f"\n{'='*60}")
                print(f"切换到训练阶段 {new_stage}")
                print(f"{'='*60}")
                
                # 设置模型阶段
                if hasattr(model, 'semantic_branch') and hasattr(model.semantic_branch, 'set_stage'):
                    model.semantic_branch.set_stage(new_stage)
                
                # 阶段3: 降低学习率
                if new_stage == 3:
                    new_lr = adjust_learning_rate_for_stage(
                        optimizer, cfg.lr, new_stage, args.stage3_lr_decay
                    )
                    print(f"学习率调整为: {new_lr:.2e}")
                
                # 重新统计可训练参数
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"当前可训练参数量: {trainable_params / 1e6:.2f}M")
                
                current_stage = new_stage
        else:
            current_stage = 1  # 非SAM模式默认阶段1
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg, 
            use_sam=args.use_sam, edge_weight=args.edge_weight, 
            alpha_div_weight=0.1, current_stage=current_stage
        )
        
        # 更新学习率（scheduler）
        if scheduler and current_stage < 3:  # 阶段3不使用scheduler，保持低学习率
            scheduler.step()
        
        # 评估
        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            eval_loss, results = evaluate(model, test_loader, criterion, device, cfg)
            miou = results['mIoU'] * 100
            
            # 获取 alpha 信息
            alpha_info = ""
            if args.use_sam and hasattr(model, 'get_last_alpha'):
                alpha = model.get_last_alpha()
                if alpha is not None:
                    alpha_info = f" | α={alpha.mean().item():.3f}"
            
            print(f"\n[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
                  f"Eval Loss: {eval_loss:.4f} | mIoU: {miou:.2f}%{alpha_info}")
            
            # 记录日志
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch}, Stage {current_stage}, Train Loss: {train_loss:.4f}, "
                       f"Eval Loss: {eval_loss:.4f}, mIoU: {miou:.2f}%{alpha_info}\n")
            
            # 保存最佳模型
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_miou,
                    os.path.join(output_dir, 'best_model.pth'),
                    model_config={
                        'use_bcw_dshr': args.use_bcw_dshr, 
                        'use_db_vcam': args.use_db_vcam,
                        'use_sam': args.use_sam,
                        'sam_checkpoint': args.sam_checkpoint,
                        'lora_rank': args.lora_rank,
                        'sam_input_size': args.sam_input_size,
                        'stage1_epochs': args.stage1_epochs,
                        'stage2_epochs': args.stage2_epochs
                    }
                )
                print(f"*** 新的最佳 mIoU: {best_miou:.2f}% ***")
        
        # 定期保存
        if epoch % cfg.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_miou,
                os.path.join(output_dir, f'checkpoint_epoch{epoch}.pth'),
                model_config={
                    'use_bcw_dshr': args.use_bcw_dshr, 
                    'use_db_vcam': args.use_db_vcam,
                    'use_sam': args.use_sam,
                    'sam_checkpoint': args.sam_checkpoint,
                    'lora_rank': args.lora_rank,
                    'sam_input_size': args.sam_input_size,
                    'stage1_epochs': args.stage1_epochs,
                    'stage2_epochs': args.stage2_epochs
                }
            )
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, scheduler, cfg.epochs, best_miou,
        os.path.join(output_dir, 'final_model.pth'),
        model_config={
            'use_bcw_dshr': args.use_bcw_dshr, 
            'use_db_vcam': args.use_db_vcam,
            'use_sam': args.use_sam,
            'sam_checkpoint': args.sam_checkpoint,
            'lora_rank': args.lora_rank,
            'sam_input_size': args.sam_input_size,
            'stage1_epochs': args.stage1_epochs,
            'stage2_epochs': args.stage2_epochs
        }
    )
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳 mIoU: {best_miou:.2f}%")
    print(f"模型保存至: {output_dir}")
    
    # 打印BCW-DSHR的gamma值
    if args.use_bcw_dshr:
        print("\nBCW-DSHR gamma 值 (学习到的高频注入强度):")
        print(f"  - bcw_dshr_1: {model.encoder.bcw_dshr_1.gamma.item():.4f}")
        print(f"  - bcw_dshr_2: {model.encoder.bcw_dshr_2.gamma.item():.4f}")
        print(f"  - bcw_dshr_3: {model.encoder.bcw_dshr_3.gamma.item():.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
