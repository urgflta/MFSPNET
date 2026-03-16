"""
测试脚本
评估模型性能并可视化结果
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import cfg
from datasets.suim_dataset import get_dataloaders, get_val_transform, SUIMDataset
from models.deeplabv3plus import get_model
from utils.losses import SegmentationMetrics


def visualize_prediction(image, pred, target, save_path, cfg):
    """
    可视化预测结果
    
    Args:
        image: (3, H, W) normalized tensor
        pred: (H, W) predicted class indices
        target: (H, W) ground truth class indices
        save_path: 保存路径
    """
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = (image * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    
    # 转换为RGB掩码
    pred_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    target_rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
    
    for class_id, color in cfg.id_to_color.items():
        pred_rgb[pred == class_id] = color
        target_rgb[target == class_id] = color
    
    # 拼接: 原图 | 真值 | 预测
    combined = np.concatenate([image, target_rgb, pred_rgb], axis=1)
    
    Image.fromarray(combined).save(save_path)


@torch.no_grad()
def test(model, test_loader, device, cfg, output_dir, visualize=True, num_vis=20):
    """测试模型"""
    model.eval()
    
    metrics = SegmentationMetrics(
        num_classes=cfg.num_classes,
        class_names=cfg.class_names
    )
    
    vis_dir = os.path.join(output_dir, 'visualizations')
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    vis_count = 0
    pbar = tqdm(test_loader, desc="Testing")
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        with autocast():
            outputs = model(images)
        
        preds = outputs.argmax(dim=1)
        metrics.update(preds, masks)
        
        # 可视化部分结果
        if visualize and vis_count < num_vis:
            for i in range(min(images.size(0), num_vis - vis_count)):
                save_path = os.path.join(vis_dir, f'vis_{vis_count:04d}.png')
                visualize_prediction(
                    images[i].cpu(),
                    preds[i].cpu().numpy(),
                    masks[i].cpu().numpy(),
                    save_path,
                    cfg
                )
                vis_count += 1
    
    # 打印详细结果
    metrics.print_metrics()
    
    return metrics.get_metrics()


def main():
    parser = argparse.ArgumentParser(description='Test DeepLabV3+ on SUIM')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--num_vis', type=int, default=20, help='可视化数量')
    # 以下参数可选，如果checkpoint中有配置则自动使用
    parser.add_argument('--use_bcw_dshr', action='store_true', help='使用BCW-DSHR模块 (可自动从checkpoint检测)')
    parser.add_argument('--use_db_vcam', action='store_true', help='使用DB-VCAM模块 (可自动从checkpoint检测)')
    parser.add_argument('--use_sam', action='store_true', help='使用A²-LoRA-SAM (可自动从checkpoint检测)')
    parser.add_argument('--sam_checkpoint', type=str, default=None, help='SAM预训练权重路径')
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA的秩')
    parser.add_argument('--sam_input_size', type=int, default=512, help='SAM输入尺寸')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载checkpoint
    print(f"\n加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # 从checkpoint中读取模型配置（如果存在）
    model_config = checkpoint.get('model_config', {})
    use_bcw_dshr = model_config.get('use_bcw_dshr', args.use_bcw_dshr)
    use_db_vcam = model_config.get('use_db_vcam', args.use_db_vcam)
    use_sam = model_config.get('use_sam', args.use_sam)
    sam_checkpoint = model_config.get('sam_checkpoint', args.sam_checkpoint)
    lora_rank = model_config.get('lora_rank', args.lora_rank)
    sam_input_size = model_config.get('sam_input_size', args.sam_input_size)
    
    print(f"模型配置: use_bcw_dshr={use_bcw_dshr}, use_db_vcam={use_db_vcam}, use_sam={use_sam}")
    
    # 创建模型
    model = get_model(
        cfg, 
        use_bcw_dshr=use_bcw_dshr, 
        use_db_vcam=use_db_vcam,
        use_sam=use_sam,
        sam_checkpoint=sam_checkpoint,
        lora_rank=lora_rank,
        sam_input_size=sam_input_size
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"模型加载完成 (来自 epoch {checkpoint.get('epoch', 'unknown')}, mIoU: {checkpoint.get('best_miou', 'unknown'):.2f}%)")
    
    # 加载数据
    _, test_loader = get_dataloaders(cfg)
    print(f"测试集: {len(test_loader.dataset)} 张图像")
    
    # 测试
    results = test(model, test_loader, device, cfg, args.output_dir, 
                   visualize=args.visualize, num_vis=args.num_vis)
    
    # 保存结果
    results_file = os.path.join(args.output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write("SUIM Test Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"mIoU: {results['mIoU'] * 100:.2f}%\n")
        f.write(f"Pixel Acc: {results['Pixel_Acc'] * 100:.2f}%\n")
        f.write(f"Mean Acc: {results['Mean_Acc'] * 100:.2f}%\n")
        f.write(f"Mean F1: {results['Mean_F1'] * 100:.2f}%\n")
        f.write("\nPer-class IoU:\n")
        for name, iou in results['IoU_per_class'].items():
            f.write(f"  {name}: {iou * 100:.2f}%\n")
    
    print(f"\n结果已保存至: {results_file}")
    if args.visualize:
        print(f"可视化结果保存至: {os.path.join(args.output_dir, 'visualizations')}")


if __name__ == "__main__":
    main()
