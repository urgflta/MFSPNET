"""
SUIM 数据集加载器
支持 RGB mask 到类别ID的转换
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import cfg


class SUIMDataset(Dataset):
    """SUIM 水下图像分割数据集"""
    
    def __init__(self, img_dir, mask_dir, transform=None, mode='train'):
        """
        Args:
            img_dir: 图像目录路径
            mask_dir: 掩码目录路径
            transform: 数据增强变换
            mode: 'train' 或 'test'
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode
        
        # 获取所有图像文件
        self.images = sorted([f for f in os.listdir(img_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        print(f"[{mode.upper()}] 加载了 {len(self.images)} 张图像")
    
    def __len__(self):
        return len(self.images)
    
    def rgb_to_class(self, mask_rgb):
        """
        将 RGB mask 转换为类别ID mask
        
        Args:
            mask_rgb: numpy array, shape (H, W, 3)
        Returns:
            mask_class: numpy array, shape (H, W), dtype=int64
        """
        mask_class = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        
        for rgb, class_id in cfg.color_map.items():
            # 创建匹配掩码
            match = np.all(mask_rgb == rgb, axis=-1)
            mask_class[match] = class_id
        
        return mask_class
    
    def __getitem__(self, idx):
        # 加载图像
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 加载掩码 - 尝试不同的扩展名
        mask_name_base = os.path.splitext(img_name)[0]
        mask_path = None
        
        for ext in ['.png', '.bmp', '.jpg', '.jpeg']:
            candidate = os.path.join(self.mask_dir, mask_name_base + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        
        if mask_path is None:
            # 如果找不到对应mask，尝试直接用同名文件
            mask_path = os.path.join(self.mask_dir, img_name)
        
        mask_rgb = Image.open(mask_path).convert('RGB')
        
        # 确保图像和掩码尺寸一致
        if image.size != mask_rgb.size:
            # 以图像尺寸为准，调整掩码尺寸 (使用最近邻插值保持类别ID)
            mask_rgb = mask_rgb.resize(image.size, Image.NEAREST)
        
        # 转换为numpy数组
        image = np.array(image)
        mask_rgb = np.array(mask_rgb)
        
        # RGB mask 转换为类别ID
        mask = self.rgb_to_class(mask_rgb)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.long()


def get_train_transform(input_size):
    """训练集数据增强"""
    return A.Compose([
        A.Resize(input_size[0], input_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(input_size):
    """验证/测试集数据变换（不做增强）"""
    return A.Compose([
        A.Resize(input_size[0], input_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_dataloaders(cfg):
    """获取训练和测试数据加载器"""
    
    train_transform = get_train_transform(cfg.input_size)
    test_transform = get_val_transform(cfg.input_size)
    
    train_dataset = SUIMDataset(
        img_dir=cfg.train_img_dir,
        mask_dir=cfg.train_mask_dir,
        transform=train_transform,
        mode='train'
    )
    
    test_dataset = SUIMDataset(
        img_dir=cfg.test_img_dir,
        mask_dir=cfg.test_mask_dir,
        transform=test_transform,
        mode='test'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # 测试数据集加载
    train_loader, test_loader = get_dataloaders(cfg)
    
    print(f"\n训练集 batches: {len(train_loader)}")
    print(f"测试集 batches: {len(test_loader)}")
    
    # 检查一个batch
    images, masks = next(iter(train_loader))
    print(f"\n图像 shape: {images.shape}")
    print(f"掩码 shape: {masks.shape}")
    print(f"掩码类别: {torch.unique(masks)}")
