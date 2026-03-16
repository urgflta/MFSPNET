"""
MFSP-Net 配置文件
Baseline: DeepLabV3+ with ResNet-50
"""

import os

class Config:
    # ==================== 数据集配置 ====================
    data_root = "/home/heyutong/bs/data/SUIM"
    train_img_dir = os.path.join(data_root, "train/images")
    train_mask_dir = os.path.join(data_root, "train/masks")
    test_img_dir = os.path.join(data_root, "test/images")
    test_mask_dir = os.path.join(data_root, "test/masks")
    
    # SUIM 类别定义 (8类)
    num_classes = 8
    class_names = [
        "Background",           # 0 - (0,0,0)
        "Human divers",         # 1 - (0,0,255)
        "Aquatic plants",       # 2 - (0,255,0)
        "Wrecks/ruins",         # 3 - (0,255,255)
        "Robots/instruments",   # 4 - (255,0,0)
        "Reefs/invertebrates",  # 5 - (255,0,255)
        "Fish/vertebrates",     # 6 - (255,255,0)
        "Sea-floor/rocks",      # 7 - (255,255,255)
    ]
    
    # RGB颜色到类别ID的映射
    color_map = {
        (0, 0, 0): 0,
        (0, 0, 255): 1,
        (0, 255, 0): 2,
        (0, 255, 255): 3,
        (255, 0, 0): 4,
        (255, 0, 255): 5,
        (255, 255, 0): 6,
        (255, 255, 255): 7,
    }
    
    # 类别ID到RGB颜色的映射 (用于可视化)
    id_to_color = {v: k for k, v in color_map.items()}
    
    # ==================== 模型配置 ====================
    backbone = "resnet50"
    output_stride = 16  # DeepLabV3+ 标准配置
    pretrained = True
    
    # ==================== 训练配置 ====================
    # 4090 24GB 显存，可以用较大batch size
    batch_size = 8
    num_workers = 4
    
    # 图像尺寸
    input_size = (480, 480)  # SUIM原图大小不一，统一resize
    
    # 优化器
    optimizer = "adamw"
    lr = 1e-4
    weight_decay = 1e-4
    
    # 学习率调度
    scheduler = "poly"
    poly_power = 0.9
    
    # 训练轮数
    epochs = 100
    
    # 损失函数
    loss_type = "ce_dice"  # "ce" / "dice" / "ce_dice"
    ce_weight = 1.0
    dice_weight = 1.0
    
    # ==================== 日志与保存 ====================
    exp_name = "deeplabv3plus_baseline"
    output_dir = "./outputs"
    save_interval = 10  # 每N个epoch保存一次
    log_interval = 20   # 每N个batch打印一次
    
    # ==================== 数据增强 ====================
    use_augmentation = True
    horizontal_flip = True
    vertical_flip = False
    random_crop = True
    color_jitter = True
    
    # ==================== 评估配置 ====================
    eval_interval = 5  # 每N个epoch评估一次


# 全局配置实例
cfg = Config()
