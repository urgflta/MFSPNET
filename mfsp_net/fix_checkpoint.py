"""
修复旧的 checkpoint 文件，添加 model_config 信息
用于处理在更新代码之前训练的模型
"""

import argparse
import torch


def fix_checkpoint(checkpoint_path, use_bcw_dshr, use_db_vcam, output_path=None):
    """
    给旧的 checkpoint 添加 model_config
    
    Args:
        checkpoint_path: 原始 checkpoint 路径
        use_bcw_dshr: 是否使用了 BCW-DSHR
        use_db_vcam: 是否使用了 DB-VCAM
        output_path: 输出路径，默认覆盖原文件
    """
    if output_path is None:
        output_path = checkpoint_path
    
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 检查是否已有配置
    if 'model_config' in checkpoint:
        print(f"已有配置: {checkpoint['model_config']}")
        print("是否覆盖? (y/n)")
        if input().lower() != 'y':
            print("取消操作")
            return
    
    # 添加配置
    checkpoint['model_config'] = {
        'use_bcw_dshr': use_bcw_dshr,
        'use_db_vcam': use_db_vcam
    }
    
    # 保存
    torch.save(checkpoint, output_path)
    print(f"已保存到: {output_path}")
    print(f"配置: use_bcw_dshr={use_bcw_dshr}, use_db_vcam={use_db_vcam}")


def auto_detect_config(checkpoint_path):
    """
    自动检测 checkpoint 中使用了哪些模块
    """
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    state_dict = checkpoint['model_state_dict']
    keys = list(state_dict.keys())
    
    # 检测 BCW-DSHR
    use_bcw_dshr = any('bcw_dshr' in k for k in keys)
    
    # 检测 DB-VCAM
    use_db_vcam = any('db_vcam' in k or 'semantic_branch' in k for k in keys)
    
    print(f"\n自动检测结果:")
    print(f"  use_bcw_dshr: {use_bcw_dshr}")
    print(f"  use_db_vcam: {use_db_vcam}")
    
    return use_bcw_dshr, use_db_vcam


def main():
    parser = argparse.ArgumentParser(description='修复旧的 checkpoint 文件')
    parser.add_argument('checkpoint', type=str, help='checkpoint 文件路径')
    parser.add_argument('--auto', action='store_true', help='自动检测并修复')
    parser.add_argument('--use_bcw_dshr', action='store_true', help='标记使用了 BCW-DSHR')
    parser.add_argument('--use_db_vcam', action='store_true', help='标记使用了 DB-VCAM')
    parser.add_argument('--output', type=str, default=None, help='输出路径 (默认覆盖原文件)')
    args = parser.parse_args()
    
    if args.auto:
        # 自动检测
        use_bcw_dshr, use_db_vcam = auto_detect_config(args.checkpoint)
        print("\n是否使用检测结果修复? (y/n)")
        if input().lower() == 'y':
            fix_checkpoint(args.checkpoint, use_bcw_dshr, use_db_vcam, args.output)
    else:
        # 手动指定
        fix_checkpoint(args.checkpoint, args.use_bcw_dshr, args.use_db_vcam, args.output)


if __name__ == "__main__":
    main()
