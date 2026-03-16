#!/usr/bin/env python
"""
GPU 显存清理工具
在训练前运行以释放显存
"""

import torch
import gc
import os

def clear_gpu_memory():
    """清理 GPU 显存"""
    print("=" * 50)
    print("GPU 显存清理")
    print("=" * 50)
    
    if torch.cuda.is_available():
        # 显示清理前状态
        print(f"\n清理前:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i} ({props.name}):")
            print(f"    总显存: {total:.2f} GB")
            print(f"    已分配: {allocated:.2f} GB")
            print(f"    已缓存: {cached:.2f} GB")
        
        # 清理
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 显示清理后状态
        print(f"\n清理后:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}:")
            print(f"    已分配: {allocated:.2f} GB")
            print(f"    已缓存: {cached:.2f} GB")
        
        print("\n✓ 清理完成")
    else:
        print("未检测到 GPU")
    
    print("=" * 50)


def show_gpu_processes():
    """显示占用 GPU 的进程"""
    print("\n当前占用 GPU 的进程:")
    os.system("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GPU 显存清理工具')
    parser.add_argument('--show-processes', action='store_true', help='显示占用GPU的进程')
    args = parser.parse_args()
    
    clear_gpu_memory()
    
    if args.show_processes:
        show_gpu_processes()
