#!/usr/bin/env python3
"""
显存泄漏检测工具
帮助识别和诊断显存泄漏问题
"""

import torch
import gc
import sys
import tracemalloc
from pathlib import Path
from typing import Dict, Tuple, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.gpu_monitor import GPUMemoryMonitor


class MemoryLeakDetector:
    """显存泄漏检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.gpu_monitor = GPUMemoryMonitor()
        self.initial_memory = None
        self.snapshots = []
        
    def start_monitoring(self):
        """开始监测"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        self.initial_memory = self.gpu_monitor.get_memory_usage()
        print(f"初始显存: {self.initial_memory['allocated_gb']:.2f} GB")
    
    def check_memory_growth(self, name: str = "checkpoint"):
        """检查显存增长"""
        current = self.gpu_monitor.get_memory_usage()
        
        if self.initial_memory is None:
            print("✗ 请先调用 start_monitoring()")
            return
        
        growth_mb = (current['allocated_gb'] - self.initial_memory['allocated_gb']) * 1024
        
        self.snapshots.append({
            'name': name,
            'memory_gb': current['allocated_gb'],
            'growth_mb': growth_mb,
        })
        
        status = "✓" if abs(growth_mb) < 100 else "⚠" if abs(growth_mb) < 500 else "✗"
        print(f"{status} {name}: {current['allocated_gb']:.2f} GB (增长: {growth_mb:+.1f} MB)")
        
        if growth_mb > 100:
            print(f"  ⚠️ 检测到显存增长，可能存在泄漏")
        
        return growth_mb
    
    def analyze_tensors(self):
        """分析当前GPU上的tensor"""
        print("\n当前GPU Tensor分析:")
        print("-" * 70)
        
        total_size = 0
        tensor_info = []
        
        # 遍历所有对象
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                size_mb = obj.numel() * obj.element_size() / 1e6
                tensor_info.append({
                    'size_mb': size_mb,
                    'shape': tuple(obj.shape),
                    'dtype': str(obj.dtype),
                })
                total_size += size_mb
        
        # 排序并显示
        tensor_info.sort(key=lambda x: x['size_mb'], reverse=True)
        
        print(f"{'大小(MB)':<12} {'形状':<30} {'数据类型':<15}")
        print("-" * 70)
        
        for info in tensor_info[:20]:  # 显示前20个
            print(f"{info['size_mb']:<12.2f} {str(info['shape']):<30} {info['dtype']:<15}")
        
        print("-" * 70)
        print(f"总计: {total_size:.2f} MB (跟踪到的tensor)")
        print()
        
        return tensor_info
    
    def check_reference_cycles(self):
        """检查引用循环"""
        print("\n引用循环检查:")
        print("-" * 70)
        
        unreachable_before = len(gc.garbage)
        collected = gc.collect()
        unreachable_after = len(gc.garbage)
        
        print(f"回收的对象数: {collected}")
        print(f"垃圾对象数: {unreachable_after}")
        
        if unreachable_after > unreachable_before:
            print("⚠️ 检测到垃圾对象（可能的引用循环）")
        else:
            print("✓ 未检测到明显的引用循环")
        print()
    
    def print_summary(self):
        """打印汇总"""
        if not self.snapshots:
            return
        
        print("\n" + "=" * 70)
        print("显存监测汇总")
        print("=" * 70)
        
        total_growth = sum(s['growth_mb'] for s in self.snapshots)
        max_memory = max(s['memory_gb'] for s in self.snapshots)
        
        print(f"检查点数: {len(self.snapshots)}")
        print(f"总显存增长: {total_growth:+.1f} MB")
        print(f"峰值显存: {max_memory:.2f} GB")
        
        if total_growth > 500:
            print("\n⚠️ 警告: 检测到显存泄漏！")
            print("建议:")
            print("1. 检查是否保留了不必要的tensor")
            print("2. 使用 tensor.detach() 破除计算图")
            print("3. 定期调用 torch.cuda.empty_cache()")
            print("4. 检查模型的forward()方法中的temp变量")
        else:
            print("\n✓ 显存使用正常")
        
        print("=" * 70 + "\n")


def test_memory_leak():
    """演示内存泄漏检测"""
    print("=" * 70)
    print("显存泄漏检测演示")
    print("=" * 70)
    print()
    
    detector = MemoryLeakDetector()
    detector.start_monitoring()
    print()
    
    # 测试1: 正常的tensor操作
    print("测试1: 正常的tensor操作")
    for i in range(5):
        x = torch.randn(1000, 1000).cuda()
        y = torch.mm(x, x.t())
        del x, y  # 显式删除
        torch.cuda.empty_cache()
        detector.check_memory_growth(f"  Step {i+1}")
    print()
    
    # 测试2: 有泄漏的操作
    print("测试2: 检测泄漏的操作（不删除tensor）")
    tensors = []
    for i in range(3):
        x = torch.randn(1000, 1000).cuda()
        tensors.append(x)  # ✗ 没有删除，显存会增长
        detector.check_memory_growth(f"  Step {i+1}")
    print()
    
    # 分析tensors
    detector.analyze_tensors()
    
    # 清理
    tensors.clear()
    torch.cuda.empty_cache()
    detector.check_memory_growth("清理后")
    print()
    
    # 汇总
    detector.print_summary()


def suggest_fixes(issue: str):
    """提供修复建议"""
    suggestions = {
        'detach': """
解决方案: 使用 .detach() 破除计算图

错误代码:
    x = model(input)
    loss = criterion(x, target)
    loss.backward()
    # x会保留计算图，占用显存

正确代码:
    with torch.no_grad():
        x = model(input)
    loss = criterion(x, target)
    loss.backward()
""",
        'no_grad': """
解决方案: 在不需要梯度时使用 torch.no_grad()

错误代码:
    for input, target in dataloader:
        output = model(input)  # 默认记录梯度
        loss = criterion(output, target)

正确代码:
    with torch.no_grad():
        for input, target in dataloader:
            output = model(input)
            loss = criterion(output, target)
""",
        'empty_cache': """
解决方案: 定期清空缓存

优化代码:
    for step in range(num_steps):
        # ... 训练代码 ...
        
        if step % 100 == 0:
            torch.cuda.empty_cache()  # 清空缓存
            gc.collect()               # 回收CPU内存
""",
        'del_batch': """
解决方案: 显式删除不需要的变量

改进代码:
    for batch in dataloader:
        # 处理batch
        x = preprocess(batch)
        output = model(x)
        loss = criterion(output, batch.target)
        loss.backward()
        
        # 删除不需要的变量
        del x, output, batch
        torch.cuda.empty_cache()
""",
    }
    
    return suggestions.get(issue, "未知问题")


if __name__ == "__main__":
    test_memory_leak()
    
    print("\n常见显存泄漏场景和修复:\n")
    
    issues = ['detach', 'no_grad', 'empty_cache', 'del_batch']
    for issue in issues:
        print(suggest_fixes(issue))
        print()
