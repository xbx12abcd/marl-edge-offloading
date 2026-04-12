"""
GPU Memory Monitoring and Optimization Utils
针对RTX 4060 Ti等中小型显卡的显存管理
"""

import torch
import psutil
import tracemalloc
from typing import Dict, Optional
import warnings


class GPUMemoryMonitor:
    """GPU显存监控工具"""
    
    def __init__(self, device: torch.device = None, warning_threshold: float = 0.85):
        """
        初始化显存监控器
        
        Args:
            device: 计算设备
            warning_threshold: 显存使用超过此比例时发出警告
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.warning_threshold = warning_threshold
        
        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9  # GB
        else:
            self.total_memory = 0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前显存使用情况"""
        if not torch.cuda.is_available():
            return {
                'allocated': 0,
                'reserved': 0,
                'free': 0,
                'total': 0,
                'utilization_percent': 0
            }
        
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated(self.device) / 1e9  # GB
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        free = self.total_memory - allocated
        
        utilization = (allocated / self.total_memory * 100) if self.total_memory > 0 else 0
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': self.total_memory,
            'utilization_percent': utilization
        }
    
    def print_memory_usage(self, prefix: str = ""):
        """打印显存使用情况"""
        mem = self.get_memory_usage()
        
        print(f"\n{prefix} GPU Memory Usage:")
        print(f"  Allocated: {mem['allocated']:.2f} GB / {mem['total']:.2f} GB")
        print(f"  Reserved:  {mem['reserved']:.2f} GB")
        print(f"  Free:      {mem['free']:.2f} GB")
        print(f"  Usage:     {mem['utilization_percent']:.1f}%")
        
        if mem['utilization_percent'] > self.warning_threshold * 100:
            warnings.warn(
                f"GPU memory usage {mem['utilization_percent']:.1f}% exceeds "
                f"threshold {self.warning_threshold * 100:.0f}%",
                RuntimeWarning
            )
    
    def clear_cache(self):
        """清空显存缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_checkpoint(self) -> Dict[str, float]:
        """保存当前显存快照用于比较"""
        return self.get_memory_usage()


class MemoryOptimizer:
    """显存优化工具"""
    
    @staticmethod
    def enable_gradient_checkpointing(model: torch.nn.Module):
        """启用梯度检查点来减少显存占用"""
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
    
    @staticmethod
    def enable_mixed_precision(scaler: Optional[torch.cuda.amp.GradScaler]) -> torch.cuda.amp.GradScaler:
        """启用混合精度训练（FP16 + FP32）"""
        if torch.cuda.is_available():
            return torch.cuda.amp.GradScaler()
        return None
    
    @staticmethod
    def reduce_network_size(hidden_dim: int, num_layers: int, reduction_factor: float = 0.5) -> Dict:
        """推荐减小网络大小的参数"""
        new_hidden = max(32, int(hidden_dim * reduction_factor))
        new_layers = max(1, int(num_layers * reduction_factor))
        
        return {
            'hidden_dim': new_hidden,
            'num_layers': new_layers,
            'estimated_param_reduction': f"{(1 - reduction_factor) * 100:.0f}%"
        }
    
    @staticmethod
    def calculate_batch_size_for_memory(
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        available_memory_gb: float = 6.0,  # RTX 4060 Ti 保留2GB给系统
        safety_factor: float = 0.8
    ) -> int:
        """
        根据设备显存计算可用的批次大小
        
        简单的启发式估计：每个参数需要约4字节(FP32)
        """
        # 估计模型大小(字节)
        param_estimate = (
            (hidden_dim * state_dim) +  # 第一层
            (hidden_dim * hidden_dim * (num_layers - 1)) +  # 隐藏层
            (hidden_dim * action_dim) +  # 输出层(actor)
            (hidden_dim * 1)  # 输出层(critic)
        ) * 4  # FP32 = 4字节
        
        # 每条轨迹需要的显存(GB)
        trajectory_memory_per_sample = (
            state_dim * 4 +  # state
            8 +  # action
            8 +  # reward
            8 +  # value
            8 +  # log_prob
            8    # done
        ) / 1e9
        
        # 可用显存(字节)
        available_bytes = available_memory_gb * 1e9 * safety_factor
        
        # 留给模型的显存
        model_memory = param_estimate * 2.5  # 考虑梯度和优化器状态
        available_for_batch = available_bytes - model_memory
        
        # 计算批次大小
        bytes_per_sample = (state_dim * 4) * 3 + 64  # 输入、梯度、激活值
        batch_size = max(4, int(available_for_batch / bytes_per_sample))
        
        return batch_size


class CPUMemoryMonitor:
    """CPU内存监控工具"""
    
    def __init__(self, warning_threshold: float = 0.85):
        self.warning_threshold = warning_threshold
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取CPU内存使用"""
        mem = psutil.virtual_memory()
        return {
            'used': mem.used / 1e9,
            'available': mem.available / 1e9,
            'total': mem.total / 1e9,
            'percent': mem.percent
        }
    
    def print_memory_usage(self, prefix: str = ""):
        """打印CPU内存使用"""
        mem = self.get_memory_usage()
        print(f"\n{prefix} CPU Memory Usage:")
        print(f"  Used:      {mem['used']:.2f} GB / {mem['total']:.2f} GB")
        print(f"  Available: {mem['available']:.2f} GB")
        print(f"  Usage:     {mem['percent']:.1f}%")
        
        if mem['percent'] > self.warning_threshold * 100:
            warnings.warn(
                f"CPU memory usage {mem['percent']:.1f}% exceeds "
                f"threshold {self.warning_threshold * 100:.0f}%",
                RuntimeWarning
            )


def print_memory_info():
    """打印所有内存信息"""
    print("\n" + "="*60)
    print("SYSTEM MEMORY INFORMATION")
    print("="*60)
    
    # GPU信息
    if torch.cuda.is_available():
        gpu_mon = GPUMemoryMonitor()
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        gpu_mon.print_memory_usage("GPU")
    else:
        print("\nGPU: Not Available (CPU mode)")
    
    # CPU信息
    cpu_mon = CPUMemoryMonitor()
    cpu_mon.print_memory_usage("CPU")
    
    print("\n" + "="*60)


def get_device_with_memory_info():
    """获取设备并打印内存信息"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"\n✓ CUDA available with {device_count} device(s)")
        
        # 优先使用设备1（通常是独立显卡），如果不存在则使用设备0
        target_device_idx = 1 if device_count > 1 else 0
        
        device = torch.device(f'cuda:{target_device_idx}')
        props = torch.cuda.get_device_properties(target_device_idx)
        
        print(f"✓ Using CUDA Device {target_device_idx}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # 打印所有可用设备信息
        if device_count > 1:
            print(f"\nAvailable CUDA devices:")
            for i in range(device_count):
                props_i = torch.cuda.get_device_properties(i)
                marker = " ← SELECTED" if i == target_device_idx else ""
                print(f"  Device {i}: {props_i.name}{marker}")
    else:
        device = torch.device('cpu')
        print("\n✓ Running on CPU (no CUDA device available)")
    
    return device
