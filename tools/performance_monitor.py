#!/usr/bin/env python3
"""
GPU训练性能监测工具
用于实时跟踪显存、CPU、训练进度等关键指标
"""

import time
import torch
import psutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.gpu_monitor import GPUMemoryMonitor


class PerformanceMonitor:
    """训练性能监测器"""
    
    def __init__(self, name: str = "training", save_interval: int = 10):
        """
        初始化监测器
        
        Args:
            name: 实验名称
            save_interval: 每多少步保存一次数据
        """
        self.name = name
        self.save_interval = save_interval
        self.gpu_monitor = GPUMemoryMonitor()
        
        # 记录的数据
        self.step = 0
        self.start_time = time.time()
        self.episode = 0
        
        self.history = {
            'timestamps': [],
            'episodes': [],
            'steps': [],
            'gpu_memory_allocated': [],
            'gpu_memory_reserved': [],
            'gpu_memory_utilization': [],
            'cpu_memory_usage': [],
            'cpu_percent': [],
            'training_time': [],
            'fps': [],  # steps per second
        }
        
        self.checkpoint_path = Path(f"results/{name}_monitor.json")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
    def update(self, episode: int, loss: float = None, reward: float = None):
        """
        更新监测数据
        
        Args:
            episode: 当前episode数
            loss: 当前loss (可选)
            reward: 当前reward (可选)
        """
        self.episode = episode
        self.step += 1
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        fps = self.step / elapsed_time if elapsed_time > 0 else 0
        
        # GPU信息
        gpu_info = self.gpu_monitor.get_memory_usage()
        
        # CPU信息
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_memory_mb = psutil.virtual_memory().used / 1e6
        
        # 记录
        self.history['timestamps'].append(datetime.now().isoformat())
        self.history['episodes'].append(episode)
        self.history['steps'].append(self.step)
        self.history['gpu_memory_allocated'].append(gpu_info['allocated_gb'])
        self.history['gpu_memory_reserved'].append(gpu_info['reserved_gb'])
        self.history['gpu_memory_utilization'].append(gpu_info['utilization_percent'])
        self.history['cpu_memory_usage'].append(cpu_memory_mb)
        self.history['cpu_percent'].append(cpu_percent)
        self.history['training_time'].append(elapsed_time)
        self.history['fps'].append(fps)
        
        # 定期保存
        if self.step % self.save_interval == 0:
            self.save_checkpoint()
            self.print_status(loss, reward)
    
    def print_status(self, loss: float = None, reward: float = None):
        """打印当前状态"""
        gpu_info = self.gpu_monitor.get_memory_usage()
        elapsed = time.time() - self.start_time
        fps = self.step / elapsed if elapsed > 0 else 0
        
        print(f"[Episode {self.episode:4d}] ", end="")
        print(f"GPU: {gpu_info['allocated_gb']:.1f}GB/{gpu_info['total_gb']:.1f}GB ({gpu_info['utilization_percent']:.0f}%) | ", end="")
        print(f"Speed: {fps:.1f} steps/s | ", end="")
        print(f"Time: {int(elapsed)}s")
        
        if loss is not None:
            print(f"  Loss: {loss:.4f} | ", end="")
        if reward is not None:
            print(f"Reward: {reward:.2f}")
        
    def save_checkpoint(self):
        """保存监测数据"""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.history, f)
    
    def get_summary(self) -> Dict:
        """获取汇总统计"""
        if not self.history['gpu_memory_allocated']:
            return {}
        
        gpu_mem_alloc = self.history['gpu_memory_allocated']
        gpu_mem_util = self.history['gpu_memory_utilization']
        fps_list = self.history['fps']
        time_total = self.history['training_time'][-1]
        
        return {
            'total_episodes': self.episode,
            'total_steps': self.step,
            'total_time_seconds': time_total,
            'total_time_hms': self._seconds_to_hms(time_total),
            'avg_fps': sum(fps_list) / len(fps_list) if fps_list else 0,
            'peak_gpu_memory_gb': max(gpu_mem_alloc),
            'avg_gpu_memory_gb': sum(gpu_mem_alloc) / len(gpu_mem_alloc),
            'max_gpu_utilization_percent': max(gpu_mem_util),
            'avg_gpu_utilization_percent': sum(gpu_mem_util) / len(gpu_mem_util),
        }
    
    def print_summary(self):
        """打印汇总统计"""
        summary = self.get_summary()
        if not summary:
            return
        
        print("\n" + "=" * 70)
        print("训练完成 - 性能汇总")
        print("=" * 70)
        print(f"总Epsiode数: {summary['total_episodes']}")
        print(f"总Step数: {summary['total_steps']}")
        print(f"总训练时间: {summary['total_time_hms']}")
        print(f"平均速度: {summary['avg_fps']:.1f} steps/sec")
        print()
        print("GPU显存统计:")
        print(f"  峰值占用: {summary['peak_gpu_memory_gb']:.2f} GB")
        print(f"  平均占用: {summary['avg_gpu_memory_gb']:.2f} GB")
        print(f"  峰值利用率: {summary['max_gpu_utilization_percent']:.1f}%")
        print(f"  平均利用率: {summary['avg_gpu_utilization_percent']:.1f}%")
        print("=" * 70 + "\n")
    
    @staticmethod
    def _seconds_to_hms(seconds: float) -> str:
        """将秒数转换为时:分:秒格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class PerformanceComparator:
    """性能对比工具"""
    
    def __init__(self):
        """初始化对比器"""
        self.experiments = {}
    
    def load_experiment(self, name: str, path: Path):
        """加载实验数据"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.experiments[name] = data
            print(f"✓ 加载: {name}")
        except Exception as e:
            print(f"✗ 失败: {name} - {e}")
    
    def compare_memory(self):
        """比较显存占用"""
        if len(self.experiments) < 2:
            print("需要至少2个实验进行对比")
            return
        
        print("\n" + "=" * 70)
        print("显存占用对比")
        print("=" * 70)
        print(f"{'实验名':<20} {'平均GB':<12} {'峰值GB':<12} {'平均利用率%':<15}")
        print("-" * 70)
        
        for name, data in self.experiments.items():
            gpu_mem = data.get('gpu_memory_allocated', [])
            gpu_util = data.get('gpu_memory_utilization', [])
            
            if gpu_mem and gpu_util:
                avg_mem = sum(gpu_mem) / len(gpu_mem)
                peak_mem = max(gpu_mem)
                avg_util = sum(gpu_util) / len(gpu_util)
                
                print(f"{name:<20} {avg_mem:<12.2f} {peak_mem:<12.2f} {avg_util:<15.1f}")
    
    def compare_speed(self):
        """比较训练速度"""
        if len(self.experiments) < 2:
            print("需要至少2个实验进行对比")
            return
        
        print("\n" + "=" * 70)
        print("训练速度对比")
        print("=" * 70)
        print(f"{'实验名':<20} {'平均速度':<15} {'总时间':<15}")
        print("-" * 70)
        
        for name, data in self.experiments.items():
            fps_list = data.get('fps', [])
            time_total = data['training_time'][-1] if data.get('training_time') else 0
            
            if fps_list:
                avg_fps = sum(fps_list) / len(fps_list)
                time_hms = self._seconds_to_hms(time_total)
                
                print(f"{name:<20} {avg_fps:<15.2f} {time_hms:<15}")
    
    def plot_memory_timeline(self):
        """绘制显存占用时间线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("需要matplotlib库: pip install matplotlib")
            return
        
        plt.figure(figsize=(12, 6))
        
        for name, data in self.experiments.items():
            episodes = data.get('episodes', [])
            gpu_mem = data.get('gpu_memory_allocated', [])
            
            if episodes and gpu_mem:
                plt.plot(episodes, gpu_mem, label=name, marker='o', markersize=2, alpha=0.7)
        
        plt.xlabel('Episode')
        plt.ylabel('GPU Memory (GB)')
        plt.title('GPU显存占用时间线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = Path("results/memory_timeline.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100)
        print(f"✓ 图表已保存: {save_path}")
        plt.close()
    
    def plot_utilization_timeline(self):
        """绘制利用率时间线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("需要matplotlib库: pip install matplotlib")
            return
        
        plt.figure(figsize=(12, 6))
        
        for name, data in self.experiments.items():
            episodes = data.get('episodes', [])
            gpu_util = data.get('gpu_memory_utilization', [])
            
            if episodes and gpu_util:
                plt.plot(episodes, gpu_util, label=name, marker='s', markersize=2, alpha=0.7)
        
        plt.axhline(y=80, color='orange', linestyle='--', label='警告线 (80%)')
        plt.axhline(y=95, color='red', linestyle='--', label='危险线 (95%)')
        
        plt.xlabel('Episode')
        plt.ylabel('GPU Utilization (%)')
        plt.title('GPU利用率时间线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 105])
        plt.tight_layout()
        
        save_path = Path("results/utilization_timeline.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100)
        print(f"✓ 图表已保存: {save_path}")
        plt.close()
    
    @staticmethod
    def _seconds_to_hms(seconds: float) -> str:
        """将秒数转换为时:分:秒格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def demo_monitor():
    """演示监测功能"""
    print("=" * 70)
    print("性能监测演示")
    print("=" * 70)
    print()
    
    monitor = PerformanceMonitor(name="demo_training")
    
    # 模拟训练过程
    print("模拟10个episode的训练...")
    for episode in range(1, 11):
        # 模拟一些工作
        dummy = torch.randn(1000, 1000).cuda()
        _ = torch.mm(dummy, dummy.t())
        
        # 模拟loss和reward
        loss = 1.0 / (episode + 0.1)
        reward = episode * 10
        
        monitor.update(episode, loss, reward)
        
        if episode % 5 == 0:
            print()
    
    monitor.print_summary()
    print(f"监测数据已保存到: {monitor.checkpoint_path}")


if __name__ == "__main__":
    # 演示
    demo_monitor()
    
    print("\n使用示例:")
    print("""
在训练代码中集成:

from tools.performance_monitor import PerformanceMonitor

# 初始化
monitor = PerformanceMonitor(name="lowmem_training")

# 在训练循环中
for episode in range(num_episodes):
    # ... 训练代码 ...
    loss = ...
    reward = ...
    
    monitor.update(episode, loss, reward)

# 训练完成
monitor.print_summary()
""")
