"""
Low Memory IPPO Training Script
针对RTX 4060 Ti等中小型显卡的优化版本

主要优化：
1. 混合精度训练 (mixed precision FP16)
2. 梯度累积 (gradient accumulation)
3. 批次大小动态调整
4. 显存监控和清理
5. CPU Fallback（显存不足时使用CPU）
"""

import os
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

from envs import EdgeComputingEnv
from agents import PPOAgent
from utils import load_config, set_seed, MetricsCollector, get_device
from utils.gpu_monitor import GPUMemoryMonitor, MemoryOptimizer, get_device_with_memory_info


class LowMemoryIPPOTrainer:
    """低显存IPPO训练器"""
    
    def __init__(self, config: dict, experiment_name: str = None):
        """
        初始化低显存训练器
        
        Args:
            config: 配置字典
            experiment_name: 实验名称
        """
        self.config = config
        
        # 设备和内存管理
        print("\n" + "="*80)
        print("INITIALIZING LOW MEMORY IPPO TRAINER")
        print("="*80)
        
        self.device = get_device_with_memory_info()
        self.gpu_monitor = GPUMemoryMonitor(device=self.device) if torch.cuda.is_available() else None
        
        # 显存优化设置
        self.use_mixed_precision = config['algorithm'].get('use_mixed_precision', True)
        self.gradient_accumulation_steps = config['algorithm'].get('gradient_accumulation_steps', 1)
        
        if self.use_mixed_precision:
            self.scaler = amp.GradScaler()
            print("✓ Mixed Precision Training Enabled (FP16)")
        else:
            self.scaler = None
        
        if self.gradient_accumulation_steps > 1:
            print(f"✓ Gradient Accumulation Enabled ({self.gradient_accumulation_steps} steps)")
        
        # 设置随机种子
        set_seed(config.get('seed', 42))
        
        # 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"IPPO_LowMem_{timestamp}"
        self.experiment_dir = Path("results") / self.experiment_name
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.log_dir = self.experiment_dir / "logs"
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n✓ Experiment Directory: {self.experiment_dir}")
        
        # 初始化环境
        print(f"\n初始化环境...")
        self.env = EdgeComputingEnv(config)
        print(f"✓ Environment initialized")
        print(f"  - End devices: {self.env.num_end_devices}")
        print(f"  - Edge servers: {self.env.num_edge_servers}")
        
        # 清空显存
        if self.gpu_monitor:
            self.gpu_monitor.clear_cache()
        
        # 初始化智能体
        print(f"\n初始化智能体...")
        self.num_agents = config['marl']['num_agents']
        self.agents = []
        
        for i in range(self.num_agents):
            try:
                agent = PPOAgent(
                    agent_id=i,
                    state_dim=config['environment']['state_dim'],
                    action_dim=self.env.action_space.n,
                    hidden_dim=config['algorithm']['hidden_dim'],
                    learning_rate=config['algorithm']['learning_rate'],
                    gamma=config['algorithm']['gamma'],
                    gae_lambda=config['algorithm']['gae_lambda'],
                    clip_ratio=config['algorithm']['clip_ratio'],
                    entropy_coeff=config['algorithm']['entropy_coeff'],
                    value_coeff=config['algorithm']['value_coeff'],
                    max_grad_norm=config['algorithm']['max_grad_norm'],
                    device=self.device
                )
                self.agents.append(agent)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"✗ CUDA out of memory! Falling back to CPU...")
                    self.device = torch.device('cpu')
                    # 重试
                    agent = PPOAgent(
                        agent_id=i,
                        state_dim=config['environment']['state_dim'],
                        action_dim=self.env.action_space.n,
                        hidden_dim=config['algorithm']['hidden_dim'],
                        learning_rate=config['algorithm']['learning_rate'],
                        gamma=config['algorithm']['gamma'],
                        gae_lambda=config['algorithm']['gae_lambda'],
                        clip_ratio=config['algorithm']['clip_ratio'],
                        entropy_coeff=config['algorithm']['entropy_coeff'],
                        value_coeff=config['algorithm']['value_coeff'],
                        max_grad_norm=config['algorithm']['max_grad_norm'],
                        device=self.device
                    )
                    self.agents.append(agent)
                else:
                    raise
        
        print(f"✓ Created {self.num_agents} agents")
        
        # 显示内存使用
        if self.gpu_monitor:
            self.gpu_monitor.print_memory_usage("After agent initialization")
        
        # 初始化指标收集器
        self.metrics_collector = MetricsCollector()
        
        # TensorBoard日志
        self.writer = SummaryWriter(str(self.log_dir))
        
        # 训练参数
        self.total_episodes = config['marl']['total_episodes']
        self.episode_length = config['marl']['episode_length']
        self.batch_size = config['algorithm']['batch_size']
        self.num_epochs = config['algorithm']['num_epochs']
        self.save_interval = config['evaluation']['save_interval']
        
        # 显存监控参数
        self.monitor_memory = config['logging'].get('monitor_memory', True)
        self.memory_check_interval = config['logging'].get('memory_check_interval', 10)
        
        # 保存配置
        with open(self.experiment_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        print(f"\n✓ Trainer initialized successfully!")
        print(f"  - Device: {self.device}")
        print(f"  - Mixed Precision: {self.use_mixed_precision}")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Hidden Dim: {config['algorithm']['hidden_dim']}")
        print(f"  - Num Epochs: {self.num_epochs}")
    
    def train_episode(self) -> float:
        """训练一个episode"""
        obs, _ = self.env.reset()
        episode_reward = 0.0
        
        for step in range(self.episode_length):
            # 获取动作
            actions = []
            for agent in self.agents:
                action, _, _ = agent.select_action(obs)
                actions.append(action)
            
            # 环境步骤
            obs, reward, terminated, truncated, info = self.env.step(actions[0])
            episode_reward += reward
            
            # 存储转移
            for i, agent in enumerate(self.agents):
                agent.store_transition(
                    state=obs,
                    action=actions[i],
                    reward=reward,
                    value=0.0,  # 简化版本
                    log_prob=0.0,
                    done=terminated
                )
            
            if terminated:
                break
        
        # 梯度累积更新
        for agent_idx, agent in enumerate(self.agents):
            if agent.trajectory['states']:
                # 使用梯度累积
                accumulated_loss = 0.0
                for accum_step in range(self.gradient_accumulation_steps):
                    # 更新只用一部分数据（梯度累积）
                    effective_batch = max(1, self.batch_size // self.gradient_accumulation_steps)
                    
                    # 使用混合精度
                    if self.use_mixed_precision:
                        with amp.autocast(dtype=torch.float16):
                            losses = agent.update(
                                batch_size=effective_batch,
                                num_epochs=1  # 每步1个epoch节省显存
                            )
                    else:
                        losses = agent.update(
                            batch_size=effective_batch,
                            num_epochs=1
                        )
                    
                    if losses:
                        accumulated_loss += losses.get('total_loss', 0)
                
                # 记录日志
                avg_loss = accumulated_loss / self.gradient_accumulation_steps
                self.writer.add_scalar(
                    f'agent_{agent_idx}/loss/total',
                    avg_loss,
                    self.episode_count
                )
        
        # 显存清理
        if self.gpu_monitor:
            self.gpu_monitor.clear_cache()
        
        return episode_reward / len(self.agents)
    
    def evaluate(self, num_episodes: int = 5) -> dict:
        """评估（简化版本）"""
        all_metrics = {
            'task_completion_rate': [],
            'energy_consumption': [],
            'average_delay': [],
            'deadline_miss_rate': [],
            'fairness_index': []
        }
        
        with torch.no_grad():
            for _ in range(num_episodes):
                obs, _ = self.env.reset()
                
                for step in range(self.episode_length):
                    actions = []
                    for agent in self.agents:
                        action, _, _ = agent.select_action(obs)
                        actions.append(action)
                    
                    obs, _, terminated, _, _ = self.env.step(actions[0])
                    
                    if terminated:
                        break
                
                metrics = self.env.get_metrics()
                for key, value in metrics.items():
                    if key in all_metrics:
                        all_metrics[key].append(value)
        
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        return avg_metrics
    
    def train(self):
        """主训练循环"""
        print("\n" + "="*80)
        print(f"Starting Training - {self.experiment_name}")
        print("="*80)
        print(f"Total Episodes: {self.total_episodes}")
        print(f"Device: {self.device}")
        
        self.episode_count = 0
        
        pbar = tqdm(total=self.total_episodes, desc="Training", unit="episode")
        
        try:
            for episode in range(self.total_episodes):
                self.episode_count = episode
                
                try:
                    # 训练
                    avg_reward = self.train_episode()
                    
                    # 日志
                    self.writer.add_scalar('training/episode_reward', avg_reward, episode)
                    
                    # 定期显示内存和评估
                    if (episode + 1) % self.memory_check_interval == 0:
                        if self.gpu_monitor:
                            self.gpu_monitor.print_memory_usage(f"Episode {episode + 1}")
                    
                    if (episode + 1) % self.save_interval == 0:
                        eval_metrics = self.evaluate(num_episodes=3)  # 少做评估节省时间
                        
                        for key, value in eval_metrics.items():
                            self.writer.add_scalar(f'evaluation/{key}', value, episode)
                        
                        self.save_checkpoint(episode)
                        
                        print(f"\nEpisode {episode + 1}/{self.total_episodes} | "
                              f"Reward: {avg_reward:.4f} | "
                              f"Completion: {eval_metrics['task_completion_rate']:.3f}")
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n✗ Out of Memory at episode {episode}!")
                        print(f"  Attempting recovery...")
                        if self.gpu_monitor:
                            self.gpu_monitor.clear_cache()
                        torch.cuda.empty_cache()
                        
                        # 减小批次大小重试
                        self.batch_size = max(2, self.batch_size // 2)
                        print(f"  Reduced batch size to {self.batch_size}")
                        continue
                    else:
                        raise
                
                pbar.update(1)
        
        finally:
            pbar.close()
            self.writer.close()
        
        # 最终评估和保存
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        
        final_metrics = self.evaluate(num_episodes=5)
        self.save_checkpoint(self.total_episodes - 1, is_final=True)
        
        print("\nFinal Metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.6f}")
        
        # 保存结果
        self.save_results(final_metrics)
    
    def save_checkpoint(self, episode: int, is_final: bool = False):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / f"episode_{episode:06d}.pt"
        
        state_dict = {
            'episode': episode,
            'agents': [agent.network.state_dict() for agent in self.agents],
            'config': self.config
        }
        
        torch.save(state_dict, checkpoint_path)
        
        if is_final:
            torch.save(state_dict, self.checkpoint_dir / "final_model.pt")
            print(f"\n✓ Final model saved to {self.checkpoint_dir / 'final_model.pt'}")
    
    def save_results(self, metrics: dict):
        """保存结果"""
        results_file = self.experiment_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        results_txt = self.experiment_dir / "results.txt"
        with open(results_txt, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Low Memory IPPO Results\n")
            f.write("="*80 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.6f}\n")
        
        print(f"\n✓ Results saved to {results_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Low Memory IPPO Training")
    parser.add_argument('--config', type=str, default='configs/lowmem_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖命令行参数
    if args.episodes:
        config['marl']['total_episodes'] = args.episodes
    
    # 创建并运行训练器
    trainer = LowMemoryIPPOTrainer(config, experiment_name=args.name)
    trainer.train()


if __name__ == '__main__':
    main()
