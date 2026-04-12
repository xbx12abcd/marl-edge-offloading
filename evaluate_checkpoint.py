"""
Evaluation script for trained IPPO models.
加载checkpoint文件并评估模型性能或进行推演演示。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm

from utils import load_config, set_seed
from envs import EdgeComputingEnv
from agents import PPOAgent
from utils.gpu_monitor import get_device_with_memory_info


def load_checkpoint(agent: PPOAgent, checkpoint_path: str):
    """加载checkpoint到agent"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # checkpoint格式: {'episode': int, 'agents': [state_dict, ...], 'config': dict}
    if 'agents' in checkpoint and isinstance(checkpoint['agents'], list):
        # 使用第一个agent的网络状态
        agent.network.load_state_dict(checkpoint['agents'][0])
        print(f"✓ Loaded checkpoint: {checkpoint_path} (episode {checkpoint.get('episode', 'unknown')})")
    elif 'network_state_dict' in checkpoint:
        agent.network.load_state_dict(checkpoint['network_state_dict'])
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    elif 'model_state_dict' in checkpoint:
        agent.network.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        # 尝试直接加载（兼容旧格式）
        try:
            agent.network.load_state_dict(checkpoint)
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
        except:
            print(f"✗ Unknown checkpoint format in {checkpoint_path}")
            print(f"Available keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'not a dict'}")
            raise

    agent.network.eval()
def evaluate_model(
    checkpoint_path: str,
    config_path: str = None,
    num_episodes: int = 10,
    render: bool = False,
    save_results: bool = True
):
    """
    评估训练好的模型

    Args:
        checkpoint_path: checkpoint文件路径
        config_path: 配置文件路径（可选，会从实验目录自动查找）
        num_episodes: 评估的episode数量
        render: 是否显示推演过程
        save_results: 是否保存结果
    """

    print("=" * 80)
    print("IPPO Model Evaluation")
    print("=" * 80)

    # 确定配置文件路径
    if config_path is None:
        # 从checkpoint路径推断实验目录
        checkpoint_path = Path(checkpoint_path)
        exp_dir = checkpoint_path.parent.parent
        config_path = exp_dir / "config.yaml"

    if not Path(config_path).exists():
        print(f"✗ Config file not found: {config_path}")
        return

    # 加载配置
    config = load_config(str(config_path))
    set_seed(config['seed'])

    # 获取设备
    device = get_device_with_memory_info()

    # 创建环境
    print("\nInitializing environment...")
    env = EdgeComputingEnv(config)

    # 创建agent
    agent = PPOAgent(
        agent_id=0,
        state_dim=config['environment']['state_dim'],
        action_dim=env.action_space.n,
        hidden_dim=config['algorithm']['hidden_dim'],
        learning_rate=config['algorithm']['learning_rate'],
        gamma=config['algorithm']['gamma'],
        gae_lambda=config['algorithm']['gae_lambda'],
        clip_ratio=config['algorithm']['clip_ratio'],
        entropy_coeff=config['algorithm']['entropy_coeff'],
        value_coeff=config['algorithm']['value_coeff'],
        max_grad_norm=config['algorithm']['max_grad_norm'],
        device=device
    )

    # 加载checkpoint
    load_checkpoint(agent, str(checkpoint_path))

    # 评估循环
    print(f"\nEvaluating {num_episodes} episodes...")
    all_metrics = {
        'task_completion_rate': [],
        'average_energy_consumption': [],
        'average_task_delay': [],
        'total_reward': []
    }

    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        step = 0

        while not done and step < config['marl']['episode_length']:
            # 选择动作
            action, _, _ = agent.select_action(obs)

            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            if render:
                print(f"Step {step}: Action={action}, Reward={reward:.4f}, Done={done}")

            obs = next_obs
            step += 1

        # 收集指标
        metrics = env.get_metrics()
        for key in all_metrics.keys():
            if key in metrics:
                all_metrics[key].append(metrics[key])
        all_metrics['total_reward'].append(episode_reward)

    # 计算平均指标
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:
            avg_metrics[key] = np.mean(values)
        else:
            avg_metrics[key] = 0.0

    # 打印结果
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.6f}")

    # 保存结果
    if save_results:
        results_dir = Path(checkpoint_path).parent.parent / "evaluation"
        results_dir.mkdir(exist_ok=True)

        checkpoint_name = Path(checkpoint_path).stem
        results_file = results_dir / f"{checkpoint_name}_evaluation.json"

        with open(results_file, 'w') as f:
            json.dump({
                'checkpoint': str(checkpoint_path),
                'num_episodes': num_episodes,
                'metrics': avg_metrics,
                'individual_episodes': all_metrics
            }, f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")

    return avg_metrics


def demonstrate_model(
    checkpoint_path: str,
    config_path: str = None,
    num_steps: int = 100,
    delay: float = 0.5
):
    """
    演示模型推演过程

    Args:
        checkpoint_path: checkpoint文件路径
        config_path: 配置文件路径
        num_steps: 演示的步数
        delay: 每步之间的延迟（秒）
    """

    print("=" * 80)
    print("IPPO Model Demonstration")
    print("=" * 80)

    # 确定配置文件路径
    if config_path is None:
        checkpoint_path = Path(checkpoint_path)
        exp_dir = checkpoint_path.parent.parent
        config_path = exp_dir / "config.yaml"

    # 加载配置和模型
    config = load_config(str(config_path))
    set_seed(config['seed'])
    device = get_device_with_memory_info()

    env = EdgeComputingEnv(config)

    agent = PPOAgent(
        agent_id=0,
        state_dim=config['environment']['state_dim'],
        action_dim=env.action_space.n,
        hidden_dim=config['algorithm']['hidden_dim'],
        learning_rate=config['algorithm']['learning_rate'],
        gamma=config['algorithm']['gamma'],
        gae_lambda=config['algorithm']['gae_lambda'],
        clip_ratio=config['algorithm']['clip_ratio'],
        entropy_coeff=config['algorithm']['entropy_coeff'],
        value_coeff=config['algorithm']['value_coeff'],
        max_grad_norm=config['algorithm']['max_grad_norm'],
        device=device
    )

    load_checkpoint(agent, str(checkpoint_path))

    # 演示
    print("\nStarting demonstration...")
    obs, _ = env.reset()
    total_reward = 0.0

    for step in range(num_steps):
        # 选择动作
        action, log_prob, value = agent.select_action(obs)

        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 显示信息
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Done: {terminated or truncated}")

        # 显示环境状态
        metrics = env.get_metrics()
        print(f"  Task Completion: {metrics.get('task_completion_rate', 0):.3f}")
        print(f"  Energy Consumption: {metrics.get('average_energy_consumption', 0):.3f}")

        if terminated or truncated:
            print("\nEpisode finished!")
            break

        obs = next_obs

        if delay > 0:
            import time
            time.sleep(delay)

    print(f"\nFinal Total Reward: {total_reward:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained IPPO models")
    parser.add_argument("checkpoint", help="Path to checkpoint file (.pt)")
    parser.add_argument("--config", help="Path to config file (optional)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render demonstration")
    parser.add_argument("--demo", action="store_true", help="Run demonstration mode")
    parser.add_argument("--steps", type=int, default=50, help="Number of demonstration steps")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between steps in demo")

    args = parser.parse_args()

    if args.demo:
        demonstrate_model(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            num_steps=args.steps,
            delay=args.delay
        )
    else:
        evaluate_model(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            num_episodes=args.episodes,
            render=args.render
        )


if __name__ == "__main__":
    main()