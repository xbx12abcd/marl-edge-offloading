"""
Comparison evaluation script for IPPO vs Explaboff.
Stage 2: Mutual Information Analysis and Performance Comparison
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import torch

from utils import load_config, set_seed, get_device
from envs import EdgeComputingEnv
from agents import PPOAgent
from agents.explaboff_agent import ExplaboffAgent


def evaluate_algorithm(
    algorithm_name: str,
    config: dict,
    num_episodes: int = 20,
    comm_enabled: bool = False
) -> dict:
    """
    Evaluate an algorithm (IPPO or Explaboff).
    
    Args:
        algorithm_name: 'IPPO' or 'Explaboff' 
        config: Configuration dictionary
        num_episodes: Number of evaluation episodes
        comm_enabled: Whether communication is enabled
    
    Returns:
        Dictionary of evaluation metrics
    """
    
    print(f"\n{'='*80}")
    print(f"Evaluating {algorithm_name} - {'with' if comm_enabled else 'without'} Communication")
    print(f"{'='*80}\n")
    
    set_seed(config['seed'])
    device = get_device()
    
    # Create environment
    env = EdgeComputingEnv(config)
    
    # Create agents
    num_agents = config['marl']['num_agents']
    agents = []
    
    for i in range(num_agents):
        if algorithm_name == 'Explaboff':
            agent = ExplaboffAgent(
                agent_id=i,
                state_dim=config['environment']['state_dim'],
                action_dim=env.action_space.n,
                hidden_dim=config['algorithm']['hidden_dim'],
                learning_rate=config['algorithm']['learning_rate'],
                mi_weight=config['algorithm'].get('mi_weight', 0.5),
                comm_enabled=comm_enabled,
                device=device
            )
        else:  # IPPO
            agent = PPOAgent(
                agent_id=i,
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
        agents.append(agent)
    
    # Evaluation metrics
    metrics = {
        'task_completion_rates': [],
        'energy_consumptions': [],
        'average_delays': [],
        'deadline_miss_rates': [],
        'fairness_indices': [],
        'episode_rewards': []
    }
    
    if comm_enabled:
        metrics['mutual_information'] = []
    
    # Run evaluation episodes
    for episode in tqdm(range(num_episodes), desc=f"{algorithm_name} Evaluation"):
        obs, _ = env.reset()
        episode_reward = 0.0
        
        for step in range(config['marl']['episode_length']):
            # Get actions from all agents
            actions = []
            all_agent_states = [obs] * num_agents  # Simplified: same obs for all
            
            for agent_idx, agent in enumerate(agents):
                if isinstance(agent, ExplaboffAgent) and comm_enabled:
                    action, _, _, mi_reward = agent.select_action_with_communication(
                        obs, all_agent_states, top_k=3
                    )
                    metrics['mutual_information'].append(mi_reward)
                else:
                    action, _, _ = agent.select_action(obs)
                
                actions.append(action)
            
            # Environment step
            obs, reward, terminated, truncated, info = env.step(actions[0])
            episode_reward += reward
            
            if terminated:
                break
        
        # Collect metrics
        env_metrics = env.get_metrics()
        metrics['task_completion_rates'].append(env_metrics['task_completion_rate'])
        metrics['energy_consumptions'].append(env_metrics['energy_consumption'])
        metrics['average_delays'].append(env_metrics['average_delay'])
        metrics['deadline_miss_rates'].append(env_metrics['deadline_miss_rate'])
        metrics['fairness_indices'].append(env_metrics['fairness_index'])
        metrics['episode_rewards'].append(episode_reward)
    
    # Aggregate metrics
    results = {
        'algorithm': algorithm_name,
        'communication': comm_enabled,
        'task_completion_rate': {
            'mean': float(np.mean(metrics['task_completion_rates'])),
            'std': float(np.std(metrics['task_completion_rates']))
        },
        'energy_consumption': {
            'mean': float(np.mean(metrics['energy_consumptions'])),
            'std': float(np.std(metrics['energy_consumptions']))
        },
        'average_delay': {
            'mean': float(np.mean(metrics['average_delays'])),
            'std': float(np.std(metrics['average_delays']))
        },
        'deadline_miss_rate': {
            'mean': float(np.mean(metrics['deadline_miss_rates'])),
            'std': float(np.std(metrics['deadline_miss_rates']))
        },
        'fairness_index': {
            'mean': float(np.mean(metrics['fairness_indices'])),
            'std': float(np.std(metrics['fairness_indices']))
        },
        'episode_reward': {
            'mean': float(np.mean(metrics['episode_rewards'])),
            'std': float(np.std(metrics['episode_rewards']))
        }
    }
    
    if comm_enabled and metrics['mutual_information']:
        results['mutual_information'] = {
            'mean': float(np.mean(metrics['mutual_information'])),
            'std': float(np.std(metrics['mutual_information']))
        }
    
    return results, metrics


def print_comparison_results(ippo_results: dict, explaboff_results: dict):
    """Print comparison results."""
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: IPPO vs Explaboff")
    print("="*80)
    
    # Metrics to compare
    metrics_to_show = [
        ('task_completion_rate', 'Task Completion Rate', 'higher'),
        ('energy_consumption', 'Energy Consumption', 'lower'),
        ('average_delay', 'Average Delay', 'lower'),
        ('deadline_miss_rate', 'Deadline Miss Rate', 'lower'),
        ('fairness_index', 'Fairness Index', 'higher'),
        ('episode_reward', 'Episode Reward', 'higher')
    ]
    
    print(f"\n{'Metric':<30} {'IPPO':>20} {'Explaboff':>20} {'Improvement':>15}")
    print("-" * 85)
    
    for metric_key, metric_name, direction in metrics_to_show:
        ippo_mean = ippo_results[metric_key]['mean']
        explaboff_mean = explaboff_results[metric_key]['mean']
        
        if direction == 'higher':
            improvement = ((explaboff_mean - ippo_mean) / (abs(ippo_mean) + 1e-8)) * 100
            better = "✓" if improvement > 0 else "✗"
        else:
            improvement = ((ippo_mean - explaboff_mean) / (abs(ippo_mean) + 1e-8)) * 100
            better = "✓" if improvement > 0 else "✗"
        
        print(f"{metric_name:<30} {ippo_mean:>20.6f} {explaboff_mean:>20.6f} {better} {improvement:>12.2f}%")
    
    # Mutual information
    if 'mutual_information' in explaboff_results:
        print(f"\n{'Metric':<30} {'Value':>20}")
        print("-" * 50)
        print(f"{'Mutual Information (Explaboff)':<30} {explaboff_results['mutual_information']['mean']:>20.6f}")
    
    print("\n" + "="*80)


def plot_comparison(ippo_metrics: dict, explaboff_metrics: dict, output_dir: Path):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('IPPO vs Explaboff Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Task Completion Rate
    axes[0, 0].plot(ippo_metrics['task_completion_rates'], label='IPPO', alpha=0.7)
    axes[0, 0].plot(explaboff_metrics['task_completion_rates'], label='Explaboff', alpha=0.7)
    axes[0, 0].set_title('Task Completion Rate')
    axes[0, 0].set_ylabel('Rate')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Energy Consumption
    axes[0, 1].plot(ippo_metrics['energy_consumptions'], label='IPPO', alpha=0.7)
    axes[0, 1].plot(explaboff_metrics['energy_consumptions'], label='Explaboff', alpha=0.7)
    axes[0, 1].set_title('Energy Consumption')
    axes[0, 1].set_ylabel('Energy (J)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average Delay
    axes[0, 2].plot(ippo_metrics['average_delays'], label='IPPO', alpha=0.7)
    axes[0, 2].plot(explaboff_metrics['average_delays'], label='Explaboff', alpha=0.7)
    axes[0, 2].set_title('Average Task Delay')
    axes[0, 2].set_ylabel('Delay (time slots)')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Deadline Miss Rate
    axes[1, 0].plot(ippo_metrics['deadline_miss_rates'], label='IPPO', alpha=0.7)
    axes[1, 0].plot(explaboff_metrics['deadline_miss_rates'], label='Explaboff', alpha=0.7)
    axes[1, 0].set_title('Deadline Miss Rate')
    axes[1, 0].set_ylabel('Miss Rate')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Fairness Index
    axes[1, 1].plot(ippo_metrics['fairness_indices'], label='IPPO', alpha=0.7)
    axes[1, 1].plot(explaboff_metrics['fairness_indices'], label='Explaboff', alpha=0.7)
    axes[1, 1].set_title('Fairness Index')
    axes[1, 1].set_ylabel('Index')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Episode Reward
    axes[1, 2].plot(ippo_metrics['episode_rewards'], label='IPPO', alpha=0.7)
    axes[1, 2].plot(explaboff_metrics['episode_rewards'], label='Explaboff', alpha=0.7)
    axes[1, 2].set_title('Episode Reward')
    axes[1, 2].set_ylabel('Reward')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_plots.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plots saved to: {output_dir}/comparison_plots.png")


def main():
    """Run comparison evaluation."""
    
    # Load config
    config = load_config('configs/default_config.yaml')
    
    # Create output directory
    output_dir = Path('results/comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate IPPO
    ippo_results, ippo_metrics = evaluate_algorithm(
        'IPPO',
        config,
        num_episodes=20,
        comm_enabled=False
    )
    
    # Evaluate Explaboff
    explaboff_results, explaboff_metrics = evaluate_algorithm(
        'Explaboff',
        config,
        num_episodes=20,
        comm_enabled=True
    )
    
    # Print comparison
    print_comparison_results(ippo_results, explaboff_results)
    
    # Create plots
    try:
        plot_comparison(ippo_metrics, explaboff_metrics, output_dir)
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
    
    # Save results
    results = {
        'ippo': ippo_results,
        'explaboff': explaboff_results
    }
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to: {output_dir}/comparison_results.json")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
