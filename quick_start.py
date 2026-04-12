"""
Quick start script for IPPO baseline training.
Minimal configuration for fast testing.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm
import json
from pathlib import Path

from utils import load_config, set_seed, get_device
from envs import EdgeComputingEnv
from agents import PPOAgent


def quick_train(num_episodes: int = 50, episode_length: int = 50):
    """Quick training for testing baseline IPPO."""
    
    print("=" * 80)
    print("IPPO Baseline - Quick Training Test")
    print("=" * 80)
    
    # Load config
    config = load_config('configs/default_config.yaml')
    config['marl']['total_episodes'] = num_episodes
    config['marl']['episode_length'] = episode_length
    
    set_seed(config['seed'])
    device = get_device()
    
    # Create environment
    print("\nInitializing environment...")
    env = EdgeComputingEnv(config)
    
    # Create agents
    num_agents = config['marl']['num_agents']
    agents = [
        PPOAgent(
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
        for i in range(num_agents)
    ]
    
    print(f"✓ Created {num_agents} agents on device: {device}")
    
    # Training loop
    print("\nStarting training...")
    episode_rewards = []
    
    for episode in tqdm(range(num_episodes), desc="Episodes"):
        obs, _ = env.reset()
        episode_reward = 0.0
        
        for step in range(episode_length):
            # Get actions from all agents
            actions = []
            for agent in agents:
                action, _, _ = agent.select_action(obs)
                actions.append(action)
            
            # Take environment step (use first agent's action)
            obs, reward, terminated, truncated, info = env.step(actions[0])
            episode_reward += reward
            
            # Store transitions
            for i, agent in enumerate(agents):
                state_for_action = np.zeros_like(obs)  # Simplified for testing
                agent.store_transition(obs, actions[i], reward, 0.0, 0.0, terminated)
            
            if terminated:
                break
        
        # Update agents
        for agent in agents:
            if agent.trajectory['states']:
                agent.update(
                    batch_size=config['algorithm']['batch_size'],
                    num_epochs=config['algorithm']['num_epochs']
                )
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} - Avg Reward: {avg_reward:.4f}")
    
    # Results
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    results = {
        'total_episodes': num_episodes,
        'episode_length': episode_length,
        'average_reward': float(np.mean(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'final_10_avg': float(np.mean(episode_rewards[-10:]))
    }
    
    print("\nTraining Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save results
    results_dir = Path("results/quick_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to: {results_dir / 'results.json'}")
    
    # Get final metrics
    print("\nFinal Environment Metrics:")
    metrics = env.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick IPPO training")
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--length', type=int, default=50, help='Episode length')
    
    args = parser.parse_args()
    
    try:
        quick_train(num_episodes=args.episodes, episode_length=args.length)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
