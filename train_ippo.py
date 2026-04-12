"""
IPPO Training Script - Independent PPO for Multi-Agent Edge Offloading
Baseline implementation without inter-agent communication.

This implements Stage 1 (30%) of the project: Build a simulation field and 
use IPPO algorithm as baseline without any communications.
"""

import os
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from envs import EdgeComputingEnv
from agents import PPOAgent
from utils import load_config, set_seed, MetricsCollector, get_device


class IPPOTrainer:
    """IPPO Trainer for multi-agent edge offloading."""
    
    def __init__(self, config: dict, experiment_name: str = None):
        """
        Initialize IPPO trainer.
        
        Args:
            config: Configuration dictionary
            experiment_name: Name for the experiment
        """
        self.config = config
        self.device = get_device()
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"IPPO_{timestamp}"
        self.experiment_dir = Path("results") / self.experiment_name
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.log_dir = self.experiment_dir / "logs"
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment
        self.env = EdgeComputingEnv(config)
        
        # Initialize agents
        self.num_agents = config['marl']['num_agents']
        self.agents = []
        for i in range(self.num_agents):
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
        
        # Initialize metrics
        self.metrics_collector = MetricsCollector()
        
        # TensorBoard writer
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Training parameters
        self.total_episodes = config['marl']['total_episodes']
        self.episode_length = config['marl']['episode_length']
        self.batch_size = config['algorithm']['batch_size']
        self.num_epochs = config['algorithm']['num_epochs']
        self.save_interval = config['evaluation']['save_interval']
        
        # Save config
        with open(self.experiment_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f)
    
    def train_episode(self) -> float:
        """
        Train one episode.
        
        Returns:
            Average reward for the episode
        """
        obs, _ = self.env.reset()
        episode_reward = 0.0
        agent_rewards = [0.0] * self.num_agents
        
        for step in range(self.episode_length):
            # Select actions for each agent
            actions = []
            values = []
            log_probs = []
            
            for agent_idx, agent in enumerate(self.agents):
                action, log_prob, value = agent.select_action(obs)
                actions.append(action)
                values.append(value)
                log_probs.append(log_prob)
            
            # Environment step (using first agent's action for now)
            next_obs, reward, terminated, truncated, info = self.env.step(actions[0])
            
            # Store transitions for each agent
            for agent_idx, agent in enumerate(self.agents):
                # Each agent gets the same reward (global reward) for simplicity
                agent.store_transition(
                    state=obs,
                    action=actions[agent_idx],
                    reward=reward,
                    value=values[agent_idx],
                    log_prob=log_probs[agent_idx],
                    done=terminated
                )
                agent_rewards[agent_idx] += reward
            
            episode_reward += reward
            obs = next_obs
            
            if terminated:
                break
        
        # Update agents
        for agent_idx, agent in enumerate(self.agents):
            losses = agent.update(batch_size=self.batch_size, num_epochs=self.num_epochs)
            if losses:
                self.writer.add_scalar(f'agent_{agent_idx}/loss/total', losses['total_loss'], self.episode_count)
                self.writer.add_scalar(f'agent_{agent_idx}/loss/policy', losses['policy_loss'], self.episode_count)
                self.writer.add_scalar(f'agent_{agent_idx}/loss/value', losses['value_loss'], self.episode_count)
        
        return episode_reward / len(self.agents)
    
    def evaluate(self, num_episodes: int = 10) -> dict:
        """
        Evaluate trained agents.
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation metrics
        """
        all_metrics = {
            'task_completion_rate': [],
            'energy_consumption': [],
            'average_delay': [],
            'deadline_miss_rate': [],
            'fairness_index': []
        }
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            
            for step in range(self.episode_length):
                # Select actions using greedy policy
                actions = []
                with torch.no_grad():
                    for agent in self.agents:
                        state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                        action_probs, _ = agent.network(state_tensor)
                        action = torch.argmax(action_probs, dim=-1).item()
                        actions.append(action)
                
                # Environment step
                next_obs, _, terminated, _, _ = self.env.step(actions[0])
                obs = next_obs
                
                if terminated:
                    break
            
            # Collect metrics
            metrics = self.env.get_metrics()
            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        return avg_metrics
    
    def train(self):
        """Run training loop."""
        print("=" * 80)
        print(f"IPPO Training - Experiment: {self.experiment_name}")
        print("=" * 80)
        print(f"Total Episodes: {self.total_episodes}")
        print(f"Episode Length: {self.episode_length}")
        print(f"Number of Agents: {self.num_agents}")
        print(f"Device: {self.device}")
        print("=" * 80)
        
        self.episode_count = 0
        best_eval_reward = float('-inf')
        
        # Training loop
        pbar = tqdm(total=self.total_episodes, desc="Training")
        
        for episode in range(self.total_episodes):
            self.episode_count = episode
            
            # Train one episode
            avg_reward = self.train_episode()
            
            # Log training metrics
            self.writer.add_scalar('training/episode_reward', avg_reward, episode)
            
            # Periodic evaluation and checkpoint
            if (episode + 1) % self.save_interval == 0:
                eval_metrics = self.evaluate(num_episodes=10)
                
                # Log evaluation metrics
                for key, value in eval_metrics.items():
                    self.writer.add_scalar(f'evaluation/{key}', value, episode)
                
                # Save checkpoint
                self.save_checkpoint(episode)
                
                # Print progress
                print(f"\nEpisode {episode + 1}/{self.total_episodes}")
                print(f"  Avg Training Reward: {avg_reward:.4f}")
                print(f"  Task Completion Rate: {eval_metrics['task_completion_rate']:.4f}")
                print(f"  Energy Consumption: {eval_metrics['energy_consumption']:.4f}")
                print(f"  Average Delay: {eval_metrics['average_delay']:.4f}")
                print(f"  Fairness Index: {eval_metrics['fairness_index']:.4f}")
            
            pbar.update(1)
        
        pbar.close()
        
        # Final evaluation
        print("\n" + "=" * 80)
        print("Final Evaluation")
        print("=" * 80)
        final_metrics = self.evaluate(num_episodes=100)
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save final model and results
        self.save_checkpoint(self.total_episodes - 1, is_final=True)
        self.save_results(final_metrics)
        
        self.writer.close()
    
    def save_checkpoint(self, episode: int, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"episode_{episode:06d}.pt"
        
        state_dict = {
            'episode': episode,
            'agents': [agent.network.state_dict() for agent in self.agents],
            'config': self.config
        }
        
        torch.save(state_dict, checkpoint_path)
        
        if is_final:
            torch.save(state_dict, self.checkpoint_dir / "final_model.pt")
    
    def save_results(self, metrics: dict):
        """Save final results to file."""
        results_file = self.experiment_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Also save as text
        results_txt = self.experiment_dir / "results.txt"
        with open(results_txt, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"IPPO Baseline Results - {self.experiment_name}\n")
            f.write("=" * 80 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.6f}\n")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="IPPO Training for Edge Offloading")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.episodes:
        config['marl']['total_episodes'] = args.episodes
    
    # Create and run trainer
    trainer = IPPOTrainer(config, experiment_name=args.name)
    trainer.train()


if __name__ == '__main__':
    main()
