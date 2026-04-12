"""
Quick test script to verify environment setup and basic functionality.
Tests Stage 1: Simulation field with IPPO baseline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from utils import load_config, set_seed
from envs import EdgeComputingEnv
from agents import PPOAgent

def test_environment():
    """Test basic environment functionality."""
    print("=" * 80)
    print("Testing Edge Computing Environment")
    print("=" * 80)
    
    # Load configuration
    config = load_config('configs/default_config.yaml')
    set_seed(config['seed'])
    
    # Create environment
    env = EdgeComputingEnv(config)
    print(f"✓ Environment created successfully")
    print(f"  - End devices: {env.num_end_devices}")
    print(f"  - Edge servers: {env.num_edge_servers}")
    print(f"  - Action space: {env.action_space.n}")
    print(f"  - Observation space: {env.observation_space.shape}")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Observation range: [{obs.min():.4f}, {obs.max():.4f}]")
    
    # Test a few steps
    print(f"\n✓ Running 10 test steps:")
    cumulative_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        if step % 5 == 0:
            print(f"  - Step {step}: Reward={reward:.4f}, Completed tasks={info['completed_tasks']}")
    
    print(f"\n✓ Environment test passed!")
    print(f"  - Cumulative reward: {cumulative_reward:.4f}")
    
    return env


def test_agent():
    """Test PPO agent functionality."""
    print("\n" + "=" * 80)
    print("Testing PPO Agent")
    print("=" * 80)
    
    import torch
    
    # Create agent
    agent = PPOAgent(
        agent_id=0,
        state_dim=64,
        action_dim=6,
        hidden_dim=128,
        device=torch.device('cpu')
    )
    print(f"✓ Agent created successfully")
    
    # Test action selection
    state = np.random.randn(64).astype(np.float32)
    action, log_prob, value = agent.select_action(state)
    print(f"✓ Action selection successful")
    print(f"  - Action: {action}")
    print(f"  - Log prob: {log_prob:.6f}")
    print(f"  - Value estimate: {value:.6f}")
    
    # Test trajectory storage
    agent.store_transition(state, action, 1.0, value, log_prob, False)
    print(f"✓ Trajectory storage successful")
    
    # Test update
    for i in range(5):
        next_state = np.random.randn(64).astype(np.float32)
        action, log_prob, value = agent.select_action(next_state)
        agent.store_transition(next_state, action, 0.5, value, log_prob, i == 4)
    
    losses = agent.update(batch_size=32, num_epochs=2)
    print(f"✓ Model update successful")
    print(f"  - Total loss: {losses['total_loss']:.6f}")
    print(f"  - Policy loss: {losses['policy_loss']:.6f}")
    print(f"  - Value loss: {losses['value_loss']:.6f}")
    
    print(f"\n✓ Agent test passed!")


def main():
    """Run all tests."""
    try:
        env = test_environment()
        test_agent()
        
        print("\n" + "=" * 80)
        print("All tests passed! Environment is ready for training.")
        print("=" * 80)
        print("\nNext step: Run training with:")
        print("  python train_ippo.py --config configs/default_config.yaml --episodes 100")
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
