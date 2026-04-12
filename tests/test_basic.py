"""
Basic tests for MARL Edge Offloading
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from utils.helpers import get_device
from utils.gpu_monitor import GPUMemoryMonitor


class TestDeviceManagement:
    """Test device management utilities"""

    def test_get_device(self):
        """Test device selection"""
        device = get_device()
        assert isinstance(device, torch.device)

        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"

    def test_gpu_monitor_initialization(self):
        """Test GPU monitor initialization"""
        device = get_device()
        monitor = GPUMemoryMonitor(device)

        assert monitor.device == device
        assert isinstance(monitor.warning_threshold, float)
        assert monitor.warning_threshold == 0.85


class TestEnvironmentBasics:
    """Test basic environment functionality"""

    def test_environment_import(self):
        """Test environment can be imported"""
        try:
            from envs.edge_env import EdgeComputingEnv
            assert True
        except ImportError:
            pytest.fail("Failed to import EdgeComputingEnv")

    def test_config_loading(self):
        """Test configuration loading"""
        try:
            from utils import load_config
            config = load_config('configs/default_config.yaml')
            assert isinstance(config, dict)
            assert 'environment' in config
            assert 'algorithm' in config
        except Exception as e:
            pytest.fail(f"Config loading failed: {e}")


class TestAgentBasics:
    """Test basic agent functionality"""

    def test_ppo_agent_import(self):
        """Test PPO agent can be imported"""
        try:
            from agents.ppo_agent import PPOAgent
            assert True
        except ImportError:
            pytest.fail("Failed to import PPOAgent")

    def test_agent_initialization(self):
        """Test agent initialization with mock parameters"""
        from agents.ppo_agent import PPOAgent

        # Mock minimal config
        config = {
            'algorithm': {
                'hidden_dim': 32,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_ratio': 0.2,
                'entropy_coeff': 0.01,
                'value_coeff': 0.5,
                'max_grad_norm': 0.5
            }
        }

        device = get_device()

        try:
            agent = PPOAgent(
                agent_id=0,
                state_dim=32,
                action_dim=4,
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
            assert agent.device == device
            assert agent.agent_id == 0
        except Exception as e:
            pytest.fail(f"Agent initialization failed: {e}")


class TestTrainingBasics:
    """Test basic training functionality"""

    def test_training_script_import(self):
        """Test training script can be imported"""
        try:
            import train_ippo_lowmem
            assert True
        except ImportError:
            pytest.fail("Failed to import training script")

    def test_quick_start_import(self):
        """Test quick start script can be imported"""
        try:
            import quick_start
            assert True
        except ImportError:
            pytest.fail("Failed to import quick start script")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])