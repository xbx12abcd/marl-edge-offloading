"""
PPO Agent implementation for MARL edge offloading.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Any
import numpy as np
from .networks import ActorCriticNetwork


class PPOAgent:
    """
    PPO Agent for independent learners in MARL.
    Each agent learns independently with its own policy and value network.
    """
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize PPO agent.
        
        Args:
            agent_id: Unique agent identifier
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE parameter
            clip_ratio: PPO clip ratio
            entropy_coeff: Entropy weighting coefficient
            value_coeff: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on (cpu or cuda)
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Trajectory buffer
        self.trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Current state
        
        Returns:
            Tuple of (action, log_probability, value_estimate)
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action_probs, value = self.network(state_tensor)
            
            # Sample action from policy
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """
        Store transition in trajectory buffer.
        
        Args:
            state: State
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode is done
        """
        self.trajectory['states'].append(state)
        self.trajectory['actions'].append(action)
        self.trajectory['rewards'].append(reward)
        self.trajectory['values'].append(value)
        self.trajectory['log_probs'].append(log_prob)
        self.trajectory['dones'].append(done)
    
    def compute_gae_advantages(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            next_value: Value estimate for next state (for bootstrapping)
        
        Returns:
            Tuple of (advantages, returns)
        """
        rewards = np.array(self.trajectory['rewards'])
        values = np.array(self.trajectory['values'] + [next_value])
        dones = np.array(self.trajectory['dones'])
        
        # Compute TD residuals
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Compute returns
        returns = advantages + values[:-1]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, batch_size: int = 64, num_epochs: int = 4) -> Dict[str, float]:
        """
        Update policy using collected trajectories.
        
        Args:
            batch_size: Batch size for training
            num_epochs: Number of training epochs
        
        Returns:
            Dictionary of loss values
        """
        if not self.trajectory['states']:
            return {}
        
        # Prepare data
        states = np.array(self.trajectory['states'])
        actions = np.array(self.trajectory['actions'])
        old_log_probs = np.array(self.trajectory['log_probs'])
        
        advantages, returns = self.compute_gae_advantages()
        
        states_tensor = torch.from_numpy(states).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)
        old_log_probs_tensor = torch.from_numpy(old_log_probs).float().to(self.device)
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
        returns_tensor = torch.from_numpy(returns).float().to(self.device)
        
        # Training loop
        losses = {'total': [], 'policy': [], 'value': [], 'entropy': []}
        num_samples = len(states)
        
        for epoch in range(num_epochs):
            # Shuffle indices
            indices = np.random.permutation(num_samples)
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Forward pass
                action_probs, values = self.network(batch_states)
                
                # Compute policy loss
                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(batch_actions)
                
                # PPO loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy loss
                entropy = action_dist.entropy().mean()
                entropy_loss = -self.entropy_coeff * entropy
                
                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Total loss
                total_loss = policy_loss + self.value_coeff * value_loss + entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Record losses
                losses['total'].append(total_loss.item())
                losses['policy'].append(policy_loss.item())
                losses['value'].append(value_loss.item())
                losses['entropy'].append(entropy_loss.item())
        
        # Clear trajectory buffer
        self.trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # Return average losses
        return {
            'total_loss': np.mean(losses['total']),
            'policy_loss': np.mean(losses['policy']),
            'value_loss': np.mean(losses['value']),
            'entropy_loss': np.mean(losses['entropy'])
        }
    
    def save_model(self, path: str):
        """Save model to file."""
        torch.save(self.network.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model from file."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))
