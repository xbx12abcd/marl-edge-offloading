"""
Actor-Critic Network Architecture for MARL.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class ActorNetwork(nn.Module):
    """
    Actor network for policy gradient methods.
    Outputs action probabilities.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        """
        Initialize actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super(ActorNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            Action probabilities [batch_size, action_dim]
        """
        return self.network(state)


class CriticNetwork(nn.Module):
    """
    Critic network for estimating state value.
    Outputs scalar value estimates.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        """
        Initialize critic network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super(CriticNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            Value estimates [batch_size, 1]
        """
        return self.network(state)


class ActorCriticNetwork(nn.Module):
    """
    Combined actor-critic network for efficient computation.
    Shares feature extraction layers between actor and critic.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        """
        Initialize actor-critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        shared_layers = []
        shared_layers.append(nn.Linear(state_dim, hidden_dim))
        shared_layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            Tuple of (action_probabilities, values)
            - action_probabilities: [batch_size, action_dim]
            - values: [batch_size, 1]
        """
        features = self.shared_network(state)
        action_probs = self.actor_head(features)
        values = self.critic_head(features)
        return action_probs, values
