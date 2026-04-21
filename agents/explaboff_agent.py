"""
Explaboff Scheme Implementation with Mutual Information Exchange.
Stage 2: Communication and Mutual Information Maximization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any
import numpy as np


class MutualInformationEstimator(nn.Module):
    """
    Estimate mutual information between agents using Noise-Contrastive Estimation (NCE).
    
    Estimates I(X;Y) where X and Y are random variables from different agents.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        """
        Initialize MI estimator.
        
        Args:
            state_dim: Dimension of agent state
            hidden_dim: Hidden layer dimension
        """
        super(MutualInformationEstimator, self).__init__()
        
        # Network for jointly processing states
        self.joint_network = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Network for marginal distribution
        self.marginal_network = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
        """
        Estimate MI using NCE.
        
        Args:
            state1: State from agent 1 [batch_size, state_dim]
            state2: State from agent 2 [batch_size, state_dim]
        
        Returns:
            Estimated mutual information
        """
        # Concatenate states
        joint = torch.cat([state1, state2], dim=-1)
        
        # Shuffle state2 for negative samples
        indices = torch.randperm(state2.shape[0])
        state2_shuffled = state2[indices]
        marginal = torch.cat([state1, state2_shuffled], dim=-1)
        
        # Score joint and marginal
        score_joint = self.joint_network(joint)
        score_marginal = self.marginal_network(marginal)
        
        # NCE loss (maximize MI means minimizing this)
        mi_estimate = torch.mean(score_joint) - torch.log(torch.mean(torch.exp(score_marginal)))
        
        return mi_estimate


class AttentionCommunicationModule(nn.Module):
    """
    Attention-based communication module for agents.
    Allows agents to selectively communicate using attention weights.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128, num_agents: int = 10):
        """
        Initialize communication module.
        
        Args:
            state_dim: Dimension of state
            hidden_dim: Hidden layer dimension
            num_agents: Total number of agents
        """
        super(AttentionCommunicationModule, self).__init__()
        
        self.state_dim = state_dim
        self.num_agents = num_agents
        
        # Encoder for agent state
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Message processing
        self.message_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim // 2)
        )
        
        # Decoder for communication output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + state_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(
        self,
        agent_states: torch.Tensor,
        agent_id: int,
        top_k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process communication among agents.
        
        Args:
            agent_states: States of all agents [num_agents, state_dim]
            agent_id: ID of the agent requesting communication
            top_k: Number of agents to communicate with
        
        Returns:
            Tuple of (updated_state, communication_weights)
        """
        # Encode all agent states
        encoded_states = self.encoder(agent_states)  # [num_agents, hidden_dim]
        
        # Get current agent's encoded state
        agent_encoded = encoded_states[agent_id].unsqueeze(0)  # [1, hidden_dim]
        
        # Apply attention
        other_states = encoded_states.unsqueeze(0)  # [1, num_agents, hidden_dim]
        attn_output, attn_weights = self.attention(
            agent_encoded.unsqueeze(0),
            other_states,
            other_states
        )  # attn_output: [1, 1, hidden_dim], attn_weights: [1, 1, num_agents]
        
        # Extract attention weights and select top-k agents
        weights = attn_weights.squeeze()  # [num_agents]
        _, top_indices = torch.topk(weights, min(top_k, self.num_agents))
        
        # Process messages from top-k agents
        top_states = encoded_states[top_indices]  # [top_k, hidden_dim]
        messages = self.message_processor(top_states)  # [top_k, state_dim//2]
        aggregated_message = torch.mean(messages, dim=0)  # [state_dim//2]
        
        # Decode to get updated state
        decoder_input = torch.cat([attn_output.squeeze(), aggregated_message])
        updated_state = self.decoder(decoder_input.unsqueeze(0))  # [1, state_dim]
        
        return updated_state.squeeze(0), weights


class ExplaboffAgent:
    """
    Explaboff Agent with Mutual Information-based Communication.
    Extends PPO with communication capabilities.
    """
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        mi_weight: float = 0.5,
        comm_enabled: bool = True,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize Explaboff agent.
        
        Args:
            agent_id: Unique agent ID
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate
            mi_weight: Weight for mutual information loss
            comm_enabled: Whether communication is enabled
            device: Device to run on
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.mi_weight = mi_weight
        self.comm_enabled = comm_enabled
        
        # Import PPO agent components
        from agents.networks import ActorCriticNetwork
        
        # Actor-Critic network (same as IPPO)
        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Communication module
        if comm_enabled:
            self.communication_module = AttentionCommunicationModule(
                state_dim, hidden_dim
            ).to(device)
            
            # MI estimator
            self.mi_estimator = MutualInformationEstimator(state_dim, hidden_dim).to(device)
        
        # Optimizer
        params = list(self.network.parameters())
        if comm_enabled:
            params += list(self.communication_module.parameters())
            params += list(self.mi_estimator.parameters())
        
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        
        # Trajectory buffer
        self.trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'mi_values': []  # For MI-based rewards
        }
    
    def select_action_with_communication(
        self,
        state: np.ndarray,
        all_agent_states: List[np.ndarray] = None,
        top_k: int = 3
    ) -> Tuple[int, float, float, float]:
        """
        Select action with optional communication.
        
        Args:
            state: Current agent state
            all_agent_states: States of all agents for communication
            top_k: Number of agents to communicate with
        
        Returns:
            Tuple of (action, log_prob, value, communication_reward)
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            
            # If communication enabled and other agents available
            if self.comm_enabled and all_agent_states is not None:
                # Convert all states to tensor
                all_states_tensor = torch.from_numpy(
                    np.array(all_agent_states)
                ).float().to(self.device)
                
                # Get communication
                comm_state, _ = self.communication_module(
                    all_states_tensor, self.agent_id, top_k
                )
                
                # Blend original state with communication
                combined_state = 0.7 * state_tensor + 0.3 * comm_state.unsqueeze(0)
            else:
                combined_state = state_tensor
            
            # Get action from network
            action_probs, value = self.network(combined_state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Estimate MI with other agents (for reward)
            comm_reward = 0.0
            if self.comm_enabled and all_agent_states is not None and len(all_agent_states) > 1:
                # Sample a random other agent
                other_agent_idx = np.random.randint(0, len(all_agent_states))
                if other_agent_idx != self.agent_id:
                    other_state_tensor = torch.from_numpy(
                        all_agent_states[other_agent_idx]
                    ).float().unsqueeze(0).to(self.device)
                    
                    with torch.enable_grad():
                        mi_estimate = self.mi_estimator(state_tensor, other_state_tensor)
                        comm_reward = mi_estimate.item()
            
            return action.item(), log_prob.item(), value.item(), comm_reward
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        mi_value: float = 0.0
    ):
        """Store transition with MI information."""
        self.trajectory['states'].append(state)
        self.trajectory['actions'].append(action)
        self.trajectory['rewards'].append(reward)
        self.trajectory['values'].append(value)
        self.trajectory['log_probs'].append(log_prob)
        self.trajectory['dones'].append(done)
        self.trajectory['mi_values'].append(mi_value)
    
    def compute_rewards_with_mi(self) -> np.ndarray:
        """
        Compute rewards incorporating mutual information.
        
        Returns:
            Adjusted rewards with MI bonus
        """
        rewards = np.array(self.trajectory['rewards'])
        mi_values = np.array(self.trajectory['mi_values'])
        
        # Normalize MI values
        if np.max(np.abs(mi_values)) > 0:
            mi_values = mi_values / (np.max(np.abs(mi_values)) + 1e-8)
        
        # Blend rewards with MI bonus
        adjusted_rewards = rewards + self.mi_weight * mi_values
        
        return adjusted_rewards
    
    def select_action(self, state: np.ndarray):
        """
        Select action without communication (standard PPO interface).
        Wraps select_action_with_communication with no peer states.

        Returns:
            Tuple of (action, log_prob, value)
        """
        action, log_prob, value, _ = self.select_action_with_communication(
            state, all_agent_states=None
        )
        return action, log_prob, value

    def compute_gae_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        next_value: float = 0.0,
    ):
        """GAE advantage estimation."""
        values_ext = np.append(values, next_value)
        deltas = rewards + gamma * values_ext[1:] * (1 - dones) - values_ext[:-1]
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(
        self,
        batch_size: int = 64,
        num_epochs: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        """
        PPO update incorporating MI-adjusted rewards.

        Args:
            batch_size: Mini-batch size
            num_epochs: Number of update epochs over collected data
            gamma / gae_lambda / clip_ratio / entropy_coeff / value_coeff / max_grad_norm:
                Standard PPO hyper-parameters (use defaults if not stored on self).

        Returns:
            Dict with average loss values, or empty dict if no data.
        """
        if not self.trajectory.get('states'):
            return {}

        states = np.array(self.trajectory['states'])
        actions = np.array(self.trajectory['actions'])
        old_log_probs = np.array(self.trajectory['log_probs'])
        values = np.array(self.trajectory['values'])
        dones = np.array(self.trajectory['dones'], dtype=np.float32)

        # MI-adjusted rewards
        rewards = self.compute_rewards_with_mi()

        advantages, returns = self.compute_gae_advantages(
            rewards, values, dones, gamma, gae_lambda
        )

        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        old_lp_t = torch.from_numpy(old_log_probs).float().to(self.device)
        adv_t = torch.from_numpy(advantages).float().to(self.device)
        ret_t = torch.from_numpy(returns).float().to(self.device)

        losses = {'total': [], 'policy': [], 'value': [], 'entropy': []}
        n = len(states)

        for _ in range(num_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                b = idx[start: start + batch_size]
                bs, ba, blp, badv, bret = (
                    states_t[b], actions_t[b], old_lp_t[b], adv_t[b], ret_t[b]
                )

                action_probs, vals = self.network(bs)
                dist = torch.distributions.Categorical(action_probs)
                new_lp = dist.log_prob(ba)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - blp)
                surr1 = ratio * badv
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * badv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = torch.nn.functional.mse_loss(vals.squeeze(), bret)
                total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in self.optimizer.param_groups for p in pg['params']],
                    max_grad_norm,
                )
                self.optimizer.step()

                losses['total'].append(total_loss.item())
                losses['policy'].append(policy_loss.item())
                losses['value'].append(value_loss.item())
                losses['entropy'].append(entropy.item())

        # Clear trajectory
        self.trajectory = {
            'states': [], 'actions': [], 'rewards': [],
            'values': [], 'log_probs': [], 'dones': [], 'mi_values': []
        }

        return {k: float(np.mean(v)) for k, v in losses.items()}

    def get_explainability_weights(self) -> np.ndarray:
        """
        Get attention weights for explainability.
        
        Returns:
            Attention weights from communication module
        """
        if not self.comm_enabled or not self.trajectory['states']:
            return np.zeros(1)
        
        with torch.no_grad():
            last_state = torch.from_numpy(
                self.trajectory['states'][-1]
            ).float().unsqueeze(0).to(self.device)
            
            # This is simplified - in full implementation would track all weights
            # For now return a placeholder
            return np.ones(1)
