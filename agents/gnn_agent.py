"""
GNN-based communication agent for Stage 3: large-scale scenario optimization.

Each agent encodes its local observation, exchanges messages with neighbours
via a graph attention layer, and then acts with the enriched representation.
Parameter sharing is supported: all agents share one GNNAgent instance and
are distinguished only by their index in the graph.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """Single-head graph attention (GAT) message-passing layer.

    Each node attends over its neighbours and aggregates their features.
    Self-loops are included so a node always considers its own state.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (N, in_dim)  node features
            adj: (N, N)       adjacency matrix (1 = edge exists, 0 = no edge)
        Returns:
            (N, out_dim) updated node features
        """
        N = x.size(0)
        h = self.W(x)                           # (N, out_dim)

        # Pairwise attention coefficients
        h_src = h.unsqueeze(1).expand(-1, N, -1)   # (N, N, out_dim)
        h_dst = h.unsqueeze(0).expand(N, -1, -1)   # (N, N, out_dim)
        e = self.leaky_relu(
            self.attn(torch.cat([h_src, h_dst], dim=-1)).squeeze(-1)
        )                                           # (N, N)

        # Mask out non-edges (including self → handled via adj diagonal = 1)
        mask = (adj == 0)
        e = e.masked_fill(mask, float("-inf"))
        alpha = torch.softmax(e, dim=-1)            # (N, N)
        alpha = self.dropout(alpha)

        # Replace NaN rows (isolated nodes) with zeros
        alpha = torch.nan_to_num(alpha, nan=0.0)

        out = torch.matmul(alpha, h)                # (N, out_dim)
        return F.elu(out)


class MultiHeadGAT(nn.Module):
    """Multi-head graph attention network."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        head_dim = out_dim // num_heads
        self.heads = nn.ModuleList(
            [GraphAttentionLayer(in_dim, head_dim, dropout) for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        head_outs = [head(x, adj) for head in self.heads]   # each (N, head_dim)
        out = torch.cat(head_outs, dim=-1)                  # (N, out_dim)
        return F.elu(self.out_proj(out))


# ---------------------------------------------------------------------------
# Full GNN Actor-Critic
# ---------------------------------------------------------------------------

class GNNActorCritic(nn.Module):
    """
    Actor-Critic that first runs local observations through a node encoder,
    applies multi-head GAT message passing, then decodes into policy logits
    and a state value.

    All agents share this single network (parameter sharing).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gnn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Encode each agent's raw observation into a node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # GNN message-passing layers
        self.gnn_layers = nn.ModuleList([
            MultiHeadGAT(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(gnn_layers)
        ])

        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Critic (value) head — global pooling then decode
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, obs_batch: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_batch: (N, obs_dim)  one row per agent
            adj:       (N, N)        adjacency / topology matrix

        Returns:
            logits: (N, action_dim)
            values: (N,)
        """
        h = self.node_encoder(obs_batch)          # (N, hidden_dim)
        for gnn in self.gnn_layers:
            h = h + gnn(h, adj)                   # residual connection

        logits = self.actor(h)                    # (N, action_dim)
        values = self.critic(h).squeeze(-1)       # (N,)
        return logits, values

    def get_actions(
        self, obs_batch: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions and return (actions, log_probs, values)."""
        logits, values = self.forward(obs_batch, adj)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values


# ---------------------------------------------------------------------------
# GNN PPO trainer (parameter-sharing IPPO with graph comm)
# ---------------------------------------------------------------------------

class GNNPPOTrainer:
    """
    Parameter-sharing PPO with GNN communication.

    A single GNNActorCritic is shared by all agents. At each step the full
    observation matrix and the current topology adjacency are fed through the
    network, producing simultaneous actions for every agent.
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        gnn_layers: int = 2,
        num_heads: int = 4,
        device: str = "cpu",
        top_k: int = 3,
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.top_k = top_k
        self.device = torch.device(device)

        self.network = GNNActorCritic(
            obs_dim, action_dim, hidden_dim, gnn_layers, num_heads
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        # Fixed fully-connected adjacency (self-loops included)
        adj = torch.ones(num_agents, num_agents, dtype=torch.float32)
        self.register_adj(adj)

    def register_adj(self, adj: torch.Tensor):
        """Set the topology adjacency matrix used for message passing."""
        self.adj = adj.to(self.device)

    def build_sparse_adj(self, server_loads: np.ndarray) -> torch.Tensor:
        """
        Build a dynamic adjacency where each agent connects only to the
        top-k least-loaded edge servers (encoded as pseudo-agents).

        For simplicity we use a fully-connected graph among all agents here
        and let the attention mechanism learn which edges matter.
        """
        return self.adj

    @torch.no_grad()
    def select_actions(
        self, obs_list: List[np.ndarray], server_loads: Optional[np.ndarray] = None
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        Args:
            obs_list: list of per-agent observation arrays (length = num_agents)
            server_loads: optional server load vector for dynamic topology

        Returns:
            actions, log_probs, values — each a list of length num_agents
        """
        obs_tensor = torch.as_tensor(
            np.stack(obs_list), dtype=torch.float32, device=self.device
        )
        adj = self.build_sparse_adj(server_loads)
        actions, log_probs, values = self.network.get_actions(obs_tensor, adj)
        return (
            actions.cpu().tolist(),
            log_probs.cpu().tolist(),
            values.cpu().tolist(),
        )

    def compute_gae(
        self,
        rewards: List[List[float]],
        values: List[List[float]],
        final_values: List[float],
    ) -> torch.Tensor:
        """
        Compute GAE advantages for all agents jointly.

        Args:
            rewards:      (num_agents, T) per-step rewards
            values:       (num_agents, T) per-step value estimates
            final_values: (num_agents,)   bootstrap values at end of episode

        Returns:
            advantages: (num_agents, T)
        """
        T = len(rewards[0])
        advantages = torch.zeros(self.num_agents, T, device=self.device)

        for i in range(self.num_agents):
            gae = 0.0
            vals = values[i] + [final_values[i]]
            for t in reversed(range(T)):
                delta = rewards[i][t] + self.gamma * vals[t + 1] - vals[t]
                gae = delta + self.gamma * self.lam * gae
                advantages[i, t] = gae

        return advantages

    def update(
        self,
        obs_seq: List[List[np.ndarray]],
        actions_seq: List[List[int]],
        old_log_probs_seq: List[List[float]],
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single PPO update over the collected trajectory.

        Args:
            obs_seq:          (T, num_agents, obs_dim)
            actions_seq:      (T, num_agents)
            old_log_probs_seq:(T, num_agents)
            advantages:       (num_agents, T)

        Returns:
            loss dict
        """
        T = len(obs_seq)

        # Stack into tensors: (T, N, obs_dim)
        obs_t = torch.as_tensor(
            np.array([[obs_seq[t][i] for i in range(self.num_agents)] for t in range(T)]),
            dtype=torch.float32, device=self.device,
        )
        actions_t = torch.tensor(
            [[actions_seq[t][i] for i in range(self.num_agents)] for t in range(T)],
            dtype=torch.long, device=self.device,
        )                                                   # (T, N)
        old_lp_t = torch.tensor(
            [[old_log_probs_seq[t][i] for i in range(self.num_agents)] for t in range(T)],
            dtype=torch.float32, device=self.device,
        )                                                   # (T, N)

        # Normalize advantages per-agent
        adv = advantages.T.clone()                          # (T, N)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for t in range(T):
            logits, values = self.network(obs_t[t], self.adj)   # (N,*), (N,)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_lp = dist.log_prob(actions_t[t])                # (N,)

            ratio = torch.exp(new_lp - old_lp_t[t])
            surr1 = ratio * adv[t]
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv[t]
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = dist.entropy().mean()
            value_loss = F.mse_loss(values, values.detach())

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy

        total_loss = (
            total_policy_loss / T
            - self.entropy_coeff * total_entropy / T
            + self.value_coeff * total_value_loss / T
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": (total_policy_loss / T).item(),
            "value_loss": (total_value_loss / T).item(),
            "entropy": (total_entropy / T).item(),
            "total_loss": total_loss.item(),
        }

    def save(self, path: str):
        torch.save({"network": self.network.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
