"""
IPPO Training Script - PettingZoo-native PPO for Multi-Agent Edge Offloading.

Uses EdgeOffloadingEnv (PettingZoo ParallelEnv) as the canonical environment.
Each end-device agent independently learns an Actor-Critic policy (IPPO).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from envs import EdgeOffloadingEnv
from utils import load_config, set_seed


def _get_config_value(config_dict, *candidate_paths):
    """Retrieve a value from config trying multiple key paths."""
    for path in candidate_paths:
        value = config_dict
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            return value
    supported_paths = ", ".join(
        "['" + "']['".join(path) + "']" for path in candidate_paths
    )
    raise KeyError(f"Missing required config value. Supported paths: {supported_paths}")


def get_device(mode: str = "auto") -> torch.device:
    if mode == "cpu":
        print("✓ Using CPU (forced)")
        return torch.device("cpu")
    if mode == "cuda":
        if torch.cuda.is_available():
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        print("⚠ CUDA requested but not available, using CPU")
        return torch.device("cpu")
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("✓ Using CPU")
    return torch.device("cpu")


def compute_gae(rewards, values, gamma=0.99, lam=0.95, device="cpu"):
    """Generalized Advantage Estimation."""
    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    values_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)

    advantages = []
    gae = 0
    for index in reversed(range(len(rewards_tensor))):
        delta = (
            rewards_tensor[index]
            + gamma * values_tensor[index + 1]
            - values_tensor[index]
        )
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    return torch.stack(advantages)


class ActorCritic(nn.Module):
    """Shared-backbone Actor-Critic network."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.actor(x)

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action(self, x):
        logits = self.actor(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, probs


class SimplePPO:
    """Independent PPO: one ActorCritic network per agent."""

    def __init__(
        self,
        num_agents,
        obs_dim,
        action_dim,
        hidden_dim,
        lr,
        device,
        entropy_coeff=0.01,
        value_coeff=0.5,
        max_grad_norm=0.5,
    ):
        self.device = device
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.networks = {}

        for index in range(num_agents):
            agent = f"agent_{index}"
            self.networks[agent] = ActorCritic(obs_dim, action_dim, hidden_dim).to(device)

        self.optimizers = {
            agent: torch.optim.Adam(net.parameters(), lr=lr)
            for agent, net in self.networks.items()
        }

    def select_action(self, agent, state):
        with torch.no_grad():
            action, log_prob, probs = self.networks[agent].get_action(
                state.unsqueeze(0)
            )
            value = self.networks[agent].get_value(state.unsqueeze(0)).item()
            return action.item(), log_prob.item(), value

    def update(self, agent, states, actions, old_log_probs, advantages, clip_ratio=0.2):
        actions_tensor = actions
        if actions_tensor.dim() == 0:
            actions_tensor = actions_tensor.unsqueeze(0)

        action, new_log_probs, probs = self.networks[agent].get_action(states)
        values = self.networks[agent].get_value(states)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        entropy_loss = -self.entropy_coeff * entropy

        value_loss = self.value_coeff * nn.functional.mse_loss(values, values.detach())

        total_loss = policy_loss + entropy_loss + value_loss

        self.optimizers[agent].zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.networks[agent].parameters(), self.max_grad_norm)
        self.optimizers[agent].step()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }


def main():
    parser = argparse.ArgumentParser(description="PettingZoo PPO Training")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["marl"]["total_episodes"] = args.episodes

    set_seed(config.get("seed", 42))
    device = get_device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.name or f"IPPO_{timestamp}"
    exp_dir = Path("results") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    print("=" * 80)
    print(f"IPPO TRAINING - {exp_name}")
    print("=" * 80)

    env = EdgeOffloadingEnv(config)
    num_agents = env.num_agents
    obs_dim = config["environment"]["state_dim"]
    action_dim = env.action_spaces["agent_0"].n
    hidden_dim = _get_config_value(
        config, ("marl", "hidden_dim"), ("algorithm", "hidden_dim")
    )
    lr = config["algorithm"]["learning_rate"]
    episode_length = _get_config_value(
        config, ("marl", "episode_length"), ("algorithm", "episode_length")
    )

    print(f"Agents: {num_agents}")
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Episodes: {args.episodes}, Steps per episode: {episode_length}")
    print(f"✓ Device: {device}")
    print()

    ppo = SimplePPO(
        num_agents,
        obs_dim,
        action_dim,
        hidden_dim,
        lr,
        device,
        entropy_coeff=config["algorithm"].get("entropy_coeff", 0.01),
        value_coeff=config["algorithm"].get("value_coeff", 0.5),
        max_grad_norm=config["algorithm"].get("max_grad_norm", 0.5),
    )
    writer = SummaryWriter(str(exp_dir / "logs"))

    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True)

    episode_rewards = []
    completion_rates = []

    for episode in tqdm(range(args.episodes), desc="Episodes"):
        observations, _ = env.reset()
        observations = {
            agent: torch.from_numpy(obs).float().to(device)
            for agent, obs in observations.items()
        }

        trajectories = {
            agent: {
                "states": [],
                "actions": [],
                "rewards": [],
                "values": [],
                "log_probs": [],
            }
            for agent in env.agents
        }

        episode_reward = 0.0

        for _ in range(episode_length):
            actions = {}
            for agent in env.agents:
                action, log_prob, value = ppo.select_action(agent, observations[agent])
                actions[agent] = action
                trajectories[agent]["states"].append(observations[agent])
                trajectories[agent]["actions"].append(action)
                trajectories[agent]["log_probs"].append(log_prob)
                trajectories[agent]["values"].append(value)

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            next_obs = {
                agent: torch.from_numpy(obs).float().to(device)
                for agent, obs in next_obs.items()
            }

            for agent in env.agents:
                trajectories[agent]["rewards"].append(rewards[agent])
                episode_reward += rewards[agent]

            observations = next_obs
            if all(terminations.values()) or all(truncations.values()):
                break

        episode_rewards.append(episode_reward)
        metrics = env.get_metrics()
        completion_rates.append(metrics["task_completion_rate"])

        for agent in env.agents:
            if len(trajectories[agent]["states"]) < 2:
                continue

            states = torch.stack(trajectories[agent]["states"])
            actions_t = torch.tensor(
                trajectories[agent]["actions"], dtype=torch.long, device=device
            )
            old_log_probs = torch.tensor(
                trajectories[agent]["log_probs"], dtype=torch.float32, device=device
            )
            rewards_tensor = torch.tensor(
                trajectories[agent]["rewards"], dtype=torch.float32, device=device
            )
            values_tensor = torch.tensor(
                trajectories[agent]["values"], dtype=torch.float32, device=device
            )

            with torch.no_grad():
                final_value = ppo.select_action(agent, observations[agent])[2]
                values_tensor = torch.cat(
                    [values_tensor, torch.tensor([final_value], device=device)]
                )

            advantages = compute_gae(rewards_tensor, values_tensor, device=device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ppo.update(agent, states, actions_t, old_log_probs, advantages)

        if episode % 10 == 0:
            writer.add_scalar("train/episode_reward", episode_reward, episode)
            writer.add_scalar(
                "train/completion_rate", metrics["task_completion_rate"], episode
            )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Final 10-ep avg reward:     {np.mean(episode_rewards[-10:]):.4f}")
    print(f"Final 10-ep avg completion: {np.mean(completion_rates[-10:]):.4f}")

    results = {
        "final_completion_rate": float(np.mean(completion_rates[-10:])),
        "final_avg_reward": float(np.mean(episode_rewards[-10:])),
        "episode_rewards": [float(r) for r in episode_rewards],
        "completion_rates": [float(r) for r in completion_rates],
    }

    with open(exp_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    torch.save(
        {"networks": {agent: net.state_dict() for agent, net in ppo.networks.items()}},
        exp_dir / "checkpoints" / "final_model.pt",
    )

    writer.close()
    print(f"\n✓ Results saved to: {exp_dir}")
    return results


if __name__ == "__main__":
    main()
