"""
Stage 3: Large-Scale Scenario Training Script.

Trains a parameter-sharing GNN-PPO agent (GNNPPOTrainer) on the
PettingZoo EdgeOffloadingEnv, supporting 50+ agents and analysing
scalability, communication overhead, and fairness.

Usage:
    python train_scalable.py --num_agents 20 --episodes 500
    python train_scalable.py --num_agents 50 --episodes 1000 --device cuda
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from envs import EdgeOffloadingEnv
from agents.gnn_agent import GNNPPOTrainer
from utils import load_config, set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(mode: str = "auto") -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scalable_config(base_config: dict, num_agents: int) -> dict:
    """Override base config for large-scale scenario."""
    cfg = {k: v.copy() if isinstance(v, dict) else v for k, v in base_config.items()}
    cfg["environment"] = dict(base_config["environment"])
    cfg["marl"] = dict(base_config["marl"])
    cfg["algorithm"] = dict(base_config["algorithm"])
    cfg["reward"] = dict(base_config["reward"])

    # Scale up topology
    cfg["environment"]["num_end_devices"] = num_agents
    cfg["environment"]["num_edge_servers"] = max(3, num_agents // 5)

    # GNN-specific marl settings
    cfg["marl"]["num_agents"] = num_agents
    cfg["marl"]["algorithm"] = "GNN-IPPO"
    cfg["marl"]["gnn_layers"] = 2
    cfg["marl"]["gnn_heads"] = 4
    cfg["marl"]["top_k_comm"] = 3

    return cfg


def print_config_summary(cfg: dict, device: torch.device):
    print("=" * 80)
    print("Stage 3: GNN-PPO Large-Scale Training")
    print("=" * 80)
    print(f"  Agents      : {cfg['environment']['num_end_devices']}")
    print(f"  Edge servers: {cfg['environment']['num_edge_servers']}")
    print(f"  Episodes    : {cfg['marl']['total_episodes']}")
    print(f"  Ep length   : {cfg['marl']['episode_length']}")
    print(f"  GNN layers  : {cfg['marl']['gnn_layers']}")
    print(f"  Device      : {device}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # ---- Config ----
    base_config = load_config(args.config)
    config = build_scalable_config(base_config, args.num_agents)
    config["marl"]["total_episodes"] = args.episodes
    set_seed(config.get("seed", 42))
    device = get_device(args.device)

    print_config_summary(config, device)

    # ---- Experiment directory ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.name or f"GNN_n{args.num_agents}_{timestamp}"
    exp_dir = Path("results") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    # ---- Environment ----
    env = EdgeOffloadingEnv(config)
    num_agents = env.num_agents
    obs_dim = config["environment"]["state_dim"]
    action_dim = env.action_spaces["agent_0"].n

    # ---- Trainer ----
    trainer = GNNPPOTrainer(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=config["marl"].get("hidden_dim") or config.get("algorithm", {}).get("hidden_dim", 128),
        lr=config["algorithm"]["learning_rate"],
        gamma=config["algorithm"]["gamma"],
        lam=config["algorithm"]["gae_lambda"],
        clip_ratio=config["algorithm"]["clip_ratio"],
        entropy_coeff=config["algorithm"]["entropy_coeff"],
        value_coeff=config["algorithm"]["value_coeff"],
        max_grad_norm=config["algorithm"]["max_grad_norm"],
        gnn_layers=config["marl"]["gnn_layers"],
        num_heads=config["marl"]["gnn_heads"],
        device=str(device),
        top_k=config["marl"]["top_k_comm"],
    )

    writer = SummaryWriter(str(exp_dir / "logs"))

    # ---- Metrics tracking ----
    episode_rewards: list = []
    completion_rates: list = []
    fairness_indices: list = []
    step_times: list = []

    total_episodes = config["marl"]["total_episodes"]
    episode_length = config["marl"]["episode_length"]

    # ---- Training loop ----
    for episode in tqdm(range(total_episodes), desc="Episodes"):
        observations, _ = env.reset()
        agent_keys = env.agents

        # Per-agent trajectory buffers
        obs_seq: list = []
        actions_seq: list = []
        log_probs_seq: list = []
        rewards_buf: list = [[] for _ in range(num_agents)]
        values_buf: list = [[] for _ in range(num_agents)]

        episode_reward = 0.0
        t0 = time.perf_counter()

        for step in range(episode_length):
            obs_list = [observations[ak] for ak in agent_keys]

            server_loads = env.server_loads if hasattr(env, "server_loads") else None
            actions, log_probs, values = trainer.select_actions(obs_list, server_loads)

            action_dict = {ak: actions[i] for i, ak in enumerate(agent_keys)}
            next_obs, rewards, terminations, truncations, _ = env.step(action_dict)

            obs_seq.append(obs_list)
            actions_seq.append(actions)
            log_probs_seq.append(log_probs)
            for i, ak in enumerate(agent_keys):
                rewards_buf[i].append(rewards[ak])
                values_buf[i].append(values[i])
                episode_reward += rewards[ak]

            observations = next_obs
            if all(terminations.values()) or all(truncations.values()):
                break

        step_times.append(time.perf_counter() - t0)

        # Bootstrap final values
        final_obs = [observations[ak] for ak in agent_keys]
        _, _, final_vals = trainer.select_actions(final_obs)

        # Compute advantages
        advantages = trainer.compute_gae(rewards_buf, values_buf, final_vals)

        # Policy update
        loss_info = trainer.update(obs_seq, actions_seq, log_probs_seq, advantages)

        # Collect metrics
        metrics = env.get_metrics()
        episode_rewards.append(episode_reward)
        completion_rates.append(metrics["task_completion_rate"])
        fairness_indices.append(metrics["fairness_index"])

        # TensorBoard
        if episode % 10 == 0:
            writer.add_scalar("train/episode_reward", episode_reward, episode)
            writer.add_scalar("train/completion_rate", metrics["task_completion_rate"], episode)
            writer.add_scalar("train/fairness_index", metrics["fairness_index"], episode)
            writer.add_scalar("train/energy", metrics["energy_consumption"], episode)
            writer.add_scalar("loss/policy", loss_info["policy_loss"], episode)
            writer.add_scalar("loss/value", loss_info["value_loss"], episode)
            writer.add_scalar("loss/entropy", loss_info["entropy"], episode)

        # Checkpoint
        if (episode + 1) % max(1, total_episodes // 5) == 0:
            ckpt_path = exp_dir / "checkpoints" / f"episode_{episode + 1:06d}.pt"
            trainer.save(str(ckpt_path))

    # ---- Final results ----
    final_metrics = env.get_metrics()
    avg_step_time = float(np.mean(step_times))
    comm_overhead_pct = _estimate_comm_overhead(num_agents, avg_step_time)

    results = {
        "num_agents": num_agents,
        "num_edge_servers": config["environment"]["num_edge_servers"],
        "total_episodes": total_episodes,
        "final_completion_rate": float(np.mean(completion_rates[-10:])),
        "final_avg_reward": float(np.mean(episode_rewards[-10:])),
        "final_fairness_index": float(np.mean(fairness_indices[-10:])),
        "avg_episode_time_s": avg_step_time,
        "comm_overhead_pct": comm_overhead_pct,
        "episode_rewards": [float(r) for r in episode_rewards],
        "completion_rates": [float(r) for r in completion_rates],
        "fairness_indices": [float(r) for r in fairness_indices],
        **{k: float(v) for k, v in final_metrics.items()},
    }

    with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    trainer.save(str(exp_dir / "checkpoints" / "final_model.pt"))
    writer.close()

    _print_results(results)
    print(f"\n✓ Results saved to: {exp_dir}")
    return results


def _estimate_comm_overhead(num_agents: int, avg_ep_time: float) -> float:
    """
    Rough estimate of GNN communication overhead as a % of total time.
    Based on O(N^2) attention vs O(N) forward pass.
    """
    if avg_ep_time <= 0:
        return 0.0
    # Heuristic: GAT attention scales as N^2, local MLP as N
    n2_fraction = (num_agents ** 2) / (num_agents ** 2 + num_agents * 10)
    return round(n2_fraction * 100, 2)


def _print_results(results: dict):
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE — Stage 3 Results")
    print("=" * 80)
    print(f"  Agents              : {results['num_agents']}")
    print(f"  Completion rate     : {results['final_completion_rate']:.4f}")
    print(f"  Avg reward (last 10): {results['final_avg_reward']:.4f}")
    print(f"  Fairness index      : {results['final_fairness_index']:.4f}")
    print(f"  Avg episode time    : {results['avg_episode_time_s']:.3f}s")
    print(f"  Comm overhead est.  : {results['comm_overhead_pct']:.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: GNN-PPO Scalable Training")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--num_agents", type=int, default=20,
                        help="Number of end-device agents (default: 20, target: 50+)")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    train(parser.parse_args())
