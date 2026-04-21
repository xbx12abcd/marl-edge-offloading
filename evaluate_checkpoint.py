"""
Evaluation script for trained models (IPPO / GNN-PPO).

Auto-detects checkpoint format and loads the correct network:
  - {"networks": {agent_id: state_dict}}  → new train_ippo.py (SimplePPO)
  - {"network": state_dict}               → train_scalable.py (GNNPPOTrainer)
  - {"agents": [state_dict, ...]}         → legacy train_ippo.py (PPOAgent)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from utils import load_config, set_seed
from envs import EdgeOffloadingEnv, EdgeComputingEnv
from utils.gpu_monitor import get_device_with_memory_info


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def _detect_format(ckpt: dict) -> str:
    """Return checkpoint format tag."""
    if "networks" in ckpt:
        return "simple_ppo"        # new train_ippo.py
    if "network" in ckpt:
        return "gnn_ppo"           # train_scalable.py
    if "agents" in ckpt and isinstance(ckpt["agents"], list):
        return "legacy_ppo"        # old train_ippo.py
    return "unknown"


def _load_config_for_checkpoint(checkpoint_path: Path, config_path: str = None) -> dict:
    if config_path and Path(config_path).exists():
        return load_config(config_path)
    # Auto-detect: walk up to experiment root
    exp_dir = checkpoint_path.parent.parent
    auto = exp_dir / "config.yaml"
    if auto.exists():
        return load_config(str(auto))
    raise FileNotFoundError(
        f"Cannot find config.yaml. Pass --config explicitly.\n"
        f"Looked in: {auto}"
    )


# ---------------------------------------------------------------------------
# Format-specific inference helpers
# ---------------------------------------------------------------------------

def _make_simple_ppo_runner(ckpt: dict, config: dict, device: torch.device):
    """Load SimplePPO networks from new train_ippo.py checkpoint."""
    from train_ippo import ActorCritic, _get_config_value

    obs_dim = config["environment"]["state_dim"]
    hidden_dim = _get_config_value(
        config, ("marl", "hidden_dim"), ("algorithm", "hidden_dim")
    )

    networks = {}
    for agent_id, state_dict in ckpt["networks"].items():
        # Infer action_dim from first Linear layer output of actor
        first_key = list(state_dict.keys())[0]
        # actor.4.weight has shape (action_dim, hidden_dim)
        action_dim = None
        for k, v in state_dict.items():
            if "actor.4.weight" in k:
                action_dim = v.shape[0]
                break
        if action_dim is None:
            # fallback: count from config
            action_dim = config["environment"]["num_edge_servers"] + 1

        net = ActorCritic(obs_dim, action_dim, hidden_dim).to(device)
        net.load_state_dict(state_dict)
        net.eval()
        networks[agent_id] = net

    def select_actions(obs_dict):
        actions = {}
        with torch.no_grad():
            for agent_id, obs in obs_dict.items():
                if agent_id not in networks:
                    actions[agent_id] = 0
                    continue
                t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = networks[agent_id].get_action(t)
                actions[agent_id] = action.item()
        return actions

    return select_actions, list(networks.keys())


def _make_gnn_runner(ckpt: dict, config: dict, device: torch.device):
    """Load GNNActorCritic from train_scalable.py checkpoint."""
    from agents.gnn_agent import GNNActorCritic, GNNPPOTrainer

    num_agents = config["environment"]["num_end_devices"]
    obs_dim = config["environment"]["state_dim"]
    action_dim = config["environment"]["num_edge_servers"] + 1
    hidden_dim = config["marl"].get("hidden_dim") or config.get("algorithm", {}).get("hidden_dim", 128)
    gnn_layers = config["marl"].get("gnn_layers", 2)
    num_heads = config["marl"].get("gnn_heads", 4)

    net = GNNActorCritic(obs_dim, action_dim, hidden_dim, gnn_layers, num_heads).to(device)
    net.load_state_dict(ckpt["network"])
    net.eval()

    adj = torch.ones(num_agents, num_agents, dtype=torch.float32, device=device)

    agent_ids = [f"agent_{i}" for i in range(num_agents)]

    def select_actions(obs_dict):
        obs_list = [obs_dict.get(aid, np.zeros(obs_dim)) for aid in agent_ids]
        obs_t = torch.as_tensor(
            np.stack(obs_list), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            actions, _, _ = net.get_actions(obs_t, adj)
        return {aid: actions[i].item() for i, aid in enumerate(agent_ids)}

    return select_actions, agent_ids


def _make_legacy_runner(ckpt: dict, config: dict, device: torch.device):
    """Load old PPOAgent checkpoint (agents list)."""
    from agents import PPOAgent

    hidden_dim = (
        config["marl"].get("hidden_dim")
        or config.get("algorithm", {}).get("hidden_dim", 128)
    )
    env_tmp = EdgeComputingEnv(config)
    action_dim = env_tmp.action_space.n
    env_tmp.close()

    agent = PPOAgent(
        agent_id=0,
        state_dim=config["environment"]["state_dim"],
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=config["algorithm"]["learning_rate"],
        gamma=config["algorithm"]["gamma"],
        gae_lambda=config["algorithm"]["gae_lambda"],
        clip_ratio=config["algorithm"]["clip_ratio"],
        entropy_coeff=config["algorithm"]["entropy_coeff"],
        value_coeff=config["algorithm"]["value_coeff"],
        max_grad_norm=config["algorithm"]["max_grad_norm"],
        device=device,
    )
    agent.network.load_state_dict(ckpt["agents"][0])
    agent.network.eval()

    # Single-agent wrapper: feed same obs to all, use agent_0 action
    def select_actions(obs_dict):
        first_obs = next(iter(obs_dict.values()))
        action, _, _ = agent.select_action(first_obs)
        return {aid: action for aid in obs_dict}

    return select_actions, list(obs_dict.keys()) if False else ["agent_0"]


# ---------------------------------------------------------------------------
# Unified evaluate / demonstrate
# ---------------------------------------------------------------------------

def _build_runner(checkpoint_path: Path, config: dict, device: torch.device):
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    fmt = _detect_format(ckpt)
    print(f"  Checkpoint format : {fmt}")

    if fmt == "simple_ppo":
        return _make_simple_ppo_runner(ckpt, config, device)
    elif fmt == "gnn_ppo":
        return _make_gnn_runner(ckpt, config, device)
    elif fmt == "legacy_ppo":
        return _make_legacy_runner(ckpt, config, device)
    else:
        raise ValueError(
            f"Unknown checkpoint format. Keys found: {list(ckpt.keys())}"
        )


def evaluate_model(
    checkpoint_path: str,
    config_path: str = None,
    num_episodes: int = 10,
    render: bool = False,
    save_results: bool = True,
):
    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)

    checkpoint_path = Path(checkpoint_path)
    config = _load_config_for_checkpoint(checkpoint_path, config_path)
    set_seed(config.get("seed", 42))
    device = get_device_with_memory_info()

    print("\nInitializing environment...")
    env = EdgeOffloadingEnv(config)
    select_actions, _ = _build_runner(checkpoint_path, config, device)

    episode_length = config["marl"]["episode_length"]
    all_metrics = {
        "task_completion_rate": [],
        "energy_consumption": [],
        "average_delay": [],
        "deadline_miss_rate": [],
        "fairness_index": [],
        "total_reward": [],
    }

    print(f"\nEvaluating {num_episodes} episodes...")
    for ep in range(num_episodes):
        observations, _ = env.reset()
        episode_reward = 0.0

        for step in range(episode_length):
            actions = select_actions(observations)
            observations, rewards, terminations, truncations, _ = env.step(actions)
            episode_reward += sum(rewards.values())

            if render:
                print(f"  ep={ep} step={step} reward={sum(rewards.values()):.3f}")

            if all(terminations.values()) or all(truncations.values()):
                break

        metrics = env.get_metrics()
        for k in all_metrics:
            if k == "total_reward":
                all_metrics[k].append(episode_reward)
            elif k in metrics:
                all_metrics[k].append(metrics[k])

    avg = {k: float(np.mean(v)) if v else 0.0 for k, v in all_metrics.items()}

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    labels = {
        "task_completion_rate": "Task Completion Rate",
        "energy_consumption":   "Energy Consumption (J)",
        "average_delay":        "Average Delay (slots)",
        "deadline_miss_rate":   "Deadline Miss Rate",
        "fairness_index":       "Fairness Index (Jain)",
        "total_reward":         "Avg Episode Reward",
    }
    for k, v in avg.items():
        print(f"  {labels.get(k, k):<30}: {v:.6f}")

    if save_results:
        out_dir = checkpoint_path.parent.parent / "evaluation"
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"{checkpoint_path.stem}_evaluation.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": str(checkpoint_path),
                       "num_episodes": num_episodes,
                       "metrics": avg}, f, indent=2)
        print(f"\n  Results saved to: {out_file}")

    return avg


def demonstrate_model(
    checkpoint_path: str,
    config_path: str = None,
    num_steps: int = 50,
    delay: float = 0.5,
):
    print("=" * 80)
    print("Model Demonstration")
    print("=" * 80)

    checkpoint_path = Path(checkpoint_path)
    config = _load_config_for_checkpoint(checkpoint_path, config_path)
    set_seed(config.get("seed", 42))
    device = get_device_with_memory_info()

    env = EdgeOffloadingEnv(config)
    select_actions, agent_ids = _build_runner(checkpoint_path, config, device)

    print(f"\n  Agents : {len(agent_ids)}")
    print(f"  Steps  : {num_steps}")
    print("\nStarting demonstration...\n")

    observations, _ = env.reset()
    total_reward = 0.0

    for step in range(num_steps):
        actions = select_actions(observations)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        step_reward = sum(rewards.values())
        total_reward += step_reward

        # Summarise this step
        completed = sum(1 for info in infos.values() if info.get("completed"))
        server_loads = infos[agent_ids[0]].get("server_loads", [])
        loads_str = " ".join(f"{x:.2f}" for x in server_loads)

        print(f"Step {step + 1:>3}  |  "
              f"reward={step_reward:+.3f}  "
              f"completed={completed}/{len(agent_ids)}  "
              f"server_loads=[{loads_str}]  "
              f"total={total_reward:+.2f}")

        if all(terminations.values()) or all(truncations.values()):
            print("\n  Episode finished early.")
            break

        if delay > 0:
            time.sleep(delay)

    print("\n" + "=" * 80)
    metrics = env.get_metrics()
    print(f"  Task Completion Rate : {metrics['task_completion_rate']:.4f}")
    print(f"  Fairness Index       : {metrics['fairness_index']:.4f}")
    print(f"  Energy Consumption   : {metrics['energy_consumption']:.2f} J")
    print(f"  Avg Delay            : {metrics['average_delay']:.4f} slots")
    print(f"  Total Reward         : {total_reward:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate or demonstrate a trained MARL checkpoint"
    )
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument("--config", default=None,
                        help="Config YAML (auto-detected from experiment dir if omitted)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--render", action="store_true",
                        help="Print per-step rewards during evaluation")
    parser.add_argument("--demo", action="store_true",
                        help="Run demonstration mode (step-by-step output)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of steps in demo mode (default: 50)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between steps in demo mode (default: 0.5)")

    args = parser.parse_args()

    if args.demo:
        demonstrate_model(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            num_steps=args.steps,
            delay=args.delay,
        )
    else:
        evaluate_model(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            num_episodes=args.episodes,
            render=args.render,
        )


if __name__ == "__main__":
    main()
