"""
Comparison evaluation script for IPPO vs Explaboff.
Stage 2: Mutual Information Analysis and Performance Comparison

Workflow per algorithm:
  - If a checkpoint path is given: load weights directly, skip training
  - Otherwise: train for `train_episodes` episodes first
  Then evaluate for `eval_episodes` episodes with frozen weights and compare.

Supported checkpoint formats
  {"networks": {agent_id: state_dict}}  new train_ippo.py  (SimplePPO)
  {"agents":   [state_dict, ...]}       old train_ippo.py  (PPOAgent)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from agents import PPOAgent
from agents.explaboff_agent import ExplaboffAgent
from envs import EdgeComputingEnv
from utils import load_config, set_seed, get_device


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_weights_into_agents(agents: list, checkpoint_path: str, device):
    """
    Load a checkpoint into a list of PPOAgent / ExplaboffAgent instances.

    Supported formats:
      {"networks": {"agent_0": sd, ...}}  — new train_ippo.py
      {"agents":   [sd, sd, ...]}         — old train_ippo.py
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "networks" in ckpt:
        # New SimplePPO format: keyed by agent name
        for i, agent in enumerate(agents):
            key = f"agent_{i}"
            if key in ckpt["networks"]:
                # SimplePPO ActorCritic has actor/critic; PPOAgent has a unified network.
                # Try direct load; fall back to extracting actor weights.
                sd = ckpt["networks"][key]
                try:
                    agent.network.load_state_dict(sd, strict=False)
                except Exception:
                    # Extract only keys that exist in PPOAgent's network
                    own_keys = set(agent.network.state_dict().keys())
                    filtered = {k: v for k, v in sd.items() if k in own_keys}
                    agent.network.load_state_dict(filtered, strict=False)
        print(f"  ✓ Loaded 'networks' checkpoint: {checkpoint_path}")

    elif "agents" in ckpt and isinstance(ckpt["agents"], list):
        # Legacy format: list of state dicts
        for i, agent in enumerate(agents):
            if i < len(ckpt["agents"]):
                agent.network.load_state_dict(ckpt["agents"][i], strict=False)
        print(f"  ✓ Loaded 'agents' checkpoint: {checkpoint_path}")

    else:
        raise ValueError(
            f"Unrecognised checkpoint format. Keys: {list(ckpt.keys())}\n"
            "Expected 'networks' (new train_ippo.py) or 'agents' (old train_ippo.py)."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agents(algorithm_name, config, env, device):
    num_agents = config['marl']['num_agents']
    hidden_dim = config['marl'].get('hidden_dim') or config['algorithm'].get('hidden_dim', 128)
    agents = []
    for i in range(num_agents):
        if algorithm_name == 'Explaboff':
            agent = ExplaboffAgent(
                agent_id=i,
                state_dim=config['environment']['state_dim'],
                action_dim=env.action_space.n,
                hidden_dim=hidden_dim,
                learning_rate=config['algorithm']['learning_rate'],
                mi_weight=config['algorithm'].get('mi_weight', 0.5),
                comm_enabled=True,
                device=device,
            )
        else:
            agent = PPOAgent(
                agent_id=i,
                state_dim=config['environment']['state_dim'],
                action_dim=env.action_space.n,
                hidden_dim=hidden_dim,
                learning_rate=config['algorithm']['learning_rate'],
                gamma=config['algorithm']['gamma'],
                gae_lambda=config['algorithm']['gae_lambda'],
                clip_ratio=config['algorithm']['clip_ratio'],
                entropy_coeff=config['algorithm']['entropy_coeff'],
                value_coeff=config['algorithm']['value_coeff'],
                max_grad_norm=config['algorithm']['max_grad_norm'],
                device=device,
            )
        agents.append(agent)
    return agents


def _run_episode(env, agents, config, train_mode: bool):
    """Run one episode. If train_mode=True, update agent weights afterwards."""
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = config['marl']['episode_length']
    all_agent_states = [obs] * len(agents)
    mi_rewards = []

    for _ in range(episode_length):
        actions = []
        step_mi = []   # per-agent MI values for this step
        for agent in agents:
            if isinstance(agent, ExplaboffAgent):
                action, _, _, mi_r = agent.select_action_with_communication(
                    obs, all_agent_states, top_k=3
                )
                mi_rewards.append(mi_r)
                step_mi.append(mi_r)
            else:
                action, _, _ = agent.select_action(obs)
                step_mi.append(0.0)
            actions.append(action)

        obs, reward, terminated, truncated, _ = env.step(actions[0])
        episode_reward += reward
        all_agent_states = [obs] * len(agents)

        for i, agent in enumerate(agents):
            if isinstance(agent, ExplaboffAgent):
                agent.store_transition(obs, actions[i], reward, 0.0, 0.0, terminated,
                                       mi_value=step_mi[i])
            else:
                agent.store_transition(obs, actions[i], reward, 0.0, 0.0, terminated)

        if terminated or truncated:
            break

    if train_mode:
        for agent in agents:
            if agent.trajectory.get('states'):
                agent.update(
                    batch_size=config['algorithm']['batch_size'],
                    num_epochs=config['algorithm']['num_epochs'],
                )

    return episode_reward, mi_rewards


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def run_algorithm(
    algorithm_name: str,
    config: dict,
    train_episodes: int = 100,
    eval_episodes: int = 20,
    checkpoint_path: str = None,
) -> tuple:
    """
    Load (or train) then evaluate one algorithm.

    If checkpoint_path is given the training phase is skipped entirely.

    Returns:
        results dict, per-episode metrics dict (for plotting), train_rewards list
    """
    print(f"\n{'='*80}")
    if checkpoint_path:
        print(f"{algorithm_name}  |  checkpoint={checkpoint_path}  eval={eval_episodes} ep")
    else:
        print(f"{algorithm_name}  |  train={train_episodes} ep  eval={eval_episodes} ep")
    print(f"{'='*80}")

    set_seed(config['seed'])
    device = get_device()
    env = EdgeComputingEnv(config)
    agents = _make_agents(algorithm_name, config, env, device)

    # ---- Load checkpoint OR train from scratch ----
    train_rewards = []
    if checkpoint_path:
        _load_weights_into_agents(agents, checkpoint_path, device)
        print("  Skipping training (checkpoint loaded).")
    else:
        print("Training...")
        for ep in tqdm(range(train_episodes), desc=f"  Train {algorithm_name}"):
            r, _ = _run_episode(env, agents, config, train_mode=True)
            train_rewards.append(r)

    # ---- Evaluation phase (weights frozen) ----
    print("Evaluating...")
    for agent in agents:
        agent.network.eval()

    metrics = {
        'task_completion_rates': [],
        'energy_consumptions': [],
        'average_delays': [],
        'deadline_miss_rates': [],
        'fairness_indices': [],
        'episode_rewards': [],
        'mutual_information': [],
    }

    for ep in tqdm(range(eval_episodes), desc=f"  Eval  {algorithm_name}"):
        r, mi_list = _run_episode(env, agents, config, train_mode=False)
        env_m = env.get_metrics()
        metrics['task_completion_rates'].append(env_m['task_completion_rate'])
        metrics['energy_consumptions'].append(env_m['energy_consumption'])
        metrics['average_delays'].append(env_m['average_delay'])
        metrics['deadline_miss_rates'].append(env_m['deadline_miss_rate'])
        metrics['fairness_indices'].append(env_m['fairness_index'])
        metrics['episode_rewards'].append(r)
        if mi_list:
            metrics['mutual_information'].append(float(np.mean(mi_list)))

    def _stat(values):
        return {'mean': float(np.mean(values)), 'std': float(np.std(values))}

    results = {
        'algorithm': algorithm_name,
        'train_episodes': train_episodes,
        'eval_episodes': eval_episodes,
        'task_completion_rate': _stat(metrics['task_completion_rates']),
        'energy_consumption':   _stat(metrics['energy_consumptions']),
        'average_delay':        _stat(metrics['average_delays']),
        'deadline_miss_rate':   _stat(metrics['deadline_miss_rates']),
        'fairness_index':       _stat(metrics['fairness_indices']),
        'episode_reward':       _stat(metrics['episode_rewards']),
    }
    if metrics['mutual_information']:
        results['mutual_information'] = _stat(metrics['mutual_information'])

    return results, metrics, train_rewards


# ---------------------------------------------------------------------------
# Print & plot
# ---------------------------------------------------------------------------

def print_comparison(ippo: dict, explaboff: dict):
    print("\n" + "=" * 85)
    print("PERFORMANCE COMPARISON: IPPO (no comm) vs Explaboff (MI comm)")
    print("=" * 85)

    rows = [
        ('task_completion_rate', 'Task Completion Rate',   'higher'),
        ('energy_consumption',   'Energy Consumption (J)', 'lower'),
        ('average_delay',        'Average Delay',          'lower'),
        ('deadline_miss_rate',   'Deadline Miss Rate',     'lower'),
        ('fairness_index',       'Fairness Index',         'higher'),
        ('episode_reward',       'Episode Reward',         'higher'),
    ]

    print(f"\n  {'Metric':<28} {'IPPO':>14} {'Explaboff':>14} {'Change':>10}")
    print("  " + "-" * 70)
    for key, label, direction in rows:
        iv = ippo[key]['mean']
        ev = explaboff[key]['mean']
        if direction == 'higher':
            pct = (ev - iv) / (abs(iv) + 1e-8) * 100
            sign = "+" if pct >= 0 else ""
            mark = "✓" if pct > 0 else "✗"
        else:
            pct = (iv - ev) / (abs(iv) + 1e-8) * 100
            sign = "+" if pct >= 0 else ""
            mark = "✓" if pct > 0 else "✗"
        print(f"  {label:<28} {iv:>14.4f} {ev:>14.4f}  {mark} {sign}{pct:.1f}%")

    if 'mutual_information' in explaboff:
        mi = explaboff['mutual_information']['mean']
        print(f"\n  {'Mutual Information (Explaboff)':<28} {mi:>14.6f}")

    print("=" * 85)


def plot_comparison(ippo_m, explaboff_m, train_ippo, train_explaboff, out_dir: Path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle('IPPO vs Explaboff — Training & Evaluation Comparison', fontsize=14)

    def _smooth(x, w=5):
        return np.convolve(x, np.ones(w)/w, mode='valid') if len(x) >= w else x

    # Training curves
    axes[0, 0].plot(_smooth(train_ippo), label='IPPO', color='steelblue')
    axes[0, 0].plot(_smooth(train_explaboff), label='Explaboff', color='darkorange')
    axes[0, 0].set_title('Training Reward (smoothed)')
    axes[0, 0].set_xlabel('Episode'); axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # Eval metrics
    eval_pairs = [
        (ippo_m['task_completion_rates'],  explaboff_m['task_completion_rates'],  'Task Completion Rate',  (0, 1)),
        (ippo_m['energy_consumptions'],    explaboff_m['energy_consumptions'],    'Energy Consumption (J)', None),
        (ippo_m['average_delays'],         explaboff_m['average_delays'],         'Average Delay',          None),
        (ippo_m['deadline_miss_rates'],    explaboff_m['deadline_miss_rates'],    'Deadline Miss Rate',     (0, 1)),
        (ippo_m['fairness_indices'],       explaboff_m['fairness_indices'],       'Fairness Index',         (0, 1)),
        (ippo_m['episode_rewards'],        explaboff_m['episode_rewards'],        'Episode Reward',         None),
        (ippo_m['mutual_information'] or [0],
         explaboff_m['mutual_information'] or [0],                               'Mutual Information',     None),
    ]

    positions = [(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)]
    for (r,c), (iv, ev, title, ylim) in zip(positions, eval_pairs):
        axes[r, c].plot(iv, label='IPPO', color='steelblue', alpha=0.8)
        axes[r, c].plot(ev, label='Explaboff', color='darkorange', alpha=0.8)
        axes[r, c].set_title(title)
        axes[r, c].set_xlabel('Eval Episode')
        if ylim:
            axes[r, c].set_ylim(*ylim)
        axes[r, c].legend(fontsize=8)
        axes[r, c].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / 'comparison_plots.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="IPPO vs Explaboff comparison")
    parser.add_argument('--train', type=int, default=100,
                        help='Training episodes per algorithm when no checkpoint given (default: 100)')
    parser.add_argument('--eval', type=int, default=20,
                        help='Evaluation episodes per algorithm (default: 20)')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--ippo_ckpt', type=str, default=None,
                        help='Path to a pre-trained IPPO checkpoint (.pt). '
                             'If given, training is skipped for IPPO.')
    parser.add_argument('--explaboff_ckpt', type=str, default=None,
                        help='Path to a pre-trained Explaboff checkpoint (.pt). '
                             'If given, training is skipped for Explaboff.')
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path('results/comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    ippo_results, ippo_metrics, train_ippo = run_algorithm(
        'IPPO', config,
        train_episodes=args.train,
        eval_episodes=args.eval,
        checkpoint_path=args.ippo_ckpt,
    )

    explaboff_results, explaboff_metrics, train_explaboff = run_algorithm(
        'Explaboff', config,
        train_episodes=args.train,
        eval_episodes=args.eval,
        checkpoint_path=args.explaboff_ckpt,
    )

    print_comparison(ippo_results, explaboff_results)

    try:
        plot_comparison(
            ippo_metrics, explaboff_metrics,
            train_ippo, train_explaboff,
            out_dir,
        )
    except Exception as e:
        print(f"  Warning: Could not save plot: {e}")

    with open(out_dir / 'comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump({'ippo': ippo_results, 'explaboff': explaboff_results}, f, indent=2)
    print(f"  Results saved to: {out_dir}/comparison_results.json")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n✗ Failed: {e}")
        traceback.print_exc()
        exit(1)
