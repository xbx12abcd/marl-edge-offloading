"""
Edge computing environments for MARL edge offloading.

The canonical implementation is a PettingZoo ParallelEnv used by the main
training script. A compatibility wrapper keeps the older single-agent API
available for low-memory scripts.
"""

from typing import Any, Dict, Optional
import random

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class EdgeOffloadingEnv(ParallelEnv):
    """PettingZoo Parallel API edge offloading environment."""

    metadata = {"name": "edge_offloading_v0"}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()

        num_agents = config["environment"]["num_end_devices"]
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        self.num_edge_servers = config["environment"]["num_edge_servers"]
        self.episode_length = config["marl"]["episode_length"]
        self.state_dim = config["environment"]["state_dim"]

        self.device_cpu_capacity = config["environment"]["device_cpu_capacity"]
        self.server_cpu_capacity = config["environment"]["server_cpu_capacity"]
        self.bandwidth_wireless = config["environment"]["bandwidth_wireless"]
        self.transmission_delay = config["environment"]["transmission_delay"]
        self.device_energy_per_cpu = config["environment"]["device_energy_per_cpu"]
        self.server_energy_per_cpu = config["environment"]["server_energy_per_cpu"]
        self.device_energy_per_bit = config["environment"]["device_energy_per_bit"]

        self.task_cpu_range = (
            config["environment"]["task_cpu_min"],
            config["environment"]["task_cpu_max"],
        )
        self.task_data_range = (
            config["environment"]["task_data_min"],
            config["environment"]["task_data_max"],
        )
        self.task_deadline_range = (
            config["environment"]["task_deadline_min"],
            config["environment"]["task_deadline_max"],
        )

        self.reward_weights = config["reward"]
        self._adjacency_matrix = self._compute_distance_matrix()

        self._action_space = spaces.Discrete(self.num_edge_servers + 1)
        self._observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        self.observation_spaces = {
            agent: self._observation_space for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self._action_space for agent in self.possible_agents
        }

        self.current_step = 0
        self._reset_internal()
        self.render_mode = None

    @property
    def num_agents(self) -> int:
        return len(self.possible_agents)

    def _validate_config(self):
        for key in ["environment", "marl", "reward"]:
            if key not in self.config:
                raise ValueError(f"Missing config key: {key}")

    def _compute_distance_matrix(self) -> np.ndarray:
        n = len(self.possible_agents) + self.num_edge_servers
        adj = np.zeros((n, n), dtype=np.float32)
        locations = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = locations[i][0] - locations[j][0]
                    dy = locations[i][1] - locations[j][1]
                    adj[i, j] = np.sqrt(dx * dx + dy * dy)
        return adj

    def _reset_internal(self):
        self.tasks = {
            agent: self._generate_task(int(agent.split("_")[1]))
            for agent in self.possible_agents
        }
        self.server_loads = np.zeros(self.num_edge_servers, dtype=np.float32)
        self.server_energies = np.zeros(self.num_edge_servers, dtype=np.float32)
        self.agent_queues = {agent: [] for agent in self.possible_agents}
        self.agent_energies = {agent: 100.0 for agent in self.possible_agents}
        self.current_step = 0
        self.episode_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.completed_tasks = {agent: 0 for agent in self.possible_agents}
        self.failed_tasks = {agent: 0 for agent in self.possible_agents}
        self.total_energy = 0.0
        self.total_delay = 0.0

    def _generate_task(self, agent_id: int = 0) -> Dict[str, Any]:
        return {
            "id": agent_id,
            "cpu_cycles": random.uniform(
                self.task_cpu_range[0], self.task_cpu_range[1]
            ),
            "data_size": random.uniform(
                self.task_data_range[0], self.task_data_range[1]
            ),
            "deadline": random.randint(
                self.task_deadline_range[0], self.task_deadline_range[1]
            ),
            "arrival_time": self.current_step,
            "completed": False,
            "completion_time": None,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._reset_internal()
        self.agents = self.possible_agents[:]
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_observation(self, agent: str) -> np.ndarray:
        obs = np.zeros(self.state_dim, dtype=np.float32)
        task = self.tasks[agent]
        obs[0] = np.tanh(task["cpu_cycles"] / 1000.0)
        obs[1] = np.tanh(task["data_size"] / 50.0)
        obs[2] = np.tanh(task["deadline"] / 50.0)
        obs[3] = np.tanh((self.current_step - task["arrival_time"]) / 100.0)
        obs[4] = np.tanh(len(self.agent_queues[agent]) / 10.0)
        for i in range(min(self.num_edge_servers, self.state_dim - 5)):
            obs[5 + i] = np.tanh(self.server_loads[i])
        obs[-2] = np.tanh(self.current_step / max(self.episode_length, 1))
        remaining = sum(1 for item in self.tasks.values() if not item["completed"])
        obs[-1] = np.tanh(remaining / max(len(self.tasks), 1))
        return obs

    def step(self, actions: Dict[str, int]):
        if not set(actions.keys()).issubset(set(self.agents)):
            raise ValueError(
                "Action keys must be a subset of active agents; extra keys are not "
                "allowed and missing agents default to action 0"
            )

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        completed_flags = {}
        for agent in self.agents:
            action = actions.get(agent, 0)
            reward = self._execute_action(agent, action)
            completed_flags[agent] = self.tasks[agent]["completed"]
            rewards[agent] = reward
            terminations[agent] = False
            truncations[agent] = False
            self.episode_rewards[agent] += reward

        self.server_loads *= 0.95
        self.server_loads = np.clip(self.server_loads, 0, 1)

        self.current_step += 1

        for agent in self.agents:
            if self.tasks[agent]["completed"]:
                self.tasks[agent] = self._generate_task(int(agent.split("_")[1]))

        if self.current_step >= self.episode_length:
            for agent in self.agents:
                truncations[agent] = True

        for agent in self.agents:
            observations[agent] = self._get_observation(agent)
            infos[agent] = {
                "completed": completed_flags[agent],
                "server_loads": self.server_loads.copy(),
            }

        return observations, rewards, terminations, truncations, infos

    def _execute_action(self, agent: str, action: int) -> float:
        task = self.tasks[agent]
        if task["completed"]:
            return 0.0

        if action == 0:
            return self._local_execution(agent, task)

        server_id = max(0, min(self.num_edge_servers - 1, action - 1))
        return self._offload_execution(agent, task, server_id)

    def _local_execution(self, agent: str, task: Dict[str, Any]) -> float:
        compute_time = task["cpu_cycles"] / max(self.device_cpu_capacity, 1e-6)
        energy = (
            task["cpu_cycles"] * self.device_energy_per_cpu
            + task["data_size"] * self.device_energy_per_bit
        )

        self.total_energy += energy
        self.total_delay += compute_time

        success = compute_time <= task["deadline"]
        if success:
            self.completed_tasks[agent] += 1
            task["completed"] = True
            reward = (
                self.reward_weights.get("task_completion_weight", 1.0)
                - abs(self.reward_weights.get("energy_penalty_per_unit", 0.001))
                * energy
                - abs(self.reward_weights.get("deadline_penalty_weight", 0.1))
                * compute_time
            )
        else:
            self.failed_tasks[agent] += 1
            reward = self.reward_weights.get("deadline_miss_penalty", -10.0)
        return float(reward)

    def _offload_execution(
        self, agent: str, task: Dict[str, Any], server_id: int
    ) -> float:
        agent_index = int(agent.split("_")[1])
        distance = self._adjacency_matrix[
            agent_index, len(self.possible_agents) + server_id
        ]
        tx_delay = (
            task["data_size"] / max(self.bandwidth_wireless, 1e-6)
            + self.transmission_delay
            + distance * 1e-4
        )
        queue_penalty = self.server_loads[server_id] * 2.0
        compute_time = task["cpu_cycles"] / max(self.server_cpu_capacity, 1e-6)
        total_delay = tx_delay + queue_penalty + compute_time
        energy = (
            task["data_size"] * self.device_energy_per_bit
            + task["cpu_cycles"] * self.server_energy_per_cpu
        )

        self.server_loads[server_id] += min(
            0.2, task["cpu_cycles"] / max(self.server_cpu_capacity, 1e-6)
        )
        self.server_energies[server_id] += energy
        self.total_energy += energy
        self.total_delay += total_delay

        success = total_delay <= task["deadline"]
        if success:
            self.completed_tasks[agent] += 1
            task["completed"] = True
            reward = (
                self.reward_weights.get("task_completion_weight", 1.0)
                + self.reward_weights.get("fairness_weight", 0.1)
                * (1.0 - float(np.std(self.server_loads)))
                - abs(self.reward_weights.get("energy_penalty_per_unit", 0.001))
                * energy
                - abs(self.reward_weights.get("deadline_penalty_weight", 0.1))
                * total_delay
            )
        else:
            self.failed_tasks[agent] += 1
            reward = self.reward_weights.get("deadline_miss_penalty", -10.0)
        return float(reward)

    def get_metrics(self) -> Dict[str, float]:
        completed = sum(self.completed_tasks.values())
        failed = sum(self.failed_tasks.values())
        total = completed + failed
        completion_rate = completed / max(total, 1)
        deadline_miss_rate = failed / max(total, 1)
        avg_delay = self.total_delay / max(total, 1)
        loads = self.server_loads + 1e-8
        fairness = (np.sum(loads) ** 2) / (len(loads) * np.sum(loads ** 2))
        return {
            "task_completion_rate": float(completion_rate),
            "energy_consumption": float(self.total_energy),
            "average_delay": float(avg_delay),
            "deadline_miss_rate": float(deadline_miss_rate),
            "fairness_index": float(fairness),
        }

    def render(self):
        return None

    def close(self):
        return None


class EdgeComputingEnv:
    """Compatibility wrapper for older single-agent scripts."""

    def __init__(self, config: Dict[str, Any]):
        self._env = EdgeOffloadingEnv(config)
        self.num_end_devices = len(self._env.possible_agents)
        self.num_edge_servers = self._env.num_edge_servers
        self.num_time_slots = config["environment"].get(
            "num_time_slots", self._env.episode_length
        )
        self.num_tasks = config["environment"].get("num_tasks", self.num_end_devices)
        self.action_space = self._env.action_spaces[self._env.possible_agents[0]]
        self.observation_space = self._env.observation_spaces[
            self._env.possible_agents[0]
        ]
        self.metadata = getattr(self._env, "metadata", {})

    def reset(self, seed: Optional[int] = None):
        observations, infos = self._env.reset(seed=seed)
        first_agent = self._env.possible_agents[0]
        return observations[first_agent], infos[first_agent]

    def step(self, action: int):
        first_agent = self._env.possible_agents[0]
        actions = {agent: 0 for agent in self._env.possible_agents}
        actions[first_agent] = action
        observations, rewards, terminations, truncations, infos = self._env.step(
            actions
        )
        return (
            observations[first_agent],
            rewards[first_agent],
            terminations[first_agent],
            truncations[first_agent],
            infos[first_agent],
        )

    def get_metrics(self) -> Dict[str, float]:
        return self._env.get_metrics()

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()
