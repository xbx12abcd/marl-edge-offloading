"""
Edge Computing Network Environment for MARL.
This environment simulates the task offloading problem in edge computing.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
from utils import (
    Task, Device, create_task_batch, 
    calculate_transmission_delay, calculate_computation_time,
    calculate_energy_consumption, compute_fairness_index
)


class EdgeComputingEnv(gym.Env):
    """
    Multi-agent edge computing environment.
    
    This environment represents a network with:
    - Multiple end devices generating computational tasks
    - Multiple edge servers for task offloading
    - Communication network with limited bandwidth
    - Energy constraints for devices
    
    Observation space:
    - Task information (CPU, data, deadline)
    - Device resource status (available CPU, energy)
    - Network state (bandwidth, queue length)
    
    Action space:
    - Binary offloading decision per task
    - Target edge server selection
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the environment.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Extract configuration
        self.num_edge_servers = config['environment']['num_edge_servers']
        self.num_end_devices = config['environment']['num_end_devices']
        self.num_time_slots = config['environment']['num_time_slots']
        self.episode_length = config['marl']['episode_length']
        
        # Task parameters
        self.num_tasks = config['environment']['num_tasks']
        self.task_cpu_range = (
            config['environment']['task_cpu_min'],
            config['environment']['task_cpu_max']
        )
        self.task_data_range = (
            config['environment']['task_data_min'],
            config['environment']['task_data_max']
        )
        self.task_deadline_range = (
            config['environment']['task_deadline_min'],
            config['environment']['task_deadline_max']
        )
        
        # Network parameters
        self.bandwidth_wireless = config['environment']['bandwidth_wireless']
        self.bandwidth_wired = config['environment']['bandwidth_wired']
        self.transmission_delay = config['environment']['transmission_delay']
        
        # Computation resources
        self.device_cpu_capacity = config['environment']['device_cpu_capacity']
        self.server_cpu_capacity = config['environment']['server_cpu_capacity']
        
        # Energy parameters
        self.device_energy_per_cpu = config['environment']['device_energy_per_cpu']
        self.device_energy_per_bit = config['environment']['device_energy_per_bit']
        self.server_energy_per_cpu = config['environment']['server_energy_per_cpu']
        
        # Reward weights
        self.reward_weights = config['reward']
        
        # Initialize devices
        self.end_devices: List[Device] = []
        self.edge_servers: List[Device] = []
        self.all_devices: List[Device] = []
        self._init_devices()
        
        # Initialize tasks
        self.tasks: List[Task] = []
        self.task_queue: List[Task] = []
        
        # Observation and action spaces
        self.state_dim = config['environment']['state_dim']
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Action space: offload decision and target server selection
        self.action_space = spaces.Discrete(self.num_edge_servers + 1)  # 0=local, 1..n=edge servers
        
        # Statistics
        self.current_time = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_energy = 0.0
        self.total_delay = 0.0
        
        # Network topology
        self.network_graph = self._build_network_graph()
    
    def _init_devices(self):
        """Initialize end devices and edge servers."""
        # Initialize end devices
        for i in range(self.num_end_devices):
            device = Device(
                device_id=i,
                device_type='end_device',
                cpu_capacity=self.device_cpu_capacity,
                energy_budget=float('inf'),  # No energy constraint for now
                location=(np.random.uniform(0, 100), np.random.uniform(0, 100))
            )
            self.end_devices.append(device)
            self.all_devices.append(device)
        
        # Initialize edge servers
        for i in range(self.num_edge_servers):
            device = Device(
                device_id=self.num_end_devices + i,
                device_type='edge_server',
                cpu_capacity=self.server_cpu_capacity,
                energy_budget=float('inf'),
                location=(np.random.uniform(0, 100), np.random.uniform(0, 100))
            )
            self.edge_servers.append(device)
            self.all_devices.append(device)
    
    def _build_network_graph(self) -> nx.Graph:
        """Build the network topology graph."""
        G = nx.Graph()
        
        # Add all devices as nodes
        for device in self.all_devices:
            G.add_node(device.device_id)
        
        # Add edges between devices (fully connected for simplicity)
        for i in range(len(self.all_devices)):
            for j in range(i + 1, len(self.all_devices)):
                distance = np.linalg.norm(
                    np.array(self.all_devices[i].location) - 
                    np.array(self.all_devices[j].location)
                )
                G.add_edge(i, j, weight=distance)
        
        return G
    
    def _get_communication_delay(self, src_device_id: int, dst_device_id: int, data_size: float) -> float:
        """Get communication delay between two devices."""
        src_device = self.all_devices[src_device_id]
        dst_device = self.all_devices[dst_device_id]
        
        # Determine bandwidth based on device types
        if src_device.device_type == 'end_device' and dst_device.device_type == 'edge_server':
            bandwidth = self.bandwidth_wireless
        elif src_device.device_type == 'edge_server' and dst_device.device_type == 'end_device':
            bandwidth = self.bandwidth_wireless
        else:
            bandwidth = self.bandwidth_wired
        
        # Calculate distance-based delay
        distance = np.linalg.norm(
            np.array(src_device.location) - np.array(dst_device.location)
        )
        propagation_delay = distance / 1e8  # Light speed in medium
        
        # Calculate transmission delay
        transmission_delay = calculate_transmission_delay(data_size, bandwidth)
        
        return propagation_delay + transmission_delay
    
    def _generate_tasks(self, num_tasks: int, current_time: int) -> List[Task]:
        """Generate new tasks at current time step."""
        tasks = create_task_batch(
            num_tasks=num_tasks,
            cpu_range=self.task_cpu_range,
            data_range=self.task_data_range,
            deadline_range=self.task_deadline_range,
            task_id_start=len(self.tasks)
        )
        
        for task in tasks:
            task.arrival_time = current_time
        
        return tasks
    
    def _get_state(self, task: Task, device: Device) -> np.ndarray:
        """Get normalized state representation for a task at a device."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Task features (normalized)
        state[0] = np.tanh(task.cpu_cycles / 1000.0)
        state[1] = np.tanh(task.data_size / 50.0)
        state[2] = np.tanh(task.deadline / 50.0)
        state[3] = np.tanh((self.current_time - task.arrival_time) / 100.0)
        
        # Current device status
        cpu_util = device.current_cpu_usage / device.cpu_capacity
        state[4] = np.tanh(cpu_util)
        state[5] = np.tanh(device.get_available_cpu() / device.cpu_capacity)
        state[6] = np.tanh(device.energy_consumed / max(device.energy_budget, 1.0))
        
        # Queue information
        queue_length = len(device.tasks_queue)
        state[7] = np.tanh(queue_length / 10.0)
        
        # Network status for each edge server
        for i, edge_server in enumerate(self.edge_servers):
            if i < self.state_dim - 8:
                server_util = edge_server.current_cpu_usage / edge_server.cpu_capacity
                state[8 + i] = np.tanh(server_util)
        
        # Padding
        state = state[:self.state_dim]
        
        return state
    
    def reset(self, seed: Optional[int] = None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset all devices
        for device in self.all_devices:
            device.reset()
        
        # Clear task list and queue
        self.tasks.clear()
        self.task_queue.clear()
        
        # Reset statistics
        self.current_time = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_energy = 0.0
        self.total_delay = 0.0
        
        # Generate initial tasks
        initial_tasks = self._generate_tasks(self.num_tasks, self.current_time)
        self.tasks.extend(initial_tasks)
        self.task_queue.extend(initial_tasks)
        
        return self._get_state(self.task_queue[0], self.end_devices[0]), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Offloading decision (0=local, 1..n=edge server i-1)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Process current task scheduling decision
        if self.task_queue:
            task = self.task_queue.pop(0)
            reward = self._schedule_task(task, action)
        else:
            reward = -1.0
        
        # Simulate one time step
        self._simulate_time_step()
        
        # Check termination
        terminated = self.current_time >= self.episode_length or not self.task_queue
        truncated = False
        
        # Get next observation
        if self.task_queue:
            next_obs = self._get_state(self.task_queue[0], self.end_devices[0])
        else:
            next_obs = np.zeros(self.state_dim, dtype=np.float32)
        
        # Prepare info dict
        info = {
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'total_energy': self.total_energy,
            'total_delay': self.total_delay
        }
        
        return next_obs, float(reward), terminated, truncated, info
    
    def _schedule_task(self, task: Task, action: int) -> float:
        """
        Schedule a task to a device and calculate reward.
        
        Args:
            task: Task to schedule
            action: Target device (0=local, 1..n=edge server)
        
        Returns:
            Reward for this action
        """
        if action == 0:
            # Execute locally on the source device (first end device for now)
            target_device = self.end_devices[0]
        else:
            # Offload to edge server
            target_device = self.edge_servers[min(action - 1, len(self.edge_servers) - 1)]
        
        # Check if device has sufficient resources
        if target_device.get_available_cpu() < task.cpu_cycles:
            # Cannot schedule - queue task
            self.task_queue.append(task)
            reward = -1.0
        else:
            # Schedule task
            task.offloaded_to = target_device.device_id
            target_device.tasks_queue.append(task)
            target_device.current_cpu_usage += task.cpu_cycles
            reward = 0.0  # Successful scheduling reward
        
        return reward
    
    def _simulate_time_step(self):
        """Simulate one time step: process tasks and update device states."""
        self.current_time += 1
        
        # Process tasks on each device
        for device in self.all_devices:
            if device.tasks_queue:
                task = device.tasks_queue[0]
                
                # Calculate execution time
                if device.device_type == 'end_device':
                    # device_cpu_capacity is in million cycles per time slot
                    available_cycles = self.device_cpu_capacity - device.current_cpu_usage
                    if available_cycles >= task.cpu_cycles:
                        execution_time = 1.0  # Complete in this time slot
                    else:
                        execution_time = 2.0  # Need more time
                    energy_per_cpu = self.device_energy_per_cpu
                else:
                    # server_cpu_capacity is in million cycles per time slot
                    available_cycles = self.server_cpu_capacity - device.current_cpu_usage
                    if available_cycles >= task.cpu_cycles:
                        execution_time = 1.0  # Complete in this time slot
                    else:
                        execution_time = 2.0  # Need more time
                    energy_per_cpu = self.server_energy_per_cpu
                
                # If task completes
                if execution_time <= 1.0:  # Simplified: 1 time slot
                    device.tasks_queue.pop(0)
                    task.completed = True
                    task.completion_time = self.current_time
                    
                    # Calculate metrics
                    task.latency = task.completion_time - task.arrival_time
                    task.energy_consumed = calculate_energy_consumption(
                        task.cpu_cycles, energy_per_cpu,
                        task.data_size * 8e6, self.device_energy_per_bit
                    )
                    
                    # Update statistics
                    if not task.is_deadline_missed(task.completion_time):
                        task.deadline_met = True
                        self.completed_tasks += 1
                    else:
                        self.failed_tasks += 1
                    
                    self.total_energy += task.energy_consumed
                    self.total_delay += task.latency
                    device.energy_consumed += task.energy_consumed
                    device.current_cpu_usage = max(0, device.current_cpu_usage - task.cpu_cycles)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        total_tasks = self.completed_tasks + self.failed_tasks
        
        metrics = {
            'task_completion_rate': self.completed_tasks / max(total_tasks, 1),
            'energy_consumption': self.total_energy,
            'average_delay': self.total_delay / max(total_tasks, 1),
            'deadline_miss_rate': self.failed_tasks / max(total_tasks, 1),
        }
        
        # Fairness index
        device_energy_list = [d.energy_consumed for d in self.all_devices]
        metrics['fairness_index'] = compute_fairness_index(device_energy_list)
        
        return metrics
