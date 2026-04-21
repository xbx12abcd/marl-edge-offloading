"""
Utility functions for the MARL edge offloading project.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import random
import torch

# Import Task class
from .task_device import Task


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file with automatic type conversion."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = _convert_scientific_notation(config)
    return config


def _convert_scientific_notation(obj):
    """Recursively convert scientific notation strings to floats."""
    import re
    scientific_pattern = re.compile(r"^[+-]?(\d+\.?\d*|\d*\.?\d+)[eE][+-]?\d+$")
    if isinstance(obj, dict):
        return {k: _convert_scientific_notation(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_scientific_notation(item) for item in obj]
    elif isinstance(obj, str):
        if scientific_pattern.match(obj):
            return float(obj)
        return obj
    else:
        return obj


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_task_batch(
    num_tasks: int,
    cpu_range: Tuple[float, float],
    data_range: Tuple[float, float],
    deadline_range: Tuple[int, int],
    task_id_start: int = 0
) -> List[Task]:
    """Create a batch of computation tasks."""
    tasks = []
    for i in range(num_tasks):
        task = Task(
            task_id=task_id_start + i,
            cpu_cycles=np.random.uniform(cpu_range[0], cpu_range[1]),  # Million cycles
            data_size=np.random.uniform(data_range[0], data_range[1]),  # MB
            deadline=np.random.randint(deadline_range[0], deadline_range[1]),  # Time slots
            arrival_time=0,  # Will be set later
        )
        tasks.append(task)
    return tasks


def calculate_transmission_delay(
    data_size: float,
    bandwidth: float,
    additional_delay: float = 0.0
) -> float:
    """
    Calculate transmission delay.
    
    Args:
        data_size: Data size in MB
        bandwidth: Bandwidth in Mbps
        additional_delay: Additional network delay
    
    Returns:
        Delay in seconds
    """
    transmission_time = (data_size * 8) / (bandwidth * 1e6)  # Convert to seconds
    return transmission_time + additional_delay


def calculate_computation_time(
    cpu_cycles: float,
    cpu_frequency: float
) -> float:
    """
    Calculate computation time.
    
    Args:
        cpu_cycles: CPU cycles in millions
        cpu_frequency: CPU frequency in MHz (Million cycles per second)
    
    Returns:
        Computation time in seconds
    """
    return (cpu_cycles * 1e6) / (cpu_frequency * 1e6)


def calculate_energy_consumption(
    cpu_cycles: float,
    energy_per_cycle: float,
    data_transmission: float = 0.0,
    energy_per_bit: float = 0.0
) -> float:
    """
    Calculate total energy consumption.
    
    Args:
        cpu_cycles: CPU cycles in millions
        energy_per_cycle: Energy per cycle in Joules
        data_transmission: Data transmitted in bits
        energy_per_bit: Energy per bit in Joules
    
    Returns:
        Total energy in Joules
    """
    cpu_energy = cpu_cycles * 1e6 * energy_per_cycle
    transmission_energy = data_transmission * energy_per_bit
    return cpu_energy + transmission_energy


class MetricsCollector:
    """Collect and aggregate metrics during training."""
    
    def __init__(self):
        self.episodes = []
        self.current_episode = {
            'task_completion_rate': [],
            'energy_consumption': [],
            'average_delay': [],
            'deadline_miss_rate': [],
            'reward': [],
            'fairness_index': []
        }
    
    def add_step_metrics(self, metrics: Dict[str, float]):
        """Add metrics for a single step."""
        for key, value in metrics.items():
            if key in self.current_episode:
                self.current_episode[key].append(value)
    
    def finalize_episode(self):
        """Finalize current episode and compute aggregate metrics."""
        episode_summary = {}
        for key, values in self.current_episode.items():
            if values:
                episode_summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        self.episodes.append(episode_summary)
        self.current_episode = {
            'task_completion_rate': [],
            'energy_consumption': [],
            'average_delay': [],
            'deadline_miss_rate': [],
            'reward': [],
            'fairness_index': []
        }
    
    def get_statistics(self, last_n: int = 100) -> Dict[str, Any]:
        """Get aggregated statistics over last N episodes."""
        if not self.episodes:
            return {}
        
        episodes_to_analyze = self.episodes[-last_n:] if last_n > 0 else self.episodes
        stats = {}
        
        for key in episodes_to_analyze[0].keys():
            means = [ep[key]['mean'] for ep in episodes_to_analyze]
            stats[key] = {
                'mean': np.mean(means),
                'std': np.std(means),
                'min': np.min(means),
                'max': np.max(means)
            }
        
        return stats


def compute_fairness_index(values: List[float]) -> float:
    """
    Compute fairness index using Jain's fairness index.
    
    Args:
        values: List of individual metrics (e.g., energy per agent)
    
    Returns:
        Fairness index in range [0, 1]
    """
    if len(values) == 0:
        return 1.0
    if len(values) == 1:
        return 1.0
    
    values = np.array(values)
    n = len(values)
    numerator = (np.sum(values)) ** 2
    denominator = n * np.sum(values ** 2)
    
    return numerator / denominator if denominator > 0 else 0.0


def normalize_observation(obs: np.ndarray, obs_mean: np.ndarray = None, obs_std: np.ndarray = None) -> np.ndarray:
    """Normalize observation using running statistics."""
    if obs_mean is not None and obs_std is not None:
        return (obs - obs_mean) / (obs_std + 1e-8)
    return obs


def get_device() -> torch.device:
    """Get the appropriate device (GPU or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
