"""
Task class for edge offloading simulation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Task:
    """Represents a computation task to be offloaded."""
    
    task_id: int
    cpu_cycles: float  # Million cycles
    data_size: float  # MB
    deadline: int  # Time slots
    arrival_time: int = 0  # Time slot when task arrives
    
    # State tracking
    offloaded_to: Optional[int] = None  # Edge server ID or None for local
    completed: bool = False
    completion_time: Optional[int] = None
    energy_consumed: float = 0.0
    latency: float = 0.0
    
    # Performance metrics
    deadline_met: bool = False
    
    def __post_init__(self):
        """Validate task parameters."""
        assert self.cpu_cycles > 0, "CPU cycles must be positive"
        assert self.data_size > 0, "Data size must be positive"
        assert self.deadline > 0, "Deadline must be positive"
    
    def is_deadline_missed(self, completion_time: int) -> bool:
        """Check if task deadline is missed."""
        return completion_time > (self.arrival_time + self.deadline)
    
    def reset(self):
        """Reset task to initial state."""
        self.offloaded_to = None
        self.completed = False
        self.completion_time = None
        self.energy_consumed = 0.0
        self.latency = 0.0
        self.deadline_met = False


@dataclass
class Device:
    """Represents an edge device (end device or edge server)."""
    
    device_id: int
    device_type: str  # 'end_device' or 'edge_server'
    cpu_capacity: float  # Million cycles per time slot
    energy_budget: float  # Joules
    
    # Location (for communication delay calculation)
    location: tuple = field(default_factory=lambda: (0.0, 0.0))
    
    # State tracking
    current_cpu_usage: float = 0.0
    energy_consumed: float = 0.0
    tasks_queue: list = field(default_factory=list)
    
    def __post_init__(self):
        """Validate device parameters."""
        assert self.cpu_capacity > 0, "CPU capacity must be positive"
        assert self.energy_budget > 0, "Energy budget must be positive"
    
    def reset(self):
        """Reset device to initial state."""
        self.current_cpu_usage = 0.0
        self.energy_consumed = 0.0
        self.tasks_queue.clear()
    
    def get_available_cpu(self) -> float:
        """Get available CPU capacity."""
        return max(0, self.cpu_capacity - self.current_cpu_usage)
    
    def get_energy_remaining(self) -> float:
        """Get remaining energy budget."""
        return max(0, self.energy_budget - self.energy_consumed)
