"""
Metadata schema for activation capture - Stage 3B
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from datetime import datetime, timezone


@dataclass
class CaptureMetadata:
    """Metadata associated with each capture step."""
    # Condition information
    condition_id: str
    perturbation_type: str = "clean"  # "clean", "patch", "background", "combined"
    perturbation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Task/trial information
    task_id: Optional[str] = None
    task_name: Optional[str] = None
    episode_id: Optional[int] = None
    trial: Optional[int] = None
    timestep: Optional[int] = None
    camera: str = "primary"  # "primary" or "wrist"
    
    # Model/checkpoint information
    checkpoint_id: Optional[str] = None
    model_name: Optional[str] = None
    run_id: Optional[str] = None
    seed: Optional[int] = None
    
    # Outcome metadata (when available)
    action: Optional[Any] = None
    action_deviation: Optional[float] = None
    success: Optional[bool] = None
    
    # Capture metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
