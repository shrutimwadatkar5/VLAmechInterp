"""
Configuration for activation capture - Stage 3B
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal


@dataclass
class HookSpec:
    """Specification for a hook point verified by Stage 3A audit."""
    pattern: str
    name: Optional[str] = None
    reducer: Literal["none", "mean_seq", "last_token", "flatten", "mean"] = "mean_seq"
    dtype: Literal["keep", "float16", "float32"] = "float16"
    include_input: bool = False
    max_elements: Optional[int] = None
    detach: bool = True
    store_on_cpu: bool = True


@dataclass
class ActivationCaptureConfig:
    """Configuration for activation capture."""
    hook_specs: List[HookSpec] = field(default_factory=lambda: [
        HookSpec(pattern="vision_backbone*", name="vision_backbone"),
        HookSpec(pattern="projector", name="projector"),
        HookSpec(pattern="language_model", name="language_model"),
        HookSpec(pattern="language_model.model.layers.*", name="language_model_layers"),
        HookSpec(pattern="language_model.model.norm", name="language_model_norm"),
    ])
    extra_modules: Dict[str, Any] = field(default_factory=dict)
    
    store_on_cpu: bool = True
    detach: bool = True
    store_tensors: bool = True
    max_tensors_per_step: Optional[int] = None
    
    strict: bool = True
    enabled: bool = True
    
    output_dir: Optional[str] = None
    save_manifest: bool = True
    save_metadata: bool = True
