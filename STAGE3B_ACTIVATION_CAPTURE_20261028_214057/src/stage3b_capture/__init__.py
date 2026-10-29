"""
Stage 3B Activation Capture Infrastructure

Minimal, reusable PyTorch activation capture for OpenVLA-OFT LIBERO experiments.
Only uses hook locations verified by Stage 3A audit.
"""

from .config import ActivationCaptureConfig, HookSpec
from .recorder import ActivationRecorder
from .metadata import CaptureMetadata

__version__ = "0.1.0"
__all__ = ["ActivationCaptureConfig", "HookSpec", "ActivationRecorder", "CaptureMetadata"]
