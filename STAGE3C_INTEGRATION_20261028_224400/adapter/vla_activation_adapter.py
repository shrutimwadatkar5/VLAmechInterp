#!/usr/bin/env python3
"""
Stage 3C Integration Adapter - OpenVLA-OFT Activation Capture
Minimal, reversible wrapper. Uses only verified hook locations from Stage 3A.
Does not modify existing files.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add Stage 3B source to path
STAGE3B_SRC = Path("/home/shrutiw/projects/openvla-oft/STAGE3B_ACTIVATION_CAPTURE_20261028_214057/src")
sys.path.insert(0, str(STAGE3B_SRC))

import torch
from stage3b_capture import ActivationRecorder, ActivationCaptureConfig, HookSpec, CaptureMetadata


class VLAActivationAdapter:
    """
    Wrapper for OpenVLA-OFT model to capture activations.
    Uses verified hook locations from Stage 3A audit.
    """
    
    def __init__(self, vla, action_head=None, proprio_projector=None, config=None):
        self.vla = vla
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        
        if config is None:
            config = ActivationCaptureConfig(
                hook_specs=[
                    HookSpec(pattern="vision_backbone*", name="vision_backbone", reducer="flatten", dtype="float16"),
                    HookSpec(pattern="projector", name="projector", reducer="mean_seq", dtype="float16"),
                    HookSpec(pattern="language_model.model.norm", name="language_model_norm", reducer="mean_seq", dtype="float16"),
                    HookSpec(pattern="language_model.model.layers.*", name="language_model_layers", reducer="mean_seq", dtype="float16"),
                ],
                strict=False,  # Don't fail if some hooks don't match
                store_on_cpu=True,
                detach=True,
            )
        self.config = config
        
        extra_modules = {}
        if action_head is not None:
            extra_modules["action_head"] = action_head
        if proprio_projector is not None:
            extra_modules["proprio_projector"] = proprio_projector
        
        self.recorder = ActivationRecorder(vla, self.config, extra_modules=extra_modules if extra_modules else None)
    
    def get_hook_info(self):
        return self.recorder.get_hook_info()
    
    def capture_step(self, metadata: CaptureMetadata, step_id: int = None):
        return self.recorder.capture_step(metadata, step_id)
    
    def get_current_capture(self):
        return self.recorder.get_current_capture()
    
    def disable(self):
        self.recorder.disable()
    
    def enable(self):
        self.recorder.enable()
    
    def remove_hooks(self):
        self.recorder.remove_hooks()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False
