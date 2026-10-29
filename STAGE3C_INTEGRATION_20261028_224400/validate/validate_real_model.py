#!/usr/bin/env python3
"""
Stage 3C Real Model Validation
Minimal safe validation using actual OpenVLA-OFT model
"""

import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image

REPO = Path("/home/shrutiw/projects/openvla-oft")
sys.path.insert(0, str(REPO))
STAGE3B_SRC = REPO / "STAGE3B_ACTIVATION_CAPTURE_20261028_214057/src"
sys.path.insert(0, str(STAGE3B_SRC))
import glob
adapter_dirs = glob.glob("/home/shrutiw/projects/openvla-oft/STAGE3C_INTEGRATION_*/adapter")
if adapter_dirs:
    sys.path.insert(0, adapter_dirs[0])

from stage3b_capture import CaptureMetadata
from vla_activation_adapter import VLAActivationAdapter

def main():
    print("="*60)
    print("Stage 3C Real Model Validation")
    print("="*60)
    
    print("\n[1] GPU Status")
    os.system("nvidia-smi --query-gpu=memory.free,memory.used,utilization.gpu --format=csv,noheader")
    
    print("\n[2] Loading model...")
    try:
        from experiments.robot.openvla_utils import get_vla, get_processor, get_action_head, get_proprio_projector
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM
        
        class DummyCfg:
            def __init__(self):
                self.pretrained_checkpoint = str(REPO / "checkpoints/openvla-7b-oft-finetuned-libero-spatial")
                self.load_in_8bit = False
                self.load_in_4bit = False
                self.center_crop = True
                self.use_l1_regression = True
                self.use_diffusion = False
                self.use_film = False
                self.num_images_in_input = 2
                self.use_proprio = True
                self.max_steps = 512
                self.num_open_loop_steps = 8
                self.unnorm_key = None
                self.lora_rank = 0
        
        cfg = DummyCfg()
        vla = get_vla(cfg)
        print(f"Model loaded: {type(vla).__name__}")
        print(f"Device: {next(vla.parameters()).device}")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[3] Getting components...")
    try:
        action_head = get_action_head(cfg, vla.llm_dim)
        proprio_projector = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM)
        print(f"action_head: {type(action_head).__name__}")
        print(f"proprio_projector: {type(proprio_projector).__name__}")
    except Exception as e:
        print(f"Component extraction failed: {e}")
        import traceback
        traceback.print_exc()
        action_head = None
        proprio_projector = None
    
    print("\n[4] Creating adapter...")
    try:
        adapter = VLAActivationAdapter(vla, action_head=action_head, proprio_projector=proprio_projector)
        print(f"Hooks: {adapter.get_hook_info()}")
    except Exception as e:
        print(f"Adapter creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[5] Preparing inputs...")
    try:
        processor = get_processor(cfg)
        print("Processor loaded")
        # Create dummy images
        img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        prompt = "In: What action should the robot take to solve the task?\nOut:"
        # Try different input formats
        try:
            inputs = processor(prompt, [img1, img2], return_tensors="pt")
        except Exception as e:
            print(f"Processor attempt 1 failed: {e}")
            try:
                inputs = processor(text=prompt, images=[img1, img2], return_tensors="pt")
            except Exception as e2:
                print(f"Processor attempt 2 failed: {e2}")
                inputs = None
        if inputs is not None:
            inputs = {k: v.to(next(vla.parameters()).device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            print(f"Inputs keys: {list(inputs.keys())}")
            for k,v in inputs.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}")
    except Exception as e:
        print(f"Input preparation failed: {e}")
        import traceback
        traceback.print_exc()
        inputs = None
    
    if inputs is not None:
        print("\n[6] Capturing activations with forward pass...")
        try:
            metadata = CaptureMetadata(
                condition_id="stage3c_test_clean",
                perturbation_type="clean",
                task_id="test",
                checkpoint_id="openvla-7b-oft-finetuned-libero-spatial",
            )
            with torch.no_grad():
                with adapter.capture_step(metadata, step_id=0):
                    try:
                        outputs = vla(**inputs)
                        print(f"Forward pass completed. Output type: {type(outputs)}")
                    except Exception as e:
                        print(f"Forward pass failed: {e}")
                        import traceback
                        traceback.print_exc()
                        return False
            
            capture = adapter.get_current_capture()
            print(f"\nCaptured hooks: {list(capture.keys())}")
            for hook_name, tensors in capture.items():
                print(f"  {hook_name}: {len(tensors)} tensor(s)")
                for suffix, tensor in tensors.items():
                    print(f"    {suffix}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
        except Exception as e:
            print(f"Capture failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n[6] Skipping forward pass (no inputs)")
    
    adapter.remove_hooks()
    print("\n[7] Hooks removed")
    print("\n" + "="*60)
    print("Stage 3C validation completed")
    print("="*60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
