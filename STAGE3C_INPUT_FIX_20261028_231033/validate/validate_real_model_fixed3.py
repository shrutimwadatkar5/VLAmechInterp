#!/usr/bin/env python3
"""
Stage 3C - Fixed Real Model Validation using actual get_vla_action pathway
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

from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
)
from experiments.robot.robot_utils import get_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM

def check_unnorm_key(cfg, model):
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key

def main():
    print("="*60)
    print("Stage 3C - Fixed Real Model Validation")
    print("="*60)
    
    print("\n[1] GPU Status")
    os.system("nvidia-smi --query-gpu=memory.free,memory.used,utilization.gpu --format=csv,noheader")
    
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
            self.unnorm_key = ""
            self.lora_rank = 0
            self.model_family = "openvla"
            self.task_suite_name = "libero_spatial"
    
    cfg = DummyCfg()
    
    print("\n[2] Loading components...")
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM)
    check_unnorm_key(cfg, vla)
    print(f"unnorm_key: {cfg.unnorm_key}")
    print(f"vla: {type(vla).__name__}, device: {next(vla.parameters()).device}")
    
    print("\n[3] Creating adapter with hooks...")
    adapter = VLAActivationAdapter(vla, action_head=action_head, proprio_projector=proprio_projector)
    print(f"Hooks: {adapter.get_hook_info()}")
    
    print("\n[4] Creating synthetic observation...")
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    wrist_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    from experiments.robot.openvla_utils import resize_image_for_policy
    img_resized = resize_image_for_policy(img, 224)
    wrist_img_resized = resize_image_for_policy(wrist_img, 224)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.zeros(8),
    }
    print("Observation created")
    
    print("\n[5] Running inference with capture...")
    task_label = "pick up the black bowl and place it on the plate"
    metadata = CaptureMetadata(
        condition_id="stage3c_fixed_test_clean",
        perturbation_type="clean",
        task_id="test",
        task_name=task_label,
        checkpoint_id="openvla-7b-oft-finetuned-libero-spatial",
    )
    
    try:
        with torch.no_grad():
            with adapter.capture_step(metadata, step_id=0):
                action = get_vla_action(
                    cfg, vla, processor, observation, task_label,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,
                    use_film=False,
                )
                print(f"Action shape: {np.array(action).shape}")
    except Exception as e:
        print(f"INFERENCE FAILED: {e}")
        import traceback
        traceback.print_exc()
        adapter.remove_hooks()
        return False
    
    capture = adapter.get_current_capture()
    print(f"\n[6] Captured hooks: {list(capture.keys())}")
    for hook_name, tensors in capture.items():
        print(f"  {hook_name}: {len(tensors)} tensor(s)")
        for suffix, tensor in tensors.items():
            print(f"    {suffix}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    
    import glob
    outdirs = glob.glob("/home/shrutiw/projects/openvla-oft/STAGE3C_INPUT_FIX_*/captures")
    if outdirs:
        outdir = Path(outdirs[0])
        torch.save({
            "metadata": metadata.to_dict(),
            "action": action,
            "activations": capture,
            "hook_names": list(capture.keys()),
        }, outdir / "stage3c_fixed_capture.pt")
        print(f"\nSaved to {outdir / 'stage3c_fixed_capture.pt'}")
        loaded = torch.load(outdir / "stage3c_fixed_capture.pt", map_location="cpu", weights_only=False)
        print(f"Loaded back successfully. Keys: {list(loaded.keys())}")
    
    adapter.remove_hooks()
    print("\n[7] Hooks removed")
    print("\n" + "="*60)
    print("SUCCESS - Real model activation capture working")
    print("="*60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
