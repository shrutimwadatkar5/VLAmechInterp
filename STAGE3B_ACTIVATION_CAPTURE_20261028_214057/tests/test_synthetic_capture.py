#!/usr/bin/env python3
"""
Synthetic test for Stage 3B activation capture.
Uses only PyTorch - no OpenVLA model, no LIBERO.
"""

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
OUTDIR = Path(__file__).parent.parent
sys.path.insert(0, str(OUTDIR / "src"))

from stage3b_capture import ActivationRecorder, ActivationCaptureConfig, HookSpec, CaptureMetadata


class SyntheticModel(nn.Module):
    """Synthetic model mimicking OpenVLA structure names."""
    def __init__(self):
        super().__init__()
        self.vision_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14))
        )
        self.projector = nn.Linear(64, 512)
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([
            nn.Linear(512, 512),
            nn.Linear(512, 512),
        ])
        self.language_model.model.norm = nn.LayerNorm(512)
        self.action_head = nn.Linear(512, 7)
        self.proprio_projector = nn.Linear(8, 512)
    
    def forward(self, x, proprio=None):
        # Vision
        v = self.vision_backbone(x)  # B, C, H, W
        v = v.flatten(2).transpose(1, 2)  # B, N, C
        v = self.projector(v)  # B, N, 512
        
        # Language model
        h = v
        for layer in self.language_model.model.layers:
            h = layer(h)
        h = self.language_model.model.norm(h)
        
        # Action head
        out = self.action_head(h[:, -1])
        return out


def test_hook_registration():
    """Test 1: Hook registration with verified patterns"""
    model = SyntheticModel()
    config = ActivationCaptureConfig(
        hook_specs=[
            HookSpec(pattern="vision_backbone*", name="vision_backbone", reducer="flatten"),
            HookSpec(pattern="projector", name="projector", reducer="mean_seq"),
            HookSpec(pattern="language_model.model.layers.*", name="language_model_layers", reducer="mean_seq"),
            HookSpec(pattern="language_model.model.norm", name="language_model_norm", reducer="mean_seq"),
        ],
        strict=True
    )
    recorder = ActivationRecorder(model, config)
    info = recorder.get_hook_info()
    print(f"Registered hooks: {info}")
    assert len(info) >= 3, f"Expected >=3 hooks, got {len(info)}"
    recorder.remove_hooks()
    print("✓ Test 1: Hook registration passed")
    return True


def test_capture_with_metadata():
    """Test 2: Capture with metadata association"""
    model = SyntheticModel()
    model.eval()
    config = ActivationCaptureConfig(
        hook_specs=[
            HookSpec(pattern="vision_backbone*", name="vision_backbone", reducer="flatten"),
            HookSpec(pattern="projector", name="projector", reducer="mean_seq"),
            HookSpec(pattern="language_model.model.norm", name="language_model_norm", reducer="mean_seq"),
        ],
        strict=True,
        store_on_cpu=True,
        detach=True,
    )
    recorder = ActivationRecorder(model, config)
    
    x = torch.randn(1, 3, 224, 224)
    metadata = CaptureMetadata(
        condition_id="test_clean_001",
        perturbation_type="clean",
        task_id="libero_spatial_task_0",
        episode_id=1,
        trial=1,
        timestep=0,
        camera="primary",
        checkpoint_id="openvla-7b-oft-finetuned-libero-spatial",
        seed=7,
    )
    
    with torch.no_grad():
        with recorder.capture_step(metadata, step_id=0):
            out = model(x)
    
    capture = recorder.get_current_capture()
    print(f"Captured keys: {list(capture.keys())}")
    assert len(capture) >= 2, "Should capture at least 2 hooks"
    for k, v in capture.items():
        for sk, tensor in v.items():
            assert tensor.is_cpu, f"Tensor should be on CPU: {k}.{sk}"
            assert not tensor.requires_grad, f"Tensor should not require grad: {k}.{sk}"
    
    recorder.remove_hooks()
    print("✓ Test 2: Capture with metadata passed")
    return True


def test_extra_modules():
    """Test 3: Extra modules (action_head, proprio_projector)"""
    model = SyntheticModel()
    config = ActivationCaptureConfig(
        hook_specs=[
            HookSpec(pattern="projector", name="projector"),
        ],
        extra_modules={"action_head": model.action_head, "proprio_projector": model.proprio_projector},
        strict=False,
    )
    recorder = ActivationRecorder(model, config)
    info = recorder.get_hook_info()
    print(f"Hooks with extras: {info}")
    recorder.remove_hooks()
    print("✓ Test 3: Extra modules passed")
    return True


def test_capture_disabled_equivalence():
    """Test 4: Output equivalence with capture enabled vs disabled"""
    model = SyntheticModel()
    model.eval()
    config = ActivationCaptureConfig(
        hook_specs=[HookSpec(pattern="vision_backbone*", name="vision_backbone")],
        strict=True,
    )
    
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out1 = model(x.clone())
    
    recorder = ActivationRecorder(model, config)
    recorder.disable()
    with torch.no_grad():
        out2 = model(x.clone())
    recorder.remove_hooks()
    
    assert torch.allclose(out1, out2, atol=1e-6), "Outputs differ with capture disabled"
    print("✓ Test 4: Output equivalence passed")
    return True


def test_cleanup():
    """Test 5: Clean hook removal"""
    model = SyntheticModel()
    config = ActivationCaptureConfig(
        hook_specs=[HookSpec(pattern="projector", name="projector")],
    )
    recorder = ActivationRecorder(model, config)
    assert len(recorder._hooks) > 0
    recorder.remove_hooks()
    assert len(recorder._hooks) == 0
    assert not recorder.enabled
    # Forward pass should still work
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape[0] == 1
    print("✓ Test 5: Cleanup passed")
    return True


def test_save_to_disk():
    """Test 6: Save to disk with manifest"""
    model = SyntheticModel()
    model.eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ActivationCaptureConfig(
            hook_specs=[HookSpec(pattern="projector", name="projector", reducer="mean_seq")],
            output_dir=tmpdir,
            save_manifest=True,
        )
        recorder = ActivationRecorder(model, config)
        x = torch.randn(1, 3, 224, 224)
        metadata = CaptureMetadata(condition_id="test_001", perturbation_type="clean")
        with torch.no_grad():
            with recorder.capture_step(metadata, step_id=0):
                model(x)
        recorder.remove_hooks()
        
        # Check files
        step_file = Path(tmpdir) / "step_000000.pt"
        manifest_file = Path(tmpdir) / "manifest.jsonl"
        assert step_file.exists()
        assert manifest_file.exists()
        
        # Load and verify
        payload = torch.load(step_file, map_location="cpu", weights_only=False)
        assert "activations" in payload
        assert "metadata" in payload
        assert payload["metadata"]["condition_id"] == "test_001"
        print("✓ Test 6: Save to disk with manifest passed")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 3B Synthetic Tests")
    print("=" * 60)
    try:
        test_hook_registration()
        test_capture_with_metadata()
        test_extra_modules()
        test_capture_disabled_equivalence()
        test_cleanup()
        test_save_to_disk()
        print("=" * 60)
        print("ALL TESTS PASSED (synthetic)")
        print("Note: These tests do not validate the full OpenVLA-OFT pipeline")
        print("=" * 60)
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
