"""
Utilities for activation capture - Stage 3B
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_manifest(manifest_path: str) -> list:
    """Load capture manifest."""
    records = []
    with Path(manifest_path).open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compare_captures(clean_capture: Dict, perturbed_capture: Dict) -> Dict[str, Any]:
    """Compare clean vs perturbed activations."""
    import torch
    results = {}
    for key in clean_capture:
        if key in perturbed_capture:
            results[key] = {}
            for suffix in clean_capture[key]:
                if suffix in perturbed_capture[key]:
                    c = clean_capture[key][suffix].float()
                    p = perturbed_capture[key][suffix].float()
                    if c.shape == p.shape:
                        diff = p - c
                        results[key][suffix] = {
                            "l2": torch.norm(diff).item(),
                            "cosine": torch.nn.functional.cosine_similarity(
                                c.flatten().unsqueeze(0), p.flatten().unsqueeze(0)
                            ).item(),
                            "shape": list(c.shape),
                        }
    return results
