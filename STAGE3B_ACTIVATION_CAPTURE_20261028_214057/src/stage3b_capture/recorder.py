"""
Activation recorder with verified hook locations - Stage 3B
"""

import fnmatch
import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import torch
from torch import nn

from .config import ActivationCaptureConfig, HookSpec
from .metadata import CaptureMetadata


def _matches(name: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return re.search(pattern[3:], name) is not None
    if any(char in pattern for char in "*?[]"):
        return fnmatch.fnmatchcase(name, pattern)
    return name == pattern or name.endswith("." + pattern)


def _iter_modules(
    root_module: nn.Module,
    extra_modules: Optional[Mapping[str, nn.Module]] = None,
) -> Iterator[Tuple[str, nn.Module]]:
    for name, module in root_module.named_modules():
        if name:
            yield name, module
    for prefix, module in (extra_modules or {}).items():
        prefix = str(prefix)
        for name, child in module.named_modules():
            full_name = prefix if not name else f"{prefix}.{name}"
            yield full_name, child


def _extract_tensors(value: Any, prefix: str) -> Iterator[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix, value
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _extract_tensors(child, f"{prefix}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _extract_tensors(child, f"{prefix}.{index}")
        return
    to_tuple = getattr(value, "to_tuple", None)
    if callable(to_tuple):
        try:
            converted = to_tuple()
        except Exception:
            return
        yield from _extract_tensors(converted, prefix)


def _reduce_tensor(tensor: torch.Tensor, spec: HookSpec) -> torch.Tensor:
    if getattr(spec, "detach", True):
        tensor = tensor.detach()
    if tensor.is_floating_point():
        if spec.dtype == "float16":
            tensor = tensor.half()
        elif spec.dtype == "float32":
            tensor = tensor.float()
    if getattr(spec, "store_on_cpu", True):
        tensor = tensor.cpu()
    tensor = tensor.contiguous()

    if spec.reducer == "none":
        reduced = tensor
    elif spec.reducer == "mean_seq":
        reduced = tensor
        if tensor.ndim >= 3:
            dims = tuple(range(1, tensor.ndim - 1))
            if dims:
                reduced = tensor.mean(dim=dims)
    elif spec.reducer == "last_token":
        reduced = tensor
        if tensor.ndim >= 3:
            reduced = tensor[:, -1]
        elif tensor.ndim == 2:
            reduced = tensor[-1]
    elif spec.reducer == "flatten":
        if tensor.ndim == 0:
            reduced = tensor.reshape(1)
        elif tensor.ndim >= 2:
            reduced = tensor.reshape(tensor.shape[0], -1)
        else:
            reduced = tensor.reshape(1, -1)
    elif spec.reducer == "mean":
        reduced = tensor.mean()
    else:
        raise ValueError(f"Unsupported reducer: {spec.reducer!r}")

    if spec.max_elements is not None and reduced.numel() > spec.max_elements:
        raise ValueError(
            f"Captured tensor has {reduced.numel()} elements, "
            f"above max_elements={spec.max_elements}"
        )
    return reduced.contiguous()


class ActivationRecorder:
    def __init__(
        self,
        root_module: nn.Module,
        config: Optional[ActivationCaptureConfig] = None,
        extra_modules: Optional[Mapping[str, nn.Module]] = None,
    ) -> None:
        from .config import ActivationCaptureConfig
        self.root_module = root_module
        self.config = config or ActivationCaptureConfig()
        if extra_modules:
            self.config.extra_modules = dict(extra_modules)
        
        self.enabled = self.config.enabled
        self._capture: Dict[str, Dict[str, torch.Tensor]] = {}
        self._hooks: List[Any] = []
        self._bindings: Dict[str, HookSpec] = {}
        self._current_metadata: Optional[CaptureMetadata] = None
        self._current_step: Optional[int] = None
        self._last_capture: Dict[str, Dict[str, torch.Tensor]] = {}
        
        if self.config.output_dir:
            self.output_dir = Path(self.config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._manifest_path = self.output_dir / "manifest.jsonl"
        else:
            self.output_dir = None
            self._manifest_path = None
        
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        matched = 0
        for name, module in _iter_modules(self.root_module, self.config.extra_modules):
            for spec in self.config.hook_specs:
                if _matches(name, spec.pattern):
                    hook_name = spec.name or name
                    self._bindings[hook_name] = spec
                    handle = module.register_forward_hook(self._make_hook(hook_name, spec))
                    self._hooks.append(handle)
                    matched += 1
                    break
        
        if self.config.strict and matched == 0:
            raise ValueError("No modules matched the hook specifications")
    
    def _make_hook(self, hook_name: str, spec: HookSpec):
        recorder = self
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            if not recorder.enabled:
                return
            if not getattr(recorder.config, "store_tensors", True):
                return
            bucket = recorder._capture.setdefault(hook_name, {})
            if spec.include_input:
                for suffix, tensor in _extract_tensors(inputs, "in"):
                    try:
                        bucket[suffix] = _reduce_tensor(tensor, spec)
                    except Exception:
                        if recorder.config.strict:
                            raise
            for suffix, tensor in _extract_tensors(output, "out"):
                try:
                    bucket[suffix] = _reduce_tensor(tensor, spec)
                except Exception:
                    if recorder.config.strict:
                        raise
        return hook
    
    @contextmanager
    def capture_step(self, metadata: CaptureMetadata, step_id: Optional[int] = None):
        self._capture = {}
        self._current_metadata = metadata
        self._current_step = step_id
        try:
            yield self
            self._last_capture = {k: {sk: v.clone() for sk, v in bv.items()} for k, bv in self._capture.items()}
            if self.config.output_dir and self._capture:
                self._save_step()
        finally:
            self._current_metadata = None
            self._current_step = None
            self._capture = {}
    
    def _save_step(self) -> Optional[Path]:
        if not self.output_dir or not self._capture:
            return None
        
        step_id = self._current_step if self._current_step is not None else 0
        path = self.output_dir / f"step_{step_id:06d}.pt"
        
        payload = {
            "step_id": step_id,
            "metadata": self._current_metadata.to_dict() if self._current_metadata else {},
            "activations": self._capture,
            "hook_names": list(self._capture.keys()),
        }
        torch.save(payload, path)
        
        if getattr(self.config, "save_manifest", True) and self._manifest_path:
            record = {
                "step_id": step_id,
                "metadata": self._current_metadata.to_dict() if self._current_metadata else {},
                "path": str(path),
                "keys": {key: sorted(value.keys()) for key, value in self._capture.items()},
                "shapes": {
                    key: {suffix: list(tensor.shape) for suffix, tensor in value.items()}
                    for key, value in self._capture.items()
                },
                "dtypes": {
                    key: {suffix: str(tensor.dtype) for suffix, tensor in value.items()}
                    for key, value in self._capture.items()
                },
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            with self._manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        
        return path
    
    def get_current_capture(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self._capture:
            return self._capture
        return self._last_capture
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def remove_hooks(self):
        for handle in self._hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self._hooks.clear()
        self._bindings.clear()
        self.enabled = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False
    
    def get_hook_info(self) -> Dict[str, str]:
        return {name: spec.pattern for name, spec in self._bindings.items()}
