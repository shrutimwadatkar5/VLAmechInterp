# Stage 3B Report: Activation Capture Infrastructure

## Objective
Implement minimal, reusable PyTorch activation-capture module for OpenVLA-OFT LIBERO Spatial experiments.

## Verified Hook Locations (from Stage 3A)
From Stage 3A audit of `prismatic/extern/hf/modeling_prismatic.py`:
- `PrismaticVisionBackbone` (line ~67, forward ~186) - vision encoder
- `PrismaticProjector` (line ~231, forward ~250) - projector  
- `PrismaticForConditionalGeneration` (line ~317+) - contains `language_model`
- `OpenVLAForActionPrediction` (line ~720, `predict_action` ~944) - action prediction
- External: `get_action_head()` in `experiments/robot/openvla_utils.py:464`, `get_proprio_projector()` in `openvla_utils.py:393`

Hook patterns used:
- `vision_backbone*` - vision encoder
- `projector` - multimodal projector
- `language_model` / `language_model.model.layers.*` / `language_model.model.norm` - LLM backbone
- External modules registered via `extra_modules` with prefixes (`action_head`, `proprio_projector`)

## Capture Schema
Per-step capture saved as `step_XXXXXX.pt`:
```python
{
    "step_id": int,
    "metadata": CaptureMetadata.to_dict(),  # condition_id, perturbation_type/params, task/episode/trial/timestep, camera, checkpoint_id, seed, etc.
    "activations": {hook_name: {suffix: tensor}},  # detached, CPU
    "hook_names": list
}
```

Manifest (`manifest.jsonl`): one record per step with metadata, keys, shapes, dtypes, path.
Run metadata optional.

## Memory Strategy
- `detach=True` by default - no gradient retention
- `store_on_cpu=True` by default - move tensors to CPU
- Configurable reducers: `none`, `mean_seq` (mean over sequence dims), `last_token`, `flatten`, `mean`
- Configurable dtypes: `keep`, `float16`, `float32`
- `max_elements` limit to prevent excessive size
- `store_tensors=False` mode available for summary-only capture
- No silent discarding - raises on `max_elements` exceeded (if strict)

## Hook Management
- Hooks registered once via `register_forward_hook`
- `remove_hooks()` removes all hooks cleanly
- Context manager `capture_step()` ensures cleanup
- `enable()`/`disable()` toggle without removing hooks
- Exception handling with `strict` flag

## Tests (Synthetic Only)
All tests use synthetic PyTorch model, no OpenVLA checkpoint, no LIBERO episodes:
1. Hook registration with verified patterns
2. Capture with metadata association
3. Extra modules (action_head, proprio_projector)
4. Output equivalence with capture enabled vs disabled
5. Clean hook removal
6. Save to disk with manifest

**Note:** Synthetic tests do not validate full OpenVLA-OFT pipeline.

## Limitations
- Tensor shapes not validated against actual OpenVLA runtime (must verify at runtime)
- Hook timing (pre/post) not configurable yet - uses forward hooks
- Some modules may need input hooks depending on analysis needs
- No integration with eval loop (separate concern)
- No perturbation implementation (as requested)

## Safety Compliance
- No model loading (7B checkpoint not loaded)
- No inference/episodes run
- No packages installed/updated
- No existing files modified (baseline launcher, checkpoints, rollouts, logs untouched)
- New files only in timestamped Stage 3B directory
- Read-only inspection of repo
