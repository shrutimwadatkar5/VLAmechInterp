# Stage 3C Input Fix Report

## Status
SUCCESS - Real model activation capture validated on OpenVLA-OFT

## Root cause of formatting failure
Previous attempt used `processor(prompt, images)` incorrectly. The correct pathway is `get_vla_action()` from `experiments/robot/openvla_utils.py` which:
1. Builds observation with `full_image`, `wrist_image`, `state` (as in `prepare_observation`)
2. Calls `prepare_images_for_vla()` on all images
3. Pops primary image, builds prompt `f"In: What action should the robot take to {task_label.lower()}?\nOut:"`
4. Calls `processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)`
5. Concatenates wrist images if `cfg.num_images_in_input > 1`
6. Processes proprio via `proprio_projector` if `cfg.use_proprio`
7. Calls `vla.predict_action()` with `unnorm_key` from `norm_stats`

Also needed `check_unnorm_key()` equivalent to set `cfg.unnorm_key` (copied from `run_libero_eval.py`).

## Source files used
- experiments/robot/openvla_utils.py: get_vla_action (715-799), get_vla, get_processor, get_action_head, get_proprio_projector, prepare_images_for_vla, resize_image_for_policy
- experiments/robot/libero/run_libero_eval.py: prepare_observation (243-263), check_unnorm_key logic
- prismatic/vla/constants.py
- checkpoints/openvla-7b-oft-finetuned-libero-spatial/

## Files created (new timestamped directory only)
- STAGE3C_INPUT_FIX_20261028_231033/validate/validate_real_model_fixed3.py
- STAGE3C_INPUT_FIX_20261028_231033/captures/stage3c_fixed_capture.pt
- STAGE3C_INPUT_FIX_20261028_231033/STAGE3C_INPUT_FIX_REPORT.md
- Reused existing Stage 3B infrastructure and Stage 3C adapter (not modified)

## Commands executed
- nvidia-smi checks
- Created new directory
- Ran validate_real_model_fixed3.py with .venv_libero

## Real-model validation results
- Model: OpenVLAForActionPrediction loaded on cuda:0
- Components: action_head L1RegressionActionHead, proprio_projector ProprioProjector
- unnorm_key: libero_spatial_no_noops
- Inference: successful, action shape (8,7)
- Hooks attached: vision_backbone, projector, language_model_layers, language_model_norm (4 hooks)
- All hooks removed cleanly

## Captured activations
- vision_backbone: out shape [1, 1114112], dtype float16, device cpu
- projector: out shape [1, 4096], dtype float16, device cpu  
- language_model_layers: out shape [1, 4096], out.0 shape [1, 4096] (2 tensors), dtype float16
- language_model_norm: out shape [1, 4096], dtype float16

All detached, CPU, reduced via mean_seq/flatten as configured. Metadata includes condition_id, perturbation_type, task_name, checkpoint_id, run_id not set.

## Artifacts
- Capture: STAGE3C_INPUT_FIX_20261028_231033/captures/stage3c_fixed_capture.pt
- Keys: ['metadata', 'action', 'activations', 'hook_names']
- Verified readable with torch.load

## Camera/view
Camera/view identity not explicitly tracked from observation dict beyond full_image/wrist_image presence. Observation used both images but hook names don't distinguish camera - remains unresolved.

## Action head/proprio projector
Retrieved successfully but not hooked via pattern matching. Their forward passes not captured in this test - unavailable.

## Output equivalence
Not tested. Would require running same inference with capture disabled/enabled.

## GPU usage
Peak ~18GB during load, inference completed successfully.

## Remaining uncertainties
1. Camera/view association in captures
2. Action head and proprio_projector hooks not captured
3. Hook timing (forward hooks) vs pre/post
4. Layer selection - captured all matched language_model layers (got 2 tensors)
5. Tensor interpretation (visual vs fused vs language) needs analysis

## Next recommended step
Test with real LIBERO environment observation, add explicit hooks for action_head/proprio_projector if their inputs/outputs are needed, and verify output equivalence.
