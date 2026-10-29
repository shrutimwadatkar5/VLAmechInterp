# EXPERIMENT 1 SETUP AUDIT (Stage 1)

## Repository
- Target directory: /home/shrutiw/projects/openvla-oft
- Git URL: https://github.com/moojink/openvla-oft.git
- Branch: main
- Commit hash: e4287e94541f459edc4feabc4e181f537cd569a8
- Clone date: 2026-10-24 16:51:20.563304330 +0530

## Verified files present
- SETUP.md: OK
- LIBERO.md: OK
- experiments/robot/libero/libero_requirements.txt: OK
- pyproject.toml: OK
- experiments/robot/libero/run_libero_eval.py: OK

## Python / PyTorch / Transformers (from repo)
- requires-python: >=3.8 (classifiers include 3.8/3.9/3.10)
- PyTorch: torch==2.2.0, torchvision==0.17.0, torchaudio==2.2.0 (as specified in pyproject.toml)
- Transformers: transformers @ git+https://github.com/moojink/transformers-openvla-oft.git (fork for bidirectional attention)
- Flash Attention: flash-attn==2.5.5 (installed after editable install; packaging, ninja required)
- Key deps: accelerate>=0.25.0, peft==0.11.1, timm==0.9.10, tokenizers==0.19.1, sentencepiece==0.1.99, draccus==0.8.0, diffusers==0.30.3, tensorflow==2.15.0, tensorflow_datasets==4.9.3, tensorflow_graphics==2021.12.3, dlimp @ git+https://github.com/moojink/dlimp_openvla
- LIBERO deps (libero_requirements.txt): imageio[ffmpeg], robosuite==1.4.1, bddl, easydict, cloudpickle, gym

## Installation commands (documented)
- conda create -n openvla-oft python=3.10 -y && conda activate openvla-oft
- pip3 install torch torchvision torchaudio (repo suggests PyTorch.org specific command; pyproject pins 2.2.0)
- git clone https://github.com/moojink/openvla-oft.git && cd openvla-oft && pip install -e .
- pip install packaging ninja && ninja --version && pip install "flash-attn==2.5.5" --no-build-isolation

## run_libero_eval.py configuration (verified)
- Pretrained checkpoint: --pretrained_checkpoint (required)
- Task suite: --task_suite_name (TaskSuite enum: libero_spatial, libero_object, libero_goal, libero_10, libero_90). Default: libero_spatial
- Trials per task: --num_trials_per_task (default 50)
- Seed: --seed (default 7)
- Center crop: --center_crop (default True; required if model trained with image aug)
- Logging: --local_log_dir ./experiments/logs (default), --use_wandb False, --wandb_project/--wandb_entity optional
- Max steps per suite: LIBERO_SPATIAL 220, LIBERO_OBJECT 280, LIBERO_GOAL 300, LIBERO_10 520, LIBERO_90 400
