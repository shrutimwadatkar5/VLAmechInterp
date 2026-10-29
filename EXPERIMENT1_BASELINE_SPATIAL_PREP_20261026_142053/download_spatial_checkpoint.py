from huggingface_hub import snapshot_download

repo_id = "moojink/openvla-7b-oft-finetuned-libero-spatial"
revision = "6d0231af0e48c5985f1ff86908f4674b84bc049b"
local_dir = "/home/shrutiw/projects/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
allow = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "dataset_statistics.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "action_head--150000_checkpoint.pt",
    "proprio_projector--150000_checkpoint.pt",
    "configuration_prismatic.py",
    "modeling_prismatic.py",
    "processing_prismatic.py",
]
print("Downloading", repo_id, "revision", revision, flush=True)
path = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=local_dir,
    allow_patterns=allow,
    max_workers=8,
)
print("DONE", path, flush=True)
