# Changelog

All notable changes to the Gemma 4 vLLM deployment are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## Cycle 1 — Base Model Deployment (v1–v30)

### v1–v3: APT Sources Fix
- **Failed**: `apt-get update` — base image uses decommissioned `artifact-foundry-prod` mirror
- **Fixed**: OS-aware auto-heal detecting Ubuntu/Debian via `/etc/os-release` and injecting official mirrors
- **Key Learning**: Never assume base image APT sources work

### v4–v6: Dependency Triangle Discovery
- **Failed**: `ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`
- **Root Cause**: PyPI vLLM 0.19.0 requires `transformers<5` and `huggingface_hub<1.0`; base image ships `transformers 5.5.0.dev0` and `huggingface_hub 1.8.0`
- **Fixed**: Stopped installing vLLM from PyPI; use base image's pre-installed vLLM 0.17.2rc1.dev133
- **Key Learning**: The dependency triangle is unsolvable via PyPI packages

### v7–v13: Dependency Conflict Iterations
- **Failed**: Various import errors and version conflicts
- **Root Cause**: Attempting to mix PyPI packages with Google's custom base image stack
- **Key Learning**: Only add packages with `--no-deps` to avoid dependency resolution conflicts

### v14–v17: LoRA Support Attempts
- **v14**: `ValueError: Gemma4ForConditionalGeneration does not support LoRA yet`
- **v17**: `unrecognized arguments: --lora-extra-vocab-size` (flag removed in 0.18+)
- **Key Learning**: LoRA mixin for Gemma 4 MoE is incomplete in vLLM 0.17.2rc1

### v18–v21: GCSFUSE + Dependency Pinning
- **Failed**: Various GCSFUSE mount and dependency issues
- **Key Learning**: GCSFUSE v2.5.3 removed `--allow-other` flag; use `/etc/fuse.conf` instead

### v22–v23: BitsAndBytes MoE Crash
- **Failed**: `AttributeError: MoE Model does not support BitsAndBytes quantization yet. Ensure this model has 'get_expert_mapping' method.`
- **Root Cause**: vLLM's bitsandbytes loader requires `get_expert_mapping()` for MoE models; Gemma 4 doesn't implement it
- **Key Learning**: NF4 quantization is incompatible with Gemma 4 MoE architecture in current vLLM

### v24: Base Image Arg Format Mismatch
- **Failed**: `gcs_download_launcher.sh: line 319: --model=google/gemma-4-27b-a4b-it: No such file or directory`
- **Root Cause**: Base image's `gcs_download_launcher.sh` passes args via env vars, not CLI
- **Fixed**: Always use GCSFUSE container with custom `entrypoint.sh`

### v25: First Successful Deployment 🎉
- **Status**: ✅ Container boots, model loads, health check passes
- **Config**: BF16, GCSFUSE container, no LoRA, no quantization
- **But**: Inference returns HTTP 500

### v25–v26: Chat Template namespace() Crash
- **Failed**: `TypeError: object() takes no arguments` in `safe_apply_chat_template`
- **Root Cause**: Gemma 4's default `chat_template.jinja` uses Jinja2 `namespace()` which is blocked in vLLM's sandboxed environment
- **Fixed**: Created simplified chat template without `namespace()`

### v27: Silent Container Crash
- **Failed**: Container crashes within 7 minutes, zero logs
- **Root Cause**: `--chat-template` CLI flag not supported in vLLM 0.17.2rc1.dev133
- **Fixed**: Strip `--chat-template` in entrypoint.sh; use tokenizer patching approach instead

### v28: VLLM_CHAT_TEMPLATE Env Var
- **Failed**: `Unknown vLLM environment variable detected: VLLM_CHAT_TEMPLATE`
- **Root Cause**: This env var doesn't exist in vLLM 0.17.2rc1
- **Fixed**: Abandoned env var approach; use tokenizer pre-download + patch

### v29: First Successful Inference 🎉🎉
- **Status**: ✅ Model serves, inference works
- **But**: Model loops past turn boundary (generates past `<turn|>` token)
- **Config**: Tokenizer pre-download + `chat_template.jinja` patch + `--tokenizer` flag

### v30: Production Golden Configuration 🏆
- **Status**: ✅ Production-stable
- **Fix**: Added `stop_token_ids: [106]` to all inference payloads (client-side enforcement)
- **Note**: `generation_config.json` patch with `eos_token_id: [1, 106]` is cosmetic — vLLM 0.17.2rc1 doesn't honor list-type `eos_token_id`

---

## Cycle 2 — LoRA Enablement Attempts

### Attempt 1: BF16 + LoRA (24 slots, rank 64)
- **GPU**: A100 80GB
- **Status**: ❌ Error code 13 — transient GCP error, no container logs

### Attempt 2: BF16 + LoRA (4 slots, rank 32)
- **GPU**: L4 24GB (wrong GPU!)
- **Status**: ❌ `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 968.00 MiB. GPU 0 has a total capacity of 22.03 GiB`
- **Root Cause**: Deploy function defaults pointed to L4 (`g2-standard-96`), not A100 (`a2-ultragpu-1g`)
- **Key Learning**: Always check GPU capacity in OOM messages — 22GB ≠ 80GB

### Attempt 3: NF4 + LoRA (24 slots, rank 64)
- **GPU**: A100 80GB
- **Status**: ❌ `AttributeError: MoE Model does not support BitsAndBytes quantization yet`
- **Key Learning**: MoE + bitsandbytes incompatibility confirmed on both vLLM 0.17.x and 0.19.x

### Attempt 4: BF16 + LoRA (4 slots, rank 32)
- **GPU**: A100 80GB (fixed)
- **Status**: ❌ Quota exceeded — previous model still deployed
- **Key Learning**: Must undeploy before redeploy with 1× A100 quota

### Attempt 5: Direct Replacement Deploy
- **Status**: ❌ `ValueError: Gemma4ForConditionalGeneration does not support LoRA yet`
- **Key Learning**: vLLM 0.17.2rc1 LoRA mixin for Gemma4 is incomplete

### Attempt 6: vLLM 0.19.0 Upgrade
- **Status**: ❌ `The checkpoint you are trying to load has model type 'gemma4' but Transformers does not recognize this architecture`
- **Root Cause**: PyPI vLLM 0.19.0 pulls transformers 4.57.6 which lacks `gemma4` model type

### Attempt 7: vLLM 0.19.0 + transformers 5.6.0.dev0
- **Status**: ❌ `ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`
- **Root Cause**: transformers 5.x needs `huggingface_hub>=0.36+` (new import location), but vLLM 0.19.0 pins `huggingface_hub<1.0,>=0.34.0` (old location)

### Attempt 8: Version Override Attempt
- **Status**: ❌ Same `is_offline_mode` import error
- **Root Cause**: The constraint matrix is mathematically unsolvable (see Dependency Matrix)

### Resolution: Reverted to v30 Base-Only
- **Status**: ✅ Reverted to v30-equivalent BF16 base-only deployment
- **Key Learning**: LoRA for Gemma 4 MoE requires upstream fixes in vLLM AND a compatible transformers/hub version matrix

---

## [Unreleased]

### Added
- Initial repository creation with full forensic documentation
- 20 failure modes catalogued across both deployment cycles
- Container files (Dockerfile, entrypoint.sh, chat_template.jinja)
- Deployment script (upload_model.py)
- Dependency matrix proof document
- Community contribution guidelines
