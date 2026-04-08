# Forensic Runbook — Gemma 4 27B-A4B-it Deployment on Vertex AI

> **Author:** Daniel Manzela | **Date:** April 2026 | **Total Deployment Time:** 16+ hours across 2 cycles

This document catalogs every failure mode encountered during the deployment of Google's Gemma 4 27B-A4B-it Mixture-of-Experts model on Vertex AI with vLLM. Each entry includes the exact error message, root cause analysis, and the fix applied.

---

## Table of Contents

- [1. Production Configuration](#1-production-configuration)
- [2. Cycle 1 Failures (v1–v30)](#2-cycle-1-failures-v1v30)
- [3. Cycle 2 Failures (LoRA Enablement)](#3-cycle-2-failures-lora-enablement)
- [4. Architecture Decisions](#4-architecture-decisions)
- [5. Error Quick-Reference](#5-error-quick-reference)

---

## 1. Production Configuration

The final working configuration after 30+ iterations:

| Parameter | Value |
|---|---|
| **Model** | `google/gemma-4-27b-a4b-it` (MoE, 27B total, 4B active) |
| **Container** | Custom GCSFUSE-enabled container extending `pytorch-vllm-serve:gemma4` |
| **Base Image** | `us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:gemma4` |
| **vLLM** | `0.17.2rc1.dev133` (Google custom build — **DO NOT UPGRADE**) |
| **Transformers** | `5.5.0.dev0` (**DO NOT UPGRADE**) |
| **huggingface_hub** | `1.8.0` (**DO NOT UPGRADE**) |
| **Precision** | BF16 (no quantization) |
| **GPU** | 1× NVIDIA A100 80GB (`a2-ultragpu-1g`) |
| **Context** | 8,192 tokens (`--max-model-len=8192`) |
| **LoRA** | Disabled (blocked by MoE incompatibilities) |
| **KV Cache** | ~25.46 GiB allocated |
| **Model Weights** | ~48.5 GiB (BF16) |

---

## 2. Cycle 1 Failures (v1–v30)

### 2.1 APT Sources Broken in Base Image

**Versions affected:** v1–v3

```
E: The repository 'https://artifact-foundry-prod.uc.r.appspot.com/...' does not have a Release file.
E: The repository '...' is not signed.
```

**Root Cause:** The `pytorch-vllm-serve:gemma4` base image uses Google's internal `artifact-foundry-prod` as its SOLE APT package source. This mirror was decommissioned — it has no Release file, no package index, nothing. Every `apt-get update` or `apt-get install` fails.

**Fix:** OS-aware auto-heal in Dockerfile:
1. Surgically remove all `artifact-foundry` lines from APT source files
2. Check if any valid `deb` entries remain
3. If zero sources remain, detect OS via `/etc/os-release` and inject official mirrors

```bash
# Step 1: Remove dead sources
sed -i '/artifact-foundry/d' /etc/apt/sources.list 2>/dev/null || true
find /etc/apt/sources.list.d/ -type f -exec sed -i '/artifact-foundry/d' {} \; 2>/dev/null || true

# Step 2: Count remaining valid sources
REMAINING=$(cat /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null | grep -c '^deb ' || true)

# Step 3: Auto-heal if zero sources
if [ "$REMAINING" -eq 0 ]; then
    . /etc/os-release
    if [ "$ID" = "ubuntu" ]; then
        echo "deb http://archive.ubuntu.com/ubuntu ${VERSION_CODENAME} main restricted universe multiverse" > /etc/apt/sources.list
        # ... additional Ubuntu sources
    elif [ "$ID" = "debian" ]; then
        echo "deb http://deb.debian.org/debian ${VERSION_CODENAME} main contrib non-free" > /etc/apt/sources.list
        # ... additional Debian sources
    fi
fi
```

**Prevention:** Always include the auto-heal block. Never assume base image APT sources work.

---

### 2.2 Dependency Triangle (vLLM × transformers × huggingface_hub)

**Versions affected:** v4–v6

```
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```

**Root Cause:** The base image ships a custom dependency stack that is incompatible with PyPI packages:

- Base image: `vLLM 0.17.2rc1.dev133 + transformers 5.5.0.dev0 + huggingface_hub 1.8.0`
- PyPI vLLM 0.19.0 requires: `transformers<5 + huggingface_hub<1.0`

Installing PyPI vLLM downgrades transformers → loses `Gemma4ForConditionalGeneration` registration. Pinning transformers 5.5.0.dev0 back conflicts with vLLM 0.19.0's imports.

**Fix:** **Never install vLLM from PyPI.** Use the base image's pre-installed stack. Only add isolated packages with `--no-deps`.

> ⚠️ **CRITICAL:** The dependency matrix `vLLM 0.17.2rc1.dev133 + transformers 5.5.0.dev0 + huggingface_hub 1.8.0` is the ONLY known working configuration.

---

### 2.3 Gemma 4 Architecture Not Registered

**Versions affected:** v4–v5

```
KeyError: 'gemma4'
```

**Root Cause:** PyPI vLLM 0.19.0 does not have the `gemma4` model type registered. Google's custom build adds it. When PyPI vLLM is installed, it also pulls `transformers 4.57.6` which likewise lacks `Gemma4ForConditionalGeneration`.

**Fix:** Use the base image's native vLLM which has the custom model type registration.

---

### 2.4 LoRA Not Supported for Gemma4

**Versions affected:** v14–v17

```
ValueError: Gemma4ForConditionalGeneration does not support LoRA yet
```

**Root Cause:** The vLLM 0.17.2rc1.dev133 build has partial LoRA support but lacks the complete LoRA mixin for Gemma 4's MoE attention layers.

**Fix:** Currently no fix. Requires upstream vLLM development.

---

### 2.5 `--lora-extra-vocab-size` Deprecated

**Versions affected:** v17

```
unrecognized arguments: --lora-extra-vocab-size
```

**Root Cause:** This flag was removed in vLLM 0.18+. The base image's 0.17.2rc1 is technically a pre-release of 0.18 and already removed it.

**Fix:** Remove `--lora-extra-vocab-size` from all vLLM arg lists.

---

### 2.6 BitsAndBytes MoE Incompatibility

**Versions affected:** v22–v23

```
AttributeError: MoE Model Gemma4ForConditionalGeneration does not support
BitsAndBytes quantization yet. Ensure this model has 'get_expert_mapping' method.
```

**Root Cause:** vLLM's bitsandbytes loader requires `get_expert_mapping()` for MoE models to know how to quantize expert layers separately. `Gemma4ForConditionalGeneration` does not implement this method.

**Fix:** Deploy without quantization (BF16). NF4 quantization is NOT currently possible for MoE architectures in any vLLM version.

> ⚠️ **BF16 is the ONLY working precision for Gemma 4 MoE on vLLM.** Do not attempt bitsandbytes, AWQ, or GPTQ.

---

### 2.7 Base Image Arg Passing Mismatch

**Versions affected:** v24

```
gcs_download_launcher.sh: line 319: --model=google/gemma-4-27b-a4b-it: No such file or directory
```

**Root Cause:** The base `pytorch-vllm-serve:gemma4` image uses `gcs_download_launcher.sh` as its entrypoint. This script passes arguments via environment variables (like `VLLM_ARGS`), not command-line args. When container args are passed via Vertex AI model registration, the launcher treats them as shell commands.

**Fix:** Always use the custom GCSFUSE container with `entrypoint.sh`, which correctly passes CLI args to `python -m vllm.entrypoints.openai.api_server`.

---

### 2.8 GPU Quota Exhaustion

**Versions affected:** v23+

```
The following quotas are exceeded: CustomModelServingA10080GBGPUsPerProjectPerRegion
```

**Root Cause:** Failed deployments hold GPU node reservations for up to 30 minutes. With a 1× A100 80GB quota, only one deployment can exist at a time.

**Fix:** Wait ~30 minutes after a failed deploy, or explicitly undeploy the current model before retrying.

> ℹ️ A 1× A100 quota means canary (traffic-split) deployments are IMPOSSIBLE. All deployments must use direct replacement.

---

### 2.9 Chat Template Jinja2 Crash

**Versions affected:** v25–v26

```python
TypeError: object() takes no arguments
# In: safe_apply_chat_template at hf.py:483
```

**Root Cause:** Gemma 4's default `chat_template.jinja` uses Jinja2 `namespace()` objects. vLLM 0.17.2rc1.dev133's sandboxed Jinja2 environment does NOT expose the `namespace` constructor — it maps it to Python's base `object()`, which takes no arguments.

**Fix:** Created a simplified `chat_template.jinja` that handles text-only user/assistant/system turns without `namespace()`:

```jinja
{{- bos_token -}}
{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{- '<|turn>user\n' -}}
        {{- message['content'] | trim -}}
        {{- '<turn|>\n' -}}
    {%- elif message['role'] == 'assistant' or message['role'] == 'model' -%}
        {{- '<|turn>model\n' -}}
        {{- message['content'] | trim -}}
        {{- '<turn|>\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|turn>model\n' -}}
{%- endif -%}
```

> ⚠️ This template does NOT support multimodal (vision) inputs.

---

### 2.10 `--chat-template` CLI Flag Not Supported

**Versions affected:** v27

**Symptom:** Container crashes within 7 minutes — zero log output.

**Root Cause:** vLLM 0.17.2rc1.dev133 does NOT support the `--chat-template` CLI argument. Passing it causes the arg parser to crash before logging is initialized.

**Fix:** The `entrypoint.sh` strips `--chat-template=*` from container args at runtime. The chat template is injected via tokenizer patching instead:

1. Pre-download tokenizer from HuggingFace
2. Overwrite `chat_template.jinja` with simplified version
3. Pass `--tokenizer=/tmp/patched_tokenizer` to vLLM

---

### 2.11 `VLLM_CHAT_TEMPLATE` Env Var Not Recognized

**Versions affected:** v28

```
Unknown vLLM environment variable detected: VLLM_CHAT_TEMPLATE
```

**Root Cause:** This environment variable does not exist in vLLM 0.17.2rc1.dev133.

**Fix:** Do not use env vars for chat template configuration. Use the tokenizer patching approach.

---

### 2.12 GCSFUSE `--allow-other` Flag

**Versions affected:** v28–v30

```
unknown flag: --allow-other
```

**Root Cause:** GCSFUSE v2.5.3 removed the `--allow-other` flag. The functionality is now controlled via `/etc/fuse.conf` (`user_allow_other` directive).

**Fix:** Removed `--allow-other` from gcsfuse command. Added `user_allow_other` to `/etc/fuse.conf` in the Dockerfile.

---

### 2.13 End-of-Turn Token Not Stopping Generation

**Versions affected:** v29

**Symptom:** Model generates correct answer then continues looping:
```
"Paris\nuser\nWhat is the capital of..."
```

**Root Cause:** Gemma 4's `eos_token` is `<eos>` (token ID 1). The end-of-turn marker `<turn|>` (token ID 106) is a separate special token NOT configured as a stop token in vLLM.

**Fix (Client-side — MANDATORY):** Every inference payload MUST include:
```json
{"stop_token_ids": [106]}
```

**Note:** Patching `generation_config.json` with `eos_token_id: [1, 106]` is cosmetic only — vLLM 0.17.2rc1 does NOT honor list-type `eos_token_id`.

> 🚨 **If `stop_token_ids: [106]` is missing, the model will loop indefinitely until `max_tokens` is exhausted.**

---

## 3. Cycle 2 Failures (LoRA Enablement)

### 3.1 Wrong GPU Type in Deploy Defaults

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 968.00 MiB.
GPU 0 has a total capacity of 22.03 GiB
```

**Root Cause:** The deployment function had hardcoded defaults pointing to L4 GPUs (`g2-standard-96`, 8× L4 24GB) instead of A100 80GB (`a2-ultragpu-1g`). The 52GB BF16 model loaded onto a single 24GB GPU.

**Key Diagnostic:** The error message says `22.03 GiB` — that's NOT 80GB. This was a GPU mismatch, not a LoRA configuration issue.

**Fix:** Changed defaults to `a2-ultragpu-1g` + `NVIDIA_A100_80GB` + count 1.

> 🚨 **Always check GPU capacity in OOM messages.** Less than 80GB means you're NOT on an A100 80GB.

---

### 3.2 PyPI Mirror Blocked in Base Image

```
ERROR: Could not find a version that satisfies the requirement vllm>=0.19.0
```

**Root Cause:** The base image's `pip.conf` uses Google's internal PyPI mirror, not public PyPI. This mirror doesn't host most public packages.

**Fix:** Always use `--index-url https://pypi.org/simple/` for pip installs:
```dockerfile
RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple/ \
    "bitsandbytes>=0.45.0"
```

---

### 3.3 Transformers 4.x Lacks `gemma4` Model Type

```
The checkpoint you are trying to load has model type 'gemma4' but
Transformers does not recognize this architecture
```

**Root Cause:** When vLLM 0.19.0 was installed from PyPI, it pulled `transformers 4.57.6` (latest satisfying `transformers<5`). The `gemma4` model type only exists in **transformers 5.x**.

**Key Insight:** `transformers 5.5.0` exists on PyPI, but it requires `huggingface_hub>=1.5.0` which conflicts with vLLM 0.19.0's `huggingface_hub<1.0`. These ranges are mutually exclusive.

---

### 3.4 `is_offline_mode` Import Chain Error

```
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```

**Root Cause:** The `is_offline_mode` function was moved in `huggingface_hub 0.36+` from the top-level module to `huggingface_hub.utils`. The full import chain:

```
vllm → transformers_utils/config.py → transformers → transformers.hub → from huggingface_hub import is_offline_mode
```

**Key Insight:** This error appears to come from vLLM but actually originates from transformers.

---

### 3.5 Vertex AI Image Digest Pinning

**Symptom:** Container deployed with old image despite pushing new `:latest` tag.

**Root Cause:** When a Vertex AI model is registered, the `:latest` tag is resolved to a specific SHA digest **at registration time**. Subsequent pushes to `:latest` do NOT affect already-registered models.

**Fix:** After pushing a new container image, ALWAYS register a new Model version via `Model.upload()`.

> ℹ️ Use versioned tags (`:v22`) alongside `:latest` for traceability.

---

### 3.6 Empty Endpoint Traffic Split Error

**Root Cause:** Canary deployment logic tried to set 10% traffic on a new model + 90% on existing. But after undeploying to free quota, the endpoint was empty. Vertex AI requires traffic to sum to 100% and rejects splits when there's no existing model.

**Fix:** Added empty-endpoint detection:
```python
if len(existing_deployed_ids) == 0:
    traffic_percentage = 100
```

---

## 4. Architecture Decisions

### Why GCSFUSE Container for All Deploys
The base image's launcher passes args via env vars, not CLI. The GCSFUSE container's `entrypoint.sh` correctly translates CLI args to vLLM. It also includes bitsandbytes and tokenizer patching logic.

### Why BF16 (No Quantization)
NF4 via bitsandbytes crashes on MoE's `get_expert_mapping()`. AWQ/GPTQ require pre-quantized checkpoints. BF16 fits on A100 80GB with headroom.

### Why Direct Replacement (No Canary)
1× A100 quota means one model at a time. Canary requires 2× simultaneous. Direct replacement causes ~5 min downtime but is the only option.

### Why Tokenizer Patching
vLLM 0.17.2rc1 doesn't support `--chat-template` or `VLLM_CHAT_TEMPLATE`. Pre-download + patch + `--tokenizer` is the only reliable path.

### Why `--enforce-eager`
CUDA graph compilation on A100 with 26B MoE causes transient OOM spikes. `--enforce-eager` disables this.

### Why `--max-model-len=8192`
BF16 weights ~52GB on A100 80GB leaves ~24GB for KV cache. At 65536 tokens the KV cache would need ~150GB+. At 8192 it fits comfortably.

---

## 5. Error Quick-Reference

| Error Message | Root Cause | Fix | Section |
|---|---|---|---|
| `cannot import name 'is_offline_mode'` | Hub version mismatch | Use base image hub 1.8.0 | 2.2, 3.4 |
| `KeyError: 'gemma4'` | Wrong vLLM/transformers | Use base image stack | 2.3 |
| `does not support LoRA yet` | Missing LoRA mixin | Wait for upstream fix | 2.4 |
| `--lora-extra-vocab-size` | Flag removed | Remove from args | 2.5 |
| `does not support BitsAndBytes` | Missing `get_expert_mapping()` | Use BF16 | 2.6 |
| `gcs_download_launcher.sh: No such file` | Base image arg mismatch | Use GCSFUSE container | 2.7 |
| `A10080GBGPUs quota exceeded` | Zombie GPU reservation | Wait 30 min | 2.8 |
| `TypeError: object() takes no arguments` | Jinja2 namespace() | Use simplified template | 2.9 |
| Container crashes, no logs | `--chat-template` unsupported | Strip in entrypoint | 2.10 |
| `Unknown vLLM env var` | Env var doesn't exist | Use tokenizer patching | 2.11 |
| Model loops past turn boundary | Missing stop token 106 | Add `stop_token_ids: [106]` | 2.13 |
| `total capacity of 22.03 GiB` | Wrong GPU (L4 not A100) | Fix deploy defaults | 3.1 |
| `Could not find vllm>=0.19.0` | Base image PyPI mirror | Use `--index-url` | 3.2 |
| `model type 'gemma4' not recognized` | transformers 4.x | Use transformers 5.5.0 | 3.3 |
| Stale container after rebuild | Image digest pinned | Register new model | 3.5 |
| Traffic split error (empty endpoint) | No models on endpoint | Use 100% traffic | 3.6 |

---

*Document authored by [Daniel Manzela](https://github.com/Manzela). April 2026.*
