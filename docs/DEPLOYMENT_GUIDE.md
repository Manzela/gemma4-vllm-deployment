# Deployment Guide — Gemma 4 27B-A4B-it on Vertex AI

> **Author:** Daniel Manzela | **Date:** April 2026  
> **Prerequisites:** Google Cloud project with Vertex AI API enabled, A100 80GB quota, Artifact Registry repository

## Overview

This guide walks through deploying the Gemma 4 27B-A4B-it (MoE) model on Vertex AI using the GCSFUSE-enabled vLLM container. The deployment uses BF16 precision on a single NVIDIA A100 80GB GPU.

---

## Step 1: Build the Container

```bash
cd container/

# Build locally (or use Cloud Build)
docker build -t gcsfuse-vllm-gemma4:latest .
```

### What the Dockerfile Does
1. Extends Google's `pytorch-vllm-serve:gemma4` base image
2. Fixes broken APT sources (OS-aware auto-heal)
3. Installs GCSFUSE v2.5.3 for dynamic adapter loading
4. Installs bitsandbytes (for future NF4 support)
5. Copies `entrypoint.sh` and `chat_template.jinja`

### Build with Cloud Build
```bash
gcloud builds submit container/ \
  --tag YOUR_REGION-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/gcsfuse-vllm-gemma4:latest \
  --machine-type=e2-highcpu-8 \
  --timeout=1800s
```

---

## Step 2: Push to Artifact Registry

```bash
# Tag for your registry
docker tag gcsfuse-vllm-gemma4:latest \
  YOUR_REGION-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/gcsfuse-vllm-gemma4:latest

# Push
docker push \
  YOUR_REGION-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/gcsfuse-vllm-gemma4:latest
```

---

## Step 3: Configure Environment Variables

Copy and edit the example config:
```bash
cp deployment/deploy_config.env.example deployment/.env
```

Required variables:
| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace token for gated model access | (required) |
| `GCP_PROJECT` | Your GCP project ID | (required) |
| `GCP_REGION` | Deployment region | `us-central1` |
| `IMAGE_URI` | Full container image URI | (required) |
| `SLM_ENABLE_LORA` | Enable LoRA adapter loading | `false` |
| `SLM_QUANTIZATION` | Quantization method | `none` |
| `SLM_MODEL_VERSION` | Model version label | `v1` |

---

## Step 4: Register and Deploy

```bash
# Source environment variables
set -a && source deployment/.env && set +a

# Register model and deploy in one step
python deployment/upload_model.py --deploy
```

### Expected Timeline
- **Container boot:** ~10 minutes
- **Model download:** ~10 minutes (52GB BF16 weights)
- **Model loading:** ~10 minutes (GPU memory allocation + kernel compilation)
- **Total:** ~30 minutes

### Deployment Stages (Cloud Console)
1. `PREPARING_MODEL` (10 min) — Container image pulled, GCSFUSE mount attempted
2. `ADDING_NODES_TO_CLUSTER` (5 min) — A100 GPU node provisioned
3. `STARTING_MODEL_SERVER` (15 min) — Model loaded into GPU, health check passes

---

## Step 5: Verify Deployment

### Check status via CLI
```bash
gcloud ai endpoints describe YOUR_ENDPOINT_ID \
  --region=YOUR_REGION \
  --format=yaml | grep -A5 deployedModels
```

### Send test inference
```python
import json
from google.auth import default
from google.auth.transport.requests import AuthorizedSession

credentials, project = default()
session = AuthorizedSession(credentials)

url = f"https://{YOUR_REGION}-aiplatform.googleapis.com/v1/projects/{YOUR_PROJECT}/locations/{YOUR_REGION}/endpoints/{YOUR_ENDPOINT_ID}:rawPredict"

payload = {
    "model": "openapi",  # Must match --served-model-name
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "stop_token_ids": [106]  # CRITICAL: Gemma 4 end-of-turn token
}

response = session.post(url, json=payload)
print(json.dumps(response.json(), indent=2))
```

### Verify correct behavior
- ✅ Response contains `"finish_reason": "stop"` (not `"length"`)
- ✅ Response content is coherent and doesn't loop
- ✅ No HTTP 500 errors

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Container crashes in <10 min | `--chat-template` flag in args | Verify entrypoint.sh strips it |
| HTTP 500 on inference | Chat template namespace() crash | Check chat_template.jinja is simplified |
| Model loops past answer | Missing stop_token_ids: [106] | Add to every request payload |
| OOM with 22GB capacity | Wrong GPU type (L4 not A100) | Check deploy defaults |
| Quota exceeded | Previous model still deployed | Undeploy first, wait 30 min |
| Stale container | Image digest pinned | Register new model version |

---

## Critical Reminders

1. **Always include `stop_token_ids: [106]`** in every inference request
2. **Always use the GCSFUSE container** — never deploy with the base image directly
3. **Never upgrade vLLM, transformers, or huggingface_hub** inside the container
4. **Always register a new model version** after rebuilding the container
5. **Always verify GPU capacity** in OOM error messages

---

*Guide authored by [Daniel Manzela](https://github.com/Manzela). April 2026.*
