#!/bin/bash
# ─────────────────────────────────────────────────────────────
# entrypoint.sh — GCSFUSE Mount + Tokenizer Patch + vLLM Startup
#
# 1. Mounts the LoRA adapter GCS bucket to /mnt/gcs using GCSFUSE
# 2. Pre-downloads the tokenizer and patches:
#    a) chat_template.jinja → simplified version (no namespace())
#    b) generation_config.json → adds <turn|> (106) to eos_token_id
# 3. Points vLLM at the patched tokenizer via --tokenizer
# 4. Hands off to vLLM's OpenAI-compatible API server
#
# Environment Variables:
#   LORA_ADAPTER_BUCKET  — GCS bucket name (your LoRA adapter bucket)
#   HF_TOKEN             — HuggingFace auth token for gated models
#
# Fail-Safe Behavior:
#   If GCSFUSE mount fails, the script proceeds without it.
#   If tokenizer patching fails, vLLM starts with the default template.
#
# Signal Handling:
#   Uses exec to replace this shell process with vLLM, ensuring
#   SIGTERM from Kubernetes/Vertex AI reaches vLLM directly.
#
# Author: Daniel Manzela (April 2026)
# ─────────────────────────────────────────────────────────────

set -uo pipefail

BUCKET="${LORA_ADAPTER_BUCKET:-your-lora-adapters-bucket}"
MOUNT_POINT="/mnt/gcs"

echo "[entrypoint] Attempting GCSFUSE mount: gs://${BUCKET} → ${MOUNT_POINT}"

# Attempt GCSFUSE mount with fail-safe
if gcsfuse \
    --implicit-dirs \
    --log-severity=warning \
    --file-mode=444 \
    --dir-mode=555 \
    "${BUCKET}" "${MOUNT_POINT}" 2>&1; then
    echo "[entrypoint] ✅ GCSFUSE mounted: gs://${BUCKET} → ${MOUNT_POINT}"
else
    echo "[entrypoint] ⚠️  GCSFUSE mount FAILED (non-fatal). Proceeding without adapter mount."
    echo "[entrypoint]    Adapters via /mnt/gcs/ will be unavailable. Base model will serve normally."
fi

# ─────────────────────────────────────────────────────────────
# Tokenizer + Generation Config Patch
#
# Problems with the default Gemma 4 tokenizer:
# 1. chat_template.jinja uses Jinja2 namespace() → crashes in
#    vLLM 0.17.2rc1.dev133's sandboxed Jinja2 environment
# 2. generation_config.json has eos_token_id=1 (<eos>) but does
#    NOT include <turn|> (token 106) as a stop token, causing
#    the model to loop past its own turn boundary
#
# Fix: download tokenizer, replace chat_template.jinja with our
# simplified version, and patch generation_config to add 106
# as an additional EOS token.
# ─────────────────────────────────────────────────────────────
PATCHED_TOKENIZER_DIR="/tmp/patched_tokenizer"

if [ -f /chat_template.jinja ]; then
    echo "[entrypoint] Downloading tokenizer for patching..."
    python3 -c "
from huggingface_hub import snapshot_download
import json, os

# Download tokenizer files
snapshot_download(
    'google/gemma-4-27b-a4b-it',
    local_dir='${PATCHED_TOKENIZER_DIR}',
    allow_patterns=['tokenizer*', 'chat_template*', 'special_tokens*', 'added_tokens*', 'generation_config*'],
    token=os.environ.get('HF_TOKEN'),
)
print('[entrypoint] ✅ Tokenizer files downloaded')

# Patch generation_config.json to add <turn|> (106) as stop token
gen_config_path = '${PATCHED_TOKENIZER_DIR}/generation_config.json'
if os.path.exists(gen_config_path):
    with open(gen_config_path) as f:
        config = json.load(f)
    # Convert eos_token_id to list if it's a single int
    current_eos = config.get('eos_token_id', 1)
    if isinstance(current_eos, int):
        config['eos_token_id'] = [current_eos, 106]
    elif isinstance(current_eos, list) and 106 not in current_eos:
        config['eos_token_id'].append(106)
    with open(gen_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'[entrypoint] ℹ️  generation_config.json patched (cosmetic only — vLLM 0.17.2rc1 does NOT honor this; stop enforced via per-request stop_token_ids): eos_token_id={config[\"eos_token_id\"]}')
else:
    print('[entrypoint] ⚠️  generation_config.json not found, skipping EOS patch')
" 2>&1

    if [ $? -eq 0 ] && [ -d "${PATCHED_TOKENIZER_DIR}" ]; then
        # Overwrite chat template with our simplified version
        cp /chat_template.jinja "${PATCHED_TOKENIZER_DIR}/chat_template.jinja"
        echo "[entrypoint] ✅ chat_template.jinja patched with simplified version"
    else
        echo "[entrypoint] ⚠️  Tokenizer download failed. Using default template (may crash)."
        PATCHED_TOKENIZER_DIR=""
    fi
else
    echo "[entrypoint] ⚠️  /chat_template.jinja not found. Using default template."
    PATCHED_TOKENIZER_DIR=""
fi

# Filter args and inject --tokenizer if we have a patched copy
FILTERED_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --chat-template=*|--chat-template)
            echo "[entrypoint] Stripping arg: $arg (handled via tokenizer patch)"
            ;;
        *)
            FILTERED_ARGS+=("$arg")
            ;;
    esac
done

# If we patched the tokenizer, add --tokenizer pointing to local copy
if [ -n "${PATCHED_TOKENIZER_DIR}" ] && [ -d "${PATCHED_TOKENIZER_DIR}" ]; then
    FILTERED_ARGS+=("--tokenizer=${PATCHED_TOKENIZER_DIR}")
    echo "[entrypoint] Injected --tokenizer=${PATCHED_TOKENIZER_DIR}"
fi

echo "[entrypoint] Starting vLLM server with args: ${FILTERED_ARGS[*]}"

# exec replaces this process with vLLM — PID 1 for proper signal handling
exec python -m vllm.entrypoints.openai.api_server "${FILTERED_ARGS[@]}"
