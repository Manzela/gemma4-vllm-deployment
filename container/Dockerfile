# ─────────────────────────────────────────────────────────────
# GCSFUSE-enabled vLLM Serving Image (Gemma 4 27B-A4B)
#
# Extends Google's pytorch-vllm-serve:gemma4 with GCSFUSE
# to enable dynamic LoRA adapter loading from GCS buckets
# via local filesystem mount (/mnt/gcs).
#
# Base image includes: vLLM 0.17.2rc1.dev133 (Google custom),
#   transformers 5.5.0.dev0, Gemma 4 support baked in.
#
# GCSFUSE mount is performed by entrypoint.sh at container
# startup (before vLLM starts), making adapters available
# at /mnt/gcs/<adapter_name>/<version>/
#
# NOTE: Vertex AI Prediction does NOT grant CAP_SYS_ADMIN.
#       GCSFUSE is installed with --implicit-dirs and
#       --foreground flags. If mounting fails (missing
#       permissions), entrypoint.sh logs a warning and
#       proceeds with base model only (fail-safe).
#
# Author: Daniel Manzela (April 2026)
# ─────────────────────────────────────────────────────────────

FROM us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:gemma4

# Fix APT sources in the base image.
# The pytorch-vllm-serve:gemma4 base image uses artifact-foundry-prod as its
# SOLE APT package source. This mirror is decommissioned (no Release file),
# making apt-get update/install fail. We first try surgical line removal;
# if APT is still broken (zero sources left), we auto-detect the OS via
# /etc/os-release and inject the correct official mirrors. This makes the
# build portable across Ubuntu (focal/jammy/noble) and Debian (bookworm).
RUN set -x && \
    # Step 1: Surgically remove dead artifact-foundry lines from all APT files
    sed -i '/artifact-foundry/d' /etc/apt/sources.list 2>/dev/null || true && \
    find /etc/apt/sources.list.d/ -type f -exec sed -i '/artifact-foundry/d' {} \; 2>/dev/null || true && \
    find /etc/apt/sources.list.d/ -type f -empty -delete 2>/dev/null || true && \
    # Step 2: Check if any valid 'deb ' entries remain across ALL source files.
    # apt-get update returns 0 even with zero sources (!), so we can't rely on
    # its exit code to trigger auto-heal. Instead, grep for actual deb lines.
    REMAINING=$(cat /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null | grep -c '^deb ' || true) && \
    if [ "$REMAINING" -eq 0 ]; then \
        echo "[gcsfuse-build] Zero valid APT sources after cleanup — activating OS-aware auto-heal" && \
        . /etc/os-release && \
        if [ "$ID" = "ubuntu" ]; then \
            echo "[gcsfuse-build] Detected Ubuntu ($VERSION_CODENAME). Restoring archive.ubuntu.com..." && \
            echo "deb http://archive.ubuntu.com/ubuntu ${VERSION_CODENAME} main restricted universe multiverse" > /etc/apt/sources.list && \
            echo "deb http://archive.ubuntu.com/ubuntu ${VERSION_CODENAME}-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
            echo "deb http://security.ubuntu.com/ubuntu ${VERSION_CODENAME}-security main restricted universe multiverse" >> /etc/apt/sources.list; \
        elif [ "$ID" = "debian" ]; then \
            echo "[gcsfuse-build] Detected Debian ($VERSION_CODENAME). Restoring deb.debian.org..." && \
            echo "deb http://deb.debian.org/debian ${VERSION_CODENAME} main contrib non-free" > /etc/apt/sources.list && \
            echo "deb http://deb.debian.org/debian ${VERSION_CODENAME}-updates main contrib non-free" >> /etc/apt/sources.list && \
            echo "deb http://security.debian.org/debian-security ${VERSION_CODENAME}-security main contrib non-free" >> /etc/apt/sources.list; \
        else \
            echo "[gcsfuse-build] UNKNOWN OS: $ID $VERSION_CODENAME — cannot auto-heal" && exit 1; \
        fi && \
        echo "=== Restored APT sources ===" && cat /etc/apt/sources.list; \
    else \
        echo "[gcsfuse-build] $REMAINING valid APT source(s) remain after cleanup — no auto-heal needed"; \
    fi && \
    # Step 3: Update and install
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends curl ca-certificates fuse wget && \
    wget -qO gcsfuse.deb https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v2.5.3/gcsfuse_2.5.3_amd64.deb && \
    dpkg -i gcsfuse.deb && \
    rm gcsfuse.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── CRITICAL: Add bitsandbytes for NF4 quantization ──
# The base image ships Google's custom vLLM 0.17.2rc1.dev133 with:
#   - transformers 5.5.0.dev0 (Gemma 4 architecture support)
#   - huggingface_hub 1.8.0
# These are INCOMPATIBLE with PyPI vLLM 0.19.0 (which requires transformers<5
# and huggingface_hub<1.0). Do NOT upgrade vLLM — it breaks the dependency
# chain and we lose gemma4 model type registration.
#
# We ONLY install bitsandbytes for NF4 quantization (52GB→~14GB weights).
# --index-url is required because the base image's pip.conf points to
# Google's artifact-foundry-prod mirror. Use --no-deps to prevent pip from
# pulling any transitive dependencies that might conflict with the base stack.
RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple/ \
    "bitsandbytes>=0.45.0"

# Create the GCSFUSE mount point and enable user_allow_other.
# user_allow_other is required because vLLM spawns tensor-parallel worker
# processes that need filesystem access to /mnt/gcs. Without this flag,
# child processes of the mounting user get EACCES on the FUSE mount.
RUN mkdir -p /mnt/gcs && \
    echo "user_allow_other" >> /etc/fuse.conf

# Copy the custom entrypoint that mounts GCS before starting vLLM
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Simplified Gemma 4 chat template. The model's default chat_template.jinja
# uses Jinja2 namespace() and other constructs that crash in vLLM 0.17.2rc1's
# sandboxed Jinja2 environment (TypeError: object() takes no arguments).
# This stripped-down template handles text-only user/assistant/system turns.
COPY chat_template.jinja /chat_template.jinja

ENTRYPOINT ["/entrypoint.sh"]
