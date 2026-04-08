"""
upload_model.py — Upload Gemma 4 27B-A4B-it to Vertex AI Model Registry

Registers the model with the GCSFUSE-enabled vLLM container and deploys
it on Vertex AI for serving.

Supports two deployment modes via SLM_ENABLE_LORA env var:
  - "false" (default) → Base-only: Maximum context window, no adapter overhead
  - "true"            → LoRA-enabled: Reduced context, adapter slots pre-allocated

HF_TOKEN resolution:
  1. os.environ (injected by secrets manager or local .env)
  2. GCP Secret Manager fallback (projects/YOUR_PROJECT/secrets/HF_TOKEN)

Quantization is controlled via SLM_QUANTIZATION env var:
  - "none" (default) → BF16 native precision (optimal for A100 80GB)
  - "bitsandbytes"   → 4-bit NF4 (⚠️ currently broken for MoE)
  - "awq"            → Activation-aware weight quantization
  - "gptq"           → GPTQ quantization

Environment Variable Contract:
  SLM_ENABLE_LORA      "false"   Gates LoRA args (fail-closed default)
  SLM_MODEL_VERSION    "v1"      Display name suffix + traceability
  SLM_QUANTIZATION     "none"    Quantization method
  SLM_USE_GCSFUSE      "true"    Use GCSFUSE-enabled custom container
  GCP_PROJECT          —         Your GCP project ID
  GCP_REGION           —         Deployment region (default: us-central1)
  HF_TOKEN             —         HuggingFace auth token

Hazard Log (resolved from forensic gap analysis):
  H-1: --kv-cache-dtype=fp8  → PURGED (crashes A100 SM80 silicon)
  H-2: --enable-lora always   → GATED by SLM_ENABLE_LORA
  H-3: --max-loras stale      → CORRECTED to 16
  H-4: --max-cpu-loras absent → ADDED at 96
  H-5: display_name hardcoded → PARAMETERIZED via SLM_MODEL_VERSION
  H-6: VLLM_ALLOW_RUNTIME_LORA_UPDATING ungated → GATED by LoRA mode

Author: Daniel Manzela (April 2026)
"""

import logging
import os

from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Update these to match YOUR Artifact Registry repository
_GCSFUSE_IMAGE_URI = os.getenv(
    "IMAGE_URI",
    "YOUR_REGION-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/gcsfuse-vllm-gemma4:latest"
)

# GCP project and region
_GCP_PROJECT = os.getenv("GCP_PROJECT", "YOUR_PROJECT")
_GCP_REGION = os.getenv("GCP_REGION", "us-central1")

_RESOURCE_LABELS = {
    "component": "slm-serving",
    "model": "gemma-4-27b-a4b",
    "managed-by": "upload-model-py",
}


def _resolve_hf_token() -> str:
    """Resolve HF_TOKEN from environment or GCP Secret Manager."""
    token = os.getenv("HF_TOKEN", "")
    if token:
        logger.info("HF_TOKEN resolved from environment variable")
        return token

    try:
        from google.cloud import secretmanager

        sm = secretmanager.SecretManagerServiceClient()
        token = sm.access_secret_version(
            request={"name": f"projects/{_GCP_PROJECT}/secrets/HF_TOKEN/versions/latest"}
        ).payload.data.decode("UTF-8").strip()
        logger.info("HF_TOKEN resolved from GCP Secret Manager")
        return token
    except Exception as e:
        logger.error(f"Failed to resolve HF_TOKEN: {e}")
        raise RuntimeError(
            "HF_TOKEN not available via environment or Secret Manager. "
            "Set HF_TOKEN env var or ensure Secret Manager access."
        ) from e


def upload_gemma_model(auto_deploy: bool = False):
    """Upload Gemma 4 27B-A4B-it model to Vertex AI Model Registry.

    Supports two deployment modes via SLM_ENABLE_LORA env var:
    - Base-only (default): Maximum context window, no adapter overhead
    - LoRA-enabled: Reduced context, adapter slots pre-allocated

    Uses the GCSFUSE-enabled container image (vLLM 0.17.2rc1.dev133 +
    transformers 5.5.0.dev0 baked in). No runtime pip install required.
    """
    aiplatform.init(project=_GCP_PROJECT, location=_GCP_REGION)

    hf_token = _resolve_hf_token()

    # ── Deployment mode ───────────────────────────────────────
    enable_lora = os.getenv("SLM_ENABLE_LORA", "false").strip().lower() == "true"
    model_version = os.getenv("SLM_MODEL_VERSION", "v1").strip()

    # ── Quantization ──────────────────────────────────────────
    # Gemma 4 is a MoE model (Gemma4ForConditionalGeneration).
    # Default: "none" (BF16 native precision).
    #   - Proven stable on A100 80GB across 30+ production deploys.
    #   - 52GB BF16 weights → ~24GB headroom for KV cache.
    #
    # ⚠️ WARNING: "bitsandbytes" NF4 quantization is currently BROKEN
    # for MoE architectures. vLLM requires get_expert_mapping() which
    # Gemma 4 does not implement. See FORENSIC_RUNBOOK.md §2.6.
    quantization = os.getenv("SLM_QUANTIZATION", "none").strip().lower()
    valid_quantizations = {"none", "bitsandbytes", "awq", "gptq"}
    if quantization not in valid_quantizations:
        raise ValueError(
            f"Invalid SLM_QUANTIZATION='{quantization}'. "
            f"Valid options: {sorted(valid_quantizations)}. "
        )

    # ── Core vLLM args (universal, mode-independent) ──────────────
    vllm_args = [
        "--model=google/gemma-4-27b-a4b-it",
        # served-model-name MUST match the "model" field in inference requests.
        "--served-model-name=openapi",
        "--dtype=bfloat16",
        # NEVER use --kv-cache-dtype=fp8 on A100 SM80 silicon.
        # FP8 KV cache requires SM89+ (H100/L4). A100 is SM80.
        "--kv-cache-dtype=auto",
        # enforce-eager disables CUDA graph compilation. On A100 with 27B MoE,
        # graph compilation causes transient OOM spikes.
        "--enforce-eager",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--port=7080",
        "--host=0.0.0.0",
        "--disable-log-stats",
        # Explicitly set content format to bypass multimodal detection bug.
        "--chat-template-content-format=string",
        # NOTE: --chat-template is stripped by entrypoint.sh at runtime.
        # Kept here for Vertex AI metadata visibility.
        "--chat-template=/chat_template.jinja",
    ]

    # ── Mode-conditional args ──────────────────────────────────
    if enable_lora:
        logger.info("LoRA-enabled deployment mode")
        vllm_args.extend([
            "--enable-lora",
            "--max-lora-rank=32",
            "--max-loras=16",
            "--max-cpu-loras=96",
            # NOTE: --lora-extra-vocab-size REMOVED in vLLM 0.18+.
            "--gpu-memory-utilization=0.92",
            "--max-model-len=32768",
        ])
    else:
        logger.info("Base-only deployment mode (no LoRA)")
        vllm_args.extend([
            "--gpu-memory-utilization=0.95",
            # BF16 weights ~52GB. A100 80GB usable: ~76GB. KV budget: ~24GB.
            "--max-model-len=8192",
        ])

    # ── Quantization flag ─────────────────────────────────────
    if quantization != "none":
        vllm_args.append(f"--quantization={quantization}")
        if quantization == "bitsandbytes":
            vllm_args.append("--load-format=bitsandbytes")
        logger.info(f"Quantization enabled: {quantization}")
    else:
        logger.info("Quantization: none (BF16 native precision)")

    # ── Container image ──────────────────────────────────────
    # ALWAYS use the GCSFUSE-enabled container.
    image_uri = _GCSFUSE_IMAGE_URI
    logger.info(f"Container: {image_uri}")

    # ── Environment variables (mode-gated) ─────────────────────
    env_vars = {"HF_TOKEN": hf_token}
    if enable_lora:
        env_vars["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    # ── Display name ─────────────────────────────────────────
    mode_suffix = "lora" if enable_lora else "base"
    display_name = f"gemma-4-27b-a4b-{model_version}-{mode_suffix}"

    # ── Resource labels (GCP governance) ─────────────────────
    labels = {
        **_RESOURCE_LABELS,
        "model-version": model_version,
        "deployment-mode": mode_suffix,
    }

    # ── Upload ───────────────────────────────────────────────
    logger.info(f"Uploading model: {display_name}")
    logger.info(f"  vLLM args: {' '.join(vllm_args)}")
    logger.info(f"  Labels: {labels}")

    model = aiplatform.Model.upload(
        display_name=display_name,
        serving_container_image_uri=image_uri,
        serving_container_args=vllm_args,
        serving_container_environment_variables=env_vars,
        serving_container_ports=[7080],
        serving_container_health_route="/health",
        serving_container_predict_route="/v1/chat/completions",
        labels=labels,
    )

    print(f"Model Display Name: {display_name}")
    print(f"Model ID: {model.name}")
    print(f"Model Resource Name: {model.resource_name}")

    return model


def deploy_model_to_endpoint(
    model: "aiplatform.Model",
    endpoint_id: str | None = None,
    # Machine spec: a2-ultragpu-1g + 1× A100 80GB is the only single-GPU config
    # that fits Gemma 4 BF16 weights (~52GB).
    # ⚠️ WARNING: Previous default (g2-standard-96 + 8× L4) caused OOM because
    # each L4 has only 24GB VRAM and model loading targets GPU 0 only.
    machine_type: str = "a2-ultragpu-1g",
    accelerator_type: str = "NVIDIA_A100_80GB",
    accelerator_count: int = 1,
    min_replica_count: int = 1,
    max_replica_count: int = 1,
):
    """Deploy model to Vertex AI Endpoint with quota-aware logic.

    Handles three deployment scenarios:
    1. Empty endpoint → deploy at 100% traffic
    2. Existing model + quota exceeded → undeploy first, then deploy at 100%
    3. Existing model + quota available → canary split
    """
    import time

    if endpoint_id is None:
        endpoint_id = os.getenv("SLM_ENDPOINT_ID", "YOUR_ENDPOINT_ID")

    logger.info(f"[Deploy] Target endpoint: {endpoint_id}")

    endpoint = aiplatform.Endpoint(
        endpoint_name=endpoint_id,
        project=_GCP_PROJECT,
        location=_GCP_REGION,
    )

    # ── Snapshot existing deployments ─────────────────────────
    existing_deployed_ids = []
    try:
        gca_endpoint = endpoint.gca_resource
        for dm in gca_endpoint.deployed_models:
            existing_deployed_ids.append(dm.id)
        logger.info(f"[Deploy] Existing deployed models: {existing_deployed_ids}")
    except Exception as e:
        logger.warning(f"[Deploy] Could not list existing models: {e}")

    # ── Quota-aware deployment strategy ──────────────────────
    # With a 1× A100 quota, canary deployment (2 simultaneous GPUs)
    # is impossible. Fall back to direct replacement.
    quota_limited = (
        accelerator_type == "NVIDIA_A100_80GB"
        and len(existing_deployed_ids) >= accelerator_count
    )

    if quota_limited:
        logger.warning(
            "[Deploy] ⚠️ A100 quota is 1 — cannot run canary (needs 2 GPUs). "
            "Falling back to DIRECT REPLACEMENT with brief downtime."
        )
        # Undeploy old model(s) to free quota
        for old_id in existing_deployed_ids:
            logger.info(f"[Deploy] Undeploying old model {old_id} to free A100 quota...")
            try:
                from google.cloud.aiplatform_v1 import EndpointServiceClient
                from google.cloud.aiplatform_v1.types import UndeployModelRequest
                client = EndpointServiceClient(
                    client_options={"api_endpoint": f"{_GCP_REGION}-aiplatform.googleapis.com"}
                )
                request = UndeployModelRequest(
                    endpoint=endpoint.resource_name,
                    deployed_model_id=old_id,
                    traffic_split={},
                )
                operation = client.undeploy_model(request=request)
                logger.info(f"[Deploy] Undeploy LRO started for {old_id}")
                operation.result(timeout=1800)
                logger.info(f"[Deploy] ✅ Undeployed {old_id}")
            except Exception as e:
                logger.error(f"[Deploy] Failed to undeploy {old_id}: {e}")
                raise

        # Deploy new model at 100%
        logger.info("[Deploy] Deploying new model at 100% traffic (direct replacement)")
        model.deploy(
            endpoint=endpoint,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_percentage=100,
            deploy_request_timeout=1800,
        )
        logger.info(f"[Deploy] ✅ Direct replacement complete")
        return endpoint

    # ── Standard deployment (empty endpoint or quota available) ──
    traffic_pct = 100 if not existing_deployed_ids else 10
    if not existing_deployed_ids:
        logger.info("[Deploy] Empty endpoint — deploying at 100% traffic")
    else:
        logger.info(f"[Deploy] Canary deployment at {traffic_pct}% traffic")

    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        traffic_percentage=traffic_pct,
        deploy_request_timeout=1800,
    )

    logger.info(f"[Deploy] ✅ Deployment complete")
    return endpoint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload and optionally deploy Gemma 4 27B-A4B-it to Vertex AI"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="After uploading, automatically deploy to endpoint",
    )
    parser.add_argument(
        "--deploy-only",
        type=str,
        metavar="MODEL_RESOURCE_NAME",
        help="Skip upload; deploy an existing model resource name",
    )
    args = parser.parse_args()

    if args.deploy_only:
        aiplatform.init(project=_GCP_PROJECT, location=_GCP_REGION)
        existing_model = aiplatform.Model(model_name=args.deploy_only)
        deploy_model_to_endpoint(model=existing_model)
    else:
        model = upload_gemma_model(auto_deploy=args.deploy)
        if args.deploy:
            deploy_model_to_endpoint(model=model)
