# Known Limitations — Gemma 4 27B-A4B-it on Vertex AI

> **Author:** Daniel Manzela | **Date:** April 2026  
> **Status:** Tracking active blockers for full Gemma 4 feature enablement

---

## Active Blockers

### 1. ❌ No LoRA Fine-Tuning

**Status:** Blocked — requires upstream vLLM changes

**Root Cause:** vLLM 0.17.2rc1.dev133 lacks complete LoRA mixin for `Gemma4ForConditionalGeneration`. The MoE architecture's attention layers are not registered for LoRA patching.

**Impact:** Cannot apply fine-tuned adapters. All inference uses the base model only.

**Unblock Path:**
- vLLM adds LoRA mixin for Gemma 4 MoE attention layers
- OR: Monkey-patch the model class in entrypoint.sh to register LoRA targets
- OR: Use alternative serving framework (TGI, SGLang) with LoRA support

---

### 2. ❌ No NF4 Quantization

**Status:** Blocked — `get_expert_mapping()` not implemented for Gemma 4 MoE

**Root Cause:** vLLM's bitsandbytes loader requires `get_expert_mapping()` for MoE models. This method is not implemented for Gemma 4.

**Impact:** Model must run at BF16 precision (52GB weights). Cannot reduce VRAM to ~14GB via NF4, which would enable LoRA headroom on A100.

**Unblock Path:**
- Implement `get_expert_mapping()` for `Gemma4ForConditionalGeneration`
- OR: Create pre-quantized GPTQ/AWQ checkpoint (separate effort)

---

### 3. ❌ No Vision/Multimodal

**Status:** Blocked — requires chat template extension

**Root Cause:** The simplified `chat_template.jinja` only handles text turns. Gemma 4's native multimodal template uses Jinja2 `namespace()` which crashes in vLLM's sandboxed environment.

**Impact:** Cannot use Gemma 4's image understanding capabilities.

**Unblock Path:**
- Create a vision-capable chat template without `namespace()`
- This is the most achievable fix — community contribution welcome

---

### 4. ❌ No Thinking/Reasoning Mode

**Status:** Not explored — requires inference configuration

**Root Cause:** Gemma 4 supports a "thinking" mode where the model's chain-of-thought is visible. This requires specific template configuration and potentially different tokenizer settings.

**Impact:** Cannot leverage Gemma 4's explicit reasoning capabilities.

**Unblock Path:**
- Extend chat template with thinking tokens
- Test with vLLM's sampling parameters
- Community experimentation needed

---

### 5. ⚠️ 8K Context Limit

**Status:** Constrained by BF16 VRAM budget

**Root Cause:** BF16 weights consume ~52GB on A100 80GB, leaving ~24GB for KV cache. At 8192 tokens, the KV cache fits comfortably. The model's native 128K context would require ~150GB+ for KV cache alone.

**Impact:** Long document processing is limited to 8K tokens.

**Mitigation Path:**
- If NF4 quantization is enabled: weights ~14GB → ~60GB for KV cache → potentially 32K+ context
- Multi-GPU deployment (2× A100 80GB with tensor parallelism)
- Test if 16384 tokens is feasible with current BF16 budget

---

### 6. ⚠️ No Canary Deployments

**Status:** Constrained by GPU quota

**Root Cause:** 1× A100 80GB per-project quota means only one model can be deployed at a time. Canary deployment requires 2× simultaneous models.

**Impact:** All deployments use direct replacement with ~5 minutes of downtime.

**Mitigation Path:**
- Request quota increase from GCP
- Use a different project for canary testing

---

### 7. ⚠️ Stop Token Client-Side Only

**Status:** Workaround in place, but brittle

**Root Cause:** vLLM 0.17.2rc1 does not honor list-type `eos_token_id` in `generation_config.json`. The end-of-turn token (ID 106) must be passed as `stop_token_ids` in every inference request.

**Impact:** If any client omits `stop_token_ids: [106]`, the model loops indefinitely.

**Mitigation Path:**
- vLLM upgrade that supports list-type `eos_token_id`
- OpenAI-compatible proxy that injects stop tokens automatically

---

### 8. ⚠️ GCSFUSE Mount Unverified

**Status:** Installed but untested with Vertex AI permissions

**Root Cause:** Vertex AI Prediction containers may not grant CAP_SYS_ADMIN required for FUSE mounts. The `--allow-other` flag was removed in GCSFUSE v2.5.3.

**Impact:** LoRA adapter loading from GCS via `/mnt/gcs/` path may not work even if LoRA support is enabled.

**Mitigation Path:**
- Test GCSFUSE mount with Vertex AI's actual container permissions
- Use `user_allow_other` in `/etc/fuse.conf` (already configured)
- Fallback: download adapters via API instead of FUSE mount

---

## Feature Enablement Roadmap

```mermaid
gantt
    title Gemma 4 Feature Enablement
    dateFormat YYYY-MM
    axisFormat %b %Y
    
    section Blocked (Upstream)
    LoRA for MoE             :crit, lora, 2026-04, 2026-09
    NF4 Quantization (MoE)   :crit, nf4, 2026-04, 2026-09
    
    section Community Achievable
    Vision Chat Template     :active, vision, 2026-04, 2026-06
    Thinking Mode            :think, 2026-05, 2026-07
    
    section Optimization
    Context Window (16K+)    :ctx, after nf4, 2026-09, 2026-10
    Multi-GPU Deployment     :mgpu, 2026-06, 2026-08
    Canary Deployments       :canary, 2026-05, 2026-06
```

---

*Document authored by [Daniel Manzela](https://github.com/Manzela). April 2026.*
