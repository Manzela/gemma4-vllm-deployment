# Cycle 2 — LoRA Enablement Attempts Log

> **Duration:** ~6 hours | **Attempts:** 8 | **Final Status:** ❌ Reverted to Base-Only

This log documents every attempt to enable LoRA adapter support for Gemma 4 27B-A4B-it during the second deployment cycle.

---

## Summary

| Attempt | Strategy | Blocker | Resolution |
|---|---|---|---|
| 1 | BF16 + LoRA (24 slots) | GCP transient error | Retry |
| 2 | BF16 + LoRA (4 slots) | Wrong GPU (L4 vs A100) | Fixed defaults |
| 3 | NF4 + LoRA (24 slots) | MoE quant crash | Abandoned NF4 |
| 4 | BF16 + LoRA (4 slots, A100) | Quota exceeded | Direct replacement |
| 5 | Direct replacement deploy | LoRA mixin missing | Confirmed blocker |
| 6 | vLLM 0.19.0 upgrade | gemma4 not in transformers 4.x | Dependency triangle |
| 7 | vLLM 0.19.0 + transformers 5.x | is_offline_mode import | Hub version conflict |
| 8 | Hub version override | Same import error | Mathematically unsolvable |

---

## Detailed Attempt Log

### Attempt 1: BF16 + LoRA (Aggressive Config)
**Config:**
- GPU: A100 80GB
- LoRA: 24 slots, rank 64
- Context: 32768 tokens

**Status:** ❌ Error code 13 (GCP Internal)
**Analysis:** Transient GCP infrastructure error. No container logs available. This error is non-deterministic — GCP infrastructure was possibly restarting the A100 node pool.

**Learning:** GCP error code 13 is always transient. Retry after 5 minutes.

---

### Attempt 2: BF16 + LoRA (Conservative Config)
**Config:**
- GPU: *Intended* A100 80GB
- LoRA: 4 slots, rank 32
- Context: 8192 tokens

**Status:** ❌ `torch.OutOfMemoryError`
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 968.00 MiB.
GPU 0 has a total capacity of 22.03 GiB
```

**Root Cause:** The deployment function had hardcoded defaults:
```python
# BEFORE (broken)
machine_type = "g2-standard-96"    # 8× L4 GPUs!
accelerator_type = "NVIDIA_L4"     # 24GB each

# AFTER (fixed)
machine_type = "a2-ultragpu-1g"    # 1× A100 80GB
accelerator_type = "NVIDIA_A100_80GB"
```

**Key Diagnostic:** The error says `22.03 GiB`. A100 80GB has 80GB. This is an L4 (24GB, 22GB usable).

> 🚨 **ALWAYS check the GPU capacity number in OOM error messages.** If it's not 80GB, you're on the wrong GPU.

---

### Attempt 3: NF4 + LoRA
**Strategy:** If we can quantize to NF4 (52GB → ~14GB), we'd have ~60GB for LoRA buffers + KV cache.

**Config:**
- GPU: A100 80GB
- Quantization: bitsandbytes NF4
- LoRA: 24 slots, rank 64

**Status:** ❌ Runtime crash
```
AttributeError: MoE Model Gemma4ForConditionalGeneration does not support
BitsAndBytes quantization yet. Ensure this model has 'get_expert_mapping' method.
```

**Analysis:** This is the same error as Cycle 1 v22. vLLM's bitsandbytes quantizer requires:
```python
def get_expert_mapping(self) -> Dict[str, int]:
    """Return mapping of expert weight names to expert indices."""
    ...
```
Gemma 4's model class does not implement this. This was also confirmed in vLLM 0.19.0 source code — the requirement exists across all recent versions.

**Conclusion:** NF4 quantization for MoE models requires upstream changes in both vLLM and potentially the model implementation.

---

### Attempt 4: BF16 + LoRA (A100 Confirmed)
**Config:**
- GPU: A100 80GB (defaults fixed from Attempt 2)
- LoRA: 4 slots, rank 32
- Context: 8192 tokens

**Status:** ❌ Quota exceeded
```
The following quotas are exceeded: CustomModelServingA10080GBGPUsPerProjectPerRegion
```

**Root Cause:** Previous model (from Attempt 3) was still deployed — consuming the single A100 allocation.

**Fix:** Must undeploy existing model before redeploying with 1× A100 quota.

---

### Attempt 5: Direct Replacement Deploy
**Config:**
- Same as Attempt 4, but using direct replacement (undeploy first)

**Status:** ❌ LoRA mixin missing
```
ValueError: Gemma4ForConditionalGeneration does not support LoRA yet
```

**Analysis:** Now that we're actually on A100 80GB with correct config, the real blocker is exposed: vLLM 0.17.2rc1.dev133 has not registered Gemma 4's attention layers for LoRA patching.

**Conclusion:** LoRA cannot work in the base image's vLLM version. Need to either:
1. Wait for upstream vLLM fix
2. Upgrade vLLM (if they add Gemma 4 LoRA support)
3. Monkey-patch the model class at runtime

---

### Attempt 6: vLLM 0.19.0 Upgrade
**Strategy:** Maybe vLLM 0.19.0 has LoRA support for Gemma 4? Let's try upgrading.

**Status:** ❌ gemma4 not recognized
```
The checkpoint you are trying to load has model type 'gemma4' but
Transformers does not recognize this architecture
```

**Analysis:** Installing vLLM 0.19.0 from PyPI pulled `transformers 4.57.6` (auto-resolved). The `gemma4` model type only exists in `transformers 5.x`.

**Pipeline:**
```
pip install vllm==0.19.0
  → pulls transformers==4.57.6 (latest satisfying <5)
  → pulls huggingface_hub==0.35.3 (latest satisfying <1.0)
  → KeyError: 'gemma4' (model type not registered)
```

---

### Attempt 7: vLLM 0.19.0 + transformers 5.x
**Strategy:** Install vLLM 0.19.0, then force-install transformers 5.6.0.dev0 with `--no-deps`.

**Status:** ❌ `is_offline_mode` import error
```
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```

**Analysis:** The import chain:
```
vLLM 0.19.0
  → transformers_utils/config.py line 18: from transformers import AutoConfig
    → transformers/__init__.py
      → transformers/hub.py: from huggingface_hub import is_offline_mode
        ❌ FAILS: is_offline_mode moved to huggingface_hub.utils in 0.36+
```

vLLM 0.19.0 installs `huggingface_hub 0.35.3` (satisfying `<1.0`). But `transformers 5.6.0.dev0` expects `huggingface_hub >= 0.36+` where `is_offline_mode` has a new import path.

---

### Attempt 8: Hub Version Override
**Strategy:** Force `huggingface_hub==0.36.2` alongside vLLM 0.19.0 + transformers 5.x.

**Status:** ❌ Same import error (different path)
**Analysis:** vLLM 0.19.0's internal code also imports from `huggingface_hub` with assumptions about the 0.34–0.35 API. Upgrading hub to 0.36+ breaks vLLM's own imports.

**Conclusion:** The dependency triangle is **mathematically unsolvable**:
```
vLLM 0.19.0: hub < 1.0 AND transformers < 5
Gemma 4:     transformers ≥ 5.5
transformers 5.x: hub ≥ 1.5

hub < 1.0 ∩ hub ≥ 1.5 = ∅ (empty set)
transformers < 5 ∩ transformers ≥ 5.5 = ∅ (empty set)
```

### Resolution: Revert to Base-Only
**Action:** Abandoned LoRA enablement. Reverted to v30-equivalent base-only deployment.

**Rationale:**
1. LoRA requires vLLM model class support → not available in 0.17.2rc1
2. Upgrading vLLM → dependency triangle blocks Gemma 4 model loading
3. No combination of PyPI packages can satisfy all constraints simultaneously

**Path Forward:** Wait for one of:
- vLLM team adds Gemma 4 LoRA mixin + relaxes transformers version constraint
- Google releases updated `pytorch-vllm-serve:gemma4` image with LoRA support
- Community creates a vLLM fork with the necessary patches

---

## Key Learnings

1. **Always verify GPU type in OOM errors** — the capacity number tells you exactly which GPU you're on
2. **Dependency triangles can be unsolvable** — when three packages have pairwise conflicts, no valid solution exists
3. **Google's base images contain custom patches** — you cannot reproduce them from PyPI
4. **Quota management is critical** — with 1× A100, every failed deploy blocks the next one
5. **Image digest pinning catches you** — pushing `:latest` doesn't help already-registered models

---

*Log maintained by [Daniel Manzela](https://github.com/Manzela). April 2026.*
