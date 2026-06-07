# ADR-024: DeepSeek OCR Local Transformers Backend

**Status:** Accepted
**Date:** 2026-06-07
**Backfilled from:** `Docs/Design/DeepSeek_OCR_Backend.md`
**Decision owner:** TASK-2275 confirmation and TASK-2276 DeepSeek backfill scope
**Related task:** TASK-2276 (`backlog/tasks/task-2276 - Backfill-DeepSeek-OCR-backend-ADR.md`)
**Related spec/plan:** `Docs/ADR/inventory/2026-06-07-deepseek-ocr-confirmation-audit.md`

## Decision

DeepSeek OCR is supported as a local Transformers-only OCR backend named `deepseek`, using HuggingFace `AutoTokenizer`/`AutoModel` with upstream `trust_remote_code=True`, markdown-oriented Gundam-equivalent defaults, safe string output extraction, temporary result handling by default, and explicit local dependency availability gates for `torch`, `transformers`, CUDA, and FlashAttention with env-based device/attention overrides.

## Context

DeepSeek-OCR is a heavy local model integration at the OCR/provider boundary. It plugs into the generic OCR registry and PDF/evaluation OCR paths, but its upstream contract differs from lightweight CLI backends and remote LLM-style OCR backends: it loads model code through HuggingFace Transformers, requires `trust_remote_code=True`, expects GPU-oriented dependencies for the default path, and uses `model.infer(...)` with temporary image and output paths.

TASK-2275 confirmed the current implementation behavior that bounds this ADR:

- `DeepSeekOCRBackend.name` is `deepseek`.
- The backend is local Transformers-only; there is no server or remote mode in this integration.
- Model loading uses `AutoTokenizer.from_pretrained(...)` and `AutoModel.from_pretrained(...)` with `trust_remote_code=True`, optional `DEEPSEEK_OCR_MODEL_REVISION`, `DEEPSEEK_OCR_MODEL_ID`, `use_safetensors=True` first, and a fallback to `use_safetensors=False`.
- The default prompt is markdown-oriented and includes the required `<image>` token. The default sizes are equivalent to the Gundam preset: `base_size=1024`, `image_size=640`, and `crop_mode=True`.
- `available()` requires `transformers` and `torch`, defaults the device to CUDA, requires CUDA when using CUDA, and requires `flash_attn` only when CUDA plus `flash_attention_2` are selected. Env overrides can choose CPU or a different attention implementation, but CPU is not the preferred or performance-proven path.
- `ocr_image()` writes input image bytes to a temporary file, calls upstream `model.infer(...)`, and extracts string output from common return shapes before falling back to safe stringification.
- `DEEPSEEK_OCR_SAVE_RESULTS` defaults false. Output paths are temporary by default, and configured persistent output is used only when saving is explicitly enabled and `DEEPSEEK_OCR_OUTPUT_DIR` is usable.
- The OCR registry exposes explicit `ocr_backend=deepseek`, includes DeepSeek in `auto` and `auto_high_quality` ordering, and `/api/v1/ocr/backends` exposes resolved DeepSeek metadata.
- Docs describe manual dependency installation and the `trust_remote_code=True` security warning.

This ADR is intentionally bounded. It does not claim the model dependencies are packaged by a project optional extra, does not eliminate the `trust_remote_code=True` risk, does not make CPU mode a first-class performance target, does not add a server mode, does not persist OCR outputs by default, and does not claim routine test runs validate live model inference.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Expose DeepSeek OCR only as a remote/server OCR provider | Current implementation and docs are local Transformers-only, and adding a server mode would introduce separate deployment, auth, and health-check decisions. |
| Treat DeepSeek OCR like a lightweight default OCR backend | Its dependency stack is heavy, GPU-oriented, and security-sensitive because upstream loading requires `trust_remote_code=True`; availability gates must remain explicit. |
| Persist upstream DeepSeek OCR outputs by default | Persistent outputs can write model artifacts and intermediate files. The accepted default is temporary output handling unless the operator opts in with `DEEPSEEK_OCR_SAVE_RESULTS` and an output directory. |
| Make CPU/eager execution the primary supported path | Env overrides can select CPU or alternate attention behavior, but upstream guidance and current testing center on the GPU-oriented path. |
| Add a project `ocr_deepseek` optional extra as part of this decision | Current docs require manual installation of compatible `torch`, `transformers`, and FlashAttention builds. Packaging policy can be decided separately if the dependency stack stabilizes. |
| Backfill the historical registry priority phrase exactly | The design note says "after dots/points", but current `auto` and `auto_high_quality` ordering differ. This ADR accepts the registered backend and documents the actual priority behavior as a caveat rather than rewriting it into an unverified policy. |

## Consequences

DeepSeek OCR changes should preserve the `deepseek` backend identity, local Transformers ownership, resolved env configuration, and explicit availability checks unless a later ADR supersedes this one.

Security-sensitive model loading must remain visible in docs and code review. The use of `trust_remote_code=True` is accepted for this backend because upstream requires it, not because the risk is removed. Operators should enable this backend only in controlled environments and review the model code they execute.

The default deployment path remains GPU-oriented. CUDA and FlashAttention requirements are availability gates for the default configuration, while CPU or alternate-attention env overrides are compatibility escape hatches. Tests and docs should avoid implying equivalent CPU performance.

Result persistence remains opt-in. Backend output should continue to use temporary image/output paths by default, and persistent output should require explicit `DEEPSEEK_OCR_SAVE_RESULTS` plus an operator-managed output directory.

The OCR registry and backend discovery endpoint should continue to surface DeepSeek metadata, but registry priority remains current implementation behavior rather than a broad provider-ranking policy. Changes to default OCR ordering should be reviewed as registry behavior changes.

Routine CI should keep lightweight unit/registry coverage for this backend. Live model endpoint validation remains an explicitly gated integration path because it requires CUDA, local model dependencies, and operator opt-in.

## Follow-up

- Use this ADR as the covering record for INV-026.
- Keep `Docs/ADR/inventory/2026-06-07-deepseek-ocr-confirmation-audit.md` as the evidence record and caveat boundary for this backfill.
- Create separate ADRs or focused tasks if DeepSeek gains a server mode, a packaged optional dependency extra, different default registry priority, persistent artifact storage, or stronger model provenance controls.
