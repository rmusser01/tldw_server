# DeepSeek OCR Backend (Transformers)

ADR coverage: The implemented, still-governing backend decision from this design is backfilled by `Docs/ADR/024-deepseek-ocr-local-transformers-backend.md`. See that ADR for the accepted scope and caveats.

## Summary
Add a DeepSeek-OCR backend that runs local HuggingFace Transformers inference on image bytes and plugs into the OCR registry. The backend will follow the upstream `model.infer(...)` API and expose lightweight config via env vars.

## Upstream Transformers Contract (for reference)
- Load with `AutoTokenizer` + `AutoModel` using `trust_remote_code=True` and `_attn_implementation='flash_attention_2'`.
- Example device/dtype: `model.eval().cuda().to(torch.bfloat16)`.
- Inference uses `model.infer(tokenizer, prompt=..., image_file=..., output_path=..., base_size=..., image_size=..., crop_mode=..., save_results=..., test_compress=...)`.
- Prompt must include `<image>` and supports task instructions (e.g. markdown conversion, free OCR).
- Resolution presets: Tiny/Small/Base/Large (single-scale) and Gundam (multi-scale + crop mode).

Sources: https://huggingface.co/deepseek-ai/DeepSeek-OCR

## Decisions
- Backend name: `deepseek`.
- Default prompt: `<image>\n<|grounding|>Convert the document to markdown. ` (layout-aware output). Provide `Free OCR` as a plain-text option.
- Default resolution: Gundam preset (base_size=1024, image_size=640, crop_mode=True). Allow override via env.
- Output handling: return string output; if non-string, stringify safely.
- Save results: default `False` (avoid persistent writes; use temp output dir when needed).
- Availability gating:
  - Require `torch` + `transformers` imports.
  - Require CUDA availability by default (upstream docs are NVIDIA GPU focused).
  - Require `flash_attn` when `_attn_implementation='flash_attention_2'` is used.
  - Allow override via env (e.g., CPU mode, alternate attention impl) if needed later.

## Configuration (env vars)
Proposed envs (minimal and consistent with existing OCR backends):
- `DEEPSEEK_OCR_MODEL_ID` (default: `deepseek-ai/DeepSeek-OCR`)
- `DEEPSEEK_OCR_PROMPT` (default: markdown prompt)
- `DEEPSEEK_OCR_BASE_SIZE` (default: 1024)
- `DEEPSEEK_OCR_IMAGE_SIZE` (default: 640)
- `DEEPSEEK_OCR_CROP_MODE` (default: true)
- `DEEPSEEK_OCR_SAVE_RESULTS` (default: false)
- `DEEPSEEK_OCR_TEST_COMPRESS` (default: false)
- `DEEPSEEK_OCR_DTYPE` (default: bfloat16)
- `DEEPSEEK_OCR_ATTN_IMPL` (default: flash_attention_2)
- `DEEPSEEK_OCR_DEVICE` (default: cuda)

Resolution presets (optional helper mapping):
- Tiny: base_size=512, image_size=512, crop_mode=false
- Small: base_size=640, image_size=640, crop_mode=false
- Base: base_size=1024, image_size=1024, crop_mode=false
- Large: base_size=1280, image_size=1280, crop_mode=false
- Gundam: base_size=1024, image_size=640, crop_mode=true

## Integration Notes
- Implement backend in `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/deepseek_ocr.py`.
- Add to registry with priority after dots/points unless overridden by config.
- Provide `describe()` with resolved prompt, sizes, device/dtype, and attention impl.
- Use temp files for input image and output path; clean up after each call.
- Recommend `OCR_PAGE_CONCURRENCY=1` for GPU stability.

## Risks / Constraints
- Heavy GPU requirements; OOM risk with multi-page PDFs.
- `trust_remote_code=True` is required (security review needed).
- FlashAttention and CUDA versions must align with upstream guidance.
