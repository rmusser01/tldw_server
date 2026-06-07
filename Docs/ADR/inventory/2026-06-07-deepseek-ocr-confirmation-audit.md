# DeepSeek OCR ADR Candidate Confirmation Audit - 2026-06-07

**Related task:** TASK-2275
**Inventory row:** INV-026
**Candidate source:** `Docs/Design/DeepSeek_OCR_Backend.md`
**Follow-up task:** TASK-2276
**Verdict:** Ready for bounded ADR backfill.

## Confirmation Summary

INV-026 is current enough to backfill as an accepted ADR if the ADR is scoped to the implemented DeepSeek OCR integration rather than every aspirational detail in the design note.

The confirmed decision is:

> tldw_server supports a local Transformers-only DeepSeek-OCR backend named `deepseek`, using the upstream HuggingFace `AutoTokenizer`/`AutoModel` plus `model.infer(...)` contract, markdown-oriented defaults, temporary output handling by default, and explicit availability gates for the heavy GPU/FlashAttention dependency stack.

## Evidence

| Area | Current evidence | Confirmation |
| --- | --- | --- |
| Backend ownership | `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/deepseek_ocr.py` defines `DeepSeekOCRBackend` with `name = "deepseek"`. | Confirms the provider/backend naming decision. |
| Local Transformers contract | `_load_transformers()` imports `AutoTokenizer` and `AutoModel`, passes `trust_remote_code=True`, resolves `DEEPSEEK_OCR_MODEL_ID`, supports `DEEPSEEK_OCR_MODEL_REVISION`, prefers `use_safetensors=True`, falls back to `use_safetensors=False`, moves the model to the configured device, and calls `eval()`. | Confirms local HuggingFace ownership. The security caveat remains material because remote code execution is part of the upstream contract. |
| Default prompt and sizing | `_DEFAULT_PROMPT` is `<image>\n<|grounding|>Convert the document to markdown.`. `_resolve_sizes()` defaults to `base_size=1024`, `image_size=640`, and `crop_mode=True`. | Confirms the markdown default and the Gundam-equivalent sizing defaults. The implementation omits the trailing space shown in the historical design text; that is not architecturally significant. |
| Availability gates | `available()` requires `transformers` and `torch`, defaults `DEEPSEEK_OCR_DEVICE` to `cuda`, checks CUDA availability when using CUDA, and requires `flash_attn` only when CUDA plus `flash_attention_2` are selected. `_resolve_attn_impl()` switches the default attention implementation to `eager` on non-CUDA devices when no env override is set. | Confirms default CUDA/FlashAttention gating with env-based escape hatches. The ADR should not claim CPU mode is the preferred or performance-tested deployment. |
| Inference and output extraction | `ocr_image()` writes image bytes to a temporary `page.png`, calls `model.infer(...)` with prompt, image path, output path, sizes, crop mode, `save_results`, and `test_compress`, then returns `_extract_text_from_any(result)`. `_extract_text_from_any()` returns strings directly, extracts common dict/list text fields, or safely stringifies fallback values. | Confirms the upstream inference contract and safe string output decision. |
| Result persistence | `DEEPSEEK_OCR_SAVE_RESULTS` defaults false. `_resolve_output_dir()` uses a temp output path by default and only uses `DEEPSEEK_OCR_OUTPUT_DIR` when saving is enabled and the directory can be created. If saving is enabled without an output dir, it warns and still uses a temporary directory. | Confirms non-persistent-by-default output handling. The ADR should not claim persistent storage unless the user explicitly opts in. |
| Registry/API exposure | `OCR/registry.py` registers `DeepSeekOCRBackend` in `_BACKENDS`, supports explicit `ocr_backend=deepseek`, and includes DeepSeek in default `auto` and `auto_high_quality` priority lists. `/api/v1/ocr/backends` exposes DeepSeek metadata from `describe()`. | Confirms integration into the OCR registry and discovery endpoint. |
| User docs | `Docs/OCR/DeepSeek-OCR.md` and `Docs/OCR/OCR_Providers.md` describe a local Transformers-only backend, manual install, default prompt, env vars, temporary output behavior, GPU-friendly dependencies, and the `trust_remote_code=True` warning. | Confirms the operational docs match the implemented decision. |
| Tests | `test_ocr_backend_deepseek.py` covers availability returning a bool and `DEEPSEEK_OCR_SAVE_RESULTS` using a configured output dir with a stubbed model. Runtime auto-selection tests patch DeepSeek availability in registry ordering. The live OCR PDF integration test is gated by `DEEPSEEK_OCR_RUN_INTEGRATION=1`, CUDA, and local model dependencies. | Confirms local unit coverage exists and live model coverage is intentionally opt-in. |

## Caveats For ADR-Backfill Scope

- Do not claim DeepSeek OCR dependencies are provided by a project optional extra. The current docs require manual installation of the compatible `torch`, `transformers`, and FlashAttention stack.
- Do not claim the `trust_remote_code=True` risk is eliminated. The decision accepts that risk for this backend and limits it to controlled environments.
- Do not claim CUDA or FlashAttention are unconditional requirements. CUDA is the default device, and FlashAttention is required only when CUDA plus `flash_attention_2` are selected; env overrides can choose CPU or alternate attention.
- Do not claim CPU mode has equivalent support or performance. It is an escape hatch/default non-CUDA behavior, not the primary supported operating mode.
- Do not claim DeepSeek has a server/remote mode in this integration. Current docs and code are local Transformers-only.
- Do not claim persistent OCR outputs are written by default. `save_results` defaults false and uses temporary directories unless explicitly enabled with a configured output directory.
- Do not claim the registry priority exactly matches the historical design phrase "after dots/points". Current normal `auto` order is `tesseract`, `nemotron_parse`, `points`, `deepseek`, `hunyuan`, `dots`, `dolphin`, `llamacpp`, `chatllm`; `auto_high_quality` places `deepseek` after `hunyuan` and before `points`/`dots` unless config overrides priority.
- Do not claim routine test runs validate live model inference. The live endpoint integration test is explicitly gated because it requires CUDA and local model/dependency setup.

## Inventory Disposition

Update INV-026 from `Needs owner review` to `Current governing` for a bounded OCR/provider backfill. TASK-2276 should create the accepted ADR and keep the caveats above explicit.
