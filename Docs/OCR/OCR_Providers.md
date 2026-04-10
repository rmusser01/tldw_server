# OCR Providers Setup

This guide covers enabling and using optional OCR providers with the PDF pipeline. You can select the provider via the `ocr_backend` option in API/WebUI forms.

Supported backends
- mineru (document-level CLI, PDF-only)
- tesseract (built-in, CLI)
- dots (dots.ocr project)
- points (Tencent POINTS-Reader)
- deepseek (DeepSeek-OCR, Transformers)
- hunyuan (Tencent HunyuanOCR, vLLM + Transformers)
- dolphin (ByteDance Dolphin-v2, Transformers + remote server)
- llamacpp (OCR-capable llama.cpp deployment, remote/managed/cli)
- chatllm (OCR-capable ChatLLM deployment, remote/managed/cli)

Common usage
- Set `enable_ocr=true` and choose `ocr_backend` (`mineru`, `tesseract`, `dots`, `points`, `deepseek`, `hunyuan`, `dolphin`, `llamacpp`, or `chatllm`).
- `ocr_mode`: `fallback` (only pages lacking text) or `always` (force OCR).
- `ocr_dpi`: 200-300 is a good balance.
- Structured outputs:
  - `ocr_output_format`: `text|markdown|json`
  - `ocr_prompt_preset`: `general|doc|table|spotting|json`
- Parallelism: `OCR_PAGE_CONCURRENCY` controls the global per-page OCR concurrency cap (default 1). The PDF pipeline now uses `min(OCR_PAGE_CONCURRENCY, backend profile max_page_concurrency)`, so backend-local caps are respected too. Keep both values small for GPU-backed models unless you have validated higher throughput.
- Config override: `Config_Files/config.txt` supports `[OCR] backend_priority` (comma-separated list or JSON array) to control auto selection order.

MinerU-specific notes
- `ocr_backend=mineru` is PDF-only in v1 and runs once per document rather than once per page.
- MinerU is discoverable via `/api/v1/ocr/backends`, but it is opt-in only and excluded from `auto`, `auto_high_quality`, and `backend_priority`.
- `ocr_lang` and `ocr_dpi` are currently advisory only for MinerU. The PDF pipeline records them in metadata and warns when they are ignored.
- MinerU normalizes output into Markdown plus a bounded structured payload with `pages`, `tables`, and bounded excerpts from upstream artifacts.

## MinerU (backend: `mineru`)

Summary
- Document-level CLI integration for PDF OCR and layout parsing.
- Best fit when table fidelity and structured downstream payloads matter more than page-image OCR interchangeability.
- Returns Markdown as the primary OCR content and stores normalized structured artifacts under `analysis_details.ocr.structured`.

Environment
- `MINERU_CMD`: command used to launch MinerU (default `mineru`). Tokenized safely and never shell-interpolated.
- `MINERU_TIMEOUT_SEC`: whole-document timeout in seconds (default `120`).
- `MINERU_MAX_CONCURRENCY`: maximum concurrent MinerU document runs (default `1`).
- `MINERU_TMP_ROOT`: optional temp root for MinerU working directories.
- `MINERU_DEBUG_SAVE_RAW`: `true` to include full raw `content_list.json` and `middle.json` payloads in the structured artifact block.

Behavior
- Requested via `enable_ocr=true` and `ocr_backend=mineru`.
- `ocr_mode=fallback` preserves parser text on MinerU failure and appends a warning.
- `ocr_mode=always` replaces parser text when MinerU succeeds, but still preserves parser text on failure.
- `ocr_output_format=text|markdown|json` is accepted. MinerU always keeps a normalized structured JSON payload; `result["content"]` is derived from the Markdown output.

Notes
- MinerU does not participate in generic image OCR flows in v1.
- The CLI output directory may be nested; the adapter resolves the primary artifact directory automatically.

## Llama.cpp OCR (backend: `llamacpp`)

Summary
- OCR-capable llama.cpp deployments can run in `remote`, `managed`, `cli`, or `auto` mode.
- Remote mode is the preferred path when you already have a llama.cpp/OpenAI-compatible endpoint or an existing server managed elsewhere.
- Managed mode is OCR-private and single-process only in v1.

Environment
- `LLAMACPP_OCR_MODE`: `auto|remote|managed|cli`.
- `LLAMACPP_OCR_AUTO_ELIGIBLE`: opt-in flag for `auto`.
- `LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE`: opt-in flag for `auto_high_quality`.
- `LLAMACPP_OCR_MAX_PAGE_CONCURRENCY`: backend-local page concurrency cap. The PDF pipeline uses the smaller of this and `OCR_PAGE_CONCURRENCY`.

Remote
- `LLAMACPP_OCR_HOST`, `LLAMACPP_OCR_PORT`, `LLAMACPP_OCR_MODEL_PATH`, `LLAMACPP_OCR_USE_DATA_URL`.

Managed
- `LLAMACPP_OCR_ARGV`, `LLAMACPP_OCR_MODEL_PATH`, `LLAMACPP_OCR_HOST`, `LLAMACPP_OCR_PORT`, `LLAMACPP_OCR_STARTUP_TIMEOUT_SEC`.
- `LLAMACPP_OCR_ALLOW_MANAGED_START=true|false` controls whether the server may start the managed process itself.
- In `auto`, a configured managed startup profile is preferred over `remote`; set `LLAMACPP_OCR_MODE=remote` to force an external endpoint.

CLI
- `LLAMACPP_OCR_ARGV`, `LLAMACPP_OCR_MODEL_PATH`.

Docs
- See [Docs/OCR/LlamaCpp-OCR.md](./LlamaCpp-OCR.md) for the backend-specific setup guide.

Notes
- `llamacpp` supports the existing OCR presets and structured output contract.
- Managed mode should be used only in single-API-process deployments in v1.

## ChatLLM OCR (backend: `chatllm`)

Summary
- ChatLLM OCR mirrors the same OCR contract as llama.cpp, but with ChatLLM-specific runtime configuration.
- Supported modes are `remote`, `managed`, `cli`, and `auto`.
- Managed mode is OCR-private and single-process only in v1.

Environment
- `CHATLLM_OCR_MODE`: `auto|remote|managed|cli`.
- `CHATLLM_OCR_AUTO_ELIGIBLE`: opt-in flag for `auto`.
- `CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE`: opt-in flag for `auto_high_quality`.
- `CHATLLM_OCR_MAX_PAGE_CONCURRENCY`: backend-local page concurrency cap. The PDF pipeline uses the smaller of this and `OCR_PAGE_CONCURRENCY`.

Remote
- `CHATLLM_OCR_URL`, `CHATLLM_OCR_MODEL`, `CHATLLM_OCR_API_KEY`.

Managed
- `CHATLLM_OCR_SERVER_BINARY`, `CHATLLM_OCR_MODEL_PATH`, `CHATLLM_OCR_HOST`, `CHATLLM_OCR_PORT`, `CHATLLM_OCR_STARTUP_TIMEOUT_SEC`, `CHATLLM_OCR_SERVER_ARGS_JSON`, `CHATLLM_OCR_HEALTHCHECK_URL`.
- `CHATLLM_OCR_ALLOW_MANAGED_START=true|false` controls whether the server may start the managed process itself.

CLI
- `CHATLLM_OCR_CLI_BINARY`, `CHATLLM_OCR_MODEL_PATH`, `CHATLLM_OCR_CLI_ARGS_JSON`.

Docs
- See [Docs/OCR/ChatLLM-OCR.md](./ChatLLM-OCR.md) for the backend-specific setup guide.

Notes
- `chatllm` supports the existing OCR presets and structured output contract.
- Managed mode should be used only in single-API-process deployments in v1.

## dots.ocr (backend: `dots`)

Quick setup
- Install per upstream repo:
  - git clone https://github.com/rednote-hilab/dots.ocr.git && cd dots.ocr
  - Install a compatible PyTorch build (CUDA/CPU), then `pip install -e .`
  - Download model weights: `python3 tools/download_model.py` (use a directory name without periods, e.g., `DotsOCR`).
- Recommended: serve with vLLM for performance (see upstream docs).

Install via extras (optional)
- You can install the project with the dots extra:
  - `pip install .[ocr_dots]`
  - This pulls the dots.ocr repo via a VCS dependency.

Backend specifics
- CLI mode: shells out to `python -m dots_ocr.parser <image.png>` per page (or use `DOTS_OCR_CMD` to specify an explicit command).
- vLLM mode: if `DOTS_VLLM_URL` is set, calls an OpenAI-compatible endpoint (`/v1/chat/completions`). Use `DOTS_VLLM_MODEL` (served-model-name), `DOTS_VLLM_TIMEOUT`, and `DOTS_VLLM_USE_DATA_URL` (default true) to control behavior.
- Env: `DOTS_OCR_PROMPT` (default `prompt_ocr`); `DOTS_OCR_CMD` can point to a script for custom setups.

Docs
- See pyproject extras and provider docs for optional OCR dependencies.

Pros/cons
- Pros: strong layout understanding; good accuracy on scanned docs; vLLM option for scale.
- Cons: heavy install; per-page CLI call if not served; model weights management.

## POINTS-Reader (backend: `points`)

Modes
- Transformers (local): loads `tencent/POINTS-Reader` via `transformers` and WePOINTS toolkit.
- SGLang (server): calls an OpenAI-compatible `/v1/chat/completions` endpoint.

Environment
- `POINTS_MODE`: `auto` (default), `sglang`, `transformers`.
- Transformers:
  - `POINTS_MODEL_PATH` (default `tencent/POINTS-Reader`), requires `transformers` + `torch` and WePOINTS installed.
- SGLang:
  - `POINTS_SGLANG_URL` (default `http://127.0.0.1:8081/v1/chat/completions`)
  - `POINTS_SGLANG_MODEL` (default `WePoints`)
- Shared generation (both modes): `POINTS_MAX_NEW_TOKENS`, `POINTS_TEMPERATURE`, `POINTS_REPETITION_PENALTY`, `POINTS_TOP_P`, `POINTS_TOP_K`, `POINTS_DO_SAMPLE`.
- Prompt: `POINTS_PROMPT` (defaults to extracting tables as HTML and text as Markdown).

Docs
- See detailed guide at `Docs/OCR/POINTS-Reader.md`.

Install via extras (optional)
- Local transformers path: `pip install .[ocr_points_transformers]`
  - Includes `transformers`, `torch`, and the WePOINTS toolkit via VCS.
- SGLang client only: `pip install .[ocr_points_sglang]`

## DeepSeek-OCR (backend: `deepseek`)

Summary
- Local Transformers-only backend (no server mode yet).
- Requires `trust_remote_code=True` and GPU-friendly dependencies.
- Default prompt targets Markdown conversion with layout grounding; override via env.

Environment
- `DEEPSEEK_OCR_MODEL_ID` (default: `deepseek-ai/DeepSeek-OCR`)
- `DEEPSEEK_OCR_PROMPT` (default: markdown conversion prompt)
- `DEEPSEEK_OCR_BASE_SIZE`, `DEEPSEEK_OCR_IMAGE_SIZE`, `DEEPSEEK_OCR_CROP_MODE`
- `DEEPSEEK_OCR_DTYPE` (default: `bfloat16`)
- `DEEPSEEK_OCR_ATTN_IMPL` (default: `flash_attention_2`)
- `DEEPSEEK_OCR_DEVICE` (default: `cuda`)
- `DEEPSEEK_OCR_SAVE_RESULTS` (default: `false`)
- `DEEPSEEK_OCR_TEST_COMPRESS` (default: `false`)
- `DEEPSEEK_OCR_OUTPUT_DIR` (optional; used when `DEEPSEEK_OCR_SAVE_RESULTS=true`)

Install (manual)
- Install a compatible `torch`, `transformers`, and flash-attn stack for your CUDA version.
- Set `DEEPSEEK_OCR_MODEL_ID` if you want a local cache path or different repo.
- Security note: upstream requires `trust_remote_code=True`; use only in controlled environments.

Docs
- See `Docs/OCR/DeepSeek-OCR.md` for setup details and prompt tips.

## HunyuanOCR (backend: `hunyuan`)

Summary
- Supports vLLM (OpenAI-compatible) and local Transformers.
- Good for structured outputs when paired with `ocr_output_format=json` or prompt presets.

Environment
- `HUNYUAN_MODE`: `auto` (default), `vllm`, `transformers`.
- `HUNYUAN_PROMPT`: prompt override (free-form).
- `HUNYUAN_PROMPT_PRESET`: `general|doc|table|spotting|json`.

vLLM
- `HUNYUAN_VLLM_URL`: OpenAI-compatible `/v1/chat/completions` endpoint.
- `HUNYUAN_VLLM_MODEL`: served model name.
- `HUNYUAN_VLLM_TIMEOUT`: request timeout seconds.
- `HUNYUAN_VLLM_USE_DATA_URL`: `true` to send base64 image URLs (recommended).

Transformers
- `HUNYUAN_MODEL_PATH` (default: `tencent/HunyuanOCR`).
- `HUNYUAN_DEVICE` (optional: `cuda`, `cpu`, etc.).

Generation / post-processing
- `HUNYUAN_MAX_NEW_TOKENS`, `HUNYUAN_TEMPERATURE`, `HUNYUAN_DO_SAMPLE`.
- `HUNYUAN_CLEAN_REPEATS`: `true|false` (default `true`).

Docs
- See `Docs/OCR/HunyuanOCR.md` for setup details and usage notes.

## Dolphin (backend: `dolphin`)

Summary
- Supports local Transformers inference (ByteDance/Dolphin-v2) and remote Dolphin servers.
- Returns Markdown as-is; also captures JSON outputs inline when enabled.
- Opt-in only (use `ocr_backend=dolphin` or include in `OCR.backend_priority`).

Environment
- `DOLPHIN_MODE`: `auto` (default), `transformers`, `remote`.
- `DOLPHIN_PROMPT`: override main prompt.
- `DOLPHIN_PROMPT_PRESET`: `general|doc|table|json` (default picks `doc`).
- `DOLPHIN_JSON_PROMPT`: override JSON prompt (empty disables).
- `DOLPHIN_DISABLE_JSON`: `true` to skip JSON extraction pass.

Remote (Dolphin servers)
- `DOLPHIN_URL`: server base URL.
- `DOLPHIN_REMOTE_MODE`: `dolphin_vllm` (expects `encoder_prompt` + `decoder_prompt`), `dolphin_trt` (expects `prompt`), or `openai`.
- `DOLPHIN_ENCODER_PROMPT`: override encoder prompt (vLLM mode).
- `DOLPHIN_DECODER_PROMPT`: override decoder prompt (vLLM/TRT).
- `DOLPHIN_REMOTE_MODEL`: model name for OpenAI-compatible mode.
- `DOLPHIN_TIMEOUT`: request timeout seconds (default 60).
- `DOLPHIN_USE_DATA_URL`: `true` to send base64 image URLs (recommended for remote).

Local Transformers
- `DOLPHIN_MODEL_PATH` (default: `ByteDance/Dolphin-v2`).
- `DOLPHIN_DEVICE` (optional: `cuda`, `cpu`, etc.).

Generation controls
- `DOLPHIN_MAX_NEW_TOKENS`, `DOLPHIN_MAX_LENGTH`, `DOLPHIN_TEMPERATURE`
- `DOLPHIN_TOP_P`, `DOLPHIN_TOP_K`, `DOLPHIN_REPETITION_PENALTY`
- `DOLPHIN_DO_SAMPLE`, `DOLPHIN_NUM_BEAMS`

Notes
- If `ocr_output_format=json` is requested but JSON parsing fails, the backend returns Markdown and logs a warning.

## WebUI

- OCR Evaluation is available under WebUI → Evaluations → “OCR Evaluation / OCR PDF”.
- Fields map to API parameters: `enable_ocr`, `ocr_backend`, `ocr_mode`, `ocr_dpi`, etc.
- Set `ocr_backend` to `mineru`, `dots`, or `points` after installing the respective provider.

## Tips

- Prefer `fallback` mode to reduce compute and latency when PDFs already contain text.
- Lower `ocr_dpi` for speed; raise it if recognition quality needs improvement.
- Review provider-specific known issues and constraints in their docs.

## Quick comparison

| Backend   | Modes             | Best for                          | Notes                                  |
|-----------|-------------------|-----------------------------------|----------------------------------------|
| tesseract | CLI               | Light installs, fast text         | No layout semantics, basic accuracy    |
| mineru    | CLI (document)    | Tables, layout, structured OCR    | PDF-only; opt-in only; not in `auto`   |
| dots      | CLI + vLLM        | High quality OCR + layout         | Heavy; recommend vLLM; prompt-tunable  |
| points    | Transformers/SGLang | OCR + HTML tables + Markdown text | trust_remote_code required; SGLang ideal |
| deepseek  | Transformers      | Layout-rich OCR to Markdown       | Heavy GPU stack; trust_remote_code required |
| hunyuan   | vLLM/Transformers | Structured OCR + JSON outputs     | vLLM recommended for quality/speed     |
| llamacpp  | Remote/Managed/CLI | Server-owned OCR-capable llama.cpp | Backend-local page concurrency cap 1 by default |
| chatllm   | Remote/Managed/CLI | Server-owned ChatLLM OCR          | Backend-local page concurrency cap 1 by default |

Note: VCS installs (extras) require `git` and network. For airgapped deployments, install upstream repos manually, then install this project without the extras.
