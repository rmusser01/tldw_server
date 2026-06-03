# HunyuanOCR Backend

This document describes how to enable and operate the Tencent HunyuanOCR backend in tldw_server. The public backend name remains `hunyuan`, but it now supports two runtime families:

- native Hunyuan execution through vLLM or local Transformers
- Hunyuan GGUF execution through llama.cpp in `remote`, `managed`, or `cli` mode

## Summary

- Backend name: `hunyuan`
- API usage: set `enable_ocr=true` and `ocr_backend=hunyuan` in PDF processing/evaluation endpoints.
- Runtime families:
  - `native`: `vllm` (server) or `transformers` (local)
  - `llamacpp`: Hunyuan GGUF served by llama.cpp
  - `auto`: prefer a configured native runtime first, otherwise fall back to Hunyuan GGUF
- Structured output: supported via `ocr_output_format` and `ocr_prompt_preset`.
- Discovery: `GET /api/v1/ocr/backends` now returns top-level Hunyuan fields plus namespaced `native` and `llamacpp` capability sub-objects.

## Prompt presets

Presets are simple task hints. Override via `HUNYUAN_PROMPT` if you need a custom instruction.

- `general`: plain text extraction
- `doc`: Markdown text, tables as HTML
- `table`: tables focused, other text in Markdown
- `spotting`: text + bounding boxes (JSON)
- `json`: JSON output with text + blocks

## Environment variables

Family selection
- `HUNYUAN_RUNTIME_FAMILY`: `auto|native|llamacpp` (default `auto`).
  - `auto` prefers configured native Hunyuan first.
  - Native availability is now stricter: importable `transformers` dependencies alone are not enough. Set `HUNYUAN_MODE=transformers` or `HUNYUAN_MODEL_PATH` to make the Transformers path eligible.

Native core
- `HUNYUAN_MODE`: `auto|vllm|transformers` (default `auto`).
- `HUNYUAN_PROMPT`: prompt override.
- `HUNYUAN_PROMPT_PRESET`: `general|doc|table|spotting|json`.

Native vLLM
- `HUNYUAN_VLLM_URL`: OpenAI-compatible `/v1/chat/completions` endpoint.
- `HUNYUAN_VLLM_MODEL`: served model name.
- `HUNYUAN_VLLM_TIMEOUT`: request timeout seconds (default `60`).
- `HUNYUAN_VLLM_USE_DATA_URL`: `true|false` (default `true`).

Native Transformers
- `HUNYUAN_MODEL_PATH`: HF model id or local path (default: `tencent/HunyuanOCR`).
- `HUNYUAN_DEVICE`: optional device override (`cuda`, `cpu`, etc.).

Hunyuan GGUF via llama.cpp
- `HUNYUAN_LLAMACPP_MODE`: `auto|remote|managed|cli` (default `auto`).
- `HUNYUAN_LLAMACPP_AUTO_ELIGIBLE`: opt-in flag for generic OCR `auto`.
- `HUNYUAN_LLAMACPP_AUTO_HIGH_QUALITY_ELIGIBLE`: opt-in flag for `auto_high_quality`.
- `HUNYUAN_LLAMACPP_MAX_PAGE_CONCURRENCY`: backend-local page cap used with `OCR_PAGE_CONCURRENCY`.

Hunyuan GGUF remote
- `HUNYUAN_LLAMACPP_HOST`, `HUNYUAN_LLAMACPP_PORT`
- `HUNYUAN_LLAMACPP_MODEL`: logical model identifier sent to `/v1/chat/completions`
- `HUNYUAN_LLAMACPP_USE_DATA_URL`: `true|false` (default `true`)
- `HUNYUAN_LLAMACPP_TIMEOUT`, `HUNYUAN_LLAMACPP_TEMPERATURE`, `HUNYUAN_LLAMACPP_MAX_TOKENS`

Hunyuan GGUF managed
- `HUNYUAN_LLAMACPP_ALLOW_MANAGED_START`: `true|false`
- `HUNYUAN_LLAMACPP_HOST`, `HUNYUAN_LLAMACPP_PORT`
- `HUNYUAN_LLAMACPP_MODEL_PATH`: local GGUF path or `-hf` target used to start `llama-server`
- `HUNYUAN_LLAMACPP_SERVER_ARGV`: JSON argv template for the managed server process
- `HUNYUAN_LLAMACPP_STARTUP_TIMEOUT_SEC`

Hunyuan GGUF CLI
- `HUNYUAN_LLAMACPP_MODEL_PATH`
- `HUNYUAN_LLAMACPP_CLI_ARGV`: JSON argv template for per-page CLI invocation

Generation
- `HUNYUAN_MAX_NEW_TOKENS`, `HUNYUAN_TEMPERATURE`, `HUNYUAN_DO_SAMPLE`.

Post-processing
- `HUNYUAN_CLEAN_REPEATS`: `true|false` (default `true`).

Operator guidance
- Keep using `ocr_backend=hunyuan` for both native and GGUF Hunyuan deployments.
- `ocr_backend=llamacpp` remains the generic llama.cpp OCR backend. Do not point both `ocr_backend=hunyuan` and `ocr_backend=llamacpp` at the same managed deployment unless you explicitly want two separate operator surfaces.
- For multi-worker deployments, prefer `remote` GGUF mode over `managed`.

## Using with API endpoints

Any endpoint that accepts OCR options can use HunyuanOCR:
- PDF ingestion: `POST /api/v1/media/process`
- Process-only PDFs: `POST /api/v1/media/process-pdfs`
- OCR evaluation (PDFs): `POST /api/v1/evaluations/ocr-pdf`

Example (form fields)
- `enable_ocr=true`
- `ocr_backend=hunyuan`
- `ocr_mode=fallback` (or `always`)
- `ocr_dpi=300`
- `ocr_output_format=json`
- `ocr_prompt_preset=json`

## Structured output

When `ocr_output_format` or `ocr_prompt_preset` is set, the pipeline persists structured OCR output into `analysis_details.ocr.structured` alongside the plain text content.

Example curl (process PDF)

```bash
curl -s -X POST http://localhost:8000/api/v1/media/process-pdfs \
  -H "X-API-KEY: $TLDW_API_KEY" \
  -F "enable_ocr=true" \
  -F "ocr_backend=hunyuan" \
  -F "ocr_output_format=json" \
  -F "ocr_prompt_preset=json" \
  -F "files=@/path/to/sample.pdf"
```

Example response excerpt (truncated)

```json
{
  "analysis_details": {
    "ocr": {
      "backend": "hunyuan",
      "output_format": "json",
      "prompt_preset": "json",
      "structured": {
        "format": "json",
        "text": "...",
        "pages": [
          { "text": "...", "raw": { "blocks": [ { "text": "..." } ] } }
        ]
      }
    }
  }
}
```

## Performance tips

- Prefer `ocr_mode=fallback` when PDFs already contain text.
- Keep `OCR_PAGE_CONCURRENCY` low (1-2) for GPU-backed OCR.
- Lower `ocr_dpi` for speed; raise it if recognition quality needs improvement.
- vLLM is recommended for throughput and consistency.

## Troubleshooting

- If native Hunyuan is not detected, confirm `HUNYUAN_VLLM_URL` or set explicit Transformers intent with `HUNYUAN_MODE=transformers` or `HUNYUAN_MODEL_PATH`.
- If Hunyuan GGUF is not detected, confirm `HUNYUAN_LLAMACPP_MODE` plus the matching `HOST/PORT`, `MODEL`, `MODEL_PATH`, and argv settings for that mode.
- If output is verbose or noisy, try `ocr_prompt_preset=general` and reduce `ocr_dpi`.
- If JSON parsing fails, the backend returns raw text; review `analysis_details.ocr.structured.raw` for original output.
