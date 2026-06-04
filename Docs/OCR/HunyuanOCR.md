# HunyuanOCR Backend

This document describes how to enable and operate the Tencent HunyuanOCR backend in tldw_server. The backend supports both a vLLM OpenAI-compatible server and local Transformers inference.

## Summary

- Backend name: `hunyuan`
- API usage: set `enable_ocr=true` and `ocr_backend=hunyuan` in PDF processing/evaluation endpoints.
- Modes: `vllm` (server) or `transformers` (local).
- Structured output: supported via `ocr_output_format` and `ocr_prompt_preset`.

## Prompt presets

Presets are simple task hints. Override via `HUNYUAN_PROMPT` if you need a custom instruction.

- `general`: plain text extraction
- `doc`: Markdown text, tables as HTML
- `table`: tables focused, other text in Markdown
- `spotting`: text + bounding boxes (JSON)
- `json`: JSON output with text + blocks

## Environment variables

Core
- `HUNYUAN_MODE`: `auto|vllm|transformers` (default `auto`).
- `HUNYUAN_PROMPT`: prompt override.
- `HUNYUAN_PROMPT_PRESET`: `general|doc|table|spotting|json`.

vLLM
- `HUNYUAN_VLLM_URL`: OpenAI-compatible `/v1/chat/completions` endpoint.
- `HUNYUAN_VLLM_MODEL`: served model name.
- `HUNYUAN_VLLM_TIMEOUT`: request timeout seconds (default `60`).
- `HUNYUAN_VLLM_USE_DATA_URL`: `true|false` (default `true`).

Transformers
- `HUNYUAN_MODEL_PATH`: HF model id or local path (default: `tencent/HunyuanOCR`).
- `HUNYUAN_DEVICE`: optional device override (`cuda`, `cpu`, etc.).

Generation
- `HUNYUAN_MAX_NEW_TOKENS`, `HUNYUAN_TEMPERATURE`, `HUNYUAN_DO_SAMPLE`.

Post-processing
- `HUNYUAN_CLEAN_REPEATS`: `true|false` (default `true`).

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

- If the backend is not detected, confirm `HUNYUAN_VLLM_URL` (server mode) or `transformers` + `torch` + `Pillow` (local mode).
- If output is verbose or noisy, try `ocr_prompt_preset=general` and reduce `ocr_dpi`.
- If JSON parsing fails, the backend returns raw text; review `analysis_details.ocr.structured.raw` for original output.
