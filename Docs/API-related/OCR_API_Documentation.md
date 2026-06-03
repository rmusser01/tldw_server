# OCR API Documentation (v0.1) - Placeholder

Status: Initial placeholder for OCR endpoints and integration. This page will expand with full request/response schemas and examples.

## Summary

The OCR module integrates with media ingestion to extract text from scanned PDFs or images when native text is unavailable. The public API currently exposes lightweight endpoints for backend discovery and preloading, while full OCR execution is driven via the media ingestion APIs.

## Auth + Rate Limits
- Single-user: `X-API-KEY: <key>`
- Multi-user: `Authorization: Bearer <JWT>`
- Standard limits apply; OCR preloading is low-cost, while end-to-end OCR via media ingestion follows media service limits.

## Endpoints

1) GET `/api/v1/ocr/backends`
   - Lists available OCR backends with basic health info.
   - Returns a map keyed by backend name (e.g., `mineru`, `points`, `dots`, `llamacpp`, `chatllm`) including lightweight backend-specific configuration details.
   - Field shape varies by backend. Today `llamacpp` exposes `mode`, `configured_mode`, `configured`, `supports_structured_output`, `supports_json`, `model`, `configured_flags`, auto-eligibility flags, `backend_concurrency_cap`, and mode-specific flags such as `url_configured`, `managed_configured`, `managed_running`, `allow_managed_start`, and `cli_configured`; `chatllm` exposes a similar capability set with its own mode-specific details, but not every field is identical.
   - `hunyuan` exposes top-level `mode`, `runtime_family`, `configured_family`, auto-eligibility flags, `backend_concurrency_cap`, and namespaced `native` / `llamacpp` sub-objects so callers can distinguish native Hunyuan from Hunyuan GGUF runtime state.
   - Code: `tldw_Server_API/app/api/v1/endpoints/ocr.py:router.get("/backends")`

2) POST `/api/v1/ocr/points/preload`
   - Attempts to preload the POINTS Transformers model to surface errors early.
   - Returns `{ "status": "ok" | "error", ... }`.
   - Code: `tldw_Server_API/app/api/v1/endpoints/ocr.py:router.post("/points/preload")`

## OCR in Media Ingestion

OCR is typically enabled via the media ingestion request options. Key fields (see code for authoritative definitions):

- `enable_ocr` (bool) - enable OCR for scanned/low-text PDFs
- `ocr_backend` (str | null) - backend name (e.g., `tesseract`, `auto`, or module-specific)
- `ocr_lang` (str) - language (e.g., `eng`)
- `ocr_dpi` (int) - DPI for page rendering prior to OCR
- `ocr_mode` (enum) - `always` or `fallback`
- `ocr_min_page_text_chars` (int) - threshold to treat a page as “no text” for fallback OCR
- `ocr_output_format` (str | null) - `text|markdown|json` (controls structured OCR output)
- `ocr_prompt_preset` (str | null) - `general|doc|table|spotting|json` (backend-specific presets)
- The PDF pipeline stores structured OCR data under `analysis_details.ocr.structured` when the backend returns it.
- Per-page OCR concurrency is capped by the smaller of `OCR_PAGE_CONCURRENCY` and the backend profile's `max_page_concurrency`.

Reference (code): `tldw_Server_API/app/api/v1/schemas/media_request_models.py`.

### MinerU behavior

- `ocr_backend=mineru` is supported only for PDF ingestion and OCR evaluation in v1.
- MinerU is document-level, not per-page image OCR. The PDF pipeline runs it once for the whole PDF and stores the normalized result under `analysis_details.ocr.structured`.
- MinerU appears in `GET /api/v1/ocr/backends` with capability flags such as `pdf_only`, `document_level`, and `opt_in_only`.
- MinerU is excluded from `auto`, `auto_high_quality`, and `OCR.backend_priority` in v1.
- `ocr_lang` and `ocr_dpi` are advisory for MinerU and are currently recorded in metadata but not used to drive the CLI invocation.

### Llama.cpp and ChatLLM behavior

- `ocr_backend=llamacpp` and `ocr_backend=chatllm` are server-owned OCR profiles that can run in `remote`, `managed`, `cli`, or `auto` mode.
- Both backends use explicit auto-eligibility flags. They only participate in `auto` / `auto_high_quality` when the flag is enabled and the backend is locally available.
- Managed mode is single-process only in v1. For multi-worker deployments, use `remote` or `cli`.
- The PDF pipeline records the effective page concurrency it actually used, not just the global cap.

### Hunyuan family behavior

- `ocr_backend=hunyuan` remains the only public Hunyuan selector.
- `HUNYUAN_RUNTIME_FAMILY=auto|native|llamacpp` controls whether the backend uses native Hunyuan execution or Hunyuan GGUF through llama.cpp.
- In `auto`, native Hunyuan is preferred only when it is explicitly configured. Importable Transformers dependencies alone no longer make Hunyuan eligible.
- The PDF pipeline stores Hunyuan family metadata such as `runtime_family`, `configured_family`, and sanitized namespaced `native` / `llamacpp` sub-objects under `analysis_details.ocr`.

## Quick Examples

List OCR backends

```bash
curl -s http://localhost:8000/api/v1/ocr/backends | jq
```

Preload POINTS Transformers

```bash
curl -s -X POST http://localhost:8000/api/v1/ocr/points/preload | jq
```

Enable OCR in media ingestion (illustrative JSON fragment)

```json
{
  "enable_ocr": true,
  "ocr_backend": "auto",
  "ocr_lang": "eng",
  "ocr_mode": "fallback",
  "ocr_dpi": 300
}
```

Structured OCR example (process PDF + inspect `analysis_details`)

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
  "results": [
    {
      "analysis_details": {
        "ocr": {
          "backend": "hunyuan",
          "runtime_family": "llamacpp",
          "configured_family": "auto",
          "output_format": "json",
          "prompt_preset": "json",
          "llamacpp": {
            "mode": "remote",
            "configured_mode": "auto"
          },
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
  ]
}
```

MinerU PDF OCR example

```bash
curl -s -X POST http://localhost:8000/api/v1/media/process-pdfs \
  -H "X-API-KEY: $TLDW_API_KEY" \
  -F "enable_ocr=true" \
  -F "ocr_backend=mineru" \
  -F "ocr_mode=fallback" \
  -F "ocr_output_format=markdown" \
  -F "files=@/path/to/scanned-table.pdf"
```

Example MinerU discovery response excerpt (truncated)

```json
{
  "hunyuan": {
    "available": true,
    "mode": "remote",
    "runtime_family": "llamacpp",
    "configured_family": "auto",
    "backend_concurrency_cap": 2,
    "native": {
      "mode": "transformers",
      "configured": false,
      "available": false
    },
    "llamacpp": {
      "mode": "remote",
      "configured_mode": "auto",
      "configured": true,
      "url_configured": true,
      "managed_configured": true,
      "cli_configured": true
    }
  },
  "mineru": {
    "available": true,
    "pdf_only": true,
    "document_level": true,
    "opt_in_only": true,
    "mode": "cli",
    "timeout_sec": 120,
    "max_concurrency": 1
  }
}
```

## Backend Notes

- MinerU: document-level PDF OCR with bounded structured artifacts (`pages`, `tables`, artifact excerpts)
- POINTS Reader: documentation coming soon
- Llama.cpp OCR: see `Docs/OCR/LlamaCpp-OCR.md`
- ChatLLM OCR: see `Docs/OCR/ChatLLM-OCR.md`
- OCR Providers overview: see `Docs/OCR/OCR_Providers.md`

## Roadmap (Placeholder)

- Expand docs with full request/response schemas
- Add examples for common ingestion flows with OCR
- Add troubleshooting and performance tips

---

If you need additional OCR endpoints or deeper docs, please open an issue with your use case.
