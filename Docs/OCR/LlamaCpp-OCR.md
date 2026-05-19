# Llama.cpp OCR

This guide covers the `llamacpp` OCR backend. It keeps OCR server-owned: the client selects `ocr_backend=llamacpp`, while the server decides the runtime mode, model, binary, and concurrency cap.

## Modes

- `remote`: call an existing OpenAI-compatible llama.cpp endpoint.
- `managed`: let the server own a private OCR process.
- `cli`: run a one-shot local command for each OCR request.
- `auto`: choose the best configured mode from the server profile.

Managed mode is single-process only in v1. Use remote mode if you need multi-worker or externally managed deployments.

## Required Settings

- `LLAMACPP_OCR_MODE=auto|remote|managed|cli`
- `LLAMACPP_OCR_ALLOW_MANAGED_START=true|false`
- `LLAMACPP_OCR_AUTO_ELIGIBLE=true|false`
- `LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE=true|false`
- `LLAMACPP_OCR_MAX_PAGE_CONCURRENCY`

The PDF pipeline uses `min(OCR_PAGE_CONCURRENCY, LLAMACPP_OCR_MAX_PAGE_CONCURRENCY)` when dispatching page OCR work.

## Remote Mode

Use remote mode when you already have a llama.cpp endpoint or when another service manages the server process.

Typical settings:

- `LLAMACPP_OCR_HOST`
- `LLAMACPP_OCR_PORT`
- `LLAMACPP_OCR_MODEL_PATH`
- `LLAMACPP_OCR_USE_DATA_URL=true|false`

The endpoint should be OpenAI-compatible. This backend expects structured OCR output to be represented through the existing `OCRResult` contract and stored under `analysis_details.ocr.structured`.

## Managed Mode

Managed mode owns a private OCR process and is only intended for single-API-process deployments in v1.

Typical settings:

- `LLAMACPP_OCR_ARGV`
- `LLAMACPP_OCR_MODEL_PATH`
- `LLAMACPP_OCR_HOST` (optional; defaults to `127.0.0.1`)
- `LLAMACPP_OCR_PORT`
- `LLAMACPP_OCR_STARTUP_TIMEOUT_SEC`

`LLAMACPP_OCR_ARGV` must be a JSON argv array for the private server process, using placeholders such as `{model_path}`, `{host}`, and `{port}`.

The server should not auto-start this process unless `LLAMACPP_OCR_ALLOW_MANAGED_START=true`. In `auto`, a configured managed startup profile is preferred over `remote`; set `LLAMACPP_OCR_MODE=remote` to force an external endpoint instead.

## CLI Mode

CLI mode runs a single OCR invocation and exits.

Typical settings:

- `LLAMACPP_OCR_ARGV`
- `LLAMACPP_OCR_MODEL_PATH`

`LLAMACPP_OCR_ARGV` must be a JSON argv array for the one-shot OCR command. Use JSON argv arrays rather than shell strings. Example:

```json
["--model", "{model_path}", "--image", "{image_path}", "--prompt", "{prompt}"]
```

## Auto Selection

`llamacpp` participates in `auto` and `auto_high_quality` only when:

- the backend is locally available
- the matching auto-eligibility flag is enabled

The operator can still force the backend explicitly with `ocr_backend=llamacpp`.

## Operational Notes

- `LLAMACPP_OCR_ARGV` is the current local-runtime command surface for both managed and CLI modes. Configure the mode you actually intend to run; do not assume separate server/CLI env vars exist in v1.
- Keep `LLAMACPP_OCR_MAX_PAGE_CONCURRENCY` low unless you have validated higher parallelism.
- Keep temporary image files ephemeral and server-owned.
- Do not expose request-level overrides for model paths, server paths, or ports.
