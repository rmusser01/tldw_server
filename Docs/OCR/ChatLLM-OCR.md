# ChatLLM OCR

This guide covers the `chatllm` OCR backend. Like `llamacpp`, it keeps OCR server-owned: the client selects `ocr_backend=chatllm`, while the server decides the runtime mode, model, binary, and concurrency cap.

## Modes

- `remote`: call an OpenAI-compatible ChatLLM endpoint.
- `managed`: let the server own a private OCR process.
- `cli`: run a one-shot local command for each OCR request.
- `auto`: choose the best configured mode from the server profile.

Managed mode is single-process only in v1. Use remote mode if you need multi-worker or externally managed deployments.

## Required Settings

- `CHATLLM_OCR_MODE=auto|remote|managed|cli`
- `CHATLLM_OCR_ALLOW_MANAGED_START=true|false`
- `CHATLLM_OCR_AUTO_ELIGIBLE=true|false`
- `CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE=true|false`
- `CHATLLM_OCR_MAX_PAGE_CONCURRENCY`

The PDF pipeline uses `min(OCR_PAGE_CONCURRENCY, CHATLLM_OCR_MAX_PAGE_CONCURRENCY)` when dispatching page OCR work.

## Remote Mode

Use remote mode when ChatLLM is already exposed as an OpenAI-compatible service.

Typical settings:

- `CHATLLM_OCR_URL`
- `CHATLLM_OCR_MODEL`
- `CHATLLM_OCR_API_KEY`

The backend should preserve backend metadata in `analysis_details.ocr` and normalize OCR results into the existing `OCRResult` contract, including `analysis_details.ocr.structured` when structured output is available.

## Managed Mode

Managed mode owns a private OCR process and is only intended for single-API-process deployments in v1.

Typical settings:

- `CHATLLM_OCR_SERVER_BINARY`
- `CHATLLM_OCR_MODEL_PATH`
- `CHATLLM_OCR_HOST`
- `CHATLLM_OCR_PORT`
- `CHATLLM_OCR_STARTUP_TIMEOUT_SEC`
- `CHATLLM_OCR_SERVER_ARGS_JSON`
- `CHATLLM_OCR_HEALTHCHECK_URL`

The server should not auto-start this process unless `CHATLLM_OCR_ALLOW_MANAGED_START=true`.

## CLI Mode

CLI mode runs a single OCR invocation and exits.

Typical settings:

- `CHATLLM_OCR_CLI_BINARY`
- `CHATLLM_OCR_MODEL_PATH`
- `CHATLLM_OCR_CLI_ARGS_JSON`

Use JSON argv arrays rather than shell strings. Example:

```json
["--model", "{model_path}", "--image", "{image_path}", "--prompt", "{prompt}"]
```

## Auto Selection

`chatllm` participates in `auto` and `auto_high_quality` only when:

- the backend is locally available
- the matching auto-eligibility flag is enabled

The operator can still force the backend explicitly with `ocr_backend=chatllm`.

## Operational Notes

- Keep `CHATLLM_OCR_MAX_PAGE_CONCURRENCY` low unless you have validated higher parallelism.
- Keep temporary image files ephemeral and server-owned.
- Do not expose request-level overrides for model paths, server paths, or ports.
