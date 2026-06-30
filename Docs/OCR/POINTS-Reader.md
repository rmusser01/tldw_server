# POINTS-Reader OCR Backend

This document describes how to enable and operate the POINTS-Reader (Tencent) OCR backend in tldw_server. The backend supports two integration modes: local Transformers inference and SGLang server inference.

## Summary

- Backend name: `points`
- API usage: set `enable_ocr=true` and `ocr_backend=points` in PDF processing/evaluation endpoints. Optional: `ocr_mode` (`fallback`|`always`), `ocr_dpi`.
- Modes: `transformers` (local) or `sglang` (server). Default `auto` prefers SGLang if configured.

## Recommended Prompt

By default, the backend uses this extraction prompt (tuned to the upstream model):

```
Please extract all the text from the image with the following requirements:
1. Return tables in HTML format.
2. Return all other text in Markdown format.
```

Override with the env var `POINTS_PROMPT`.

## Mode Selection and Environment Variables

- `POINTS_MODE`: `auto` (default), `sglang`, or `transformers`.

Transformers (local model):
- `POINTS_MODEL_PATH`: HF model id or local path. Default: `tencent/POINTS-Reader`.
- Requirements: `transformers`, `torch` (CUDA recommended), and WePOINTS toolkit.

SGLang (server):
- `POINTS_SGLANG_URL`: server URL. Default: `http://127.0.0.1:8081/v1/chat/completions`.
- `POINTS_SGLANG_MODEL`: model field in the request. Default: `WePoints`.

Generation parameters (both modes):
- `POINTS_MAX_NEW_TOKENS` (int, default 2048)
- `POINTS_TEMPERATURE` (float, default 0.7)
- `POINTS_REPETITION_PENALTY` (float, default 1.05)
- `POINTS_TOP_P` (float, default 0.8)
- `POINTS_TOP_K` (int, default 20)
- `POINTS_DO_SAMPLE` (bool: true/false, default true)
- `POINTS_SGLANG_TIMEOUT` (int seconds, default 60)
- `POINTS_SGLANG_USE_DATA_URL` (bool: if true, send image as data URL rather than local path)

## Installation

### Local Transformers (WePOINTS + HF)

Environment (tested upstream):
- python==3.10.12
- torch==2.5.1
- transformers==4.55.2
- cuda==12.1

Steps:
1) Install WePOINTS
```
git clone https://github.com/WePOINTS/WePOINTS.git
cd WePOINTS
pip install -e .
```
2) Ensure `transformers` and `torch` match your CUDA setup.
3) Optional: set `POINTS_MODEL_PATH` to a local cache or custom path.

Manual source setup (optional):
- Install the local path dependencies:
  - `pip install "transformers>=5.5.3"`
  - Install a CUDA/CPU-compatible PyTorch build for your platform.
- Install WePOINTS manually from source with the steps above. It is not bundled in `tldw-server` package metadata because PyPI rejects direct Git dependencies.

### SGLang Server

Launch SGLang serving tencent/POINTS-Reader (see upstream docs):
```
python -m sglang.launch_server \
  --model-path tencent/POINTS-Reader \
  --tp-size 1 \
  --dp-size 1 \
  --chat-template points-v15-chat \
  --trust-remote-code \
  --port 8081
```
Then configure the client:
```
export POINTS_MODE=sglang
export POINTS_SGLANG_URL=http://127.0.0.1:8081/v1/chat/completions
export POINTS_SGLANG_MODEL=WePoints
```

Extras install (optional):
- Client-only deps can be installed with:
  - `pip install .[ocr_points_sglang]`

Security note
- The upstream model requires `trust_remote_code=True` in Transformers and SGLang launch. Only enable this in controlled environments and review the code you execute.

## Using with API Endpoints

Any endpoint that accepts OCR options can use POINTS:
- PDF ingestion: `POST /api/v1/media/process` (form fields include `enable_ocr`, `ocr_backend`, `ocr_mode`, `ocr_dpi`)
- OCR evaluation (PDFs): `POST /api/v1/evaluations/ocr-pdf`

Example (form fields):
- `enable_ocr=true`
- `ocr_backend=points`
- `ocr_mode=fallback` (or `always`)
- `ocr_dpi=300`

## Performance Tips

- Prefer SGLang for production workloads; it avoids repeated model loads and can scale.
- On local inference, ensure CUDA is available; otherwise, expect slower throughput.
- Use `ocr_mode=fallback` to skip OCR on pages with selectable text.
- Consider lowering `ocr_dpi` (e.g., 200-300) for speed if acceptable.

## Known Issues (Upstream)

- Complex layouts (e.g., newspapers) may produce repeated or missing content.
- Handwriting is challenging; expect recognition errors or omissions.
- Language coverage focuses on English and Chinese.

## Troubleshooting

- Transformers mode availability requires both `transformers` and `torch` to be importable.
- SGLang mode requires `requests` and a reachable `POINTS_SGLANG_URL`.
- Enable debug logs to inspect backend selection and errors.
- If repetition occurs, try increasing image resolution before OCR (as suggested upstream).
