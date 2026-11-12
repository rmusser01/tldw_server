# VibeVoice-ASR Getting Started

This guide shows how to run VibeVoice-ASR inside the server (local inference)
and how to route it to a separate vLLM HTTP service. It also covers the new
`hotwords` field.

VibeVoice-ASR is best suited for long-form audio, speaker-aware transcripts,
and domain-specific vocabularies.

## 1) Prerequisites

Minimum recommendations:

- A working server install:

```bash
pip install -e ".[dev,STT_All]"
```

- FFmpeg on your system path.
- A GPU with ample VRAM is strongly recommended for the 7B model.

If you do not have a GPU, you can still try the vLLM HTTP path on a different
machine.

## 2) Enable Local VibeVoice-ASR

Edit `tldw_Server_API/Config_Files/config.txt` and set:

```ini
[STT-Settings]
# Make VibeVoice the default STT provider (optional)
default_transcriber = vibevoice

# Local VibeVoice-ASR inference
vibevoice_enabled = true
vibevoice_model_id = microsoft/VibeVoice-ASR
vibevoice_device = cuda
vibevoice_dtype = bfloat16
vibevoice_cache_dir = ./models/vibevoice
vibevoice_allow_download = true
```

Notes:

- The first request may download model weights and can take a while.
- If you want to keep Whisper as the default, leave `default_transcriber`
  unchanged and request `model=vibevoice-asr` explicitly.

## 3) Optional: Use the vLLM HTTP Path

If you have a vLLM server hosting the model, enable the HTTP-only path:

```ini
[STT-Settings]
# Keep this false when you want local inference:
vibevoice_enabled = false

# Route VibeVoice to vLLM over HTTP:
vibevoice_vllm_enabled = true
vibevoice_vllm_base_url = http://127.0.0.1:8001
vibevoice_vllm_model_id = microsoft/VibeVoice-ASR
vibevoice_vllm_timeout_seconds = 600
# Optional:
# vibevoice_vllm_api_key = <token>
```

The server will only attempt the vLLM HTTP path when
`vibevoice_vllm_enabled = true`.

## 4) Sanity Check via curl

You can call the OpenAI-compatible transcription endpoint directly:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/transcriptions" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -F "file=@/absolute/path/to/sample.wav" \
  -F "model=vibevoice-asr" \
  -F "response_format=json" \
  -F "language=en" \
  -F "hotwords=[\"VibeVoice\",\"tldw_server\"]"
```

`hotwords` accepts either:

- CSV: `alpha,beta,gamma`
- JSON list: `["alpha","beta","gamma"]`

## 5) Use the Sanity Script

A small smoke script is provided:

```bash
python Helper_Scripts/Testing-related/vibevoice_asr_smoke.py \
  --api-key "$SINGLE_USER_API_KEY"
```

It generates a short test WAV when you do not provide `--audio-path`.

## 6) Health and Troubleshooting

Useful checks:

- STT health:

```bash
curl "http://127.0.0.1:8000/api/v1/audio/transcriptions/health?model=vibevoice-asr" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY"
```

Common issues:

- Model not enabled:
  - Set `vibevoice_enabled = true` (local) or
    `vibevoice_vllm_enabled = true` (vLLM HTTP).
- Long cold start:
  - Expect a large first-run download and model load.
- CUDA issues:
  - Try `vibevoice_device = cpu` to confirm the pipeline works end-to-end,
    then return to GPU.

