# PocketTTS Voice Cloning Guide

A step-by-step guide to installing PocketTTS (`pocket_tts`, the Python/ONNX runtime), cloning a voice from a reference audio sample, and setting the cloned voice as your default TTS output.

If you want the separate compiled-native runtime, use `pocket_tts_cpp` and its installer helper instead. That provider has different runtime expectations and is documented in the general TTS guides, not in this ONNX-specific guide.

## What is PocketTTS?

PocketTTS is a lightweight, ONNX-based local TTS engine with voice cloning. Key characteristics:

- **English only** - optimized for English speech synthesis
- **Voice cloning required** - every request needs a reference audio sample (1-60 seconds)
- **CPU-friendly** - INT8 quantized models run well on CPU; CUDA optional
- **Streaming support** - progressive audio delivery for low-latency playback
- **24 kHz output** - mono, 16-bit PCM internally; converted to your chosen format (MP3, WAV, OPUS, FLAC, AAC, PCM)

## Prerequisites

Before starting, make sure you have:

1. **The server repository cloned** and your Python virtual environment active
2. **FFmpeg installed** (`brew install ffmpeg` on macOS, `apt-get install -y ffmpeg` on Linux)
3. **The server starts successfully** (you don't need PocketTTS enabled yet):
   ```bash
   python -m uvicorn tldw_Server_API.app.main:app --reload
   ```
4. Note the `X-API-KEY` printed at startup (single-user mode) - you'll use it in all API calls.

## Step 1: Install Dependencies

Install PocketTTS runtime dependencies and the HuggingFace Hub CLI (for model download):

```bash
pip install -e '.[TTS_pocket_tts]'
pip install -U "huggingface_hub"
```

## Step 2: Download PocketTTS Models

### Option A: Use the Installer (Recommended)

Run from the project root:

```bash
python Helper_Scripts/TTS_Installers/install_tts_pocket_tts_onnx.py
```

This downloads the ONNX models, tokenizer, and Python module into `models/pocket_tts_onnx/` and automatically updates `tts_providers_config.yaml` to enable PocketTTS with the correct paths.

Useful flags:

| Flag | Effect |
|------|--------|
| `--force` | Re-download even if assets already exist |
| `--no-config-update` | Download models but don't touch the YAML config |
| `--output-dir DIR` | Download to a custom directory instead of `models/pocket_tts_onnx` |

### Option B: Manual Download

```bash
hf download KevinAHM/pocket-tts-onnx \
  --local-dir models/pocket_tts_onnx \
  --local-dir-use-symlinks False \
  --include "onnx/**" "tokenizer.model" "pocket_tts_onnx/**" "pocket_tts_onnx.py"
```

**What gets downloaded:**

```text
models/pocket_tts_onnx/
  onnx/
    flow_lm_main_int8.onnx      # Main language model (INT8)
    flow_lm_flow_int8.onnx      # Flow model (INT8)
    mimi_decoder_int8.onnx       # Audio decoder (INT8)
    mimi_encoder.onnx            # Audio encoder
    text_conditioner.onnx        # Text conditioning
    flow_lm_main.onnx            # FP32 variants (optional)
    flow_lm_flow.onnx
    mimi_decoder.onnx
  tokenizer.model
  pocket_tts_onnx.py (or pocket_tts_onnx/)
```

## Step 3: Enable PocketTTS in Configuration

If you used the installer without `--no-config-update`, the config is already updated. Otherwise, edit `tldw_Server_API/Config_Files/tts_providers_config.yaml`:

```yaml
providers:
  pocket_tts:
    enabled: true                                     # <-- change to true
    model_path: "models/pocket_tts_onnx/onnx"
    tokenizer_path: "models/pocket_tts_onnx/tokenizer.model"
    module_path: "models/pocket_tts_onnx"
    precision: "int8"         # "int8" (CPU optimized) or "fp32"
    device: "auto"            # "auto" | "cpu" | "cuda"
    temperature: 0.7
    lsd_steps: 10
    max_frames: 500
    sample_rate: 24000
```

### Optional Tuning

| Setting | Default | Description |
|---------|---------|-------------|
| `precision` | `int8` | Use `fp32` for slightly higher quality at the cost of speed and memory |
| `device` | `auto` | Force `cpu` or `cuda`; `auto` picks CUDA if available |
| `temperature` | `0.7` | Controls randomness in generation (lower = more deterministic) |
| `max_frames` | `500` | Maximum output length in frames; increase for longer passages |
| `lsd_steps` | `10` | Denoising steps; more steps = higher quality, slower |

## Step 4: Verify Installation

Restart the server, then run these checks:

**Check that PocketTTS appears as an available provider:**

```bash
curl -s http://127.0.0.1:8000/api/v1/audio/providers \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" | jq
```

You should see `"pocket_tts"` in the list.

**Check the voice catalog:**

```bash
curl -s http://127.0.0.1:8000/api/v1/audio/voices/catalog \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" | jq '.pocket_tts'
```

PocketTTS has no built-in voices (voice cloning is required), so this will show an empty or minimal listing. That's expected.

## Step 5: Prepare Voice Reference Audio

PocketTTS needs a reference audio clip of the voice you want to clone.

### Requirements

| Property | Requirement |
|----------|-------------|
| Duration | 1-60 seconds (5-15 seconds recommended for best quality) |
| Channels | Mono |
| Sample rate | 24 kHz (the adapter auto-converts, but native is best) |
| Format | WAV, MP3, FLAC, OGG, or M4A |
| Content | Single speaker, clear speech, minimal background noise |

### Convert with FFmpeg

If your source audio doesn't meet these specs, convert it:

```bash
ffmpeg -i input.mp3 -ar 24000 -ac 1 -c:a pcm_s16le voice_sample.wav
```

To extract a 10-second clip starting at 5 seconds:

```bash
ffmpeg -i input.mp3 -ss 5 -t 10 -ar 24000 -ac 1 -c:a pcm_s16le voice_sample.wav
```

### Tips for Best Results

- Record in a quiet room with minimal echo
- Maintain consistent volume and distance from the microphone
- Use natural, conversational speech
- Trim silence from the beginning and end
- Avoid music, other speakers, or background noise

## Step 6: Clone a Voice (Inline Reference)

With your voice sample ready, send a TTS request with the audio base64-encoded in the `voice_reference` field.

### curl Example

```bash
# Base64-encode the voice sample
VOICE_B64=$(base64 < voice_sample.wav)

# Generate speech with the cloned voice
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/speech \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"pocket_tts\",
    \"input\": \"Hello, this is my cloned voice speaking through PocketTTS.\",
    \"voice\": \"clone\",
    \"voice_reference\": \"$VOICE_B64\",
    \"response_format\": \"mp3\"
  }" \
  --output cloned_output.mp3
```

### Python Example

```python
import base64
import json
from pathlib import Path
from urllib import request, error

# Read and encode the voice sample
voice_b64 = base64.b64encode(
    Path("voice_sample.wav").read_bytes()
).decode()

# Build the request
payload = {
    "model": "pocket_tts",
    "input": "Hello, this is my cloned voice speaking through PocketTTS.",
    "voice": "clone",
    "voice_reference": voice_b64,
    "response_format": "mp3",
}

req = request.Request(
    "http://127.0.0.1:8000/api/v1/audio/speech",
    data=json.dumps(payload).encode(),
    headers={
        "Content-Type": "application/json",
        "X-API-KEY": "YOUR_API_KEY",
    },
    method="POST",
)

with request.urlopen(req) as resp:
    Path("cloned_output.mp3").write_bytes(resp.read())

print("Saved cloned_output.mp3")
```

### Streaming Variant

Add `"stream": true` to the request body for progressive audio delivery. The response will be a chunked transfer-encoded stream:

```bash
VOICE_B64=$(base64 < voice_sample.wav)

curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/speech \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"pocket_tts\",
    \"input\": \"Streaming cloned voice output from PocketTTS.\",
    \"voice\": \"clone\",
    \"voice_reference\": \"$VOICE_B64\",
    \"response_format\": \"mp3\",
    \"stream\": true
  }" \
  --output cloned_stream.mp3
```

### Extra Parameters

You can override generation settings per-request via `extra_params`:

```json
{
  "model": "pocket_tts",
  "input": "Custom generation settings.",
  "voice": "clone",
  "voice_reference": "<base64>",
  "extra_params": {
    "max_frames": 800,
    "temperature": 0.5,
    "stream_first_chunk_frames": 3,
    "stream_target_buffer_sec": 0.15,
    "stream_max_chunk_frames": 20
  }
}
```

## Step 7: Upload and Reuse a Voice (Voice Registry)

Sending base64 audio on every request is inefficient if you reuse the same voice. Upload the sample once and reference it by ID.

### Upload the Voice

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/voices/upload \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -F "file=@voice_sample.wav" \
  -F "name=MyVoice" \
  -F "provider=pocket_tts"
```

The response includes a `voice_id` (e.g., `"voice_id": "abc123..."`). Save this value.

### Reuse the Voice

Use `custom:<voice_id>` as the `voice` field - no `voice_reference` needed:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/speech \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pocket_tts",
    "input": "Reusing my uploaded voice without sending the audio again.",
    "voice": "custom:abc123...",
    "response_format": "mp3"
  }' \
  --output reused_voice.mp3
```

Voice records are persisted in the per-user voice registry database, so they survive server restarts.

## Step 8: Set PocketTTS as Default Provider

When a TTS request omits `model` and `voice`, the server uses defaults. Here are three ways to make PocketTTS your default.

### Method A: YAML Config (provider_priority)

Edit `tldw_Server_API/Config_Files/tts_providers_config.yaml` and move `pocket_tts` to the top of the priority list:

```yaml
provider_priority:
  - pocket_tts   # <-- first in line
  - openai
  - kokoro
  # ...
```

> **Note:** PocketTTS always requires a voice reference. Requests that omit both `voice_reference` and a `custom:<voice_id>` voice will fail. If you want a zero-configuration fallback, keep a non-cloning provider (like `openai` or `kokoro`) in the priority list as well.

### Method B: Environment Variables

```bash
export TTS_DEFAULT_PROVIDER=pocket_tts
export TTS_DEFAULT_VOICE=custom:abc123...   # your uploaded voice ID
```

### Method C: config.txt INI

Edit `tldw_Server_API/Config_Files/config.txt`:

```ini
[TTS-Settings]
default_provider = pocket_tts
default_voice = custom:abc123...
```

After any of these changes, restart the server for the new defaults to take effect.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `pocket_tts` not in provider list | Provider not enabled or assets missing | Verify `enabled: true` in YAML, check model paths exist, restart server |
| "PocketTTS requires voice_reference audio bytes" | No voice sample provided | Include `voice_reference` (base64) or use `custom:<voice_id>` |
| "voice reference validation failed" | Audio outside 1-60s, wrong format, or corrupt file | Check duration and format; re-encode with FFmpeg (see Step 5) |
| `ImportError` for pocket_tts_onnx | Runtime deps not installed | Run `pip install -e '.[TTS_pocket_tts]'` |
| "PocketTTS models directory not found" | Wrong `model_path` in config | Verify the path points to the `onnx/` subdirectory containing the `.onnx` files |
| Slow generation | FP32 precision or large `max_frames` | Switch to `precision: int8`, reduce `max_frames`, or try `device: cuda` if GPU available |
| Poor voice quality | Reference audio too short, noisy, or multi-speaker | Use a longer (5-15s), clean, single-speaker sample; adjust `temperature` (lower = more stable) |
| "PocketTTS ONNX assets missing" | Incomplete download | Re-run installer with `--force`, or manually verify all 5 ONNX files exist |

## Quick Reference

```
Install:     pip install -e '.[TTS_pocket_tts]' && pip install -U "huggingface_hub"
Download:    python Helper_Scripts/TTS_Installers/install_tts_pocket_tts_onnx.py
Config:      tldw_Server_API/Config_Files/tts_providers_config.yaml
Providers:   GET  /api/v1/audio/providers
Voices:      GET  /api/v1/audio/voices/catalog?provider=pocket_tts
Synthesize:  POST /api/v1/audio/speech
Upload:      POST /api/v1/audio/voices/upload
```
