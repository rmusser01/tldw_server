# Chatterbox TTS Setup Runbook

This runbook explains how to enable and operate the Chatterbox TTS provider inside the tldw_server backend. It covers installation, model downloads, configuration, API usage, WebUI steps, and troubleshooting.

## Overview
- Provider: Resemble AI Chatterbox family: Original, Multilingual, and Turbo
- Integration: Adapter `ChatterboxAdapter` maps repo requests to upstream Chatterbox `generate()` calls
- Model IDs: `chatterbox`, `chatterbox-emotion`, `chatterbox-multilingual`, `chatterbox-turbo`
- Features: emotion exaggeration for Original/Multilingual, optional voice cloning from a reference clip, multilingual synthesis for 23 language codes, Turbo paralinguistic tags such as `[laugh]` and `[cough]`, and streaming output via progressive encoding

Key files:
- Adapter: `tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py:1`
- Shared catalog: `tldw_Server_API/app/core/TTS/chatterbox_catalog.py:1`
- Provider config (YAML): `tldw_Server_API/Config_Files/tts_providers_config.yaml`
- Provider registry/mapping: `tldw_Server_API/app/core/TTS/adapter_registry.py`
- TTS endpoint (OpenAI-compatible): `POST /api/v1/audio/speech`

## Requirements
- Python 3.11 recommended
- FFmpeg installed and on PATH (audio resampling/conversion)
- PyTorch 2.0+ (CUDA or MPS optional, CPU supported)
- Internet access for first-time model download, or pre-download for offline mode

## Install Options

Option A - Install repo extras, then install the upstream Chatterbox package:
```bash
pip install -e .[TTS_chatterbox]
# Optional language preprocessing utilities for multilingual
pip install -e .[TTS_chatterbox_lang]
pip install chatterbox-tts
```

Option B - Install upstream Chatterbox from source:
```bash
git clone https://github.com/resemble-ai/chatterbox
cd chatterbox
pip install -e .
```

## Model Weights
The adapter loads upstream weights lazily through:
- `ChatterboxTTS.from_pretrained()` for `chatterbox` and `chatterbox-emotion`
- `ChatterboxMultilingualTTS.from_pretrained()` for `chatterbox-multilingual`
- `ChatterboxTurboTTS.from_pretrained()` for `chatterbox-turbo`
- `ChatterboxVC.from_pretrained()` for voice conversion

When `model_path`, `multilingual_model_path`, `turbo_model_path`, or `vc_model_path` points to an existing local path, the adapter uses the matching upstream `from_local()` loader instead of `from_pretrained()`. Repo IDs such as `ResembleAI/chatterbox` stay on the upstream pretrained loader.

Each runtime downloads assets from Hugging Face on first use unless offline mode is enabled.

Pre-download (recommended for servers/CI):
```bash
# Populates the local HF cache and optional mirror directories
hf download ResembleAI/chatterbox --local-dir ./models/chatterbox
hf download ResembleAI/chatterbox-multilingual --local-dir ./models/chatterbox-multilingual
hf download ResembleAI/chatterbox-turbo --local-dir ./models/chatterbox-turbo
```

Offline mode (use only local cache):
```bash
export CHATTERBOX_AUTO_DOWNLOAD=0
export TTS_AUTO_DOWNLOAD=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Notes:
- The adapter will set HF offline flags automatically when `auto_download` is disabled.
- Ensure your HF cache contains these files: `ve.safetensors`, `t3_cfg.safetensors`, `s3gen.safetensors`, `tokenizer.json`, `conds.pt`.

## Configuration

Primary provider config lives in YAML at `tldw_Server_API/Config_Files/tts_providers_config.yaml` under the `providers.chatterbox` section. Example:
```yaml
providers:
  chatterbox:
    enabled: true
    variant: "standard"      # standard, multilingual, or turbo
    model_path: "ResembleAI/chatterbox"
    multilingual_model_path: "ResembleAI/chatterbox-multilingual"
    turbo_model_path: "ResembleAI/chatterbox-turbo"
    vc_model_path: null       # Optional local ChatterboxVC directory; null uses upstream pretrained default
    device: "auto"           # auto selects cuda, mps, then cpu
    use_multilingual: false  # legacy compat: non-English chatterbox requests use multilingual when true
    use_bf16: false          # opt-in: false, true/on, or auto via TTS_BF16
    sample_rate: 24000
    disable_watermark: true  # Adapter replaces upstream watermarker (no watermark)
    target_latency_ms: 200   # Streaming chunk duration hint in milliseconds
    conditionals_cache_size: 16  # Prepared voice-reference conditionals cached per adapter; 0 disables retention
    auto_download: true      # Optional: let the adapter fetch models on first use
```

Additional behavior controlled by env:
- `CHATTERBOX_AUTO_DOWNLOAD` / `TTS_AUTO_DOWNLOAD` - override auto-download at runtime
- `TTS_BF16=off|on|auto` - opt into Chatterbox TTS BF16 generation when YAML does not set `use_bf16`
- `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` - force offline cache usage

What the adapter reads:
- Device selection & family mode: `chatterbox_device`, `chatterbox_variant`, `chatterbox_use_multilingual`
- Model path overrides: `chatterbox_model_path`, `chatterbox_multilingual_model_path`, `chatterbox_turbo_model_path`, `chatterbox_vc_model_path` (or unprefixed YAML equivalents)
- Watermarking toggle: `chatterbox_disable_watermark` (default true)
- BF16 inference: `chatterbox_use_bf16` or `use_bf16` (`off` by default; `auto` currently enables only when CUDA reports BF16 support)
- Voice conditionals cache: `chatterbox_conditionals_cache_size` or `conditionals_cache_size` (default 16 entries; 0 disables retention)
- Defaults for generation: `chatterbox_default_exaggeration`, `chatterbox_cfg_weight`, `chatterbox_temperature`, `chatterbox_repetition_penalty`, `chatterbox_min_p`, `chatterbox_top_p`

See adapter for details: `tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py`.

## Start the Server
```bash
python -m uvicorn tldw_Server_API.app.main:app --reload
# API docs: http://127.0.0.1:8000/docs
# Quickstart: http://127.0.0.1:8000/api/v1/config/quickstart
```

Verify provider availability:
```bash
curl http://127.0.0.1:8000/api/v1/audio/providers
curl http://127.0.0.1:8000/api/v1/audio/tts/providers/chatterbox/model-info
```
You should see Chatterbox listed with its capabilities once the adapter imports successfully.
The Chatterbox capability metadata includes family-specific model IDs, supported generation controls, Turbo-only controls, `split_text`/`chunk_size` chunking aliases, BF16 modes, and the voice-conversion endpoint. The focused model-info route returns the provider's loaded/initialized state, supported model IDs, Chatterbox family metadata, voice-conversion metadata, and unload endpoint without requiring clients to parse the full provider catalog.

Discover provider voices:
```bash
curl "http://127.0.0.1:8000/api/v1/audio/voices/catalog?provider=chatterbox"
curl "http://127.0.0.1:8000/api/v1/audio/voices/catalog?provider=chatterbox&format=openai"
```
The first response keeps tldw's provider-to-voices catalog shape. The second response returns an OpenAI-style `{ "object": "list", "data": [...] }` provider voice list for clients that expect upstream Chatterbox `/v1/audio/voices`-style discovery. `/api/v1/audio/voices` remains the authenticated user's stored custom voice list.

Release a loaded Chatterbox runtime without restarting the server:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/tts/providers/chatterbox/unload" \
  -H "Authorization: Bearer <TOKEN>"
```
The next Chatterbox speech or voice-conversion request reloads the provider on demand.

## API Usage

Streaming request (OpenAI-compatible) to `POST /api/v1/audio/speech` (`audio.py:120`):
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/speech" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatterbox",
    "input": "Hello from Chatterbox!",
    "voice": "default",
    "response_format": "mp3",
    "stream": true
  }' --output out.mp3
```

Voice cloning (send base64-encoded reference; ideal duration 5-20s at 24kHz):
```bash
BASE64_AUDIO=$(base64 -i my_voice_24k.wav)
curl -X POST "http://127.0.0.1:8000/api/v1/audio/speech" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d "{\
    \"model\": \"chatterbox\",\
    \"input\": \"This should sound like my reference.\",\
    \"voice\": \"clone\",\
    \"voice_reference\": \"$BASE64_AUDIO\",\
    \"response_format\": \"wav\"\
  }" --output clone.wav
```

Stored custom voices can also be selected with upstream-style Chatterbox fields. tldw resolves `predefined_voice_id` through the authenticated user's custom voice store; it does not read arbitrary server-side `reference_audio_filename` paths.
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/speech" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatterbox",
    "input": "Use my stored voice.",
    "voice": "default",
    "response_format": "wav",
    "stream": false,
    "extra_params": {"voice_mode": "predefined", "predefined_voice_id": "voice_abc123"}
  }' --output stored_voice.wav
```

Voice conversion uses a dedicated multipart endpoint, not a TTS model alias:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/voice-conversion" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "source_audio=@source_speech.wav;type=audio/wav" \
  -F "target_voice=@target_voice.wav;type=audio/wav" \
  -F "response_format=wav" \
  -F "stream=false" \
  --output converted.wav
```

Use a stored custom voice as the target reference by passing its voice ID instead of uploading `target_voice`:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/voice-conversion" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "source_audio=@source_speech.wav;type=audio/wav" \
  -F "target_voice_id=voice_abc123" \
  -F "response_format=wav" \
  -F "stream=false" \
  --output converted.wav
```

Notes:
- `source_audio` is required and should contain the speech to convert.
- `target_voice` is optional; omit it to let upstream Chatterbox VC use its default target reference.
- `target_voice_id` resolves a stored custom voice for the authenticated user; do not send both `target_voice` and `target_voice_id`.
- Supported output `response_format` values are `wav`, `mp3`, `flac`, `opus`, `aac`, and `pcm`; unsupported output formats return HTTP 400.
- Source and target uploads are capped at 50 MiB each.
- Virtual keys can grant this route with the `audio.voice_conversion` privilege scope.

Multilingual synthesis:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/speech" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatterbox-multilingual",
    "input": "Bonjour, comment ça va?",
    "language": "fr",
    "response_format": "mp3",
    "stream": true
  }' --output fr.mp3
```

Legacy compatibility: `model: "chatterbox"` plus `use_multilingual: true` still routes non-English language requests to the multilingual runtime. New clients should send `model: "chatterbox-multilingual"` explicitly.

Turbo synthesis with paralinguistic tags:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/speech" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatterbox-turbo",
    "input": "That is useful [chuckle], and it should run faster.",
    "voice": "default",
    "response_format": "wav",
    "stream": true,
    "extra_params": {"temperature": 0.8, "top_p": 0.95, "top_k": 1000}
  }' --output turbo.wav
```

Tuning generation for Original/Multilingual (adapter maps emotion+intensity -> `exaggeration`):
- `emotion`: one of neutral, happy, sad, angry, surprised, fearful, disgusted, excited, calm, confused
- `emotion_intensity`: 0.0-2.0 (defaults scaled to `exaggeration` in [0.0-1.0])
- `language`: upstream-style alias for `lang_code` on Chatterbox-family requests when `lang_code` is omitted
- `output_format`: upstream-style alias for `response_format` when `response_format` is omitted; `response_format` wins if both are provided
- Extra params accepted: `cfg_weight`, `temperature`, `repetition_penalty`, `min_p`, `top_p`, and `speed_factor` when the installed Chatterbox runtime supports it. Non-default OpenAI-compatible `speed` is also offered to the runtime as `speed_factor`.
- Turbo uses `temperature`, `repetition_penalty`, `top_p`, `top_k`, and `speed_factor` when supported. It intentionally ignores CFG/exaggeration/min-p controls; response metadata includes `ignored_controls` when callers send those fields.
- For long non-streaming Chatterbox requests, upstream-style `extra_params.split_text` and `extra_params.chunk_size` map to the service chunker. Set `"stream": false` so the service can generate PCM segments, stitch them, and return one final encoded response.

Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/speech" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatterbox",
    "input": "I am thrilled to be here!",
    "emotion": "excited",
    "emotion_intensity": 1.5,
    "extra_params": {"cfg_weight": 0.5, "temperature": 0.8},
    "response_format": "mp3"
  }' --output excited.mp3
```

Long text chunking example:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/speech" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatterbox",
    "input": "A long passage ...",
    "voice": "default",
    "response_format": "mp3",
    "stream": false,
    "extra_params": {"split_text": true, "chunk_size": 1200}
  }' --output long.mp3
```

## WebUI Usage
1. Start the server and open `http://127.0.0.1:8000/api/v1/config/quickstart`.
2. Go to the Audio tab, select “Chatterbox”.
3. Provide input text; optionally upload/mic-record a reference clip.
4. Adjust emotion intensity (exaggeration), CFG weight, and sampling parameters.
5. Click Generate to preview.

The WebUI uses the same `/api/v1/audio/speech` endpoint under the hood and will stream results.

## Performance & Devices
- Device selection order: configured `device` -> CUDA if available -> MPS if available -> CPU.
- Recommended: 4GB+ VRAM for smooth GPU inference; CPU and MPS are supported with higher latency.
- Adapter streams by encoding generated waveforms into chunks sized by `target_latency_ms` / `chatterbox_target_latency_ms` (default 200 ms) for immediate playback.
- Optional BF16 for TTS generation is disabled by default. When enabled, the adapter prepares the Chatterbox T3 module with `torch.bfloat16` and wraps TTS generation in `torch.autocast` when available; voice conversion remains on the upstream default precision path.

## Offline & Caching Checklist
- Pre-download model with `hf` (see above).
- Set `CHATTERBOX_AUTO_DOWNLOAD=0`, `TTS_AUTO_DOWNLOAD=0`.
- Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- Ensure HF cache contains required safetensors and tokenizer files.

## Troubleshooting
1) Import error: `chatterbox` not found
- Install upstream package: `pip install chatterbox-tts`, or use repo’s vendored module with `pip install -e .[TTS_chatterbox]`.

2) Model download blocked/offline
- Pre-download via `hf download ResembleAI/chatterbox --local-dir ./models/chatterbox` and set offline env vars.

3) Voice cloning fails or sounds wrong
- Use 5-20s single-speaker WAV/FLAC at 24kHz; avoid noisy or clipped audio.
- Convert with ffmpeg:
  ```bash
  ffmpeg -i input.wav -ar 24000 -ac 1 -t 15 ref_24k.wav
  ```

4) Latency too high
- Use `device: cuda` if available; reduce `temperature`; ensure no CPU throttling.
- Adjust CFG weight (~0.3-0.5) for more stable pacing.

5) Multilingual outputs have incorrect accent
- Set `cfg_weight: 0.0` for language transfer if the reference is a different language.

6) Turbo ignores emotion/CFG controls
- Use paralinguistic tags such as `[laugh]`, `[cough]`, and `[chuckle]` in the input text.
- Check response metadata `ignored_controls` to see which request controls were intentionally dropped.

7) Watermarking
- The adapter disables watermarking by default (`disable_watermark: true`). If you need watermarking, use the upstream models directly outside the adapter.

## Notes for Developers
- The adapter lazily loads standard, multilingual, and Turbo runtimes based on the request model id, `variant`, and legacy `use_multilingual` config. See `chatterbox_adapter.py` and `_get_model`.
- Provider model name mapping includes `chatterbox`, `chatterbox-emotion`, `chatterbox-multilingual`, and `chatterbox-turbo`.
- Voice conversion is intentionally not exposed as a TTS model alias. Use `POST /api/v1/audio/voice-conversion`, which calls `ChatterboxVC.generate(audio=..., target_voice_path=...)`.
- TTS and voice-conversion streaming both use `waveform_streamer.stream_encoded_waveform(...)` with chunk duration derived from `target_latency_ms` / `chatterbox_target_latency_ms`.
