# TTS Module - Text-to-Speech Service

## Overview

The TTS module provides a production-ready, extensible Text-to-Speech service with support for multiple providers, voice cloning, and OpenAI-compatible API endpoints. Built with an adapter pattern architecture, it offers seamless fallback between providers, comprehensive error handling, and enterprise-grade features.

Developer-oriented details (architecture, provider matrix, configuration, and test guidance) live in `tldw_Server_API/app/core/TTS/README.md`.

## Features

### Core Capabilities
- **Multi-Provider Support**: OpenAI, ElevenLabs, and local adapters (Kokoro, KittenTTS, PocketTTS, LuxTTS, Higgs, Chatterbox, Dia, VibeVoice, VibeVoice Realtime, Qwen3-TTS, IndexTTS2, NeuTTS, Supertonic, Supertonic2, EchoTTS) with a mock adapter for testing.
- **Managed Sidecars**: OmniVoice runs behind a dedicated loopback sidecar runtime instead of sharing the main server interpreter.
- **Voice Cloning**: Voice reference audio accepted by PocketTTS, LuxTTS, Higgs, Chatterbox, Dia, VibeVoice, Qwen3-TTS, NeuTTS, IndexTTS2, and EchoTTS (ElevenLabs supports user voices via API).
- **Streaming Audio**: Real-time chunked streaming across adapters; NeuTTS enables streaming when a quantized (GGUF) backbone is loaded; VibeVoice Realtime uses a WS backend.
- **Format Support**: Adapter-specific coverage spanning MP3, WAV, OPUS, FLAC, PCM, AAC, and OGG via the shared `AudioFormat` enum.
- **OpenAI Compatibility**: Drop-in replacement for OpenAI TTS API
- **Voice Management**: Upload, catalog, quota enforcement, and preview endpoints backed by the `voice_manager` service.
- **Fault Tolerance**: Circuit breaker pattern with automatic failover and exponential backoff
- **Performance Metrics**: Prometheus metrics (`tts_requests_total`, `tts_request_duration_seconds`, etc.) plus health checks
- **Transcription/Translation**: OpenAI-compatible speech-to-text endpoints

### Supported Providers

| Provider | Type | Languages | Voice Cloning | Key Features |
|----------|------|-----------|---------------|--------------|
| **OpenAI** | Commercial API | EN* | ❌ | Industry standard, HD quality |
| **ElevenLabs** | Commercial API | 29 | ✅ (Pro/user voices) | Premium quality, emotion control |
| **Kokoro** | Local ONNX | EN (US/GB) | ❌ | Lightweight, CPU-friendly, offline |
| **KittenTTS** | Local ONNX | EN | ❌ | Small local ONNX voices, first-use download, pinned HF revisions |
| **PocketTTS** | Local ONNX | EN | ✅ (reference) | Zero-shot cloning, streaming ONNX |
| **LuxTTS** | Local PyTorch | EN | ✅ (reference) | 48kHz voice cloning (ZipVoice) |
| **Higgs** | Local PyTorch | 50+ | ✅ (3-10s) | Music generation, multi-lingual |
| **Chatterbox** | Local PyTorch | EN | ✅ (5-20s) | Emotion exaggeration control |
| **Dia** | Local PyTorch | EN | ✅ (dialogue prompts) | Multi-speaker dialogue specialist |
| **VibeVoice** | Local PyTorch | 12 | ✅ (Any) | Long-form (90min), spontaneous music |
| **VibeVoice Realtime** | WS adapter | EN | ❌ | Low-latency streaming (requires realtime backend) |
| **Qwen3-TTS** | Runtime-aware local/remote | Multi (auto/zh/en/ja/ko/de/fr/ru/pt/es/it) | Runtime-dependent | One provider key with `upstream`, `mlx`, or `remote` execution |
| **IndexTTS2** | Local PyTorch | EN/zh | ✅ (reference) | Zero-shot cloning, emotion prompts, low-latency streaming |
| **NeuTTS** | Local (Hybrid) | EN | ✅ (3-15s) | Instant voice cloning, optional GGUF streaming |
| **Supertonic** | Local ONNX | Varies (model) | ❌ | User-supplied voice styles |
| **Supertonic2** | Local ONNX | Varies (model) | ❌ | User-supplied voice styles |
| **EchoTTS** | Local PyTorch | EN | ✅ (reference) | CUDA-only voice cloning |
| **OmniVoice** | Local sidecar | EN | ✅ (reference / stored custom voices) | Dedicated managed runtime, default `voice="auto"` when omitted |

\* Current adapter configuration targets English (`tts-1` / `tts-1-hd`). Additional languages depend on OpenAI model availability.

### Qwen3-TTS Runtime Modes

`qwen3_tts` keeps one public provider key and selects an internal runtime with `providers.qwen3_tts.runtime`.

- `auto`: prefers `mlx` on Apple Silicon macOS and `upstream` elsewhere
- `upstream`: in-process `qwen_tts`
- `mlx`: in-process `mlx-audio`
- `remote`: OpenAI-compatible hosted or sidecar backend

MLX v1 scope:

- preset-speaker synthesis only
- uploaded `custom:<voice_id>` voices are rejected
- `Base` and `VoiceDesign` requests are rejected
- stream requests are accepted and return buffered fallback chunks

Config examples:

```yaml
providers:
  qwen3_tts:
    enabled: true
    runtime: "mlx"
    model: "auto"
    model_path: null
    mlx_model: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    device: "mps"
```

```yaml
providers:
  qwen3_tts:
    enabled: true
    runtime: "remote"
    base_url: "http://127.0.0.1:8001/v1/audio/speech"
    api_key: "${QWEN_REMOTE_API_KEY}"
    capability_override:
      supports_streaming: true
      supports_voice_cloning: true
      supports_emotion_control: false
      supported_modes: ["custom_voice_preset", "voice_clone"]
      supported_voices: ["Cherry", "Ethan"]
      supports_uploaded_custom_voices: false
```

Remote Qwen requests extend the normal OpenAI `/audio/speech` payload through `extra_body` fields:

- `ref_audio_b64`
- `ref_text`
- `voice_clone_prompt`
- `x_vector_only_mode`
- `description`

Remote capability reporting defaults to conservative values until `capability_override` is set. This keeps `/api/v1/audio/health` and provider metadata aligned with the actual hosted backend. Voice catalogs stay empty unless `capability_override.supported_voices` is provided explicitly.

See `TTS-DEPLOYMENT.md` for Apple Silicon smoke-test steps.

### IndexTTS2 Adapter

IndexTTS2 extends the local pipeline with expressive zero-shot voice cloning and optional emotion conditioning.

- **Dependencies**: `torch`, `torchaudio`, `transformers`, `sentencepiece`, `safetensors`, plus the upstream [`index-tts`](https://github.com/index-tts/index-tts) package. Install via `pip install -e .` from the project root.
- **Model Assets**: Place checkpoints under `checkpoints/index_tts2/` (config, acoustic model, codec, Qwen emotion model). The adapter surfaces `model_dir` and `cfg_path` overrides in `tts_providers_config.yaml`.
- **Configuration Snippet**:

```yaml
providers:
  index_tts:
    enabled: true
    model_dir: "checkpoints/index_tts2"
    cfg_path: "checkpoints/index_tts2/config.yaml"
    device: "cuda"         # "cpu" works for debugging; GPU highly recommended
    use_fp16: true
    interval_silence: 200  # ms between chunks
    max_text_tokens_per_segment: 120
```

- **Voice Mapping**: The configuration ships with a `clone_required` placeholder to remind callers to include `voice_reference` audio bytes. The adapter rejects requests that omit a speaker prompt.
- **Streaming**: Uses `IndexTTS2.infer(..., stream_return=True)` under the hood with async chunk normalization and sample-rate conversion to 24 kHz when requested.
- **Emotion Controls**: Provide `extra_params` such as `emo_audio_reference`, `emo_alpha`, `emo_vector`, or `emo_text` to tap into QwenEmotion-guided delivery.

> **Manual GPU smoke test**: See `TTS-DEPLOYMENT.md` for a short checklist covering environment validation and end-to-end streaming playback on real hardware.

### EchoTTS Adapter

EchoTTS provides local voice cloning with a required reference clip and optional text chunking for long inputs.

- **Chunking controls** live under `extra_params` in requests (or `providers.echo_tts.extra_params` in config):
  - `chunk_text` (bool, default auto): enable/disable chunking.
  - `chunk_max_chars` / `chunk_max_bytes`: per-chunk limits (clamped to 768 chars / 767 bytes).
  - `interval_silence`: milliseconds of silence inserted between chunks.

Configuration snippet:

```yaml
providers:
  echo_tts:
    enabled: true
    module_path: "../echo-tts"
    device: "cuda"
    sample_rate: 44100
    extra_params:
      chunk_text: true
      chunk_max_chars: 768
      chunk_max_bytes: 767
      interval_silence: 200
```

## One-Command Installers
Run these from the project root to install a single TTS backend (deps + models where applicable):

```bash
# Kokoro (v1.0 ONNX + voices)
python Helper_Scripts/TTS_Installers/install_tts_kokoro.py
# Overwrite existing assets: add --force
# python Helper_Scripts/TTS_Installers/install_tts_kokoro.py --force

# Dia / Higgs / VibeVoice
python Helper_Scripts/TTS_Installers/install_tts_dia.py
python Helper_Scripts/TTS_Installers/install_tts_higgs.py
python Helper_Scripts/TTS_Installers/install_tts_vibevoice.py --variant 1.5B

# OmniVoice sidecar runtime
python Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py

# NeuTTS (deps; optional prefetch)
python Helper_Scripts/TTS_Installers/install_tts_neutts.py --prefetch

# IndexTTS2 (deps + checkpoints folder)
python Helper_Scripts/TTS_Installers/install_tts_index_tts2.py

# Chatterbox (deps only)
python Helper_Scripts/TTS_Installers/install_tts_chatterbox.py [--with-lang]
```

Flags:
- `TLDW_SETUP_SKIP_PIP=1` to skip pip installs
- `TLDW_SETUP_SKIP_DOWNLOADS=1` to skip HF downloads
- `TLDW_SETUP_FORCE_DOWNLOADS=1` to overwrite existing assets

Alternative helper (assets only):
```bash
python Helper_Scripts/download_kokoro_assets.py \
  --repo-id onnx-community/Kokoro-82M-v1.0-ONNX-timestamped \
  --model-path models/kokoro/onnx/model.onnx \
  --voices-dir models/kokoro/voices
```

### KittenTTS

KittenTTS is a local English-only ONNX provider that uses static preset voices and can download its model assets on first use.

- Default model alias: `KittenML/kitten-tts-nano-0.8` (normalized to the canonical fp32 repo internally)
- Supported voices: `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`, `Kiki`, `Leo`
- Supported formats: `wav`, `mp3`, `pcm`
- Required local deps: `onnxruntime`, `phonemizer-fork`, `espeakng_loader`, `huggingface_hub`
- Security model: downloads are pinned to immutable Hugging Face commit hashes through `model_revision`

Configuration example:

```yaml
providers:
  kitten_tts:
    enabled: true
    model: "KittenML/kitten-tts-nano-0.8"
    model_revision: "8d6d5a1851ffd13c894c40227c888302c2a86ef7"
    device: "cpu"
    auto_download: true
    extra_params:
      cache_dir: "cache/kitten_tts"
      clean_text: false
```

If you want to prefetch assets during setup instead of waiting for first synthesis, use the setup UI advanced TTS installer and select `kitten_tts`.

### OmniVoice

OmniVoice uses a managed sidecar process with its own virtual environment under `models/omnivoice_sidecar`.

- Installer: `python Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py`
- Preferred local source checkout: `../OmniVoice`
- Runtime layout: `.venv`, `runtime`, and `logs` under `models/omnivoice_sidecar`
- Provider requests that explicitly resolve to OmniVoice and omit `voice` normalize to `auto`
- The sidecar supervisor consumes the configured interpreter path so the dedicated installer-created runtime is the one that launches

Configuration notes (providers.kokoro in tts_providers_config.yaml):
- model_path: path to ONNX file
- voices_json / voice_dir: directory with voice profiles
- device: cpu|cuda|mps
- sample_rate: default 24000
- pause_interval_words: insert a pacing tag every N words (default 500)
- pause_tag: the pacing marker to insert (default "[pause=1.1]")

The adapter preprocesses long inputs and inserts a pause tag at least every `pause_interval_words` tokens to keep delivery natural and avoid very long continuous utterances.

### Voice Management & Cloning

- `voice_manager.py` validates uploads (extensions, duration, sample rate, size) and enforces quotas (`VOICE_RATE_LIMITS`).
- API surface:
  - `POST /api/v1/audio/voices/upload`
  - `GET /api/v1/audio/voices/catalog`
  - `GET /api/v1/audio/voices`, `GET/DELETE /api/v1/audio/voices/{voice_id}`
  - `POST /api/v1/audio/voices/{voice_id}/preview`
- Provider-specific requirements are documented in `TTS-VOICE-CLONING.md`; NeuTTS and IndexTTS2 require voice references for best results.

## Architecture

### V2 Adapter Pattern

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   API       │────▶│  TTS Service │────▶│  Adapter    │
│  Endpoint   │     │      V2      │     │  Registry   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │                     │
                    ┌───────▼───────┐    ┌───────▼───────┐
                    │Circuit Breaker│    │   Provider    │
                    │   Manager     │    │   Adapters   │
                    └───────────────┘    └───────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
              ┌─────▼─────┐            ┌────────▼────────┐           ┌───────▼───────┐
              │  OpenAI   │            │   Local Models  │           │  Commercial   │
              │  Adapter  │            │   (Kokoro,etc)  │           │   (ElevenLabs)│
              └───────────┘            └─────────────────┘           └───────────────┘
```

### Key Components

1. **TTSServiceV2** (`tts_service_v2.py`)
   - Main service orchestrator
   - Handles provider selection and fallback
   - Integrates metrics and circuit breaker

2. **Adapter Registry** (`adapter_registry.py`)
   - Provider registration and management
   - Capability discovery
   - Dynamic adapter loading

3. **Base Adapter** (`adapters/base.py`)
   - Abstract interface for all providers
   - Standard request/response formats
   - Capability reporting

4. **Provider Adapters** (`adapters/*.py`)
   - Provider-specific implementations
   - Handle authentication and API calls
   - Audio generation and streaming

5. **Circuit Breaker** (`circuit_breaker.py`)
   - Fault tolerance and recovery
   - Automatic provider failover
   - Configurable thresholds

6. **Audio Utils** (`audio_utils.py`)
   - Voice reference processing
   - Format conversion
   - Audio validation

## Installation

### Prerequisites

```bash
# System dependencies
apt-get install ffmpeg espeak-ng  # Ubuntu/Debian
brew install ffmpeg espeak         # macOS

# Python dependencies
pip install -e .
```

### Quick Start

1. **Configure API Keys**
```bash
# In config.txt
[API]
openai_api_key = sk-...
elevenlabs_api_key = xi-...
```

2. **Configure TTS Settings**
```yaml
# In tts_providers_config.yaml
provider_priority:
  - openai      # Primary provider
  - kokoro      # Fallback to local

providers:
  openai:
    enabled: true
    model: tts-1-hd

  kokoro:
    enabled: true
    use_onnx: true
    model_path: ./models/kokoro/onnx/model.onnx
    voices_json: ./models/kokoro/voices
```

3. **Start the Server**
```bash
python -m uvicorn tldw_Server_API.app.main:app --host 0.0.0.0 --port 8000
```

For NeuTTS installation and usage, see `Docs/STT-TTS/NEUTTS_TTS_SETUP.md`.

## API Usage

### Basic Text-to-Speech

```python
import json
from pathlib import Path
from urllib import request

payload = {
    "model": "tts-1",  # or an ElevenLabs model like "eleven_multilingual_v2" / "eleven_turbo_v2"
    "input": "Hello, world!",
    "voice": "alloy",
    "response_format": "mp3",
}

req = request.Request(
    "http://localhost:8000/api/v1/audio/speech",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": "Bearer your-token",
        "Content-Type": "application/json",
    },
    method="POST",
)

with request.urlopen(req) as resp:
    Path("output.mp3").write_bytes(resp.read())
```

### Voice Cloning

```python
import base64
import json
from urllib import request

# Prepare voice reference
with open("voice_sample.wav", "rb") as f:
    voice_ref = base64.b64encode(f.read()).decode()

payload = {
    "model": "higgs",  # or chatterbox, vibevoice
    "input": "This will sound like the reference voice.",
    "voice": "default",
    "voice_reference": voice_ref,  # Base64 encoded audio
    "response_format": "mp3",
}

req = request.Request(
    "http://localhost:8000/api/v1/audio/speech",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with request.urlopen(req) as resp:
    audio = resp.read()
```

### Streaming Audio

```python
import json
from urllib import request

payload = {
    "model": "kokoro",  # ElevenLabs also supports streaming
    "input": "Streaming audio test.",
    "voice": "af_bella",
    "stream": True,
}

req = request.Request(
    "http://localhost:8000/api/v1/audio/speech",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with request.urlopen(req, timeout=30) as resp, open("stream.mp3", "wb") as f:
    while True:
        chunk = resp.read(8192)
        if not chunk:
            break
        f.write(chunk)
```

### List Voices

List voices from all available providers, or filter by provider.

```bash
# All providers (catalog)
curl -s http://localhost:8000/api/v1/audio/voices/catalog | jq

# ElevenLabs only
curl -s "http://localhost:8000/api/v1/audio/voices/catalog?provider=elevenlabs" | jq
```

Example response (filtered):
```json
{
  "elevenlabs": [
    { "id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "gender": "female", "language": "en", "description": "American female voice" },
    { "id": "29vD33N1CtxCmqQRPOHJ", "name": "Drew",   "gender": "male",   "language": "en", "description": "American male voice" }
  ]
}
```

Notes
- ElevenLabs voices include your account’s user voices (cached on adapter init) plus defaults.
- Other providers expose their known/static voices where applicable.

### Bootstrap Kokoro Assets

Recommended: run the installer from the repo root (downloads v1.0 ONNX + voices and checks eSpeak):

```bash
python Helper_Scripts/TTS_Installers/install_tts_kokoro.py
```

Manual alternative using the `hf` CLI:
```bash
pip install -U "huggingface_hub"
mkdir -p models/kokoro
hf download onnx-community/Kokoro-82M-v1.0-ONNX-timestamped onnx/model.onnx --local-dir models/kokoro/
hf download onnx-community/Kokoro-82M-v1.0-ONNX-timestamped voices          --local-dir models/kokoro/
```

Update `tts_providers_config.yaml` if you use custom paths. For the PyTorch variant, set `use_onnx: false`, provide a `.pth` path, and point `voice_dir` to the `voices/` directory.

### Transcription (Speech-to-Text)

```bash
curl -s -X POST "http://localhost:8000/api/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-token" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "language=en" \
  -F "response_format=json" | jq -r '.text'
```

## Configuration

### Provider Configuration (tts_providers_config.yaml)

Location resolution:
- `tldw_Server_API/Config_Files/tts_providers_config.yaml` (project-level override)
- `tldw_Server_API/app/core/TTS/tts_providers_config.yaml` (in-repo default)
- `./tts_providers_config.yaml` (working directory)
- `~/.config/tldw/tts_providers_config.yaml` (user config)

Notes:
- Local providers will not download models unless `auto_download: true` (or `TTS_AUTO_DOWNLOAD=1`).
- VibeVoice 7B official repo: `vibevoice/VibeVoice-7B`; optional quantized 7B: `FabioSarracino/VibeVoice-Large-Q8`.

```yaml
# Provider priority (fallback order)
provider_priority:
  - openai          # Try first
  - elevenlabs      # Try second
  - kokoro          # Local fallback

# Individual provider settings
providers:
  openai:
    enabled: true
    api_key: ${OPENAI_API_KEY}  # From environment
    model: tts-1-hd
    timeout: 30

  kokoro:
    enabled: true
    use_onnx: true
    model_path: ./models/kokoro/onnx/model.onnx
    voices_json: ./models/kokoro/voices
    device: cpu  # or cuda

  higgs:
    enabled: true
    model_path: bosonai/higgs-audio-v2-generation-3B-base
    device: cuda
    use_fp16: true

  chatterbox:
    enabled: true
    model_path: resemble-ai/chatterbox
    enable_watermark: true

  vibevoice:
    enabled: true
    auto_download: false        # Explicit opt-in to avoid unsolicited network fetches
    variant: 1.5B               # or 7B, 7B-Q8
    model_path: microsoft/VibeVoice-1.5B  # or vibevoice/VibeVoice-7B, FabioSarracino/VibeVoice-Large-Q8
    device: cuda

# Fallback configuration
fallback:
  enabled: true
  max_attempts: 3
  retry_delay_ms: 1000

# Circuit breaker settings
circuit_breaker:
  failure_threshold: 5
  recovery_timeout: 60
  half_open_calls: 3

# Performance settings
performance:
  max_concurrent_generations: 4
  # When true, embed error messages into the audio stream (compat mode)
  # When false, raise errors so the API returns proper HTTP errors
  stream_errors_as_audio: true
  cache_enabled: false

# Environment override (takes precedence over YAML)
# export TTS_STREAM_ERRORS_AS_AUDIO=false
```

#### Generic vs Provider-Prefixed Keys

The registry automatically aliases common, generic keys from your YAML to the provider-prefixed keys that some adapters expect. This keeps configuration consistent across providers while satisfying adapter expectations.

- OpenAI
  - Generic: `api_key`, `base_url`, `model`, `stability`, `similarity_boost`, `style`, `speaker_boost`
  - Aliased to: `openai_api_key`, `openai_base_url`, `openai_model`

- ElevenLabs
  - Generic: `api_key`, `base_url`, `model`, `stability`, `similarity_boost`, `style`, `speaker_boost`
  - Aliased to: `elevenlabs_api_key`, `elevenlabs_base_url`, `elevenlabs_model`, `elevenlabs_stability`, `elevenlabs_similarity_boost`, `elevenlabs_style`, `elevenlabs_speaker_boost`

- Kokoro
  - Generic: `device`, `use_onnx`, `model_path`, `voices_json`, `voice_dir`
  - Aliased to: `kokoro_device`, `kokoro_use_onnx`, `kokoro_model_path`, `kokoro_voices_json`, `kokoro_voice_dir`
  - Other Kokoro options are read generically (no alias needed): `sample_rate`, `normalize_text`, `sentence_splitting`

- Higgs
  - Generic: `model_path`, `tokenizer_path`, `device`, `use_fp16`, `batch_size`
  - Aliased to: `higgs_model_path`, `higgs_tokenizer_path`, `higgs_device`, `higgs_use_fp16`, `higgs_batch_size`

- Dia
  - Generic: `model_path`, `device`, `use_safetensors`, `use_bf16`, `sample_rate`, `auto_detect_speakers`, `max_speakers`
  - Aliased to: `dia_model_path`, `dia_device`, `dia_use_safetensors`, `dia_use_bf16`, `dia_sample_rate`, `dia_auto_detect_speakers`, `dia_max_speakers`

- Chatterbox
  - Generic: `device`, `use_multilingual`, `disable_watermark`, `sample_rate`, `target_latency_ms`
  - Aliased to: `chatterbox_device`, `chatterbox_use_multilingual`, `chatterbox_disable_watermark`, `chatterbox_target_latency_ms`

- VibeVoice
  - Generic: `device`, `sample_rate`, `variant`, `model_path`, `model_dir`, `cache_dir`, `voices_dir`, `background_music`, `enable_singing`, `use_quantization`, `auto_cleanup`, `auto_download`, `enable_sage`, `attention_type`, `cfg_scale`, `diffusion_steps`, `temperature`, `top_p`, `top_k`, `stream_chunk_size`, `stream_buffer_size`
  - Aliased to the corresponding `vibevoice_*` fields

Notes
- Environment variables still work and may override YAML (e.g., `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`).
- Adapters that already read generic names (e.g., Kokoro’s `sample_rate`) don’t require aliasing for those fields.
- If you add new provider options in YAML, prefer generic names; the registry can be extended to alias them.

#### Concurrency Control

The service (`TTSServiceV2`) reads `performance.max_concurrent_generations` and sets an internal semaphore to enforce this limit at runtime. This controls total concurrent TTS generations across providers. If not specified, it defaults to `4` and is clamped to a minimum of `1`.

#### Error Streaming Policy

Set `performance.stream_errors_as_audio` to control failure behavior during streaming:
- `false` (default and recommended for production): raise provider/service errors instead. The API endpoint maps these to appropriate HTTP status codes (e.g., 400/402/429/5xx).
- `true` (compatibility mode): embed `ERROR: ...` text chunks in the audio stream and return HTTP 200. Suitable only for clients/tests that explicitly expect bytes regardless of outcome.

### Voice Cloning Requirements

| Provider | Min Duration | Max Duration | Format | Sample Rate |
|----------|-------------|--------------|--------|-------------|
| Higgs | 3s | 10s | WAV/MP3/FLAC | 24kHz |
| Chatterbox | 5s | 20s | WAV/MP3 | 24kHz+ |
| VibeVoice | 3s | 30s | WAV/MP3 | 22.05kHz |

## Monitoring

### Health Check
```bash
curl http://localhost:8000/api/v1/audio/health
```

Response:
```json
{
  "status": "healthy",
  "providers": {
    "total": 7,
    "available": 3,
    "details": {
      "openai": "available",
      "kokoro": "available",
      "higgs": "not_initialized"
    }
  },
  "circuit_breakers": {
    "openai": "closed",
    "kokoro": "closed"
  }
}
```

### List Providers
```bash
curl http://localhost:8000/api/v1/audio/providers
```

### Metrics
The service integrates with the application's metrics system:
- Request counts per provider
- Response times (p50, p95, p99)
- Error rates by category
- Active request gauges
- Audio generation sizes

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Provider not available" | Check API keys and model files |
| "Voice reference validation failed" | Ensure audio meets duration/format requirements |
| "Circuit breaker open" | Provider temporarily disabled due to failures, will auto-recover |
| "Model not found" | Download required model files (see setup guide) |
| "Out of memory" | Reduce batch size or use smaller model variant |
| "LuxTTS module not found" | Set `module_path` (or `LUX_TTS_MODULE_PATH`) to your LuxTTS checkout |

Setup guides:
- General TTS setup: `Docs/STT-TTS/TTS-SETUP-GUIDE.md`
- LuxTTS quickstart: `Docs/STT-TTS/LUXTTS_TTS_SETUP.md`

### Debug Mode

Enable detailed logging:
```yaml
# In tts_providers_config.yaml
logging:
  level: DEBUG
  include_metrics: true
```

### Voice Cloning Issues

1. **Audio too short/long**: Check provider-specific duration requirements
2. **Poor quality clone**: Ensure clean audio, single speaker, no background noise
3. **Format not supported**: Convert to WAV 24kHz mono
4. **Memory error**: Voice cloning requires more VRAM, try CPU mode

## Development

### Adding a New Provider

1. Create adapter class in `adapters/`
```python
class MyProviderAdapter(TTSAdapter):
    async def initialize(self) -> bool:
        # Load models/setup API

    async def generate(self, request: TTSRequest) -> TTSResponse:
        # Generate audio

    async def get_capabilities(self) -> TTSCapabilities:
        # Return provider capabilities
```

2. Register in `adapter_registry.py`
```python
TTSProvider.MY_PROVIDER = "my_provider"
DEFAULT_ADAPTERS[TTSProvider.MY_PROVIDER] = MyProviderAdapter
```

3. Add configuration to YAML
```yaml
providers:
  my_provider:
    enabled: true
    # provider-specific settings
```

### Testing

```bash
# Run TTS tests
pytest tests/TTS/ -v

# Test specific provider
pytest tests/TTS/test_adapters.py::test_openai_adapter -v

# Test with coverage
pytest tests/TTS/ --cov=tldw_Server_API.app.core.TTS
```

## Security Considerations

### Voice Cloning Ethics
- Only clone voices with explicit consent
- Add watermarking when available (Chatterbox)
- Implement usage logging for audit trails
- Consider rate limiting voice cloning requests

### API Security
- Always use authentication in production
- Validate and sanitize all inputs
- Limit file upload sizes
- Use HTTPS for API endpoints
- Rotate API keys regularly

## Performance Optimization

### For API Providers
- Use connection pooling
- Implement response caching
- Batch requests when possible

### For Local Models
- Use GPU acceleration (CUDA)
- Enable mixed precision (FP16/BF16)
- Pre-load models at startup
- Use ONNX runtime for CPU inference

### Voice Cloning
- Cache processed voice references
- Limit reference audio duration
- Use efficient audio formats (WAV)
- Consider CPU/GPU memory limits

## License

The TTS module follows the main project's license:
- GNU General Public License v2.0

Individual model licenses:
- Kokoro: Apache 2.0
- Higgs: Custom research license
- Chatterbox: MIT
- VibeVoice: Microsoft research license

## Support

For issues or questions:
1. Check the [troubleshooting guide](#troubleshooting)
2. Review [API documentation](http://localhost:8000/docs)
3. Check logs in `logs/tldw_server.log`
4. Report issues with full error messages

---

*Last Updated: 2025-08-31*
*Version: 2.0.0*
