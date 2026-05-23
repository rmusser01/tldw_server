# TTS Provider Setup Guide

This guide explains how to set up each TTS provider, especially the local models like Higgs, Kokoro, and VibeVoice.

## Table of Contents
- [Commercial Providers](#commercial-providers)
- [Local Model Providers](#local-model-providers)
- [Voice Cloning Setup](#voice-cloning-setup)
- [Setup Verification](#setup-verification)

## Commercial Providers

### OpenAI
```bash
# Add to config.txt or environment
OPENAI_API_KEY=your-api-key-here
```

### ElevenLabs
```bash
# Add to config.txt or environment
ELEVENLABS_API_KEY=your-api-key-here
```

## Local Model Providers

### One-Command Installers (Recommended)
Use these helpers from the repo root to install a specific backend in isolation:

```bash
# Kokoro (v1.0 ONNX + voices)
python Helper_Scripts/TTS_Installers/install_tts_kokoro.py

# Dia / Higgs / VibeVoice
python Helper_Scripts/TTS_Installers/install_tts_dia.py
python Helper_Scripts/TTS_Installers/install_tts_higgs.py
python Helper_Scripts/TTS_Installers/install_tts_vibevoice.py --variant 1.5B

# OmniVoice sidecar runtime
python Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py \
  --model-path models/omnivoice_sidecar/models/OmniVoice

# NeuTTS (deps; optional prefetch)
python Helper_Scripts/TTS_Installers/install_tts_neutts.py --prefetch

# IndexTTS2 (deps + checkpoints folder scaffold)
python Helper_Scripts/TTS_Installers/install_tts_index_tts2.py

# Chatterbox (deps only)
python Helper_Scripts/TTS_Installers/install_tts_chatterbox.py [--with-lang]
```

Flags:
- `TLDW_SETUP_SKIP_PIP=1` to skip pip installs
- `TLDW_SETUP_SKIP_DOWNLOADS=1` to skip HF downloads

### OmniVoice Setup

OmniVoice is optional, disabled by default, and runs in a dedicated sidecar runtime rather than the main server interpreter. Install the PyTorch/backend stack for your target hardware before expecting real synthesis to work; the helper installs the sidecar package/runtime wiring but does not download model assets during requests.

Preferred install path:

```bash
python Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py \
  --model-path /absolute/path/to/local/OmniVoice/model
```

Model path examples:

- Hugging Face cache snapshot: `~/.cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots/<commit-sha>`
- Repo-local operator-managed path: `models/omnivoice_sidecar/models/OmniVoice`
- Any absolute directory containing the local OmniVoice model files

What the installer provisions:

- `models/omnivoice_sidecar/.venv`
- `models/omnivoice_sidecar/runtime`
- `models/omnivoice_sidecar/logs`
- an updated `providers.omnivoice` block in `tldw_Server_API/Config_Files/tts_providers_config.yaml` with the explicit local `model_path`

Source checkout behavior:

- If `../OmniVoice` exists relative to the repo root, the installer prefers that checkout.
- Otherwise it uses `external/OmniVoice` and can clone the upstream repo there.

Runtime notes:

- The sidecar supervisor reads the configured OmniVoice interpreter path from provider config, so the dedicated `.venv` created by the installer is the runtime that gets launched.
- If you configure OmniVoice manually, set `extra_params.python_path` to an interpreter that can import `omnivoice`, `torch`, `torchaudio`, `soundfile`, `fastapi`, `uvicorn`, `httpx`, `pydantic`, and `loguru`.
- Runtime synthesis fails closed if the configured model directory is missing; it does not fetch OmniVoice model assets on demand.
- Public requests that target OmniVoice and omit `voice` normalize to `auto`.
- Explicit voices still win over the provider default.
- Supported request modes are automatic voice selection, voice design via `extra_params.instruct`, and cloning via direct `voice_reference` or stored `custom:<voice_id>` plus `extra_params.reference_text`.
- The sidecar returns native 24 kHz WAV/PCM; public `response_format` conversion and any resampling happen in the main tldw adapter.

Minimum manual provider config:

```yaml
providers:
  omnivoice:
    enabled: true
    runtime: "sidecar"
    model: "omnivoice"
    sample_rate: 24000
    max_concurrent_generations: 1
    extra_params:
      model_path: "/absolute/path/to/local/OmniVoice/model"
      python_path: "models/omnivoice_sidecar/.venv/bin/python"
      runtime_path: "models/omnivoice_sidecar/runtime"
      scratch_dir: "models/omnivoice_sidecar/runtime/scratch"
      host: "127.0.0.1"
      port: 8039
      autoselect_port: true
      idle_shutdown_seconds: 900
```

Verify managed sidecar synthesis:

```bash
python Helper_Scripts/TTS_Installers/smoke_test_omnivoice_sidecar.py \
  --model-path /absolute/path/to/local/OmniVoice/model \
  --sidecar-python models/omnivoice_sidecar/.venv/bin/python \
  --output /tmp/omnivoice-sidecar-smoke.wav \
  --num-step 8 \
  --speed 1.0
```

Successful output reports the written WAV path, byte count, 24 kHz sample rate, mono channel count, frame count, duration, RMS, and peak. The helper fails nonzero if the sidecar cannot start, the model cannot load, the output is not parseable WAV, the WAV shape is wrong, or the audio is silent.

The opt-in pytest smoke tests use `OmniVoiceRuntime` directly in the active test interpreter:

```bash
TLDW_TEST_OMNIVOICE_REAL=1 \
TLDW_OMNIVOICE_MODEL_PATH=/absolute/path/to/local/OmniVoice/model \
python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_omnivoice_real_runtime.py -q
```

Use those tests only when the active project environment itself has the OmniVoice inference dependencies installed. To verify the intended managed sidecar deployment path, use `smoke_test_omnivoice_sidecar.py` with `--sidecar-python`.

Common OmniVoice failures:

- `ModuleNotFoundError: torchaudio` or `RUNTIME_IMPORT_FAILED`: install OmniVoice and its inference dependencies into the sidecar interpreter passed with `--sidecar-python`.
- `MODEL_NOT_AVAILABLE` or "model path is not a directory": pass the resolved local model snapshot directory, not the Hugging Face model id.
- `PermissionError` while checking or binding `127.0.0.1`: run the smoke helper outside restricted sandboxes that block loopback sockets.
- HTTP timeout during synthesis: rerun with a larger `--timeout`; first model load can be much slower than later requests.

Example request:

```json
{
  "model": "omnivoice",
  "voice": "auto",
  "input": "A short test sentence.",
  "response_format": "wav",
  "stream": false,
  "extra_params": {
    "instruct": "A calm documentary narrator",
    "language_id": "en",
    "num_step": 8
  }
}
```

### Model Auto-Download Controls

Local providers (Kokoro, Higgs, Dia, Chatterbox, VibeVoice) can auto-download models the first time you use them. You can control this behavior globally or per provider.

Supported configuration sources (highest precedence last):
- YAML: `tts_providers_config.yaml` (per-provider `auto_download` flag)
- config.txt: `[TTS-Settings]` section (global and per-provider toggles)
- Environment variables

Defaults: auto-download is enabled unless overridden.

config.txt example (recommended for self-hosted setups):

```
[TTS-Settings]
# Global toggle for all local providers
auto_download_local_models = false

# Provider-specific overrides (optional)
vibevoice_auto_download = false
kokoro_auto_download = false
dia_auto_download = false
higgs_auto_download = false
chatterbox_auto_download = false
```

YAML example (per provider):

```yaml
providers:
  vibevoice:
    enabled: true
    auto_download: false
    model_path: microsoft/VibeVoice-1.5B  # or a local path
  higgs:
    enabled: true
    auto_download: true
    model_path: bosonai/higgs-audio-v2-generation-3B-base
```

Environment variables (override at runtime):
- Global: `TTS_AUTO_DOWNLOAD=0` (or `1`)
- Per provider: `VIBEVOICE_AUTO_DOWNLOAD`, `KOKORO_AUTO_DOWNLOAD`, `DIA_AUTO_DOWNLOAD`, `HIGGS_AUTO_DOWNLOAD`, `CHATTERBOX_AUTO_DOWNLOAD` (accept `0/1`, `true/false`, `yes/no`, `on/off`).

Behavior when disabled:
- VibeVoice: initialization returns unavailable if models are missing (no download).
- Dia: loads with `local_files_only` and fails fast if not cached.
- Chatterbox: runs in HF offline mode and fails if models are not local.
- Higgs: errors if a remote model path is specified while auto-download is disabled.
- Kokoro: does not auto-download; requires local files regardless of this flag.

Tip (CI/Dev): The test suite sets `TTS_AUTO_DOWNLOAD=0` to avoid network during tests.

### Qwen3-TTS Setup

Qwen3-TTS is a runtime-aware multilingual TTS provider with `upstream`, `mlx`, and `remote` execution modes.
Full runbook: `Docs/STT-TTS/QWEN3_TTS_SETUP.md`.

#### Installation
```bash
# Upstream runtime
pip install qwen-tts torch soundfile

# Apple Silicon MLX runtime
pip install mlx mlx-audio
```

If the package name differs for your environment, install from the upstream repo instead.

#### Configuration (YAML)
Enable Qwen3-TTS in `tldw_Server_API/Config_Files/tts_providers_config.yaml`:

```yaml
providers:
  qwen3_tts:
    enabled: true
    runtime: "auto"  # auto | upstream | mlx | remote
    model: "auto"  # or an explicit model id
    device: "cuda" # cpu | cuda | mps
    dtype: "float16"
    auto_download: false
    max_text_length: 5000
```

#### Usage Examples

CustomVoice (speaker + optional instruction):
```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "input": "Hello from Qwen3-TTS.",
  "voice": "Vivian",
  "response_format": "mp3",
  "stream": true,
  "extra_params": {
    "instruct": "Warm and calm delivery."
  }
}
```

VoiceDesign (instruction required):
```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
  "input": "Design a new voice sample.",
  "response_format": "wav",
  "stream": false,
  "extra_params": {
    "instruct": "A soft, narrative voice with light rasp."
  }
}
```

Base voice clone (reference audio + optional reference text):
```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
  "input": "Cloned voice output.",
  "response_format": "mp3",
  "stream": true,
  "voice_reference": "<base64 audio>",
  "extra_params": {
    "reference_text": "Transcript of the reference clip."
  }
}
```
Notes:
- `voice_reference` is always required for Base models.
- `extra_params.x_vector_only_mode=true` allows omitting `reference_text` (quality may degrade).
- `reference_duration_min` (seconds) can be provided to enforce a minimum reference clip duration.
- Base models enforce a default 3s minimum reference duration when `reference_duration_min` is omitted.
- `runtime=mlx` supports preset-speaker CustomVoice only in v1 and rejects Base/VoiceDesign/uploaded custom voices.

#### Tokenizer API Endpoints

Encode audio to tokens:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/tokenizer/encode" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@/path/to/audio.wav" \
  -F "tokenizer_model=Qwen/Qwen3-TTS-Tokenizer-12Hz"
```

Decode tokens to audio:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/tokenizer/decode" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"tokens":[1,2,3], "tokenizer_model":"Qwen/Qwen3-TTS-Tokenizer-12Hz", "response_format":"wav"}'
```

Note: tokenizer endpoints require the `audio.tokenizer` scope.

### LuxTTS Setup

LuxTTS is a ZipVoice-based 48kHz voice‑cloning TTS model. Full runbook:
`Docs/STT-TTS/LUXTTS_TTS_SETUP.md`.

### Kokoro Setup

Kokoro is a lightweight, high-quality TTS model that runs locally using ONNX Runtime or PyTorch. We recommend the v1.0 ONNX artifacts for most users.

#### Installation
Preferred:
```bash
python Helper_Scripts/TTS_Installers/install_tts_kokoro.py
```
Manual alternative:
```bash
pip install onnxruntime kokoro-onnx phonemizer espeak-phonemizer
# Optional GPU: pip install onnxruntime-gpu
# Install eSpeak NG: brew install espeak-ng  |  sudo apt-get install -y espeak-ng
# Env var only if needed: export PHONEMIZER_ESPEAK_LIBRARY=/path/to/libespeak-ng
```

#### Download Models (v1.0 ONNX)
```bash
# Create model directory
mkdir -p models/kokoro

# Use hf to fetch the model and voices
pip install -U "huggingface_hub"
hf download onnx-community/Kokoro-82M-v1.0-ONNX-timestamped onnx/model.onnx --local-dir models/kokoro/
hf download onnx-community/Kokoro-82M-v1.0-ONNX-timestamped voices          --local-dir models/kokoro/

# Optional: choose an alternate ONNX (fp16/quantized) by replacing onnx/model.onnx
# e.g., onnx/model_fp16.onnx or onnx/model_quantized.onnx
```

#### Configuration
```yaml
# In tts_providers_config.yaml
kokoro:
  enabled: true
  use_onnx: true
  model_path: ./models/kokoro/onnx/model.onnx
  voices_json: ./models/kokoro/voices   # path to voices directory for v1.0 ONNX
  device: cpu  # or cuda for GPU (onnxruntime-gpu)
```

#### PyTorch Variant (optional)
```bash
# Download from hexgrad/Kokoro-82M
hf download hexgrad/Kokoro-82M kokoro-v1_0.pth --local-dir models/kokoro/
hf download hexgrad/Kokoro-82M config.json     --local-dir models/kokoro/
hf download hexgrad/Kokoro-82M voices          --local-dir models/kokoro/

# YAML
kokoro:
  enabled: true
  use_onnx: false
  model_path: ./models/kokoro/kokoro-v1_0.pth
  voice_dir:  ./models/kokoro/voices
  device: cuda  # or mps/cpu
```

#### System Requirements
- **Disk Space**: ~300–330MB for `model.onnx`, plus voices directory
- **RAM**: 2GB minimum
- **eSpeak NG**: install system package; env var only for non-standard library paths

### PocketTTS ONNX Setup

PocketTTS ONNX provides lightweight, streaming-capable voice cloning with short reference audio samples.

#### Installation
```bash
# From repo root (runtime deps only)
pip install -e '.[TTS_pocket_tts]'
# Optional GPU: pip install onnxruntime-gpu
```
Note: PocketTTS ONNX is not published on PyPI. The runtime module and weights are downloaded separately.

#### Download Models (Scripted)
```bash
python Helper_Scripts/TTS_Installers/install_tts_pocket_tts_onnx.py --output-dir models/pocket_tts_onnx
```
The installer updates `tts_providers_config.yaml` by default and checks that `pocket_tts_onnx` can be imported. Use `--no-config-update` to skip, `--config-path` for a custom file, or `--no-import-check` to skip the import sanity check.

#### Download Models (Manual)
```bash
pip install -U "huggingface_hub"
hf download KevinAHM/pocket-tts-onnx onnx --local-dir models/pocket_tts_onnx
hf download KevinAHM/pocket-tts-onnx tokenizer.model --local-dir models/pocket_tts_onnx
# If present in the repo, also fetch the Python module:
hf download KevinAHM/pocket-tts-onnx pocket_tts_onnx --local-dir models/pocket_tts_onnx
hf download KevinAHM/pocket-tts-onnx pocket_tts_onnx.py --local-dir models/pocket_tts_onnx
```
If you store the module elsewhere, set `module_path` to that directory in the config.

#### Configuration
```yaml
# In tts_providers_config.yaml
pocket_tts:
  enabled: true
  model_path: ./models/pocket_tts_onnx/onnx
  tokenizer_path: ./models/pocket_tts_onnx/tokenizer.model
  module_path: ./models/pocket_tts_onnx
  precision: "int8"  # or "fp32"
  device: "auto"     # "cpu" | "cuda"
```

### Higgs Audio V2 Setup

Higgs is a powerful 3B parameter model supporting 50+ languages, music generation, and voice cloning.

#### Installation
```bash
# Install dependencies
pip install transformers torch torchaudio accelerate

# For optimized inference
pip install flash-attn --no-build-isolation  # Requires CUDA
```

#### Download Models
```bash
# Method 1: Automatic download (first run)
# The model will auto-download on first use (~3GB)

# Method 2: Pre-download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bosonai/higgs-audio-v2-generation-3B-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Save locally
model.save_pretrained("./models/higgs")
tokenizer.save_pretrained("./models/higgs")
```

#### Configuration
```yaml
# In tts_providers_config.yaml
higgs:
  enabled: true
  model_path: bosonai/higgs-audio-v2-generation-3B-base  # or ./models/higgs for local
  tokenizer_path: bosonai/higgs-audio-v2-tokenizer
  device: cuda  # Strongly recommended for 3B model
  use_fp16: true  # Reduces memory usage
  batch_size: 1
  # Voice cloning settings
  enable_voice_clone: true
  voice_clone_min_duration: 3.0  # seconds
  voice_clone_max_duration: 10.0  # seconds
```

#### System Requirements
- **Disk Space**: ~6GB for model
- **RAM**: 8GB minimum
- **VRAM**: 6GB+ for GPU inference (recommended)
- **CPU**: Can run on CPU but very slow
- **Voice Cloning**: Supports 3-10 second audio samples at 24kHz

### Chatterbox Setup

Chatterbox features unique emotion exaggeration control and voice cloning from Resemble AI.

#### Installation
```bash
# Install Chatterbox (when available)
pip install chatterbox-tts

# Or from source
git clone https://github.com/resemble-ai/chatterbox
cd chatterbox
pip install -e .
```

#### Download Models
```bash
# Download model
mkdir -p models/chatterbox
hf download resemble-ai/chatterbox --local-dir models/chatterbox/
```

#### Configuration
```yaml
# In tts_providers_config.yaml
chatterbox:
  enabled: true
  model_path: ./models/chatterbox  # or resemble-ai/chatterbox
  device: cuda
  use_fp16: true
  enable_watermark: true  # Perth watermarking
  target_latency_ms: 200
  # Voice cloning settings
  enable_voice_clone: true
  voice_clone_min_duration: 5.0  # seconds
  voice_clone_max_duration: 20.0  # seconds
  voice_clone_sample_rate: 24000  # Hz
```

#### System Requirements
- **Disk Space**: ~3GB for model
- **RAM**: 6GB minimum
- **VRAM**: 4GB+ for GPU inference
- **Latency**: Sub-200ms on good GPU
- **Voice Cloning**: Supports 5-20 second audio samples, single speaker

### Dia Setup

Dia specializes in multi-speaker dialogue with nonverbal cues.

#### Installation
```bash
# Install dependencies
pip install transformers torch accelerate

# For dialogue processing
pip install nltk spacy
python -m spacy download en_core_web_sm
```

#### Download Models
```bash
# Download Dia model
mkdir -p models/dia
hf download nari-labs/dia --local-dir models/dia/

# Or auto-download on first use
```

#### Configuration
```yaml
# In tts_providers_config.yaml
dia:
  enabled: true
  model_path: nari-labs/dia  # or ./models/dia for local
  device: cuda
  use_safetensors: true
  use_bf16: true  # Better than fp16 for this model
  auto_detect_speakers: true
  max_speakers: 5
```

#### System Requirements
- **Disk Space**: ~3.2GB for model
- **RAM**: 6GB minimum
- **VRAM**: 4GB+ for GPU inference
- **Best for**: Dialogue, conversations, storytelling

### VibeVoice Setup (Community Reference)

VibeVoice generates expressive, long-form, multi-speaker conversational audio with spontaneous background music and voice cloning support.

#### Installation
```bash
# Recommended: Use NVIDIA Deep Learning Container
sudo docker run --privileged --gpus all --rm -it nvcr.io/nvidia/pytorch:24.07-py3

# Install VibeVoice from GitHub (community reference)
git clone https://github.com/vibevoice-community/VibeVoice.git
cd VibeVoice/
pip install -e .
```

#### Models Available
- **VibeVoice-1.5B**: 64K context (~90 min generation)
- **VibeVoice-7B-Preview**: 32K context (~45 min generation)

Models will auto-download from HuggingFace on first use.

#### Test Installation
```bash
# Run Gradio demo for 1.5B model
python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B --share

# Run Gradio demo for 7B model (official)
python demo/gradio_demo.py --model_path vibevoice/VibeVoice-7B --share

# File-based inference (single speaker)
python demo/inference_from_file.py \
  --model_path microsoft/VibeVoice-1.5B \
  --txt_path demo/text_examples/1p_abs.txt \
  --speaker_names Alice

# File-based inference (multiple speakers)
python demo/inference_from_file.py \
  --model_path microsoft/VibeVoice-1.5B \
  --txt_path demo/text_examples/2p_music.txt \
  --speaker_names Alice Frank
```

#### Adapter: Speaker Mapping via Config

You can define a default mapping between speakers in the script and voice samples so callers don’t need to pass it on each request. The adapter reads `vibevoice_speakers_to_voices` from its provider config.

- INI (config.txt) example (store JSON as a string):

```ini
[TTS-Settings]
vibevoice_speakers_to_voices = {"1": "en-Alice_woman", "2": "/abs/path/to/frank.wav"}
```

- YAML (tts_providers_config.yaml) example:

```yaml
providers:
  vibevoice:
    enabled: true
    model_path: vibevoice/VibeVoice-1.5B
    speakers_to_voices:
      "1": en-Alice_woman
      "2": /abs/path/to/frank.wav
```

At runtime, a request can still override the defaults by passing `extra_params["speakers_to_voices"]`.

#### Configuration
```yaml
# In tts_providers_config.yaml
vibevoice:
  enabled: true
  auto_download: true
  vibevoice_variant: "1.5B"  # or "7B", "7B-Q8"
  model_path: microsoft/VibeVoice-1.5B  # or vibevoice/VibeVoice-7B (official), FabioSarracino/VibeVoice-Large-Q8 (7B-Q8)
  device: cuda  # GPU strongly recommended
  use_fp16: true
  enable_music: true  # Spontaneous background music
  max_speakers: 4
  # Voice cloning settings
  enable_voice_clone: true
  voice_clone_min_duration: 3.0  # seconds
  voice_clone_max_duration: 30.0  # seconds
  voice_clone_sample_rate: 22050  # Hz
```

#### System Requirements
- **Disk Space**: ~3GB (1.5B) or ~14GB (7B)
- **RAM**: 8GB minimum (1.5B), 16GB (7B)
- **VRAM**: 4GB+ (1.5B), 16GB+ (7B)
- **Features**:
  - Long-form generation (up to 90 min)
  - Multi-speaker (up to 4 distinct voices)
  - Spontaneous background music
  - Emergent singing capability
  - Cross-lingual transfer
  - Voice cloning with any duration audio (3-30s recommended)

## Voice Cloning Setup

Voice cloning allows you to synthesize speech using a reference voice from an audio sample. Providers include PocketTTS, Higgs, Chatterbox, and VibeVoice.

### Preparing Voice Reference Audio

#### Audio Requirements by Provider

| Provider | Min Duration | Max Duration | Sample Rate | Format | Quality Requirements |
|----------|-------------|--------------|-------------|---------|---------------------|
| **PocketTTS** | 1 second | 60 seconds | 24kHz | WAV/MP3/FLAC/OGG | Clear speech, single speaker |
| **Higgs** | 3 seconds | 10 seconds | 24kHz | WAV/MP3/FLAC | Clear speech, single speaker |
| **Chatterbox** | 5 seconds | 20 seconds | 24kHz | WAV/MP3 | No background noise/music |
| **VibeVoice** | 3 seconds | 30 seconds | 22.05kHz | WAV/MP3 | Can handle some background |

#### Preparing Audio Files

1. **Record or Select Clean Audio**:
   - Single speaker only
   - Clear speech without music
   - Minimal background noise
   - Natural speaking pace

2. **Convert Audio Format** (if needed):
```bash
# Convert to WAV with proper sample rate for Higgs/Chatterbox
ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav

# Convert for VibeVoice
ffmpeg -i input.mp3 -ar 22050 -ac 1 output.wav

# Trim audio to specific duration
ffmpeg -i input.wav -ss 0 -t 10 -ar 24000 output_10s.wav
```

### Error Semantics and Client Responsibilities

The `/api/v1/audio/speech` endpoint returns **structured HTTP errors by default**:

- Validation errors → HTTP 400 with a JSON `{"detail": "..."}`
- Model/provider configuration issues → HTTP 4xx/5xx with JSON `{"detail": "..."}`
- Provider/auth failures (e.g., invalid API key) → HTTP 5xx with JSON error detail

Clients **must not** assume a 200 response or treat all responses as audio bytes:

- Always check the HTTP status code (e.g., `response.raise_for_status()` in Python).
- Only treat the body as audio when the status is 200.
- On non-200 responses, parse the JSON body and surface the `detail` field to users/logging.

An opt-in legacy mode exists (`performance.stream_errors_as_audio: true` or `TTS_STREAM_ERRORS_AS_AUDIO=1`)
that embeds `ERROR: ...` bytes in the stream, but this mode is not recommended for production APIs.

### Text Sanitization and Strict Validation

Incoming TTS text is passed through a sanitizer (`TTSInputValidator`) that:

- Normalizes Unicode and removes HTML tags/entities.
- Strips or rejects potentially dangerous patterns (e.g., obvious SQL/command injections such as `whoami`, `curl evil`, `rm -rf /`, `../../etc/passwd`).
- Enforces provider-aware text length limits and basic repetition checks.

By default, **strict validation is enabled**:

- `strict_validation: true` in `tts_providers_config.yaml` (or omitted, since the default is true), or
- `TTS_STRICT_VALIDATION=1`

In strict mode, dangerous patterns cause a 400 error rather than being silently stripped. This is recommended for multi-tenant or untrusted deployments.

For trusted/local deployments, you can relax this behavior by setting:

```yaml
# tldw_Server_API/Config_Files/tts_providers_config.yaml
strict_validation: false
```

or via environment:

```bash
export TTS_STRICT_VALIDATION=0
```

In non-strict mode, dangerous substrings are removed, but the request is still processed. Clients should be aware that meta-text like “the `whoami` command” may be altered or rejected depending on the chosen validation mode.

3. **Validate Audio Quality**:
```python
import librosa
import numpy as np

# Load and check audio
audio, sr = librosa.load("voice_sample.wav", sr=None)
duration = len(audio) / sr

print(f"Duration: {duration:.2f} seconds")
print(f"Sample rate: {sr} Hz")
print(f"RMS energy: {np.sqrt(np.mean(audio**2)):.4f}")

# Check if too quiet or too loud
if np.max(np.abs(audio)) < 0.1:
    print("Warning: Audio may be too quiet")
elif np.max(np.abs(audio)) > 0.95:
    print("Warning: Audio may be clipping")
```

### Using Voice Cloning via API

#### Basic Voice Cloning Request

```python
import base64
import requests

# Prepare voice reference
with open("voice_sample.wav", "rb") as f:
    voice_data = base64.b64encode(f.read()).decode()

# Make TTS request with voice cloning
response = requests.post(
    "http://localhost:8000/api/v1/audio/speech",
    headers={"Authorization": "Bearer your-token"},
    json={
        "model": "higgs",  # or "chatterbox", "vibevoice"
        "input": "This text will be spoken in the cloned voice.",
        "voice": "clone",  # Use "clone" to indicate voice cloning
        "voice_reference": voice_data,  # Base64-encoded audio
        "response_format": "mp3"
    }
)

# Check for HTTP errors before treating the body as audio
response.raise_for_status()

# Save the generated audio on success
with open("cloned_output.mp3", "wb") as f:
    f.write(response.content)
```

#### Advanced Voice Cloning with Parameters

```python
# Chatterbox with emotion control
response = requests.post(
    "http://localhost:8000/api/v1/audio/speech",
    json={
        "model": "chatterbox",
        "input": "I'm so excited about this feature!",
        "voice": "clone",
        "voice_reference": voice_data,
        "extra_params": {
            "emotion": "excited",
            "emotion_intensity": 1.5,
            "enable_watermark": True  # Add Perth watermark
        }
    }
)
response.raise_for_status()

# VibeVoice with vibe control
response = requests.post(
    "http://localhost:8000/api/v1/audio/speech",
    json={
        "model": "vibevoice",
        "input": "This is a professional presentation.",
        "voice": "clone",
        "voice_reference": voice_data,
        "extra_params": {
            "vibe": "professional",
            "vibe_intensity": 1.2,
            "enable_music": False  # Disable background music
        }
    }
)
response.raise_for_status()
```

### Voice Cloning via cURL

```bash
# Encode audio file to base64
base64 voice_sample.wav > voice_base64.txt

# Create JSON payload
cat > request.json <<EOF
{
  "model": "higgs",
  "input": "Hello, this is a voice cloning test.",
  "voice": "clone",
  "voice_reference": "$(cat voice_base64.txt)",
  "response_format": "mp3"
}
EOF

# Send request
curl -X POST http://localhost:8000/api/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d @request.json \
  --output cloned.mp3

# Important: check the HTTP status code. On non-2xx, the response body will be
# a JSON error (not audio), so handle that in scripts by inspecting the status
# and printing/parsing the body instead of assuming audio.
```

### Voice Cloning Best Practices

1. **Audio Quality**:
   - Use lossless formats when possible (WAV, FLAC)
   - Record in a quiet environment
   - Use a good microphone if recording
   - Normalize audio levels

2. **Speaker Consistency**:
   - Use consistent speaking style in reference
   - Avoid emotional extremes in reference
   - Match the tone you want in output

3. **Performance Optimization**:
   - Cache processed voice references
   - Pre-process audio to correct format
   - Use appropriate model for use case

4. **Ethical Considerations**:
   - Only clone voices with explicit consent
   - Use watermarking when available (Chatterbox)
   - Document voice sources
   - Implement usage logging

### Troubleshooting Voice Cloning

#### Common Issues and Solutions

1. **"Voice reference validation failed"**:
   - Check audio duration (must be within provider limits)
   - Verify audio format and sample rate
   - Ensure single speaker in audio
   - Check for silence or corruption

2. **"Poor voice quality in output"**:
   - Improve reference audio quality
   - Use longer reference (up to max duration)
   - Ensure clear speech in reference
   - Try different provider

3. **"Voice doesn't match reference"**:
   - Some providers better for certain voice types
   - Higgs: Best for multilingual
   - Chatterbox: Best for emotional expression
   - VibeVoice: Best for natural conversation

4. **"Memory error during cloning"**:
   - Reduce batch size in config
   - Enable FP16/BF16 in provider config
   - Use CPU offloading if available
   - Try smaller model variant

### Voice Cloning Configuration

Add to `tts_providers_config.yaml`:

```yaml
voice_cloning:
  # Global settings
  enabled: true
  max_reference_size_mb: 10
  cache_processed_references: true
  cache_ttl_hours: 24

  # Processing settings
  auto_normalize: true
  remove_silence: true
  denoise: false

  # Security settings
  require_consent: true
  log_usage: true
  watermark_when_available: true

# Provider-specific overrides
providers:
  higgs:
    voice_clone_settings:
      min_duration: 3.0
      max_duration: 10.0
      preferred_format: "wav"
      sample_rate: 24000

  chatterbox:
    voice_clone_settings:
      min_duration: 5.0
      max_duration: 20.0
      enable_perth_watermark: true

  vibevoice:
    voice_clone_settings:
      min_duration: 3.0
      max_duration: 30.0
      sample_rate: 22050
      enable_speaker_embeddings: true
```

## Setup Verification

### Test Installation

```python
# test_tts_setup.py
import asyncio
from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2

async def test_providers():
    service = await get_tts_service_v2()

    # Check available providers
    status = service.get_status()
    print(f"Available providers: {status['available']}/{status['total_providers']}")

    # List capabilities
    caps = await service.get_capabilities()
    for provider, cap in caps.items():
        print(f"{provider}: {cap['status']}")
        if cap['status'] == 'available':
            print(f"  - Languages: {cap['languages']}")
            print(f"  - Formats: {cap['formats']}")

# Run test
asyncio.run(test_providers())
```

### Quick Test for Each Provider

```bash
# Test Kokoro
curl -X POST http://localhost:8000/api/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello from Kokoro local TTS",
    "voice": "af_bella"
  }' --output kokoro_test.mp3

# Test Higgs
curl -X POST http://localhost:8000/api/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs",
    "input": "Hello from Higgs multilingual TTS",
    "voice": "narrator"
  }' --output higgs_test.mp3

# Test Chatterbox with emotion
curl -X POST http://localhost:8000/api/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatterbox",
    "input": "I am so excited to test Chatterbox!",
    "voice": "energetic",
    "extra_params": {
      "emotion": "excited",
      "emotion_intensity": 1.5
    }
  }' --output chatterbox_test.mp3

# Test Dia with dialogue
curl -X POST http://localhost:8000/api/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dia",
    "input": "Alice: Hello Bob! Bob: Hi Alice, how are you? Alice: Great, thanks!",
    "voice": "auto"
  }' --output dia_test.mp3

# Test VibeVoice with vibe control
curl -X POST http://localhost:8000/api/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vibevoice",
    "input": "This is a professional announcement with VibeVoice.",
    "voice": "aurora",
    "extra_params": {
      "vibe": "professional",
      "vibe_intensity": 1.2
    }
  }' --output vibevoice_test.mp3
```

## Performance Optimization

### GPU Acceleration

For best performance with local models:

1. **Install CUDA** (if using NVIDIA GPU):
```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. **Use Mixed Precision**:
```yaml
# Enable in config
use_fp16: true  # or use_bf16 for newer GPUs
```

3. **Batch Processing**:
```yaml
# For multiple concurrent requests
batch_size: 4  # Adjust based on VRAM
```

### CPU Optimization

For CPU-only systems:

1. **Use ONNX models** (like Kokoro) when possible
2. **Enable multi-threading**:
```bash
export OMP_NUM_THREADS=4  # Adjust to CPU cores
```
3. **Use INT8 quantization** (if supported):
```yaml
use_int8: true  # Reduces model size and speeds up CPU inference
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce batch size
   - Enable FP16/BF16
   - Use CPU offloading for large models
   - Close other applications

2. **Slow Generation**:
   - Ensure GPU is being used (`nvidia-smi` should show activity)
   - Check if model is using correct device in logs
   - Consider using smaller models or ONNX versions

3. **Model Download Fails**:
   - Check internet connection
   - Verify HuggingFace token if needed
   - Try manual download with wget/curl
   - Check disk space

4. **Audio Quality Issues**:
   - Verify sample rate matches model output
   - Check audio format compatibility
   - Ensure proper audio normalization

### Debug Mode

Enable detailed logging:
```yaml
# In tts_providers_config.yaml
logging:
  level: DEBUG
  include_metrics: true
```

### Health Check

```bash
# Check provider health
curl http://localhost:8000/api/v1/audio/health

# List available providers
curl http://localhost:8000/api/v1/audio/providers
```

## Resource Requirements Summary

| Provider | Model Size | Min RAM | Recommended VRAM | Latency (GPU) | Languages |
|----------|-----------|---------|------------------|---------------|-----------|
| Kokoro | 800MB | 2GB | Optional | ~100ms | EN |
| Higgs | 6GB | 8GB | 6GB+ | ~1s | 50+ |
| Chatterbox | 3GB | 6GB | 4GB+ | ~200ms | EN |
| Dia | 3.2GB | 6GB | 4GB+ | ~500ms | EN |
| VibeVoice | 2GB | 4GB | 3GB+ | ~150ms | 12 |

## Best Practices

1. **Start with Kokoro** for testing - it's lightweight and CPU-friendly
2. **Use GPU for Higgs/Chatterbox/Dia** - CPU inference is very slow
3. **Configure fallback chains** - Commercial → Local for reliability
4. **Monitor memory usage** - Local models can be memory-intensive
5. **Pre-download models** - Avoid download delays on first use
6. **Use circuit breakers** - Prevent cascading failures
7. **Enable metrics** - Track performance and errors

---

*For additional help, check the logs in DEBUG mode or open an issue on GitHub.*
