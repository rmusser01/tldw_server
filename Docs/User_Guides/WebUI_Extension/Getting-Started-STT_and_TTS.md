# Getting Started — STT (Speech-to-Text) and TTS (Text-to-Speech)

This guide is now the quick speech verification and reference page, not the primary first-time onboarding guide.
For first-time setup, start with the hardware-specific guides:

- [First-Time Audio Setup: CPU Systems](../../Getting_Started/First_Time_Audio_Setup_CPU.md)
- [First-Time Audio Setup: GPU/Accelerated Systems](../../Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md)

It still covers `/setup`, alternative providers, direct API verification steps, and troubleshooting.

Scope:
- Use this guide for first successful STT + TTS requests end-to-end.
- Use [TTS Providers Getting Started](./TTS_Getting_Started.md) to compare providers and setup paths.
- Use [TTS Provider Setup Guide](./TTS-SETUP-GUIDE.md) for deep provider runbooks and tuning.
- Use [Persona Live Wake Phrase Guide](./Persona_Live_Wake_Phrases.md) for manually armed Persona Garden wake phrase setup and troubleshooting.

See design doc: [`Docs/Design/STT_TTS_Audio_API_Design.md`](../../Design/STT_TTS_Audio_API_Design.md) for architecture details, provider priority/retry behavior, auth mode behavior, storage header semantics, and streaming protocol/error handling.

## TL;DR Choices

- First-time CPU setup: use the CPU guide and start with `parakeet-tdt-0.6b-v3-onnx` for STT plus `supertonic` for TTS. `parakeet-onnx` remains accepted as a legacy alias.
- First-time accelerated setup: use the GPU/accelerated guide and start with `faster-whisper` on NVIDIA or `parakeet-mlx` on Apple Silicon, plus `supertonic` for TTS.
- `/setup` bundles remain optional, but use the hardware-first guides when you want explicit local-first provider selection and current verification steps.
- Use this page when you want a shorter API smoke test, `/setup` notes, or older alternative-provider examples.

## Recommended Setup Path - Use `/setup` Audio Bundles First

For new installs, prefer the guided `/setup` flow over hand-picking providers.
The setup UI now detects local hardware, recommends a curated audio bundle, provisions it, and gives you a verification + readiness report before you start making STT/TTS API calls.

Recommended operator flow:

1. Start the server and open `http://127.0.0.1:8000/setup`
2. Save any required config changes
3. In the audio stage, click `Provision recommended bundle`
4. Complete any guided prerequisites called out in the report
5. Click `Run verification`
6. Use the API examples below only after the readiness report reaches `ready` or `ready_with_warnings`

Current curated bundle matrix (generated from `Helper_Scripts/generate_audio_bundle_docs.py`):

| Bundle ID | Label | Profiles | Offline runtime after provisioning | Offline pack compatibility | Default STT | Default TTS |
| --- | --- | --- | --- | --- | --- | --- |
| `cpu_local` | CPU Local | Light, Balanced, Performance | Yes | v1 manifest import + model portability | faster_whisper [small] | kokoro |
| `apple_silicon_local` | Apple Silicon Local | Light, Balanced, Performance | Yes | v1 manifest import + model portability | faster_whisper [small] | kokoro |
| `nvidia_local` | NVIDIA Local | Light, Balanced, Performance | Yes | v1 manifest import + model portability | faster_whisper [medium] | kokoro |
| `hosted_plus_local_backup` | Hosted With Local Backup | Balanced | No | v1 manifest import only | faster_whisper [small] | kokoro |

Resource profile guidance:
- `Light`: lowest disk and memory footprint; best when you need a conservative local install.
- `Balanced`: the default for most machines and the safest recommendation when hardware signals are incomplete.
- `Performance`: larger local footprint; only pick it when disk headroom and acceleration are clearly available.

Offline pack guidance:
- `Online provisioning` is still the default. It installs Python dependencies, downloads models, and verifies the current machine.
- `Offline pack import` is now available through the setup audio pack import endpoint for manifest + model portability.
- `Offline pack` v1 does not install Python dependencies or OS prerequisites on the target machine. It validates compatibility and registers the imported pack in the readiness report.

## Prerequisites

- Python environment with project installed
  - From repo root: `pip install -e .`
- FFmpeg (required for audio I/O)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install -y ffmpeg`
  - Windows: install from ffmpeg.org and ensure it’s in PATH
- Start the server
  - `python -m uvicorn tldw_Server_API.app.main:app --reload`
  - API: <http://127.0.0.1:8000/docs>
  - WebUI (Next.js): run `apps/tldw-frontend/` or visit <http://127.0.0.1:8000/api/v1/config/quickstart>

Auth quick note
- Single-user mode: server prints an API key on startup; or set `SINGLE_USER_API_KEY`.
- Use header: `X-API-KEY: <your_key>` for all calls (or Bearer JWT in multi-user setups).

---

## Option A — OpenAI TTS (Hosted)

Best for immediate results; no local model setup.

1. Provide API key
- Export `OPENAI_API_KEY` in your shell or add it to `Config_Files/config.txt` (OpenAI section).

1. Verify TTS provider is enabled (optional)
- OpenAI TTS is enabled by default. To confirm or customize, see `tldw_Server_API/app/core/TTS/tts_providers_config.yaml` under `providers.openai`.

1. Test voice catalog

```bash
curl -s http://127.0.0.1:8000/api/v1/audio/voices/catalog \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" | jq
```

1. Generate speech

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/speech \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "tts-1",
        "voice": "alloy",
        "input": "Hello from tldw_server",
        "response_format": "mp3"
      }' \
  --output out.mp3
```

- Play `out.mp3` in your player.

1. (Optional) Return a storage download link in headers

```bash
curl -i -sS -X POST http://127.0.0.1:8000/api/v1/audio/speech \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "tts-1",
        "voice": "alloy",
        "input": "Save this to generated files storage",
        "response_format": "mp3",
        "stream": false,
        "return_download_link": true
      }' \
  --output out_saved.mp3
```

- Look for `X-Download-Path` and `X-Generated-File-Id` in the response headers.
- The download path will look like `/api/v1/storage/files/{id}/download`.
- `return_download_link` only works with `stream: false`.
- When `return_download_link: true`, the response still includes the audio bytes in the body; the headers provide a storage reference for later retrieval.

Troubleshooting
- 401/403: ensure `OPENAI_API_KEY` is set and valid, and you’re passing `X-API-KEY` (single-user) or Bearer token (multi-user).
- 429: OpenAI rate limit; retry after `retry-after` seconds.

---

## Option B — Kokoro TTS (Local, ONNX)

Offline TTS using Kokoro ONNX. Good quality and fast on CPU; optional GPU via ONNX Runtime.

1. Install (one command)

```bash
python Helper_Scripts/TTS_Installers/install_tts_kokoro.py
```

If you prefer manual steps, install dependencies instead:

```bash
# Python packages (CPU)
pip install onnxruntime kokoro-onnx phonemizer espeak-phonemizer huggingface-hub

# Optional: GPU acceleration (replace onnxruntime above)
pip install onnxruntime-gpu

# System package for phonemizer (required):
# macOS (Homebrew):
brew install espeak-ng
# Ubuntu/Debian:
sudo apt-get update && sudo apt-get install -y espeak-ng
# Windows (PowerShell, example):
#  - Install eSpeak NG (from https://github.com/espeak-ng/espeak-ng/releases)
#  - Set PHONEMIZER_ESPEAK_LIBRARY to libespeak-ng.dll path

# eSpeak NG is auto-detected on most systems. Point the phonemizer to the library only if needed
# macOS (adjust if your Homebrew prefix differs)
export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
# Linux example
export PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1
# Windows example (only if auto-detect fails)
# set PHONEMIZER_ESPEAK_LIBRARY=C:\\Program Files\\eSpeak NG\\libespeak-ng.dll
```

1. Download model files (skipped if you use the installer)
- Place files under a `models/` folder at the repo root (example paths below).
- Recommended sources:
  - ONNX: `onnx-community/Kokoro-82M-v1.0-ONNX-timestamped` (contains `onnx/model.onnx` and a `voices/` directory of voice styles)
  - PyTorch (optional): `hexgrad/Kokoro-82M` (contains `kokoro-v1_0.pth`, `config.json`, and `voices/`)

Examples

```bash
# Create a local directory
mkdir -p models/kokoro

# Option A: hf CLI (ONNX v1.0)
pip install -U "huggingface_hub"
hf download onnx-community/Kokoro-82M-v1.0-ONNX-timestamped onnx/model.onnx --local-dir models/kokoro/
hf download onnx-community/Kokoro-82M-v1.0-ONNX-timestamped voices          --local-dir models/kokoro/

# Option B: direct URLs for ONNX (if CLI unavailable)
wget https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped/resolve/main/onnx/model.onnx -O models/kokoro/onnx/model.onnx
# Then download the voices/ directory assets from the same repo (or use the hf CLI above)
```

1. Enable and point config to your files (the installer writes defaults under models/kokoro/)
- Edit `tldw_Server_API/app/core/TTS/tts_providers_config.yaml`:

```yaml
providers:
  kokoro:
    enabled: true
    use_onnx: true
    model_path: "models/kokoro/onnx/model.onnx"
    voices_json: "models/kokoro/voices"   # use voices directory for v1.0 ONNX
    device: "cpu"    # or "cuda" if using onnxruntime-gpu
```

- Optional: move Kokoro earlier in `provider_priority` to prefer it.

1. Restart server and verify

```bash
python -m uvicorn tldw_Server_API.app.main:app --reload
curl -s http://127.0.0.1:8000/api/v1/audio/voices/catalog \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" | jq '.kokoro'
```

1. Generate speech with Kokoro

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/speech \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "kokoro",
        "voice": "af_bella",
        "input": "Testing local Kokoro TTS",
        "response_format": "mp3"
      }' \
  --output kokoro.mp3
```

Troubleshooting
- Missing dependencies
  - kokoro_onnx: `pip install kokoro-onnx`
  - onnxruntime: `pip install onnxruntime` (or `onnxruntime-gpu`)
- phonemizer / espeak-phonemizer: `pip install phonemizer espeak-phonemizer`
- `voices assets not found` or `model not found`: fix `voices` directory or model path in YAML.
- `eSpeak lib not found`: install `espeak-ng` and set `PHONEMIZER_ESPEAK_LIBRARY` to the library path.
- Adapter previously failed and won’t retry: we enable retry by default (`performance.adapter_failure_retry_seconds: 300`). Or restart the server after fixing assets.

Notes
- PyTorch variant (hexgrad/Kokoro-82M): set `use_onnx: false`, set `model_path: models/kokoro/kokoro-v1_0.pth`, ensure `config.json` sits alongside it, and set `voice_dir: models/kokoro/voices`. Requires `torch` and a compatible Kokoro PyTorch package. Set `device` to `cuda` or `mps` if available.

### Kokoro Voice Mixing & Custom Voices

Kokoro supports lightweight “blended” voices and additional custom voice profiles:

- **Blend existing voices in a single request** using a mix pattern:
  - Syntax: `voice_id1(weight1)+voice_id2(weight2)+...`
  - Example (2 parts Bella, 1 part Adam):

    ```bash
    curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/speech \
      -H "X-API-KEY: $SINGLE_USER_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
            "model": "kokoro",
            "voice": "af_bella(2)+am_adam(1)",
            "input": "This is a blended Kokoro voice",
            "response_format": "mp3"
          }' \
      --output kokoro_mix.mp3
    ```

  - You can mix more voices, for example: `af_bella(3)+am_adam(1)+bf_emma(2)`. The numbers are relative weights (2:1 is the same ratio as 4:2).

- **Add custom Kokoro voices** if you have extra voice profiles from upstream Kokoro tools (e.g., Kokoro‑82M WebUI or other pipelines):
  - Place the voice files under the configured voices directory, typically:
    - ONNX: `models/kokoro/voices/*.bin`
    - PyTorch: `models/kokoro/voices/*.pt`
  - Restart the server; the adapter will auto-discover them and expose new voice IDs based on filenames (for example, `my_custom_voice.bin` → `voice: "my_custom_voice"`).
  - You can then mix them exactly like built-in voices, for example: `"my_custom_voice(2)+af_bella(1)"`.

> Tip: The `/api/v1/audio/voices/catalog` endpoint shows all available Kokoro voices, including dynamically discovered ones.

---

## Option C — faster-whisper STT (Local)

Fast, local transcription compatible with the OpenAI `/audio/transcriptions` API.

1. Install dependencies

```bash
pip install faster-whisper
# Optional (GPU): pip install torch --index-url https://download.pytorch.org/whl/cu121
```

- FFmpeg must be installed (see prerequisites).

1. Transcribe an audio file

```bash
# Replace sample.wav with your file
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/transcriptions \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Accept: application/json" \
  -F "file=@sample.wav" \
  -F "model=whisper-large-v3" \
  -F "language=en" | jq
```

- The `model` value is OpenAI-compatible; the server maps to your configured local backend.
- For simple text response, set `-H "Accept: text/plain"`.

1. Real-time streaming STT (WebSocket)
- Endpoint: `WS /api/v1/audio/stream/transcribe`
- Example (with `wscat`):

```bash
wscat -c ws://127.0.0.1:8000/api/v1/audio/stream/transcribe \
  -H "X-API-KEY: $SINGLE_USER_API_KEY"
# Then send base64-encoded audio chunks per the server protocol
```

Troubleshooting
- Long files: prefer shorter clips or chunk client-side.
- Out-of-memory: try a smaller model (e.g., `whisper-medium`), or run on GPU.

---

## Verifying Setup via WebUI

- Open the Next.js WebUI (`apps/tldw-frontend`) or visit <http://127.0.0.1:8000/api/v1/config/quickstart>.
- Tabs:
  - Audio → Transcription (STT): upload a short clip and transcribe
  - Audio → TTS: enter text, pick a voice/model, and synthesize
- In single-user mode, set `NEXT_PUBLIC_X_API_KEY` in the frontend config if you want auto-auth.

---

## Common Errors & Fixes

- 401/403 Unauthorized
  - Use `X-API-KEY` (single-user) or Bearer JWT (multi-user). Check server logs on startup.
- 404 / Model or voice not found
  - Verify provider is enabled and files exist; check YAML paths and voice IDs.
- `kokoro_onnx` or `kokoro` missing
  - `pip install kokoro-onnx` (ONNX) or install the PyTorch package for Kokoro.
- eSpeak library missing (Kokoro ONNX)
  - Install `espeak-ng` and set `PHONEMIZER_ESPEAK_LIBRARY` to the library path.
- FFmpeg not found
  - Install FFmpeg and ensure it’s accessible in PATH.
- Network/API errors with OpenAI
  - Verify `OPENAI_API_KEY`. Check rate limits; proxy/corporate networks may block.

---

## Tips & Configuration

- Provider priority
  - `tldw_Server_API/app/core/TTS/tts_providers_config.yaml` → `provider_priority`
  - Put your preferred provider first (e.g., `kokoro` before `openai`).
- Adapter retry
  - `performance.adapter_failure_retry_seconds: 300` allows periodic re-init after failures.
- Streaming errors as audio vs HTTP errors
  - `performance.stream_errors_as_audio: false` (recommended for production APIs).
- GPU acceleration
  - For PyTorch-based backends (Kokoro PT, NeMo), install appropriate CUDA builds and set `device: cuda`.

---

## Privacy & Security

- tldw_server is designed for local/self-hosted use. Audio data stays local unless you call hosted APIs (e.g., OpenAI).
- Never commit API keys; prefer environment variables or `.env`.

---

## Appendix — Sample Kokoro YAML Snippet

```yaml
provider_priority:
  - kokoro
  - openai
providers:
  kokoro:
    enabled: true
    use_onnx: true
    model_path: "models/kokoro/onnx/model.onnx"
    voices_json: "models/kokoro/voices"
    device: "cpu"
performance:
  adapter_failure_retry_seconds: 300
  stream_errors_as_audio: false
```

If you would like, we can configure a setup checker that validates models, voices, FFmpeg, and environment keys, and reports fixes before you run your first request.

---

## Additional TTS Backends (Advanced/Optional)

These providers are supported via adapters. Many require large model downloads and work best with a GPU.

### ElevenLabs (Hosted)

- Enable in YAML and set `ELEVENLABS_API_KEY`.

```yaml
providers:
  elevenlabs:
    enabled: true
    api_key: ${ELEVENLABS_API_KEY}
    model: "eleven_monolingual_v1"
```

- Test: `model: eleven_monolingual_v1`, `voice: rachel` (or a voice from your catalog).

### Higgs Audio V2 (Local)

- Deps: `pip install torch torchaudio soundfile huggingface_hub`; `pip install git+https://github.com/boson-ai/higgs-audio.git`
- YAML:

```yaml
providers:
  higgs:
    enabled: true
    model_path: "bosonai/higgs-audio-v2-generation-3B-base"
    tokenizer_path: "bosonai/higgs-audio-v2-tokenizer"
    device: "cuda"
```

- Test: `model: higgs`, `voice: narrator`.

### Dia (Local, dialogue specialist)

- Deps: `pip install torch transformers accelerate safetensors sentencepiece soundfile huggingface_hub`
- YAML:

```yaml
providers:
  dia:
    enabled: true
    model_path: "nari-labs/dia"
    device: "cuda"
```

- Test: `model: dia`, `voice: speaker1`.

### VibeVoice (Local, expressive multi-speaker)

- Deps: `pip install torch torchaudio sentencepiece soundfile huggingface_hub`
- Install (official):

  ```bash
  git clone https://github.com/microsoft/VibeVoice.git libs/VibeVoice
  cd libs/VibeVoice && pip install -e .
  cd ../..
  ```

- YAML:

```yaml
providers:
  vibevoice:
    enabled: true
    auto_download: true
    device: "cuda"  # or mps/cpu
```

- Test: `model: vibevoice`, `voice: 1` (speaker index).

### NeuTTS Air (Local, voice cloning)

- Deps: `pip install neucodec>=0.0.4 librosa phonemizer transformers` (optional streaming: `pip install llama-cpp-python`)
- YAML:

```yaml
providers:
  neutts:
    enabled: true
    backbone_repo: "neuphonic/neutts-air"
    backbone_device: "cpu"
    codec_repo: "neuphonic/neucodec"
    codec_device: "cpu"
```

- Test: `model: neutts` and provide a base64 `voice_reference` in the JSON body.

### IndexTTS2 (Local, expressive zero-shot)

- Place checkpoints under `checkpoints/index_tts2/`.
- YAML:

```yaml
providers:
  index_tts:
    enabled: true
    model_dir: "checkpoints/index_tts2"
    cfg_path: "checkpoints/index_tts2/config.yaml"
    device: "cuda"
```

- Test: `model: index_tts` (some voices require reference audio).

---

## Additional STT Backends (Advanced/Optional)

### NVIDIA NeMo — Parakeet and Canary

- Deps (standard backend): `pip install 'nemo_toolkit[asr]'>=1.23.0`
- Alternative backends (optional):
  - ONNX: `pip install onnxruntime>=1.16.0 huggingface_hub soundfile librosa numpy`
  - MLX (Apple Silicon): `pip install mlx mlx-lm`
- Usage with `/api/v1/audio/transcriptions`:
  - `model=nemo-parakeet-1.1b` or `model=nemo-canary`
  - Language: set `language=en` (or appropriate code) when known.

### Qwen2Audio (Local)

- Deps: `pip install torch transformers accelerate soundfile sentencepiece`
- Optional: use the setup installer to prefetch assets.
- Usage with `/api/v1/audio/transcriptions`:
  - `model=qwen2audio`

Notes
- Some media endpoints expose more granular backend choices (e.g., Parakeet backends); for `/audio/transcriptions` the `model` is typically sufficient.

---

## Model Hints (At-a-Glance)

- TTS models: `tts-1` (OpenAI), `kokoro`, `eleven_monolingual_v1`, `higgs`, `dia`, `vibevoice`, `neutts`, `index_tts`.
- STT models: `whisper-1` (faster-whisper), `whisper-large-v3` and `*-ct2` variants, `nemo-canary`, `nemo-parakeet-1.1b`, `qwen2audio`, `vibevoice-asr`.
