# First-Time Audio Setup: GPU / Accelerated Systems

Use this guide if you are setting up speech features on:

- NVIDIA GPU systems
- Apple Silicon systems

This guide supports:

- `make`-driven local setup
- manual/local Python setup
- Docker + WebUI setup

Important: the stock Docker quickstart is not a turnkey GPU-enabled audio profile. If you want the fastest first successful accelerated setup, local Python or `make` is the better path today.

## What We Recommend on Accelerated Hardware

| Hardware | Recommended STT | Fallback STT | Recommended TTS | Why |
| --- | --- | --- | --- | --- |
| NVIDIA | faster-whisper | `parakeet-tdt-0.6b-v3-onnx` | `supertonic` | best first-run accelerated STT path in current repo, with a simpler local TTS path |
| Apple Silicon | `parakeet-mlx` | `parakeet-tdt-0.6b-v3-onnx` | `supertonic` | makes MLX the primary speech acceleration path while keeping TTS local-first |

Alternatives:

- If you need local voice cloning on day one: `pocket_tts`
- If you want a better but more demanding TTS stack after the basics work: `qwen3_tts`

Important current-repo realities:

- The shipped `config.txt` defaults use `parakeet-tdt-0.6b-v3-onnx` for STT (the CPU-friendly default). The shorter `parakeet-onnx` alias remains supported for older configs. This guide shows you how to change those defaults to GPU-optimized engines.
- The `/setup` bundle docs may recommend a different first-run STT path for some hardware classes.
- Stock Docker CPU/default audio works with bundled dependencies, but the stock Docker profile is not a ready-made GPU-accelerated audio path. Host-side config or model edits require a rebuild, `Dockerfiles/docker-compose.host-storage.yml`, or a custom image path.

## Choose Your Hardware Lane First

### NVIDIA lane

Use this if:

- `nvidia-smi` works on the host
- you want accelerated faster-whisper first

### Apple Silicon lane

Use this if:

- you are on an M-series Mac
- you want MLX-based Parakeet as the main STT path

## Before You Start

### Shared prerequisites

- Git
- Python 3.10+ for local/manual or `make`
- `ffmpeg`
- `git-lfs` if you want the recommended `supertonic` path

### NVIDIA-specific prerequisites

- current NVIDIA drivers
- a working `nvidia-smi`
- CUDA-capable runtime for your chosen environment

Check this first:

```bash
nvidia-smi
```

### Apple Silicon-specific prerequisites

- Apple Silicon Mac
- Python 3.10+
- ability to install MLX packages in the active environment

### OS notes

Linux:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git git-lfs python3 python3-venv
git lfs install
```

macOS:

```bash
brew install ffmpeg git git-lfs python@3.12
git lfs install
```

Windows:

- install Python 3.10+
- install FFmpeg
- install Git and Git LFS
- for NVIDIA, confirm `nvidia-smi` works in PowerShell

Then:

```powershell
git lfs install
```

## Step 1: Choose Your Base Setup Path

If your server is already running, skip to [Step 2](#step-2-configure-accelerated-stt).

### Option A: `make` Local Setup

```bash
git clone https://github.com/rmusser01/tldw_server.git
cd tldw_server
make install-local
make setup-local-single
make start-local-single
```

### Option B: Manual / Local Python Setup

Linux/macOS:

```bash
git clone https://github.com/rmusser01/tldw_server.git
cd tldw_server
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
python -m uvicorn tldw_Server_API.app.main:app --reload
```

Windows PowerShell:

```powershell
git clone https://github.com/rmusser01/tldw_server.git
cd tldw_server
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m uvicorn tldw_Server_API.app.main:app --reload
```

### Option C: Docker + WebUI Setup

```bash
git clone https://github.com/rmusser01/tldw_server.git
cd tldw_server
cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env
```

Set `AUTH_MODE=single_user` and `SINGLE_USER_API_KEY=...`, then:

```bash
docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.single-user.yml \
  -f Dockerfiles/docker-compose.webui.yml \
  up -d --build
```

Important Docker note:

- stock Docker CPU/default audio works with bundled dependencies
- the default compose profile is not a ready-made accelerated audio profile
- the app service does not declare GPU runtime reservations in the stock compose file
- host-side `Config_Files` and `models/` changes require a rebuild, `Dockerfiles/docker-compose.host-storage.yml`, or a custom image path

For accelerated audio, local/manual or `make` is the recommended first path.

## Step 2: Configure Accelerated STT

## NVIDIA: faster-whisper first

Edit [config.txt](../../tldw_Server_API/Config_Files/config.txt):

```ini
[STT-Settings]
default_batch_transcription_model = whisper-1
default_streaming_transcription_model = whisper-1
default_transcriber = faster-whisper
```

Notes:

- `whisper-1` is the simplest OpenAI-compatible starting point and maps to the faster-whisper Whisper path.
- If your GPU is smaller and `whisper-1` is too heavy, switch both defaults to a smaller faster-whisper model such as `medium`.
- If accelerated Whisper setup becomes unstable, fall back to `parakeet-tdt-0.6b-v3-onnx`.

## Apple Silicon: `parakeet-mlx` first

Install the MLX STT extras in your active environment:

```bash
pip install -e '.[STT_Parakeet_MLX]'
```

Then edit [config.txt](../../tldw_Server_API/Config_Files/config.txt):

```ini
[STT-Settings]
default_batch_transcription_model = parakeet-mlx
default_streaming_transcription_model = parakeet-mlx
default_transcriber = parakeet
nemo_model_variant = mlx
```

## Accelerated fallback: `parakeet-tdt-0.6b-v3-onnx`

If your accelerated path is not stable yet, use:

```ini
[STT-Settings]
default_batch_transcription_model = parakeet-tdt-0.6b-v3-onnx
default_streaming_transcription_model = parakeet-tdt-0.6b-v3-onnx
default_transcriber = parakeet
nemo_model_variant = onnx
```

If you are on Docker and you edited the host config, rebuild the app image.

## Step 3: Configure the Recommended TTS Path (`supertonic`)

The accelerated guide still recommends `supertonic` as the first local TTS path because it stays much simpler than the heavier TTS stacks.

### 3A. Install the assets

```bash
python Helper_Scripts/TTS_Installers/install_tts_supertonic.py
```

### 3B. Enable the provider

Edit [tts_providers_config.yaml](../../tldw_Server_API/Config_Files/tts_providers_config.yaml):

```yaml
providers:
  supertonic:
    enabled: true
    model_path: "models/supertonic/onnx"
    sample_rate: 24000
    device: "cpu"
    extra_params:
      voice_styles_dir: "models/supertonic/voice_styles"
      default_voice: "supertonic_m1"
      voice_files:
        supertonic_m1: "M1.json"
        supertonic_f1: "F1.json"
      default_total_step: 5
      default_speed: 1.05
      n_test: 1
```

### 3C. Make it the default TTS provider

Edit [config.txt](../../tldw_Server_API/Config_Files/config.txt):

```ini
[TTS-Settings]
default_provider = supertonic
default_voice = supertonic_m1
```

Restart the server after changes.

## Step 4: First Successful Verification

Verify the accelerated lane you intended, then verify real TTS and STT.

Choose one reusable auth header before running the commands.

Single-user auth mode:

```bash
AUTH_HEADER=(-H "X-API-KEY: $SINGLE_USER_API_KEY")
```

Multi-user auth mode:

```bash
JWT=$(
  curl -sS -X POST http://127.0.0.1:8000/api/v1/auth/login \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=$ADMIN_USERNAME" \
    -d "password=$ADMIN_PASSWORD" | jq -r '.access_token'
)
AUTH_HEADER=(-H "Authorization: Bearer $JWT")
```

### 4A. TTS health and voice catalog

```bash
curl -sS http://127.0.0.1:8000/api/v1/audio/health \
  "${AUTH_HEADER[@]}"
```

```bash
curl -sS http://127.0.0.1:8000/api/v1/audio/voices/catalog \
  "${AUTH_HEADER[@]}" | jq '.supertonic'
```

### 4B. Generate a short test file with TTS

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/speech \
  "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "tts-supertonic-1",
        "voice": "supertonic_m1",
        "input": "This is the accelerated audio setup smoke test.",
        "response_format": "wav",
        "stream": false
      }' \
  --output accelerated_audio_smoke.wav
```

### 4C. Verify your STT backend

#### NVIDIA

Host check:

```bash
nvidia-smi
```

STT readiness:

```bash
curl -sS "http://127.0.0.1:8000/api/v1/audio/transcriptions/health?model=whisper-1&warm=true" \
  "${AUTH_HEADER[@]}"
```

You want to see Whisper reported as usable and warm initialization succeeding.

#### Apple Silicon

STT readiness:

```bash
curl -sS "http://127.0.0.1:8000/api/v1/audio/transcriptions/health?model=parakeet-mlx" \
  "${AUTH_HEADER[@]}"
```

You want to see:

- `"provider": "parakeet"`
- `"alias": "parakeet-mlx"`
- `"usable": true` or `"available": true`

### 4D. Transcribe the generated file back through STT

#### NVIDIA

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/transcriptions \
  "${AUTH_HEADER[@]}" \
  -F "file=@accelerated_audio_smoke.wav" \
  -F "model=whisper-1"
```

#### Apple Silicon

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/transcriptions \
  "${AUTH_HEADER[@]}" \
  -F "file=@accelerated_audio_smoke.wav" \
  -F "model=parakeet-mlx"
```

Success means:

- the request completes
- the `text` field is close to `This is the accelerated audio setup smoke test`
- the backend matches the path you intended

## Optional Alternatives: PocketTTS Runtimes

Use a PocketTTS runtime instead of `supertonic` if local voice cloning matters more than the simplest first-run TTS path.

Use:

- [PocketTTS Voice Cloning Guide](../User_Guides/WebUI_Extension/PocketTTS_Voice_Cloning_Guide.md) for `pocket_tts` (Python/ONNX)
- `python Helper_Scripts/TTS_Installers/install_tts_pocket_tts_cpp.py` for `pocket_tts_cpp` (compiled native runtime)

Tradeoffs:

- `pocket_tts` is the ONNX/Python runtime and is the simplest PocketTTS path to read and debug.
- `pocket_tts_cpp` is a separate compiled runtime and uses a different installer and runtime layout.
- Both are excellent if voice cloning is the point.
- Both are worse than the default first-sound path because you still need either a direct `voice_reference` clip or a stored `custom:<voice_id>` voice.
- `pocket_tts_cpp` streaming is only available when the local CLI probe proves incremental on this install; otherwise streaming requests fail closed.

## Better But More Demanding: `qwen3_tts`

After the basic accelerated stack works, move to:

- [QWEN3_TTS_SETUP.md](../STT-TTS/QWEN3_TTS_SETUP.md)

Treat it as the advanced upgrade path, not the baseline.

## Troubleshooting

### NVIDIA path keeps using CPU or fails to warm

- verify `nvidia-smi` on the host first
- keep `whisper-1` only if your card can handle it; otherwise switch to `medium`
- if the accelerated Whisper path is still unstable, switch to `parakeet-tdt-0.6b-v3-onnx` and get speech working first

### Apple Silicon path fails on `parakeet-mlx`

- confirm you installed:

```bash
pip install -e '.[STT_Parakeet_MLX]'
```

- verify the config really says `parakeet-mlx`
- if MLX still does not initialize, fall back to `parakeet-tdt-0.6b-v3-onnx`

### The server is using the wrong STT model

- make the defaults explicit in [config.txt](../../tldw_Server_API/Config_Files/config.txt)
- do not rely on implicit provider selection if you care which backend is used
- verify with `/api/v1/audio/transcriptions/health?model=...`

### Docker accelerated path does not see GPU changes or host config changes

- the stock app compose profile is not a GPU-optimized audio compose file
- host-side config changes require an image rebuild
- host-side model assets are not automatically mounted into the app container

If you want the least frustrating accelerated first run today, prefer local/manual or `make`.

### `/setup` chose a different first-run path than this guide

That can happen today.

Use `/setup` when you want guided provisioning, then manually set:

- your STT defaults in [config.txt](../../tldw_Server_API/Config_Files/config.txt)
- your TTS provider in [config.txt](../../tldw_Server_API/Config_Files/config.txt)
- your enabled provider block in [tts_providers_config.yaml](../../tldw_Server_API/Config_Files/tts_providers_config.yaml)
