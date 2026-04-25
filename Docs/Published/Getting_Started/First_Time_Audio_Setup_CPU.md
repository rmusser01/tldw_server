# First-Time Audio Setup: CPU Systems

Use this guide if you are setting up speech features on:

- Linux
- Windows
- macOS on Intel

If you are on Apple Silicon or an NVIDIA GPU box, use [First-Time Audio Setup: GPU / Accelerated Systems](./First_Time_Audio_Setup_GPU_Accelerated.md) instead.

This guide supports three base setup paths:

- `make`-driven local setup
- manual/local Python setup
- Docker + WebUI setup

## What We Recommend on CPU

For a local-first CPU setup in the current repo:

| Goal | STT | TTS | Why |
| --- | --- | --- | --- |
| Recommended first local stack | `parakeet-tdt-0.6b-v3-onnx` | `supertonic` | Keeps the stack local-first and avoids mandatory voice-cloning input on every TTS request |
| If you need local voice cloning immediately | `parakeet-tdt-0.6b-v3-onnx` | `pocket_tts` | Still local-first, but every request needs reference audio |
| Better but more demanding | `parakeet-tdt-0.6b-v3-onnx` or `faster-whisper` | `qwen3_tts` | Strong upgrade path after the basic stack already works |

Important current-repo realities:

- The shipped explicit STT defaults are currently `parakeet-tdt-0.6b-v3-onnx` for batch and streaming. The shorter `parakeet-onnx` alias remains supported for older configs.
- The current `/setup` audio bundle docs still describe a different first-run path in some places.
- Stock Docker CPU/default audio works with bundled dependencies. Host-side config or model edits are not visible inside the container until you rebuild, use `Dockerfiles/docker-compose.host-storage.yml`, or build a custom image path.

If your only goal is "make sound come out as fast as possible", the current `/setup` bundle path may still be less manual than the exact `supertonic` path in this guide. This guide is the better fit when you want a local-first stack that you understand and can control.

## Before You Start

You need:

- Git
- Python 3.10+ if you are using `make` or manual/local Python
- `ffmpeg`
- `git-lfs` if you want the recommended `supertonic` path

Recommended host prerequisites by OS:

### Linux

- `ffmpeg`
- `git`
- `git-lfs`
- Python 3.10+

Typical packages:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git git-lfs python3 python3-venv
git lfs install
```

### macOS (Intel)

- `ffmpeg`
- `git`
- `git-lfs`
- Python 3.10+

Typical packages:

```bash
brew install ffmpeg git git-lfs python@3.12
git lfs install
```

### Windows

Install:

- Python 3.10+
- FFmpeg
- Git
- Git LFS

Use `winget` or the official installers, then run:

```powershell
git lfs install
```

## Step 1: Choose Your Base Setup Path

If your server is already running, skip to [Step 2](#step-2-set-the-cpu-stt-defaults).

### Option A: `make` Local Setup

Use this when you want a local Python install but do not want to do the venv/bootstrap steps by hand.

```bash
git clone https://github.com/rmusser01/tldw_server.git
cd tldw_server
make install-local
make setup-local-single
make start-local-single
```

### Option B: Manual / Local Python Setup

Use this when you want full control over the virtual environment and installed extras.

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

Use this when you want the containerized first-run path.

```bash
git clone https://github.com/rmusser01/tldw_server.git
cd tldw_server
cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env
```

Set `AUTH_MODE=single_user` and `SINGLE_USER_API_KEY=...` in `tldw_Server_API/Config_Files/.env`, then:

```bash
docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.single-user.yml \
  -f Dockerfiles/docker-compose.webui.yml \
  up -d --build
```

Or, if you prefer the Makefile wrapper:

```bash
make quickstart
```

Important Docker note:

- Stock Docker CPU/default audio works with bundled dependencies.
- The stock container image does not bind-mount `Config_Files` or `models/`.
- Host-side edits to `tldw_Server_API/Config_Files/config.txt` or local model assets require a rebuild, `Dockerfiles/docker-compose.host-storage.yml`, or a custom image path.
- If you use `/setup` inside the running container, those changes are container-local unless you also persist or reproduce them in your chosen image/storage path.

## Step 2: Set the CPU STT Defaults

Edit [config.txt](../../../tldw_Server_API/Config_Files/config.txt) and make the STT defaults explicit:

```ini
[STT-Settings]
default_batch_transcription_model = parakeet-tdt-0.6b-v3-onnx
default_streaming_transcription_model = parakeet-tdt-0.6b-v3-onnx
default_transcriber = parakeet
nemo_model_variant = onnx
```

Why set all four?

- `default_batch_transcription_model` and `default_streaming_transcription_model` remove ambiguity.
- `default_transcriber` and `nemo_model_variant` keep older compatibility paths aligned with the intended backend.

If you are on the stock Docker path, rebuild the app image after editing the file on the host:

```bash
docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.single-user.yml \
  -f Dockerfiles/docker-compose.webui.yml \
  up -d --build
```

## Step 3: Set Up the Recommended CPU TTS Path (`supertonic`)

### Why `supertonic` here

This guide recommends `supertonic` as the main local-first CPU TTS path because:

- it stays local
- it does not require reference audio on every request
- it already has an installer helper and provider support in the repo

### 3A. Install the Supertonic assets

Run from the repo root:

```bash
python Helper_Scripts/TTS_Installers/install_tts_supertonic.py
```

What this does:

- clones the upstream model repo
- copies ONNX assets into `models/supertonic/onnx`
- copies voice-style JSON files into `models/supertonic/voice_styles`

This path currently assumes:

- `git` is available
- `git-lfs` is installed and initialized

### 3B. Enable the provider

Edit [tts_providers_config.yaml](../../../tldw_Server_API/Config_Files/tts_providers_config.yaml):

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

### 3C. Make `supertonic` the default TTS provider

Edit [config.txt](../../../tldw_Server_API/Config_Files/config.txt):

```ini
[TTS-Settings]
default_provider = supertonic
default_voice = supertonic_m1
local_device = cpu
```

You do not have to reorder `provider_priority` if you set `default_provider` explicitly, but it is still a good idea to make the YAML reflect your preferred path long term.

### 3D. Restart the server

Local / `make` paths:

```bash
# stop the server, then start it again
make start-local-single
```

or

```bash
python -m uvicorn tldw_Server_API.app.main:app --reload
```

Docker paths:

```bash
docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.single-user.yml \
  -f Dockerfiles/docker-compose.webui.yml \
  up -d --build
```

## Step 4: First Successful Verification

Do not stop at `/health`. Verify one real TTS request and one real STT request.

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

### 4A. Confirm TTS health

```bash
curl -sS http://127.0.0.1:8000/api/v1/audio/health \
  "${AUTH_HEADER[@]}"
```

What you want to see:

- overall health is not `unhealthy`
- `supertonic` appears under the provider details

### 4B. Confirm the Supertonic voice catalog

```bash
curl -sS http://127.0.0.1:8000/api/v1/audio/voices/catalog \
  "${AUTH_HEADER[@]}" | jq '.supertonic'
```

You should see voices such as `supertonic_m1` and `supertonic_f1`.

### 4C. Generate a short audio file with TTS

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/speech \
  "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "tts-supertonic-1",
        "voice": "supertonic_m1",
        "input": "This is the CPU audio setup smoke test.",
        "response_format": "wav",
        "stream": false
      }' \
  --output cpu_audio_smoke.wav
```

### 4D. Confirm STT health

```bash
curl -sS "http://127.0.0.1:8000/api/v1/audio/transcriptions/health?model=parakeet-tdt-0.6b-v3-onnx" \
  "${AUTH_HEADER[@]}"
```

What you want to see:

- `"provider": "parakeet"`
- `"alias": "parakeet-tdt-0.6b-v3-onnx"` or `"alias": "parakeet-onnx"`
- `"usable": true` or `"available": true`

### 4E. Transcribe the generated audio back through STT

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/transcriptions \
  "${AUTH_HEADER[@]}" \
  -F "file=@cpu_audio_smoke.wav" \
  -F "model=parakeet-tdt-0.6b-v3-onnx"
```

Success means:

- the request returns JSON
- the `text` field is close to `This is the CPU audio setup smoke test`
- the server does not silently switch to the wrong provider/model

## Optional Alternative: `pocket_tts` When Voice Cloning Matters More

Choose `pocket_tts` instead of `supertonic` if you specifically need local voice cloning on day one.

Use:

- [PocketTTS Voice Cloning Guide](../User_Guides/WebUI_Extension/PocketTTS_Voice_Cloning_Guide.md)

Important tradeoff:

- `pocket_tts` is not a better "first sound out of the box" default
- it is a better local-first choice when cloning is mandatory
- every request needs a reference clip or a previously prepared voice path

## Better But More Demanding: `qwen3_tts`

Use `qwen3_tts` after the basic CPU stack already works.

Use:

- [QWEN3_TTS_SETUP.md](../../../Docs/STT-TTS/QWEN3_TTS_SETUP.md)

Treat it as a second-step upgrade, not the first-run baseline.

## Troubleshooting

### `ffmpeg` errors or audio conversion failures

- Run `ffmpeg -version`
- Install FFmpeg on the host
- Restart the server after fixing PATH issues on Windows

### Supertonic does not appear in `/audio/health`

- confirm `providers.supertonic.enabled: true` in [tts_providers_config.yaml](../../../tldw_Server_API/Config_Files/tts_providers_config.yaml)
- confirm the asset directories exist:
  - `models/supertonic/onnx`
  - `models/supertonic/voice_styles`
- restart the server after changing config

### Supertonic voice catalog is empty

- re-run the installer
- verify `voice_files` still point to `M1.json` and `F1.json`
- check server logs for missing ONNX or style files

### STT health shows the wrong model/provider

- re-open [config.txt](../../../tldw_Server_API/Config_Files/config.txt)
- make sure both `default_batch_transcription_model` and `default_streaming_transcription_model` are set to `parakeet-tdt-0.6b-v3-onnx`
- make sure `default_transcriber = parakeet`
- restart the server

### Docker keeps ignoring host config changes

- the stock Docker image bakes in `Config_Files` at build time
- rebuild the app image after host edits:

```bash
docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.single-user.yml \
  -f Dockerfiles/docker-compose.webui.yml \
  up -d --build
```

### You want the easiest guided path, not the exact stack from this guide

Use `/setup`, accept the current recommended audio bundle, and verify speech first.

Then come back to this guide if you want to move from the bundle defaults to:

- `parakeet-tdt-0.6b-v3-onnx` (`parakeet-onnx` remains accepted as a legacy alias)
- `supertonic`
- `pocket_tts`
- `qwen3_tts`
