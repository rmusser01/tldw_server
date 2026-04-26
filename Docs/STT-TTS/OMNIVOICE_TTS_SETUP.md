# OmniVoice Setup for tldw_server

This guide is for self-hosted `tldw_server` users who want to enable the local OmniVoice provider.

OmniVoice is integrated as a first-class TTS provider, but it does not run inside the main API process. `tldw_server` starts it as a dedicated loopback-only sidecar with its own Python environment.

## What You Get

- Provider key: `omnivoice`
- Runtime model: `k2-fsa/OmniVoice`
- Runtime mode: managed sidecar
- Default sample rate: `24000`
- Voice modes:
  - automatic voice selection
  - voice design via `extra_params.instruct`
  - voice cloning from a direct reference clip
  - stored `custom:<voice_id>` voices through the normal voice manager

## Prerequisites

Before enabling OmniVoice:

- You are in the `tldw_server` repo root.
- The main project virtual environment exists.
- You activate the main project virtual environment before running setup commands.
- You have roughly 5 to 6 GB of free disk space for:
  - the OmniVoice sidecar virtual environment
  - the Hugging Face model cache
- You are comfortable letting the first model download come from Hugging Face, or you plan to pre-download it manually.

Recommended local source layout:

- preferred checkout: `../OmniVoice`
- fallback installer target: `external/OmniVoice`

## Step 1: Activate the Main Project Environment

```bash
source .venv/bin/activate
```

Run the remaining setup steps from the repo root with the main `tldw_server` environment active.

## Step 2: Install the OmniVoice Sidecar Runtime

If you already cloned OmniVoice one directory above this repo:

```bash
python Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py --skip-clone --install-inference-deps
```

If you want the installer to clone OmniVoice for you:

```bash
python Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py --install-inference-deps
```

What this provisions:

- `models/omnivoice_sidecar/.venv`
- `models/omnivoice_sidecar/runtime`
- `models/omnivoice_sidecar/logs`
- the `providers.omnivoice` block in `tldw_Server_API/Config_Files/tts_providers_config.yaml`

Important:

- `--install-inference-deps` is required for real synthesis.
- Without `--install-inference-deps`, the sidecar can start, but real OmniVoice generation will fail with dependency-related errors.

## Step 3: Enable the Provider

Open `tldw_Server_API/Config_Files/tts_providers_config.yaml` and make sure the OmniVoice block looks roughly like this:

```yaml
providers:
  omnivoice:
    enabled: true
    runtime: "sidecar"
    model: "omnivoice"
    sample_rate: 24000
    max_concurrent_generations: 1
    extra_params:
      repo_path: "../OmniVoice"
      runtime_mode: "real"
      model_id: "k2-fsa/OmniVoice"
      python_path: "models/omnivoice_sidecar/.venv/bin/python"
      runtime_path: "models/omnivoice_sidecar/runtime"
      logs_path: "models/omnivoice_sidecar/logs"
      host: "127.0.0.1"
      port: 8039
      autoselect_port: true
      warmup_on_startup: false
      idle_shutdown_seconds: 900
      resident_mode: false
```

Notes:

- Keep `runtime: "sidecar"`.
- Use `runtime_mode: "real"` for actual synthesis.
- Leave `host` on loopback only.
- `autoselect_port: true` is the safest default.
- `max_concurrent_generations: 1` is intentional for the current runtime.

## Step 4: Download Model Weights

OmniVoice does not pre-download model weights during the sidecar installer step.

You have two choices:

### Option A: Let the First Real Request Download Them

This is the default behavior. The first warmup or synthesis request downloads the model assets through Hugging Face.

### Option B: Pre-download Them Up Front

Use the sidecar runtime directly:

```bash
models/omnivoice_sidecar/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("k2-fsa/OmniVoice"))
PY
```

This downloads the main model snapshot into your Hugging Face cache.

If you use the WebUI admin installer, the same workflow is now exposed in the existing audio installer panel:

- `Pre-download weights` fetches the OmniVoice model snapshot into the Hugging Face cache
- `Warm up sidecar` starts the managed sidecar and loads the model without sending a full synthesis request

## Step 5: Restart tldw_server

After enabling the provider, restart the API process so it reloads the TTS provider configuration.

Typical local start command:

```bash
source .venv/bin/activate
python -m uvicorn tldw_Server_API.app.main:app --reload
```

## Step 6: Smoke Test the Provider

Run a simple automatic-voice request:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/speech" \
  -H "X-API-KEY: ${SINGLE_USER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "omnivoice",
    "input": "Hello from OmniVoice inside tldw server.",
    "response_format": "wav",
    "stream": false
  }' --output omnivoice_smoke.wav
```

If you run `tldw_server` in multi-user JWT mode, replace the `X-API-KEY` header with your normal `Authorization: Bearer <token>` header.

Expected result:

- the request returns `200`
- `omnivoice_smoke.wav` is a playable WAV file
- the first request may take longer than usual if the model is not already warm

## File Layout and Runtime Behavior

The OmniVoice sidecar is intentionally isolated from the main server process.

Important paths:

- source checkout: `../OmniVoice` or `external/OmniVoice`
- sidecar environment: `models/omnivoice_sidecar/.venv`
- sidecar runtime dir: `models/omnivoice_sidecar/runtime`
- sidecar logs dir: `models/omnivoice_sidecar/logs`

Runtime behavior:

- the sidecar binds to loopback only
- the sidecar uses token-authenticated internal requests
- requests that resolve to OmniVoice and omit `voice` normalize to `voice: "auto"`
- the sidecar returns OmniVoice-native WAV, and the main app handles final response behavior

## Hardware and Performance Notes

- CPU works, but warmup and synthesis are slower.
- `cuda` or `mps` should be faster if your environment supports them.
- The sidecar chooses a default dtype based on device:
  - `float16` for `cuda` and `mps`
  - `float32` for `cpu`
- The current integration is intentionally conservative and does not expose streaming synthesis.

## Troubleshooting

### Real synthesis fails immediately with missing imports

Cause:

- the sidecar runtime was installed without the full inference dependency set

Fix:

```bash
source .venv/bin/activate
python Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py --skip-clone --install-inference-deps
```

### First request is slow

Cause:

- model download
- model warmup
- CPU inference

This is normal for a cold OmniVoice runtime.

### OmniVoice returns an unsupported instruct error

Cause:

- `extra_params.instruct` included tokens OmniVoice does not recognize

Fix:

- use only supported OmniVoice attributes
- keep instructs consistently English or consistently Chinese
- for English, separate items with `, `

Example of a valid English instruct:

```json
{
  "extra_params": {
    "instruct": "female, british accent, low pitch"
  }
}
```

### Voice cloning fails

Check:

- reference audio is valid audio
- the clip is roughly 3 to 10 seconds when possible
- the request includes either:
  - direct `voice_reference`, or
  - `voice: "custom:<voice_id>"`

`reference_text` is optional for OmniVoice, but providing it is usually faster and more predictable than relying on auto-transcription.

### Sidecar starts but provider stays unavailable

Check:

- `providers.omnivoice.enabled: true`
- `providers.omnivoice.runtime: "sidecar"`
- `providers.omnivoice.extra_params.runtime_mode: "real"`
- `providers.omnivoice.extra_params.python_path` points to the installer-created sidecar interpreter

## Related Docs

- `Docs/STT-TTS/OMNIVOICE_TTS_USER_GUIDE.md`
- `Docs/STT-TTS/TTS-SETUP-GUIDE.md`
- `Docs/API-related/TTS_API.md`
- `tldw_Server_API/app/core/TTS/TTS-README.md`
