# Setup Wizard Guide

## Overview

`/setup` is the operator-facing first-run workflow for `tldw_server`.
It now covers both configuration review and curated audio provisioning, so the default path is:

1. Start the server
2. Open `http://127.0.0.1:8000/setup`
3. Save any required config changes
4. Accept or change the recommended audio bundle
5. Provision, verify, and review the audio readiness report
6. Mark setup complete

Use this guide when you want the shortest supported path to a working local or hybrid speech stack without hand-picking every STT/TTS provider.

## Public Setup Profiles

The public onboarding docs present three peer profiles. Use one profile end-to-end before opening `/setup`:

| Profile | Prepare | Start | Verify |
| --- | --- | --- | --- |
| Docker single-user + WebUI | `make setup-docker-single` | `make start-docker-single` | `make verify-docker-single` |
| Docker multi-user + Postgres | Generate admin vars, then `make setup-docker-multi` | `make start-docker-multi` | `make verify-docker-multi` |
| Local single-user | `make install-local` then `make setup-local-single` | `make start-local-single` | `make verify-local-single` |

For Docker multi-user prepare, keep the generated variables in the same shell so later login checks can reuse them:

```bash
export ADMIN_USERNAME=tldw-admin
export ADMIN_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
make setup-docker-multi
```

`make quickstart` remains the shortest alias for Docker single-user + WebUI. It is not the multi-user or local start command.

## Before You Start

Make sure the machine can run the installer and any local speech backends you want:

1. Activate the project environment and install the project:
   ```bash
   source .venv/bin/activate
   pip install -e .
   ```
2. Copy the environment template if you have not done it yet:
   ```bash
   cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env
   ```
3. Install FFmpeg:
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt-get install -y ffmpeg`
4. If you want local Kokoro TTS, install eSpeak NG as well:
   - macOS: `brew install espeak-ng`
   - Ubuntu/Debian: `sudo apt-get install -y espeak-ng`

## Recommended Flow

1. Start the API server:
   ```bash
   source .venv/bin/activate
   python -m uvicorn tldw_Server_API.app.main:app --reload
   ```
2. Open `http://127.0.0.1:8000/setup`.
3. Work through the guided questions so the setup UI can highlight the sections that matter for your deployment.
4. In the audio stage:
   - review the detected machine profile
   - accept the recommended bundle and profile, or choose a different curated bundle
   - click `Provision recommended bundle` or `Provision selected bundle`
   - complete any guided prerequisites the report calls out
   - click `Run verification`
   - inspect `View readiness report`
5. Save config changes, then click `Mark Setup Complete`.

The setup UI tracks audio separately from global setup completion. You can safely rerun audio provisioning or verification without reopening the whole first-run flow.

## Curated Audio Bundles

_Generated from `Helper_Scripts/generate_audio_bundle_docs.py` and the setup bundle catalog._

| Bundle ID | Label | Profiles | Offline runtime after provisioning | Offline pack compatibility | Default STT | Default TTS | Automatic steps | Guided prerequisites |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cpu_local` | CPU Local | Light, Balanced, Performance | Yes | v1 manifest import + model portability | faster_whisper [small] | kokoro | Install CPU-local Python dependencies, Download CPU-local speech model assets | Install FFmpeg, Install eSpeak NG |
| `apple_silicon_local` | Apple Silicon Local | Light, Balanced, Performance | Yes | v1 manifest import + model portability | faster_whisper [small] | kokoro | Install Apple Silicon Python dependencies, Download Apple Silicon speech model assets | Install FFmpeg, Install eSpeak NG |
| `nvidia_local` | NVIDIA Local | Light, Balanced, Performance | Yes | v1 manifest import + model portability | faster_whisper [medium] | kokoro | Install NVIDIA local Python dependencies, Download NVIDIA speech model assets | Install FFmpeg, Install eSpeak NG |
| `hosted_plus_local_backup` | Hosted With Local Backup | Balanced | No | v1 manifest import only | faster_whisper [small] | kokoro | Install hybrid Python dependencies, Download local fallback speech model assets | Install FFmpeg, Install eSpeak NG |

### `cpu_local`

- Label: CPU Local
- Resource profiles: Light, Balanced, Performance
- Offline runtime after provisioning: Yes
- Offline pack compatibility: v1 manifest import + model portability
- Default STT: faster_whisper [small]
- Default TTS: kokoro
- Automatic steps: Install CPU-local Python dependencies, Download CPU-local speech model assets
- Guided prerequisites: Install FFmpeg, Install eSpeak NG

| Profile | Resource class | Estimated disk | STT plan | TTS plan |
| --- | --- | --- | --- | --- |
| Light | low | 1.0 GB | faster_whisper [tiny] | kokoro |
| Balanced | medium | 2.0 GB | faster_whisper [small] | kokoro |
| Performance | high | 4.5 GB | faster_whisper [medium] | kokoro |

### `apple_silicon_local`

- Label: Apple Silicon Local
- Resource profiles: Light, Balanced, Performance
- Offline runtime after provisioning: Yes
- Offline pack compatibility: v1 manifest import + model portability
- Default STT: faster_whisper [small]
- Default TTS: kokoro
- Automatic steps: Install Apple Silicon Python dependencies, Download Apple Silicon speech model assets
- Guided prerequisites: Install FFmpeg, Install eSpeak NG

| Profile | Resource class | Estimated disk | STT plan | TTS plan |
| --- | --- | --- | --- | --- |
| Light | low | 1.0 GB | faster_whisper [tiny] | kokoro |
| Balanced | medium | 2.0 GB | faster_whisper [small] | kokoro |
| Performance | high | 4.0 GB | faster_whisper [medium] | kokoro |

### `nvidia_local`

- Label: NVIDIA Local
- Resource profiles: Light, Balanced, Performance
- Offline runtime after provisioning: Yes
- Offline pack compatibility: v1 manifest import + model portability
- Default STT: faster_whisper [medium]
- Default TTS: kokoro
- Automatic steps: Install NVIDIA local Python dependencies, Download NVIDIA speech model assets
- Guided prerequisites: Install FFmpeg, Install eSpeak NG

| Profile | Resource class | Estimated disk | STT plan | TTS plan |
| --- | --- | --- | --- | --- |
| Light | low | 2.0 GB | faster_whisper [small] | kokoro |
| Balanced | medium | 4.0 GB | faster_whisper [medium] | kokoro |
| Performance | high | 8.0 GB | faster_whisper [large-v3] | kokoro |

### `hosted_plus_local_backup`

- Label: Hosted With Local Backup
- Resource profiles: Balanced
- Offline runtime after provisioning: No
- Offline pack compatibility: v1 manifest import only
- Default STT: faster_whisper [small]
- Default TTS: kokoro
- Automatic steps: Install hybrid Python dependencies, Download local fallback speech model assets
- Guided prerequisites: Install FFmpeg, Install eSpeak NG

| Profile | Resource class | Estimated disk | STT plan | TTS plan |
| --- | --- | --- | --- | --- |
| Balanced | medium | 2.5 GB | faster_whisper [small] | kokoro |

## Provisioning Modes

- `Online provisioning`: `/setup` installs Python dependencies, downloads model assets, and verifies the selected bundle/profile on the current machine.
- `Offline pack import`: the setup audio pack import endpoint validates a manifest stored under `Config_Files/audio_packs/` against local platform, arch, and Python compatibility, then registers the imported pack in the readiness report.
- `Offline pack` v1 scope: manifest + model portability only. It does not install Python dependencies or OS prerequisites on the target machine.

## Verification After Provisioning

The bundle stage is only considered healthy after verification succeeds. Use these checkpoints:

- In `/setup`, click `Run verification` and confirm the readiness report reaches `ready` or `ready_with_warnings`.
- Review guided prerequisite items if the report is `partial` or `failed`.
- Keep a note of the selected bundle id if you plan to reproduce the same setup on another machine.

For API-level verification after the setup UI completes, choose one reusable auth header first.

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

```bash
curl -s http://127.0.0.1:8000/api/v1/audio/voices/catalog \
  "${AUTH_HEADER[@]}" | jq
```

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/transcriptions \
  "${AUTH_HEADER[@]}" \
  -F "file=@sample.wav" \
  -F "model=whisper-1"
```

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/speech \
  "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "kokoro",
        "voice": "af_bella",
        "input": "Setup verification complete",
        "response_format": "mp3"
      }' \
  --output setup-check.mp3
```

Stock Docker CPU/default audio works with bundled dependencies. If you edit host-side audio config or add host-side model files, rebuild the image or use the documented host-storage/custom image path so those changes are visible inside the container.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| No audio recommendation appears | Make sure `/setup` can reach `/api/v1/setup/audio/recommendations`; reload the page and inspect the setup message panel. |
| Provisioning returns `partial` | Guided prerequisites are still missing. Install FFmpeg and/or eSpeak NG, then run `Safe rerun` and `Run verification`. |
| Verification returns `failed` | Check the readiness report for remediation items such as missing FFmpeg, missing eSpeak NG, or an unusable STT/TTS path. |
| Offline pack import is rejected | Re-export the pack from a machine with the same platform, arch, and Python minor version. v1 imports validate compatibility but do not install missing Python dependencies. |
| `/setup` redirects away immediately | The global setup flow is already marked complete. Re-enable the first-time setup flag if you need the guided UI again. |
| Local speech is required offline later | Prefer `cpu_local`, `apple_silicon_local`, or `nvidia_local` instead of `hosted_plus_local_backup`. |
