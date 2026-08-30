# Getting Started with VibeVoice (TTS)

This guide walks you through installing, configuring, and using the VibeVoice text-to-speech provider inside tldw_server. You’ll generate speech, stream audio, use voice cloning, and map multiple speakers to specific voices.

## 1) Prerequisites

- Python 3.11+
- ffmpeg installed and on `PATH`
- GPU optional (CUDA recommended for performance).
- Sufficient disk space to cache models under `./models/vibevoice`.
  - Note: In tldw_server, auto-download is disabled by default. Enable per-provider in YAML via `auto_download: true` or set `VIBEVOICE_AUTO_DOWNLOAD=1`.

## 2) Install Dependencies

- Core extras for VibeVoice:

```bash
pip install -e ".[TTS_vibevoice]"
```

- Install the official VibeVoice package from source:

```bash
git clone https://github.com/microsoft/VibeVoice.git libs/VibeVoice
cd libs/VibeVoice && pip install -e .
cd ../..
```

- Optional performance add-ons (CUDA):

```bash
# 4-bit quantization (CUDA only)
pip install bitsandbytes

# Flash Attention 2 (CUDA)
pip install ninja
pip install flash-attn --no-build-isolation
```

## 3) Configure VibeVoice

You can configure via `tldw_Server_API/Config_Files/config.txt` or YAML. Below are INI examples.

```ini
[TTS-Settings]
# Model selection
vibevoice_variant = 1.5B              # 1.5B or 7B
vibevoice_device = auto               # auto, cuda, mps, or cpu

# Performance / memory
vibevoice_use_quantization = True     # CUDA-only 4-bit quantization (bitsandbytes)
vibevoice_attention_type = auto       # auto, flash_attention_2, sdpa, eager
vibevoice_auto_cleanup = True         # Free VRAM between generations

# Model downloads
vibevoice_auto_download = True        # Auto-download from Hugging Face if missing
vibevoice_model_dir = ./models/vibevoice
vibevoice_cache_dir = ./cache/vibevoice

# Voices
vibevoice_voices_dir = ./voices       # Folder scanned for voice samples
# Default per-speaker mapping (IDs or file paths). Request can override.
vibevoice_speakers_to_voices = {"1":"en-Alice_woman"}

# Optional: tiny warmup forward to catch lazy init issues
vibevoice_enable_warmup_forward = false
```

YAML alternative (`tldw_Server_API/app/core/TTS/tts_providers_config.yaml`):

```yaml
providers:
  vibevoice:
    enabled: true
    auto_download: true
    model_path: microsoft/VibeVoice-1.5B  # or vibevoice/VibeVoice-7B, FabioSarracino/VibeVoice-Large-Q8
    device: auto
    use_quantization: true
    voices_dir: ./voices
    speakers_to_voices:
      "1": en-Alice_woman
```

## 4) Run the Server

```bash
python -m uvicorn tldw_Server_API.app.main:app --reload
# API docs:   http://127.0.0.1:8000/docs
# Quickstart: http://127.0.0.1:8000/api/v1/config/quickstart
```

## 5) Quick Start: Generate Speech (Streaming)

POST `POST /api/v1/audio/speech` with streaming enabled. Example using `curl`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/audio/speech \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -o speech.wav \
  -d '{
    "model": "vibevoice",
    "input": "Speaker 1: Welcome to our show!\nSpeaker 2: Thanks for having me.",
    "voice": "speaker_1",
    "response_format": "wav",
    "stream": true
  }'
```

- `input`: Use the "Speaker N:" format for multi-speaker dialogue.
- `stream`: When `true`, the server streams audio chunks to the client.

## 6) Voice Cloning Options

- Zero-shot reference (3-10 seconds recommended):
  - Send `voice_reference` as base64-encoded audio in the request body. The adapter validates duration, truncates to 10s, and resamples to 24kHz.

- Uploaded voices:
  - Upload a sample to use later:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/audio/voices/upload \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "file=@/path/to/voice.wav" \
  -F "name=Frank" \
  -F "provider=vibevoice"
```

  - Use it by setting `voice` to `custom:<voice_id>` in subsequent TTS requests.

- Voices folder (`./voices`):
  - Drop files (e.g., `en-Alice_woman.wav`) in the folder. The adapter auto-detects and maps them as available voices.

## 7) Explicit Speaker-to-Voice Mapping

You can explicitly map speakers in a script to voice IDs (from `./voices` or uploaded) or direct file paths. Pass mapping via request `extra_params` or set a default in config.

Request example:

```json
{
  "model": "vibevoice",
  "input": "Speaker 1: Welcome!\nSpeaker 2: Great to be here.",
  "voice": "speaker_1",
  "response_format": "wav",
  "stream": true,
  "extra_params": {
    "speakers_to_voices": {
      "1": "en-Alice_woman",
      "2": "/abs/path/to/frank.wav"
    }
  }
}
```

Notes:
- Speaker IDs can be 0- or 1-based; the adapter normalizes them.
- If not enough voice samples exist for all speakers, the adapter disables cloning gracefully (still generates speech).

## 8) Python Client (httpx)

```python
import httpx

TOKEN = "YOUR_TOKEN"
url = "http://127.0.0.1:8000/api/v1/audio/speech"

payload = {
    "model": "vibevoice",
    "input": "Speaker 1: Welcome!\nSpeaker 2: Glad to join.",
    "voice": "speaker_1",
    "response_format": "wav",
    "stream": True,
}

with httpx.Client(timeout=None) as client:
    with client.stream("POST", url, headers={"Authorization": f"Bearer {TOKEN}"}, json=payload) as r:
        r.raise_for_status()
        with open("speech.wav", "wb") as f:
            for chunk in r.iter_bytes():
                if chunk:
                    f.write(chunk)
```

## 9) Tips for Quality & Performance

- Reference length: 3-10 seconds is ideal.
- Punctuation matters for clarity. Break long text into turns (multi-speaker) for more natural output.
- CUDA + FlashAttention + 4-bit quantization offers best throughput with moderate VRAM.
- Apple Silicon (MPS): works; quantization is disabled; uses SDPA attention fallback.

## 10) Troubleshooting

- "Required libraries not installed" on init:
  - Verify you installed `libs/VibeVoice` and ran `pip install -e .`.
  - Check `pip install -e ".[TTS_vibevoice]"` completed successfully.

- OOM / CUDA memory errors:
  - Set `vibevoice_use_quantization = true` (CUDA only) and/or reduce `vibevoice_variant` to `1.5B`.

- Model not downloaded:
  - Ensure `vibevoice_auto_download = true`, or pre-download with `hf download` and point `vibevoice_model_dir`.

- Streaming starts late:
  - Current adapter streams chunks after generation; for true model-level streaming (immediate playback), ask us to enable VibeVoice `AudioStreamer` integration in the adapter.

## 11) References

- Community repo (source models & demos):
  - `https://github.com/vibevoice-community/VibeVoice`
- Project installation & advanced tuning:
  - `Docs/STT-TTS/VIBEVOICE_INSTALLATION.md`
- Adapter configuration (speaker mapping, warmup):
  - `Docs/STT-TTS/TTS-SETUP-GUIDE.md`

## 12) Web UI Vignette (Click-Through)

Use the built-in Web UI to try VibeVoice quickly without writing code:

1. Start the API server and Web UI
   - API: `python -m uvicorn tldw_Server_API.app.main:app --reload`
   - Web UI: `http://localhost:8000/api/v1/config/quickstart`

2. Open Audio → Text-to-Speech
   - In the Web UI top tabs, click “Audio” and find the TTS request form for `POST /api/v1/audio/speech`.

3. Fill the request
   - Model: `vibevoice`
   - Input: e.g.
     ```
     Speaker 1: Welcome to our podcast.
     Speaker 2: Thanks for inviting me!
     ```
   - Voice: `speaker_1` (acts as primary/fallback)
   - Response format: `wav`
   - Stream: checked (to receive audio as it’s generated)

4. Optional: Voice cloning
   - Provide a short reference clip (3-10s) in the “voice_reference” field (base64) or use the “Voices” section to upload a sample; then set `voice` to `custom:<voice_id>`.
   - Alternatively, drop files into `./voices` and reference those IDs in `speakers_to_voices`.

5. Optional: Speaker mapping (Advanced)
   - In “extra_params JSON”, add:
     ```json
     {
       "speakers_to_voices": {
         "1": "en-Alice_woman",
         "2": "/abs/path/to/frank.wav"
       }
     }
     ```
   - This explicitly binds each speaker turn to a particular voice sample or file path.

6. Send the request
   - Click “Send Request”. Audio should begin downloading/streaming.
   - Check the Response panel for headers and content-type.

Tips
- If VibeVoice isn’t listed or errors on init, confirm installation steps and that models can download (or are pre-cached under `./models/vibevoice`).
- Use shorter scripts first; then scale up once the pipeline is working.
