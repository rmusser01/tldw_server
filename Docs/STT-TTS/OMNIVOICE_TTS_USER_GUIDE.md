# OmniVoice User Guide for tldw_server

This guide explains how to use OmniVoice after it has been installed and enabled in `tldw_server`.

## Overview

OmniVoice is exposed through the normal `POST /api/v1/audio/speech` route. You do not call the sidecar directly.

Public behavior in `tldw_server`:

- provider key: `omnivoice`
- output sample rate: `24000`
- native sidecar output: WAV
- public request route: `/api/v1/audio/speech`
- public voices:
  - `auto`
  - `clone`
  - `custom:<voice_id>`

If a request explicitly resolves to OmniVoice and omits `voice`, `tldw_server` normalizes it to `voice: "auto"`.

Operational shortcut:

- In the WebUI admin audio installer panel, use `Pre-download weights` to fill the model cache and `Warm up sidecar` to load OmniVoice before the first user-facing request.

## Supported Usage Patterns

OmniVoice currently supports these user-visible flows inside `tldw_server`:

- automatic voice selection
- voice design with `extra_params.instruct`
- direct voice cloning with `voice_reference`
- stored voice reuse with `voice: "custom:<voice_id>"`

Current limitation:

- non-streaming only in v1

## 1. Automatic Voice

This is the simplest request shape. You can omit `voice` entirely.

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/speech" \
  -H "X-API-KEY: ${SINGLE_USER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "omnivoice",
    "input": "This is an automatic OmniVoice sample.",
    "response_format": "wav",
    "stream": false
  }' --output omnivoice_auto.wav
```

If you run `tldw_server` in multi-user JWT mode, replace the `X-API-KEY` header with your normal `Authorization: Bearer <token>` header.

Equivalent explicit version:

```json
{
  "model": "omnivoice",
  "input": "This is an automatic OmniVoice sample.",
  "voice": "auto",
  "response_format": "wav",
  "stream": false
}
```

## 2. Voice Design

Voice design uses `extra_params.instruct`.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/speech" \
  -H "X-API-KEY: ${SINGLE_USER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "omnivoice",
    "input": "This is a voice design sample from OmniVoice.",
    "response_format": "wav",
    "stream": false,
    "extra_params": {
      "instruct": "female, british accent, low pitch",
      "duration": 4.0,
      "num_step": 8
    }
  }' --output omnivoice_design.wav
```

Important:

- OmniVoice validates `instruct` tokens.
- Unsupported items cause the request to fail.
- Example of an invalid token: `warm narrator`

### Supported English Instruct Items

These are the main English items you can combine:

- `female`
- `male`
- `child`
- `teenager`
- `young adult`
- `middle-aged`
- `elderly`
- `very low pitch`
- `low pitch`
- `moderate pitch`
- `high pitch`
- `very high pitch`
- `whisper`
- `american accent`
- `australian accent`
- `british accent`
- `canadian accent`
- `chinese accent`
- `indian accent`
- `japanese accent`
- `korean accent`
- `portuguese accent`
- `russian accent`

Formatting rules:

- use only English or only Chinese items in one prompt
- English items should use `, ` as the separator
- Chinese items should use full-width commas

## 3. Direct Voice Cloning

Direct cloning uses `voice_reference` audio in the public request. `reference_text` is optional, but recommended.

Example request shape:

```json
{
  "model": "omnivoice",
  "input": "This is a cloned voice sample.",
  "voice": "clone",
  "response_format": "wav",
  "stream": false,
  "voice_reference": "<base64 wav or mp3 bytes>",
  "extra_params": {
    "reference_text": "Transcript of the reference clip.",
    "duration": 3.0,
    "num_step": 8
  }
}
```

Practical notes:

- aim for roughly 3 to 10 seconds of reference audio
- `reference_text` improves predictability and avoids an extra auto-transcription step
- if you omit `reference_text`, OmniVoice may auto-transcribe the reference clip

## 4. Stored Voices with `custom:<voice_id>`

If you upload a voice through the normal voice manager endpoints, you can reuse it with OmniVoice using `custom:<voice_id>`.

### Upload a Stored OmniVoice Reference

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audio/voices/upload" \
  -H "X-API-KEY: ${SINGLE_USER_API_KEY}" \
  -F "name=OmniVoice Reference" \
  -F "provider=omnivoice" \
  -F "reference_text=This is the transcript of the stored reference clip." \
  -F "file=@/path/to/reference.wav"
```

Then use the returned voice id:

```json
{
  "model": "omnivoice",
  "input": "This sample uses a stored OmniVoice reference.",
  "voice": "custom:VOICE_ID",
  "response_format": "wav",
  "stream": false
}
```

`tldw_server` resolves the stored reference and injects the OmniVoice-compatible data internally.

## Generation Controls

OmniVoice-specific controls are passed under `extra_params`.

Commonly useful keys:

- `instruct`
- `reference_text`
- `duration`
- `speed`
- `num_step`
- `guidance_scale`
- `t_shift`
- `denoise`
- `postprocess_output`
- `preprocess_prompt`
- `layer_penalty_factor`
- `position_temperature`
- `class_temperature`
- `audio_chunk_duration`
- `audio_chunk_threshold`

Example:

```json
{
  "model": "omnivoice",
  "input": "Fine tune this sample.",
  "response_format": "wav",
  "stream": false,
  "extra_params": {
    "instruct": "female, british accent",
    "duration": 3.0,
    "speed": 1.05,
    "num_step": 16,
    "guidance_scale": 1.7,
    "denoise": true
  }
}
```

## Format and Response Notes

- OmniVoice sidecar output is native WAV.
- The public API still accepts the standard TTS request shape.
- For the safest first validation, use `response_format: "wav"` and `stream: false`.

## Recommended First-Use Flow

For a new OmniVoice user in `tldw_server`, the simplest progression is:

1. send an automatic voice request
2. send a voice design request with a valid `instruct`
3. upload a reference clip and reuse it as `custom:<voice_id>`
4. move to direct per-request cloning only if you need ad hoc references

## Known Caveats

- OmniVoice is not exposed as streaming TTS in the current integration.
- CPU inference works, but it is slower than GPU or Apple Silicon acceleration.
- Invalid `instruct` strings return explicit generation errors.
- The first request after a cold start can be noticeably slower.
- Reference text is optional for cloning, but recommended.

## Troubleshooting

### Request fails with unsupported instruct items

Cause:

- one or more `extra_params.instruct` tokens are not part of OmniVoice's supported attribute set

Fix:

- simplify the prompt
- use only supported attributes
- keep the prompt entirely English or entirely Chinese

### Clone request fails

Check:

- `voice_reference` is valid audio
- the reference clip is long enough
- `voice` is either `clone` or `custom:<voice_id>`

### OmniVoice feels slower than another local provider

That is expected in many CPU-only setups. OmniVoice is heavier than lightweight ONNX providers.

## Related Docs

- `Docs/STT-TTS/OMNIVOICE_TTS_SETUP.md`
- `Docs/STT-TTS/TTS-SETUP-GUIDE.md`
- `Docs/API-related/TTS_API.md`
