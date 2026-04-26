# Audio Ingestion Pipeline

## Overview

Transcribes audio inputs (URLs or local files), optionally chunks text, and runs analysis/summarization. Batch-oriented, DB-agnostic; returns structured results for each input.

## Primary Function

`tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files.process_audio_files`

Signature (abbreviated):

```
process_audio_files(
  inputs: List[str],
  transcription_model: str,
  transcription_language: Optional[str] = 'en',
  perform_chunking: bool = True,
  chunk_method: Optional[str] = None,
  max_chunk_size: int = 500,
  chunk_overlap: int = 200,
  use_adaptive_chunking: bool = False,
  use_multi_level_chunking: bool = False,
  chunk_language: Optional[str] = None,
  diarize: bool = False,
  vad_use: bool = False,
  timestamp_option: bool = True,
  perform_analysis: bool = True,
  api_name: Optional[str] = None,
  custom_prompt_input: Optional[str] = None,
  system_prompt_input: Optional[str] = None,
  summarize_recursively: bool = False,
  use_cookies: bool = False,
  cookies: Optional[str] = None,
  keep_original: bool = False,
  custom_title: Optional[str] = None,
  author: Optional[str] = None,
  temp_dir: Optional[str] = None,
) -> Dict[str, Any]
```

### Parameters

 - inputs: URLs (including YouTube) or absolute local paths to audio files.
 - transcription_model: Supports multiple providers via naming scheme (parsed via the same logic as the HTTP `/audio/transcriptions` endpoint):
   - faster-whisper models (e.g., `base`, `medium`, `large-v3`, `distil-whisper-large-v3`, a local path, or HF hub id)
   - NVIDIA NeMo Parakeet: `parakeet`, `parakeet-standard`, `parakeet-tdt-0.6b-v3-onnx`, `parakeet-onnx` (legacy alias), `parakeet-mlx`, or any string where `parse_transcription_model` resolves the provider to `"parakeet"`
   - NVIDIA NeMo Canary: `canary` or related aliases (provider `"canary"`)
   - Qwen2Audio: `qwen2audio` or `qwen2audio-*` (provider `"qwen2audio"`)
 - transcription_language: target/source language (default `en`). When `None`, the underlying STT provider is allowed to auto-detect; this matches the behavior of `speech_to_text(..., selected_source_lang=None)` and the `/audio/transcriptions` endpoint.
 - diarize: enable speaker diarization; `vad_use`: enable voice activity detection.
 - perform_chunking: chunk transcript; `chunk_method`: e.g., `sentences`.
 - perform_analysis: use LLM summarization via `analyze`; `api_name` selects provider (keys from server config).
 - summarize_recursively: combine per-chunk summaries into a higher-level summary.
 - temp_dir: parent directory for temporary work files.

 Tip: To check if a model is ready/downloaded before processing, use `check_transcription_model_status(model_name)` from the same module. `process_audio_files` performs a preflight check for Whisper models and surfaces the status as a warning in each item result.

 Performance notes:
 - `convert_to_wav(...)` performs a preflight validation using `ffprobe` by default (via `validate_audio_file(...)`) and then runs the actual `ffmpeg` conversion. In high-throughput deployments you can disable this pre-validation by setting either:
   - `[STT-Settings] skip_audio_prevalidation = true` in `config.txt`, or
   - `STT_SKIP_AUDIO_PREVALIDATION=1` in the environment.
   In that mode, invalid files are left for `ffmpeg` to handle directly.
 - Transcript cache pruning (`prune_transcript_cache(...)`) runs inline after successful STT when transcript persistence is enabled. To avoid any pruning work on the hot path (for example, when you manage cache directories via an external job), set:
   - `[STT-Settings] disable_transcript_cache_pruning = true`, or
   - `STT_DISABLE_TRANSCRIPT_CACHE_PRUNING=1`.
   The server will continue to write transcripts but skip age/size-based cleanup.
   When pruning is enabled and no explicit limits are set, the server applies
   conservative defaults (max age ≈ 30 days, total cache size ≈ 512MB, and
   up to ≈ 32 transcript files per source). These defaults can be tuned via
   `[STT-Settings] transcript_cache_max_age_days`, `transcript_cache_max_total_mb`,
   and `transcript_cache_max_files_per_source`.

Canonical STT entrypoints:
- `speech_to_text(...)` (in `Audio_Transcription_Lib.py`) is the file/segment-based helper used by ingestion. It returns a list of segment dicts (or `(segments, language)` when `return_language=True`) and is responsible for transcript caching and pruning.
- `transcribe_audio(...)` (in `Audio_Transcription_Lib.py`) is the waveform-based sink used when you already have NumPy audio (for example, in speech-chat or streaming sinks). It routes to the appropriate provider and returns a single transcript string. Errors from providers are returned as sentinel strings such as `"[Transcription error] ..."`; callers should detect these via `is_transcription_error_message(...)` and convert them into structured errors instead of treating them as user content.

### Return Structure (batch)

```
{
  "processed_count": int,
  "errors_count": int,
  "errors": List[str],
  "results": [
    {
      "status": "Success"|"Warning"|"Error",
      "input_ref": str,                 # original URL/path
      "processing_source": str,         # local path actually processed
      "media_type": "audio",
      "metadata": dict,
      "content": str,                   # transcript text (optionally with timestamps)
      "segments": Optional[List[Dict]], # typical keys: start_seconds, end_seconds, Text, optional words[]
      "chunks": Optional[List[Dict]],
      "analysis": Optional[str],
      "analysis_details": Dict,
      "error": Optional[str],
      "warnings": Optional[List[str]]
    }, ...
  ]
}
```

## Example

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files import process_audio_files

res = process_audio_files(
    inputs=["https://soundcloud.com/...", "/abs/path/audio.wav"],
    transcription_model="large-v3",
    transcription_language="en",
    diarize=False,
    vad_use=True,
    perform_chunking=True,
    chunk_method="sentences",
    max_chunk_size=1200,
    chunk_overlap=200,
    perform_analysis=True,
    api_name="openai",
    summarize_recursively=True,
)
print(res["processed_count"], res["errors_count"])  # batch summary
for item in res.get("results", []):
    print(item["input_ref"], item["status"], len(item.get("chunks") or []))
```

## Endpoint Integration

- `POST /api/v1/media/process-audios` (modular endpoint in `endpoints/media/process_audios.py`) adapts form data, saves validated uploads, and calls `audio_batch.run_audio_batch(...)`, which in turn uses `process_audio_files`.
- Persistent audio ingestion via `POST /api/v1/media/add` uses the shared `process_batch_media(...)` helper in `core.Ingestion_Media_Processing.persistence`, which wraps `process_audio_files` and calls `persist_primary_av_item(...)` to write results to the Media DB.

### Endpoint Examples

- Auth headers
  - Single-user: add `X-API-KEY: <your_key>`
  - Multi-user: add `Authorization: Bearer <jwt>`

- URLs only (multipart form):

```
curl -X POST "http://127.0.0.1:8000/api/v1/media/process-audios" \
  -H "X-API-KEY: $API_KEY" \
  -F "urls=https://example.com/audio1.mp3" \
  -F "urls=https://soundcloud.com/user/track" \
  -F "transcription_model=distil-whisper-large-v3" \
  -F "transcription_language=en" \
  -F "perform_chunking=true" \
  -F "perform_analysis=true" \
  -F "api_name=openai"
```

- File uploads (multipart form):

```
curl -X POST "http://127.0.0.1:8000/api/v1/media/process-audios" \
  -H "Authorization: Bearer $JWT" \
  -F "files=@/abs/path/audio.wav" \
  -F "files=@/abs/path/audio2.m4a" \
  -F "transcription_model=deepdml/faster-distil-whisper-large-v3.5" \
  -F "vad_use=true" \
  -F "diarize=false" \
  -F "timestamp_option=true"
```

- Python (requests):

```python
import requests

url = "http://127.0.0.1:8000/api/v1/media/process-audios"
headers = {"X-API-KEY": "<api-key>"}
data = {
    "urls": ["https://example.com/audio.mp3"],
    "transcription_model": "distil-whisper-large-v3",
    "perform_analysis": True,
    "api_name": "openai",
}
files = []  # e.g., [("files", ("local.wav", open("/abs/path/local.wav","rb"), "audio/wav"))]
resp = requests.post(url, headers=headers, data=data, files=files)
print(resp.status_code)
print(resp.json())
```

Notes:
- Returns 200 when all items succeed, 207 for mixed outcomes, or 400 if nothing was processed.
- `transcription_model` must be one of the allowed values in the OpenAPI (see `TranscriptionModel` in schemas) or it will fallback to a default.

### OpenAPI (minimal)

```yaml
openapi: 3.0.3
paths:
  /api/v1/media/process-audios:
    post:
      summary: Transcribe / chunk / analyse audio and return full artefacts (no DB write)
      tags: ["Media Processing (No DB)"]
      requestBody:
        required: false
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                urls:
                  type: array
                  items: { type: string, format: uri }
                files:
                  type: array
                  items: { type: string, format: binary }
                transcription_model: { type: string }
                transcription_language: { type: string }
                diarize: { type: boolean }
                vad_use: { type: boolean }
                timestamp_option: { type: boolean }
                perform_chunking: { type: boolean }
                perform_analysis: { type: boolean }
                api_name: { type: string }
                summarize_recursively: { type: boolean }
                use_cookies: { type: boolean }
                cookies: { type: string }
      responses:
        "200": { description: OK }
        "207": { description: Multi-Status (mixed outcomes) }
        "400": { description: Bad Request }
        "422": { description: Validation Error }
```

### Response Example

```json
{
  "processed_count": 1,
  "errors_count": 1,
  "errors": [
    "Download failed for https://example.com/bad.mp3. Reason: 404"
  ],
  "results": [
    {
      "status": "Success",
      "input_ref": "local_audio.wav",
      "processing_source": "/tmp/process_audio_abc123/local_audio.wav",
      "media_type": "audio",
      "metadata": {"title": "Local Audio", "author": null},
      "content": "[00:00:00-00:00:04] This text was transcribed using whisper model: distil-whisper-large-v3\nDetected language: en\n\nHello and welcome...",
      "segments": [
        {"start_seconds": 0.0, "end_seconds": 2.1, "Text": "Hello and welcome"},
        {"start_seconds": 2.1, "end_seconds": 4.0, "Text": "to the sample recording"}
      ],
      "chunks": [
        {"index": 0, "text": "Hello and welcome to the sample recording", "start": 0, "end": 1200}
      ],
      "analysis": "This recording greets the listener and introduces a sample.",
      "analysis_details": {"analysis_model": "openai"},
      "error": null,
      "warnings": null,
      "db_id": null,
      "db_message": "Processing only endpoint.
",
      "message": null
    },
    {
      "status": "Error",
      "input_ref": "https://example.com/bad.mp3",
      "processing_source": "https://example.com/bad.mp3",
      "media_type": "audio",
      "metadata": {},
      "content": "",
      "segments": null,
      "chunks": null,
      "analysis": null,
      "analysis_details": {},
      "error": "Download failed for https://example.com/bad.mp3. Reason: 404",
      "warnings": null,
      "db_id": null,
      "db_message": "Processing only endpoint.",
      "message": "Invalid processing result."
    }
  ]
}
```

## Dependencies & Config

- Requires `ffmpeg` for conversion/transcoding.
- Uses `yt-dlp` to handle YouTube URLs.
- Uses `requests` for direct HTTP downloads (supports cookie-based sessions).
- Summarization uses providers configured in server config; API keys are not passed to this function.
- Chunking uses `tldw_Server_API.app.core.Chunking` utilities.

## Error Handling & Notes

- Download errors, file size limits, cookie format issues, or conversion problems are mapped into per-item `status` and `error`.
- `results` may include structured warnings even when `status` is `Success`.
- Keep temp files by setting `keep_original=True` in the caller-managed temp dir.

### Timestamps
- When `timestamp_option=True`, the `content` string includes `HH:MM:SS-HH:MM:SS` prefixes per segment line.

### Cookies
- For sites requiring authentication, pass `use_cookies=True` and `cookies` as a JSON string or dict; invalid formats yield a clear per-item error.
