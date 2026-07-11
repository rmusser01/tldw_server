# Video Ingestion Pipeline

## Overview

Downloads videos (yt-dlp) or uses local files, extracts audio (audio-only by default), then transcribes, optionally chunks the transcript, and runs analysis/summarization. Batch-oriented and DB-agnostic.

## Primary Functions

Module: `tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib`

- `process_videos(inputs, start_time, end_time, diarize, vad_use, transcription_model, transcription_language, perform_analysis, custom_prompt, system_prompt, perform_chunking, chunk_method, max_chunk_size, chunk_overlap, use_adaptive_chunking, use_multi_level_chunking, chunk_language, summarize_recursively, api_name, use_cookies, cookies, timestamp_option, perform_confabulation_check, temp_dir=None, keep_original=False, perform_diarization=False) -> Dict[str, Any]`
- `process_single_video(...) -> Dict[str, Any]` (internal worker)

### Parameters (selected)

- inputs: URLs or local paths.
- start_time/end_time: optional partial transcription windows.
- transcription_model/language: passed to STT backend.
- perform_chunking/analysis/summarize_recursively: chunk and summarize transcript.
- use_cookies/cookies: for authenticated downloads.
- timestamp_option: include timestamps in transcript.
- temp_dir: directory managed by caller for downloads/intermediates.

Notes:
- `temp_dir` is required by the library function; the API endpoint always supplies and manages it.
- Chunking is performed only when `perform_analysis=True` (library behavior).
- `start_time` is used as an offset; `end_time` is currently not applied in transcription.
- Diarization is controlled by `diarize`; the `perform_diarization` flag is currently unused.

### Return Structure (batch)

Same pattern as audio pipeline: `processed_count`, `errors_count`, `errors`, `results` (per item dict with transcript/chunks/analysis), and optional `confabulation_results`.

## Example

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib import process_videos

out = process_videos(
    inputs=["https://www.youtube.com/watch?v=...", "/abs/local/video.mp4"],
    start_time=None,
    end_time=None,
    diarize=False,
    vad_use=True,
    transcription_model="medium",
    transcription_language="en",
    perform_analysis=True,
    custom_prompt="Summarize as steps",
    system_prompt=None,
    perform_chunking=True,
    chunk_method="sentences",
    max_chunk_size=1000,
    chunk_overlap=150,
    use_adaptive_chunking=False,
    use_multi_level_chunking=False,
    chunk_language="en",
    summarize_recursively=False,
    api_name="openai",
    use_cookies=False,
    cookies=None,
    timestamp_option=True,
    perform_confabulation_check=False,
    temp_dir=None,
    keep_original=False,
    perform_diarization=False,
)
print(out["processed_count"], out["errors"])  # batch summary
```

## Endpoint Integration

- `POST /api/v1/media/process-videos` (modular endpoint in `endpoints/media/process_videos.py`) prepares uploads and URLs, then calls `video_batch.run_video_batch(...)`, which uses `process_videos` for the core work.
- Persistent video ingestion via `POST /api/v1/media/add` uses the shared `process_batch_media(...)` helper in `core.Ingestion_Media_Processing.persistence`, which wraps `process_videos` and calls `persist_primary_av_item(...)` to write results to the Media DB.

Endpoint specifics:
- Uses a managed temporary directory (`TempDirManager`) and passes its path to `process_videos`.
- Provider API keys are read from server configuration; the library calls do not require an `api_key` argument.
- Uploaded files are validated for allowed video types; remote URLs are downloaded via yt-dlp inside the library.
- Playlist URLs (e.g., YouTube playlists) are expanded server-side into per-video entries before processing begins.

## Dependencies & Config

- Requires `ffmpeg` and `yt-dlp`.
- Existing environments are not updated by pulling new code; update the active environment with `pip install -U "yt-dlp>=2026.7.4"`.
- Summarization provider is chosen via `api_name`; credentials come from server config.

## Error Handling & Notes

- Download failures, missing ffmpeg, or unsupported formats produce per-item errors.
- If `temp_dir` is not supplied by the caller, the endpoint creates and manages a temp directory (library requires it).
- Audio is extracted by yt-dlp (audio-only by default); ffmpeg must be available in PATH for post-processing.
- Chunking occurs only when analysis is requested; otherwise, no chunks are produced.
- Per-item results include: `content` (transcript), `segments` (with timestamps if requested), `chunks` (when chunking/analysis enabled), and `analysis` (summary).
- `start_time` and `end_time` accept integers or `HH:MM:SS(.sss)` values; invalid timestamps are rejected with a validation error instead of failing mid-run.
