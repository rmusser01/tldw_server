# Ingestion_Media_Processing

The ingestion and media processing hub for video, audio, PDF, EPUB, documents (txt/md/html/xml/docx/rtf), email, and MediaWiki dumps. Processors are DB‑agnostic: they validate inputs, extract content/metadata, optionally chunk and analyze, and return structured results. FastAPI endpoints handle auth, quotas/rate‑limits, and persistence.


## 1. Descriptive of Current Feature Set

- Purpose: Safe, pluggable ingestion for heterogeneous media; consistent outputs for downstream RAG, search, and storage.
- Capabilities:
  - Video/audio: yt‑dlp downloads; ffmpeg conversion; STT via faster‑whisper and NVIDIA NeMo/Parakeet; optional diarization/VAD; timestamps; batch processing.
  - PDFs: PyMuPDF/pymupdf4llm and Docling parsing; optional OCR (Tesseract, dots.ocr, POINTS-Reader, HunyuanOCR, Nemotron-Parse) and VLM table extraction; robust metadata extraction.
  - Documents: txt/md/html/xml/docx/rtf conversion (pandoc for RTF/DOCX); HTML/XML sanitized; consistent text output.
  - Books: EPUB extraction; TOC → Markdown; bulk ZIP of EPUBs.
  - MediaWiki: streaming dump parsing with NDJSON events; optional persistence and checkpointing.
  - Email: RFC822/mbox/PST/OST parsing; header/body extraction.
  - Claims extraction: optional ingestion‑time claims from chunks for downstream evaluation.
  - Streaming STT: real‑time WebSocket audio transcription.
- Inputs/Outputs:
  - Inputs: file uploads (multipart), URLs, bytes/paths for library calls.
  - Outputs: dict aligned to MediaItemProcessResponse: `status`, `input_ref`, `processing_source`, `media_type`, `metadata`, `content` or `segments`, `chunks`, `analysis`, `analysis_details`, `keywords`, `warnings`, `error`.
- Related Endpoints (examples):
  - `tldw_Server_API/app/api/v1/endpoints/media.py:7467` — POST `/api/v1/media/process-pdfs`
  - `tldw_Server_API/app/api/v1/endpoints/media.py:6892` — POST `/api/v1/media/process-documents`
  - `tldw_Server_API/app/api/v1/endpoints/media.py:5535` — POST `/api/v1/media/process-audios`
  - `tldw_Server_API/app/api/v1/endpoints/media.py:5073` — POST `/api/v1/media/process-videos`
  - `tldw_Server_API/app/api/v1/endpoints/media.py:6102` — POST `/api/v1/media/process-ebooks`
  - `tldw_Server_API/app/api/v1/endpoints/media.py:7951` — POST `/api/v1/media/mediawiki/process-dump` (ephemeral stream)
  - `tldw_Server_API/app/api/v1/endpoints/media.py:7875` — POST `/api/v1/media/mediawiki/ingest-dump` (persist)
  - `tldw_Server_API/app/api/v1/endpoints/audio.py:502` — POST `/api/v1/audio/transcriptions` (file STT)
  - `tldw_Server_API/app/api/v1/endpoints/audio.py:1196` — WS `/api/v1/audio/stream/transcribe` (real‑time STT)


## 2. Technical Details of Features

- Architecture & Data Flow:
  - Processors are pure functions where feasible; DB write boundaries are at the API layer. Media is validated → normalized → parsed → chunked (optional) → analyzed (optional) → result dict.
  - Chunking integrates with `app/core/Chunking` for consistent strategies and metadata.
  - OCR/VLM are pluggable via registries; backends are optional and auto‑detected.
- Key Classes/Functions (entry points):
  - `tldw_Server_API/app/core/Ingestion_Media_Processing/PDF/PDF_Processing_Lib.py:202` — `process_pdf(...)`
  - `tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py:264` — `process_document_content(...)`
  - `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py:325` — `process_audio_files(...)`
  - `tldw_Server_API/app/core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py:771` — `process_videos(...)`
  - `tldw_Server_API/app/core/Ingestion_Media_Processing/Books/Book_Processing_Lib.py:1` — EPUB helpers incl. `process_epub(...)`
  - `tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py:636` — `import_mediawiki_dump(...)` (yields progress/events)
- Schemas (requests/responses):
  - `tldw_Server_API/app/api/v1/schemas/media_request_models.py:1`
  - `tldw_Server_API/app/api/v1/schemas/media_response_models.py:1`
- Validation & Security:
  - Upload gate: `tldw_Server_API/app/core/Ingestion_Media_Processing/Upload_Sink.py:1` via `FileValidator` (puremagic/python‑magic fallback), MIME/extension allowlists, size caps, optional YARA scanning, safe ZIP/TAR extraction with nesting/size limits.
  - API dependency: `tldw_Server_API/app/api/v1/API_Deps/validations_deps.py:1` provides a singleton validator.
  - Hardened parsers for XML/HTML; executables/scripts blocked by default.
  - SSRF guard: outbound media URLs are checked against egress policy before download (audio/video + document-like URLs).
    In tests, hostname-based DNS checks are relaxed to avoid flaky resolution; literal IP checks remain enforced.
- Configuration:
  - Env vars: `MAGIC_FILE_PATH` (libmagic), `YARA_RULES_PATH` (malware rules); set via `.env`/env.
  - OCR backends (env): `OCR_PAGE_CONCURRENCY` and backend-specific settings. Nemotron-Parse uses:
    `NEMOTRON_MODE`, `NEMOTRON_PROMPT`, `NEMOTRON_VLLM_URL`, `NEMOTRON_VLLM_MODEL`,
    `NEMOTRON_VLLM_TIMEOUT`, `NEMOTRON_VLLM_USE_DATA_URL`, `NEMOTRON_MODEL_PATH`,
    `NEMOTRON_DEVICE`, `NEMOTRON_DTYPE`, `NEMOTRON_MAX_NEW_TOKENS`, `NEMOTRON_TEMPERATURE`,
    `NEMOTRON_REPETITION_PENALTY`, `NEMOTRON_TOP_P`, `NEMOTRON_TOP_K`, `NEMOTRON_DO_SAMPLE`,
    `NEMOTRON_SKIP_SPECIAL_TOKENS`, `NEMOTRON_TEXT_FORMAT`, `NEMOTRON_TABLE_FORMAT`,
    `NEMOTRON_KEEP_RAW_OUTPUT`, `NEMOTRON_RESIZE_MODE`, `NEMOTRON_POSTPROCESSOR_MODULE`.
    `NEMOTRON_SKIP_SPECIAL_TOKENS` defaults to false to preserve layout tags for structured output.
  - Request-level OCR params (per job): `ocr_output_format` and `ocr_prompt_preset` influence structured OCR output and prompt selection when supported by the backend; env defaults are used if these are omitted.
  - `loaded_config_data['media_processing']` keys honored by validator, e.g.: `max_audio_file_size_mb`, `max_video_file_size_mb`, `max_document_file_size_mb`, `max_archive_file_size_mb`, `max_archive_internal_files`, `max_archive_uncompressed_size_mb`, `max_archive_member_uncompressed_size_mb`, `max_archive_nesting_depth`, `yara_fail_open`.
  - Chunking defaults via `chunking_config` in config; see `app/core/Chunking/__init__.py:1`.
- Concurrency & Performance:
  - Batch processing for audio/video lists; temp directories for large artifacts; ffmpeg pipelines.
  - Chunking supports adaptive/multi‑level strategies; STT supports VAD/diarization.
  - Rate limits/quotas enforced at endpoints (RBAC scopes + limiter hooks).
- Error Handling:
  - Functions collect non‑fatal issues in `warnings`; fatal errors set `status='Error'` with `error` details. Endpoint maps exceptions to HTTP status with consistent messages.
- Security:
  - Media ingestion endpoints enforce claim-first `RequirePermission(MEDIA_CREATE)` + `rbac_rate_limit('media.create')`; API keys/JWTs required. Never accept client‑supplied provider API keys; providers are resolved server‑side.


## 3. Developer‑Related/Relevant Information for Contributors

- Folder Structure:
  - `Audio/`, `Video/`, `PDF/`, `Books/`, `Plaintext/`, `MediaWiki/`, `Email/`, `OCR/`, `VLM/`, `Claims/`, plus `Upload_Sink.py`, `Media_Update_lib.py`, `XML_Ingestion_Lib.py`.
- Extension Points:
  - Add a new pipeline folder with a `process_*` entry that returns the standard result dict.
  - Register new OCR/VLM backends under `OCR/backends` or `VLM/backends`; expose via registry.
  - If supporting new file types, extend allowlists in `Upload_Sink.py` and update `EXT_TO_MEDIA_TYPE_KEY`.
  - Wire a new endpoint in `tldw_Server_API/app/api/v1/endpoints/media.py` and add Pydantic models in schemas.
- Coding Patterns:
  - Keep processors DB‑agnostic; delegate persistence to API layer (`DB_Management`).
  - Use `loguru` via `app/core/Utils/Utils.py` helpers; return predictable result keys.
  - Respect config via `app/core/config.py`; do not log secrets.
- Tests:
  - Endpoint tests: `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py:354`, `.../test_add_media_endpoint.py:56` and related.
  - MediaWiki: `tldw_Server_API/tests/test_mediawiki_ephemeral_smoke.py:2`, `tldw_Server_API/tests/test_mediawiki_security.py:2`.
  - Use markers `unit`/`integration`; mock external downloads (yt‑dlp), STT/LLM calls.
- Local Dev Tips:
  - Install: `pip install -e .[dev]` and optional deps (`pypandoc`, `docling`, `yara`, `python-magic`, `puremagic`). Ensure `ffmpeg`, `yt-dlp`, optional `tesseract` are on PATH.
  - Start API: `python -m uvicorn tldw_Server_API.app.main:app --reload` (docs at /docs). Use `X-API-KEY` or JWT depending on `AUTH_MODE`.
  - Quick curls: see “Related Endpoints” and try `process-*` routes for ephemeral processing.
- Pitfalls & Gotchas:
  - Large archives: enforce nesting/uncompressed limits to avoid zip bombs; prefer ephemeral temp dirs.
  - OCR/Docling are optional; guard imports and provide fallbacks with clear warnings.
  - Never accept client API keys; select provider/model from server config only.
  - Chunk overlap must be < size; enforced by chunker validators.
- Follow-up candidates:
  - Consolidate duplicate parsing paths; unify analysis prompt profiles.
  - Expand email/PST parsing; add more robust HTML sanitization policies.
  - Improve adapter coverage for additional VLMs and OCR engines.

References
- `Docs/Code_Documentation/Ingestion_Media_Processing.md`
- `Docs/Code_Documentation/Ingestion_Pipeline_PDF.md`
- `Docs/Code_Documentation/Ingestion_Pipeline_Documents.md`
- `Docs/Code_Documentation/Ingestion_Pipeline_Audio.md`
- `Docs/Code_Documentation/Ingestion_Pipeline_Video.md`
- `Docs/Code_Documentation/Ingestion_Pipeline_Ebooks.md`
- `Docs/Code_Documentation/Ingestion_Pipeline_MediaWiki.md`

Example Library Usage
```python
# PDF (bytes → dict)
from pathlib import Path
from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf
data = Path('sample.pdf').read_bytes()
res = process_pdf(data, filename='sample.pdf', parser='pymupdf4llm', perform_chunking=True)
print(res['status'], len(res.get('chunks') or []))

# Documents (path → dict)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files import process_document_content
res = process_document_content(Path('sample.txt'), True, {'method':'recursive','max_size':1000,'overlap':200}, False, False, None, None, None, None)
print(res['status'], res['metadata'])

# Audio (batch)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files import process_audio_files
res = process_audio_files(inputs=['sample.mp3'], transcription_model='base.en', transcription_language='en', perform_analysis=False, perform_chunking=True)
print(res['processed_count'])

# Video (YouTube URL)
import tempfile
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib import process_videos
tmp = tempfile.mkdtemp(prefix='vid_')
out = process_videos(
    inputs=['https://www.youtube.com/watch?v=dQw4w9WgXcQ'],
    start_time=None,
    end_time=None,
    diarize=False,
    vad_use=False,
    transcription_model='small',
    transcription_language='en',
    perform_analysis=False,
    custom_prompt=None,
    system_prompt=None,
    perform_chunking=True,
    chunk_method='sentences',
    max_chunk_size=1000,
    chunk_overlap=200,
    use_adaptive_chunking=False,
    use_multi_level_chunking=False,
    chunk_language='en',
    summarize_recursively=False,
    api_name=None,
    use_cookies=False,
    cookies=None,
    timestamp_option=False,
    perform_confabulation_check=False,
    temp_dir=tmp,
)
print(out['processed_count'])
```
