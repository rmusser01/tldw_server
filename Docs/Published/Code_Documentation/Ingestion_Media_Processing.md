# Ingestion_Media_Processing Module Guide

This document describes the Ingestion_Media_Processing module: responsibilities, submodules, key functions, how FastAPI endpoints call into it, and security/validation behavior. It reflects the current code in `tldw_Server_API/app/core/Ingestion_Media_Processing/`.

## Overview

- Purpose: Ingest media from various sources (audio, video, PDF, EPUB, document/HTML/XML/RTF/DOCX, MediaWiki dumps), extract text and metadata, optionally chunk and analyze content, and return structured results. Most functions here are DB-agnostic; persistence is handled at the API layer.
- Output shape: Processing functions return dicts aligned with the API’s response schema: keys commonly include `status`, `input_ref`, `processing_source`, `media_type`, `metadata`, `content`, `segments/chunks`, `analysis`, `analysis_details`, `keywords`, `error`, `warnings`.
- Security: Upload validation, MIME checks, Yara scanning, size limits, and safe archive scanning are supported.
  - MIME detection in the validator uses `puremagic.from_file(..., mime=True)`; the API layer can optionally
    configure `python-magic` via `MAGIC_FILE_PATH`, but the upload validator itself does not depend on it.

## Deprecation Policy

Ingestion API changes follow a one-release compatibility window. A deprecated
field or endpoint will continue to function through the current release and is
eligible for removal in Release N+1. During that window, responses may include
the `Deprecation`, `Sunset`, and `Link` headers to communicate migration timing
and replacement guidance.

## Directory Structure

```
tldw_Server_API/app/core/Ingestion_Media_Processing/
├── Audio/                       # STT, streaming/live transcription, diarization, utilities
├── Books/                       # EPUB and related formats ingestion
├── Claims/                      # Claim extraction at ingestion time
├── MediaWiki/                   # MediaWiki XML dump processor
├── OCR/                         # OCR backend interface/registry
├── PDF/                         # PDF parsing/processing (with optional OCR)
├── Plaintext/                   # .txt/.md/.html/.xml/.docx/.rtf processing
├── Video/                       # Video/playlist download + transcription pipeline
├── Media_Update_lib.py          # Helpers for updating media records (DB-level)
├── Upload_Sink.py               # Secure upload validation/sanitization
└── XML_Ingestion_Lib.py         # Legacy XML import helper (DB-writing)
```

- Submodule docs and references:
  - `Audio/` ([docs](Docs/Code_Documentation/Ingestion_Pipeline_Audio.md))
  - `Books/` ([docs](Docs/Code_Documentation/Ingestion_Pipeline_Ebooks.md))
  - `Claims/` ([docs](Docs/Code_Documentation/Claims_Extraction.md))
  - `MediaWiki/` ([docs](Docs/Code_Documentation/Ingestion_Pipeline_MediaWiki.md))
  - `OCR/` ([docs](Docs/API-related/OCR_API_Documentation.md))
  - `PDF/` ([docs](Docs/Code_Documentation/Ingestion_Pipeline_PDF.md))
  - `Plaintext/` ([docs](Docs/Code_Documentation/Ingestion_Pipeline_Documents.md))
  - `Video/` ([docs](Docs/Code_Documentation/Ingestion_Pipeline_Video.md))
  - `Media_Update_lib.py` ([code](tldw_Server_API/app/core/Ingestion_Media_Processing/Media_Update_lib.py))
  - `Upload_Sink.py` ([code](tldw_Server_API/app/core/Ingestion_Media_Processing/Upload_Sink.py))
  - `XML_Ingestion_Lib.py` ([code](tldw_Server_API/app/core/Ingestion_Media_Processing/XML_Ingestion_Lib.py))

## Validation & Security: `Upload_Sink.py`

- Core types:
  - `ValidationResult`: result bag with `is_valid`, `issues`, `file_path`, `detected_mime_type`, `detected_extension`.
  - `FileValidator`: main validator with per-type config and Yara integration.
- MIME detection: Uses `puremagic.from_file(..., mime=True)`. If `puremagic` is unavailable, MIME checks are skipped; extensions and size are still enforced. The API layer may preconfigure `python-magic` via `MAGIC_FILE_PATH` (see `api/v1/API_Deps/validations_deps.py`), but `Upload_Sink.py` uses `puremagic` by default.
- Yara scanning: Optional; if `YARA_RULES_PATH` provided and `yara` installed, rules are compiled and used by `_scan_file_with_yara`.
- Per-type policy: Defaults in `DEFAULT_MEDIA_TYPE_CONFIG` (audio, video, image, document, ebook, pdf, html, xml, archive). Limits (e.g., `max_pdf_file_size_mb`) read from `loaded_config_data['media_processing']`.
- Extension → media key mapping is defined in `EXT_TO_MEDIA_TYPE_KEY`.
- Key functions:
  - `validate_file(path, original_filename, media_type_key, ...) -> ValidationResult`: existence, size, allowed extension (by claimed filename), MIME (if available), Yara results.
  - `validate_archive_contents(path) -> ValidationResult`: safe ZIP extraction in a temp dir with path-traversal checks, total member count/size limits, and per-member validation via `validate_file`.
  - `process_and_validate_file(path, validator, original_filename=None, media_type_key_override=None) -> ValidationResult`: dispatch by extension to proper media_type, archive scanning when configured.
- Sanitization is active by default:
  - `sanitize_html_content` removes dangerous blocks (`script`/`style`/`noscript`), cleans tags/attributes/protocols, and strips unsafe content.
  - `sanitize_xml_content` uses guarded parsing (`defusedxml`) and strips comments/processing instructions by default.
  - Upload sanitization defaults are controlled by `media_processing.sanitize_html_uploads` and `media_processing.sanitize_xml_uploads`.

Upload flow in endpoints
- Endpoints use `file_validator_instance` (see `api/v1/API_Deps/validations_deps.py`) which optionally configures `python-magic` via `MAGIC_FILE_PATH` and enables Yara via `YARA_RULES_PATH`.
- Uploaded files are saved to per-request temp dirs, validated, then dispatched to the appropriate processing library.

## FastAPI Endpoints (Media)
Base prefix: `/api/v1/media`
- `POST /add` - Ingest URLs/files, process via the core libraries, and persist to the package-native MediaDatabase-backed media store (with versioning, metadata, keywords).
- Processing-only (no DB writes):
  - `POST /process-videos`
  - `POST /process-audios`
  - `POST /process-documents`
  - `POST /process-pdfs`
  - `POST /process-ebooks`
  - `POST /process-emails`
- MediaWiki (streaming):
  - `POST /mediawiki/ingest-dump` - Process and persist; streams item events.
  - `POST /mediawiki/process-dump` - Process only; streams item events.
- Web content ingestion:
  - `POST /ingest-web-content` - Multi-mode scraping (individual, sitemap, url_level, recursive) with optional analysis/chunking and persistence.
  - `POST /process-web-scraping` - Process scraping jobs without persistence.

### Media Item Details

- `GET  /api/v1/media/{id}` - Retrieve rich media details
  - Query params:
    - `include_content` (bool, default: true): include main content text
    - `include_versions` (bool, default: true): include versions list
    - `include_version_content` (bool, default: false): include per-version content
  - Response: unified `MediaDetailResponse` (also used by PUT and POST version endpoints)

Notes
- API does not accept provider API keys in requests; credentials are read from server configuration.
- Audio/Video processing requires ffmpeg in PATH.
- Chunking uses the v2 chunker via `improved_chunking_process` (structure-aware/hierarchical templates supported).

## `/media/add` Persistence Helpers

The `/api/v1/media/add` endpoint no longer implements its pipeline directly
inside the legacy `_legacy_media.py` module. Instead it uses core helpers
under `tldw_Server_API.app.core.Ingestion_Media_Processing.persistence`:

- `add_media_persist(...)` – entry point used by `endpoints/media/add.py`; prepares request context and delegates into `add_media_orchestrate(...)`.
- `add_media_orchestrate(...)` – orchestrates TempDir management, upload saving, quota checks, per-type dispatch, and final status selection for `/media/add`.
- `process_batch_media(...)` – canonical batch helper for audio/video items; wraps `Video_DL_Ingestion_Lib.process_videos` and `Audio_Files.process_audio_files` and then calls `persist_primary_av_item(...)` to write results to the Media DB.
- `process_document_like_item(...)` – canonical helper for document-like items (PDFs, documents/HTML/XML/RTF/DOCX, ebooks, JSON, emails and email archives); performs download/validation, processor dispatch, and then calls `persist_doc_item_and_children(...)`.
- `persist_primary_av_item(...)` / `persist_doc_item_and_children(...)` – handle Media DB writes and ingestion-time claims persistence via `Claims_Extraction.claims_utils.persist_claims_if_applicable(...)`, keeping envelopes (`status`, `input_ref`, `processing_source`, `media_type`, `content`/`transcript`, `chunks`, `analysis`, `claims`, `db_id`, `db_message`, `media_uuid`) consistent across A/V and document flows.

The legacy `_legacy_media.py` file is now a compatibility shim that exposes
shared constants, Pydantic forms, and thin delegates for tests and older
imports; `/media` business logic lives in the core helpers and modular
endpoints under `api/v1/endpoints/media/`.

## Audio Ingestion: `Audio/`

- Primary orchestration: `Audio_Files.py`:
  - `process_audio_files(inputs, transcription_model, transcription_language='en', perform_chunking=True, chunk_method=None, max_chunk_size=500, chunk_overlap=200, use_adaptive_chunking=False, use_multi_level_chunking=False, chunk_language=None, diarize=False, vad_use=False, timestamp_option=True, perform_analysis=True, api_name=None, custom_prompt_input=None, system_prompt_input=None, summarize_recursively=False, use_cookies=False, cookies=None, keep_original=False, custom_title=None, author=None, temp_dir=None) -> Dict[str, Any]`
  - Pipeline (per input): download (if URL) → optional conversion → STT → optional chunking → optional analysis (LLM via `analyze`). Returns structured batch results; no DB writes.
  - STT engines/related utilities live in `Audio_Transcription_Lib.py` (faster_whisper + model mgmt), `Audio_Transcription_Nemo.py`/`Parakeet_*.py`/`Qwen2Audio`, and diarization in `Diarization_Lib.py`.
- Other notable modules: streaming/live transcription (`Audio_Streaming_Unified.py`, Parakeet/Nemo streaming), buffered capture.

Chunking integration
- All media processors call the unified chunker (`improved_chunking_process`). For structure-aware/hierarchical chunking, templates live in `app/core/Chunking/templates.py`. See the Chunking Module docs for strategies and options.

## Video Ingestion: `Video/Video_DL_Ingestion_Lib.py`

- Orchestration functions:
  - `process_videos(...) -> Dict[str, Any]`: handles URLs or local paths. Uses yt-dlp for download (`download_video`), then transcribes/segments/analyzes similar to audio. DB-agnostic.
  - `process_single_video(...) -> Dict[str, Any]`: worker for one input, used internally by `process_videos`.
- Helpers: metadata extraction, playlist expansion, timecode URL generation, cross-platform ffmpeg path resolution.

## PDF Processing: `PDF/PDF_Processing_Lib.py`

- Text extraction options:
  - `pymupdf4llm_parse_pdf(path)`: high-level markdown conversion.
  - `extract_text_and_format_from_pdf(path)`: page/block/span iteration via PyMuPDF with simple formatting heuristics.
  - `docling_parse_pdf(path)`: optional, if `docling` installed.
- OCR: `_ocr_pdf_pages(...)` uses `OCR/registry.py` to resolve an OCR backend (Tesseract CLI supported) and renders pages with PyMuPDF.
- Main processors:
  - `process_pdf(file_input: str|bytes|Path, filename, parser='pymupdf4llm', ..., enable_ocr=False, ocr_backend=None, ocr_lang='eng', ocr_dpi=300, ocr_mode='fallback', ocr_min_page_text_chars=40) -> Dict[str, Any]`: accepts bytes or path, writes bytes to a temp file for parsers that need a path, extracts content/metadata, optional chunking and analysis, returns result dict; no DB writes.
  - `async process_pdf_task(file_bytes, filename, ...) -> Dict[str, Any]`: async wrapper around `process_pdf` used in the API.
- Config: respects `media_processing.max_pdf_file_size_mb` and `pdf_conversion_timeout_seconds` from config.
- Metrics: emits counters/histograms via `metrics_logger`.

## EPUB and Books: `Books/Book_Processing_Lib.py`

- Extraction methods:
  - `'filtered'` (default): spine-based read with front-matter filtering.
  - `'markdown'`: full EPUB→Markdown (TOC + content).
  - `'basic'`: simple tag extraction fallback.
- Main processor:
  - `process_epub(file_path, title_override=None, author_override=None, keywords=None, custom_prompt=None, system_prompt=None, perform_chunking=True, chunk_options=None, perform_analysis=False, api_name=None, api_key=None, summarize_recursively=False, extraction_method='filtered') -> Dict[str, Any]`.
- Returns dict with extracted content, metadata (`title`, `author`, `raw`), optional chunks/analysis, warnings/errors.

## Documents/Markup: `Plaintext/Plaintext_Files.py`

- Conversion support: `.txt`, `.md`, `.html`, `.htm`, `.xml`, `.docx`, `.rtf` (Pandoc required for RTF). Uses BeautifulSoup/html2text and docx2txt where applicable.
- Key functions:
  - `convert_document_to_text(path: Path) -> Tuple[str, str, Dict[str, Any]]`: returns `(text, source_format, raw_metadata)`.
  - `process_document_content(doc_path: Path, perform_chunking, chunk_options, perform_analysis, summarize_recursively, api_name, api_key, custom_prompt, system_prompt, title_override=None, author_override=None, keywords=None) -> Dict[str, Any]`.
- Exceptions: `PandocMissing` for missing Pandoc on RTF paths; other conversion errors reported in `warnings`/`error`.

## MediaWiki Dumps: `MediaWiki/Media_Wiki.py`

- Evented pipeline:
  - `import_mediawiki_dump(file_path, wiki_name, namespaces=None, skip_redirects=False, chunk_options_override=None, progress_callback=None, store_to_db=True, store_to_vector_db=True, api_name_vector_db=None, api_key_vector_db=None) -> Iterator[Dict[str, Any]]`.
- Emits `progress_total`, `progress_item`, `item_result`, and final `summary` events. When `store_to_db=True`, it persists via a `MediaDatabase` instance (e.g., `db = create_media_database(...); db.add_media_with_keywords(...)`); vector store saving uses ChromaDBManager with embeddings configured from the MediaWiki embeddings settings, with `api_name_vector_db`/`api_key_vector_db` overriding the provider/model or API key as needed.
- Safety: filename/path validation, checkpointing (atomic save/cleanup), chunking (`optimized_chunking`).

## Claims Extraction: `Claims/`

- Runtime extraction on chunks:
  - `extract_claims_for_chunks(chunks, extractor_mode='heuristic', max_per_chunk=3) -> List[Dict]`: heuristic sentence snippets by default; LLM provider mode available via the unified chat API if configured.
  - `store_claims(db: MediaDatabase, media_id, chunk_texts_by_index, claims, extractor='heuristic', extractor_version='v1') -> int`: computes chunk hashes and upserts via `MediaDatabase.upsert_claims`.

## Legacy XML Helper: `XML_Ingestion_Lib.py`

- `xml_to_text(xml_file)` and `import_xml_handler(...)`: legacy code path that parses XML and immediately writes to DB via `add_media_with_keywords`. Newer API endpoints prefer using the DB-agnostic processors and then the API layer persists.

## How Endpoints Use This Module

From `app/api/v1/endpoints/media.py`:

- Ephemeral processing (no DB writes):
  - `POST /api/v1/media/process-audios` → `Audio_Files.process_audio_files`
  - `POST /api/v1/media/process-videos` → `Video_DL_Ingestion_Lib.process_videos`
  - `POST /api/v1/media/process-pdfs` → `PDF_Processing_Lib.process_pdf_task`
  - `POST /api/v1/media/process-ebooks` → `Books.process_epub`
  - `POST /api/v1/media/process-documents` → `Plaintext.process_document_content`
- MediaWiki:
  - `POST /api/v1/media/mediawiki/ingest-dump` → `MediaWiki.import_mediawiki_dump` with `store_to_db=True`.
  - `POST /api/v1/media/mediawiki/process-dump` → same iterator with `store_to_db=False` (ephemeral).
- Upload validation: endpoints use `file_validator_instance` from `api/v1/API_Deps/validations_deps.py`, which creates a `FileValidator` (optionally configuring libmagic via `MAGIC_FILE_PATH`, and Yara via `YARA_RULES_PATH`). Uploaded files are saved into temp dirs and validated before processing.

## Configuration & Dependencies

- Config keys (via `loaded_config_data['media_processing']` where applicable):
  - Size limits by type: `max_*_file_size_mb` (e.g., `max_pdf_file_size_mb`).
  - Archive limits:
    - `max_archive_internal_files`: maximum number of members scanned.
    - `max_archive_uncompressed_size_mb`: aggregate uncompressed size of all members (enforced in `validate_archive_contents`).
    - `max_archive_member_uncompressed_size_mb`: optional per-member uncompressed size cap (enforced in `validate_archive_contents`).
    - `max_archive_file_size_mb` (aka `archive_file_size_mb` default): compressed archive file size limit (applies to the uploaded `.zip`/`.tar*` file, enforced in `validate_file`).
    - `validate_email_archive_contents`: when true (default), URL-based email ZIP ingestion runs full archive-content validation parity.
  - Sanitization defaults:
    - `sanitize_html_uploads`: sanitize uploaded/downloaded HTML before processor execution (default: true).
    - `sanitize_xml_uploads`: sanitize uploaded/downloaded XML/SVG before processor execution (default: true).
    - `sanitize_email_html_bodies`: sanitize email `text/html` bodies before HTML-to-text conversion (default: true).
  - PDF conversion: `pdf_conversion_timeout_seconds`.
- External tooling: `ffmpeg` and `yt-dlp` required for A/V; optional `yara`, `puremagic` (MIME detection), `docling` (PDF), `pypandoc` (RTF), and system `tesseract` for OCR.

## Usage Examples

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator, process_and_validate_file

validator = FileValidator()  # Yara rules optional
res = process_and_validate_file("/tmp/upload.pdf", validator, original_filename="report.pdf")
if res:
    print("Valid!", res.detected_mime_type)
else:
    print("Invalid:", res.issues)
```

```python
# PDF (async wrapper used by API)
from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task

result = await process_pdf_task(file_bytes, filename="paper.pdf", parser="pymupdf4llm", perform_chunking=True)
print(result["status"], len(result.get("chunks") or []))
```

```python
# Audio batch (DB-agnostic)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files import process_audio_files

out = process_audio_files(
    inputs=["https://youtu.be/…"],
    transcription_model="large-v3",
    perform_analysis=True,
    api_name="openai",
)
print(out["processed_count"], out["errors"])
```

### Per Media Type Examples

#### Audio

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files import process_audio_files

result = process_audio_files(
    inputs=[
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",   # URL
        "/abs/path/to/local_audio.mp3",                  # Local file
    ],
    transcription_model="large-v3",
    transcription_language="en",
    diarize=False,
    vad_use=False,
    perform_chunking=True,
    chunk_method="sentences",
    max_chunk_size=1200,
    chunk_overlap=200,
    perform_analysis=True,
    api_name="openai",            # API key is read from server configuration
    custom_prompt_input="Summarize for a technical reader",
    system_prompt_input=None,
    summarize_recursively=True,
)

print(result["processed_count"], result["errors_count"])  # batch-level
for item in result.get("results", []):
    print(item["input_ref"], item["status"], len(item.get("chunks") or []))
```

#### Video

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib import process_videos

out = process_videos(
    inputs=["https://youtu.be/…", "/abs/path/to/local_video.mp4"],
    start_time=None,
    end_time=None,
    diarize=False,
    vad_use=True,
    transcription_model="medium",
    transcription_language="en",
    perform_analysis=True,
    custom_prompt="List 5 key takeaways",
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

#### PDF

```python
from pathlib import Path
from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf

pdf_bytes = Path("/abs/path/to/report.pdf").read_bytes()
res = process_pdf(
    file_input=pdf_bytes,
    filename="report.pdf",
    parser="pymupdf4llm",
    perform_chunking=True,
    chunk_options={"method": "recursive", "max_size": 1500, "overlap": 200},
    perform_analysis=True,
    api_name="openai",
    api_key=None,  # API keys are read from the server config; can be left None
    custom_prompt="Provide a concise executive summary",
    system_prompt=None,
    summarize_recursively=True,
    enable_ocr=False,
)
print(res["status"], len(res.get("chunks") or []))
```

#### EPUB / Books

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Books.Book_Processing_Lib import process_epub

book = process_epub(
    file_path="/abs/path/to/book.epub",
    title_override=None,
    author_override=None,
    keywords=["biology", "genetics"],
    perform_chunking=True,
    chunk_options={"method": "ebook_chapters", "max_size": 1500, "overlap": 200},
    perform_analysis=True,
    api_name="openai",
    summarize_recursively=True,
    extraction_method="filtered",  # or "markdown" | "basic"
)
print(book["status"], book.get("metadata", {}).get("title"))
```

#### Documents / Markup

```python
from pathlib import Path
from tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files import process_document_content

doc_path = Path("/abs/path/to/article.docx")
doc_res = process_document_content(
    doc_path=doc_path,
    perform_chunking=True,
    chunk_options={"method": "sentences", "max_size": 1200, "overlap": 200},
    perform_analysis=True,
    summarize_recursively=False,
    api_name="openai",
    api_key=None,
    custom_prompt="Summarize in bullet points",
    system_prompt=None,
    title_override=None,
    author_override=None,
    keywords=["policy", "research"],
)
print(doc_res["status"], doc_res["source_format"])  # e.g., docx, html, rtf
```

#### MediaWiki (Evented)

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.MediaWiki.Media_Wiki import import_mediawiki_dump

events = import_mediawiki_dump(
    file_path="/abs/path/to/enwiki-latest-pages-articles.xml.bz2",
    wiki_name="enwiki",
    namespaces=[0],            # article namespace
    skip_redirects=True,
    chunk_options_override={"method": "recursive", "max_size": 1500, "overlap": 200},
    store_to_db=False,         # set True to persist via a MediaDatabase instance
    store_to_vector_db=False,
)

for ev in events:
    if ev.get("type") == "item_result":
        page = ev.get("data", {})
        print(page.get("title"), page.get("status"))
```

## Notes & Limitations

- HTML/XML sanitization in `Upload_Sink.py` are placeholders (return original content with a warning).
- MediaWiki vector store saving is implemented via ChromaDBManager; it requires embeddings configuration, and when embeddings are unavailable vector store saving is skipped with a warning while the rest of ingestion proceeds.
- `XML_Ingestion_Lib.py` follows an older pattern that writes to DB directly; newer endpoints prefer DB-agnostic processors.
- Some modules rely on optional dependencies; functions degrade gracefully with warnings when a dependency is absent.
- Ingestion-time claim extraction is available and wired in the embeddings pipeline (see `ChromaDB_Library.py`) behind `ENABLE_INGESTION_CLAIMS`; it is not run for every add-media path by default.

---

Maintainers: keep this page aligned with the code. If you add new formats or alter return shapes, update both function docstrings and this guide.

## Further Reading

- Audio Pipeline: `./Ingestion_Pipeline_Audio.md`
- Video Pipeline: `./Ingestion_Pipeline_Video.md`
- PDF Pipeline: `./Ingestion_Pipeline_PDF.md`
- EPUB Pipeline: `./Ingestion_Pipeline_Ebooks.md`
- Documents Pipeline: `./Ingestion_Pipeline_Documents.md`
- MediaWiki Pipeline: `./Ingestion_Pipeline_MediaWiki.md`
 - Chunking Module: `../Chunking-Module.md`
 - Claims design: `../Design/ingestion_claims.md`
