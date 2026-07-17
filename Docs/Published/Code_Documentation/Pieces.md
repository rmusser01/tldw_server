# Pieces (Semantic Units)

This page explains what “pieces” are in the current tldw_server architecture, where they are created, how they flow through the system, and which APIs and modules use them. It replaces legacy notes about a Gradio UI and `summarize.py` which are no longer part of the core.

Last updated: v0.1.0

## What Are “Pieces”?

- Pieces are the semantically meaningful units of text the system works with end-to-end. In code we usually call them “chunks,” “segments,” or “claims,” depending on the stage and granularity:
  - Transcript segments: time-aligned snippets from audio/video transcription (pre-chunking stage).
  - Chunks: text units produced by the Chunking module (words/sentences/paragraphs/tokens/semantic/templates/etc.). These are the primary “pieces” used for embeddings and RAG.
  - Claims/Propositions: concise statements extracted from chunks (used in evaluation and fine-grained retrieval/QA).

- These pieces carry metadata (e.g., offsets, method, type, language, provenance) to support search, embeddings, evaluations, RAG, and re-summarization.

## Where Pieces Come From

1) Chunking module (primary)
   - Location: `tldw_Server_API/app/core/Chunking/`
   - Entry points: `Chunker.process_text`, `improved_chunking_process`, strategy classes in `strategies/` and the template system.
   - Output: standardized list of chunk dicts or `ChunkResult` objects with text and metadata. See Code Docs: “Chunking Module” and “Chunking Templates”.

2) Audio/video transcription
   - Location: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/` and `.../Video/`
   - Functions: transcription and segmentation (e.g., Whisper models) yield segments which can be joined or further chunked.

3) Document parsing
   - PDF: `.../PDF/PDF_Processing_Lib.py` (PyMuPDF4LLM, PyMuPDF, optional OCR) → text → chunking
   - EPUB/Books: `.../Books/Book_Processing_Lib.py` (chapter/section splitters) → chunking
   - Plaintext/HTML/XML/RTF/DOCX: `.../Plaintext/Plaintext_Files.py` → normalization → chunking
   - Web pages: `core/Web_Scraping/Article_Extractor_Lib.py` → markdown text → chunking

4) Claims/Propositions
   - Location: `tldw_Server_API/app/core/RAG` and `core/Chunking/strategies/propositions.py` (plus services)
   - Created from chunks using heuristic, spaCy, or LLM-based extraction.

## How Pieces Flow Through the System

1) Ingestion (API) → Processing libraries → Chunking
   - Endpoints in `tldw_Server_API/app/api/v1/endpoints/media.py` accept uploads/URLs and route to the appropriate processor (audio, video, pdf, ebooks, documents, web content). Chunking can be enabled per request.

2) Optional analysis/summarization
   - `core/LLM_Calls/Summarization_General_Lib.analyze` can generate summaries over raw content or over chunked text (direct, chunked, or recursive options).

3) Persistence to the Media DB (when using /media/add or /media/mediawiki/ingest-dump)
   - Database: the package-native MediaDatabase layer stores versions and pieces:
     - `DocumentVersions`: versioned content + analysis
     - `UnvectorizedMediaChunks`: chunk text, indices, offsets, type, metadata
     - `MediaChunks`: legacy/simple chunk storage
     - `Claims`: proposition-level statements tied to chunk indices

4) Embeddings and RAG
   - Chunks become embedding units (on-demand) and are retrieved by the RAG service (`api/v1/rag/*`).

## Key APIs That Produce or Consume Pieces

- Media processing (no DB writes):
  - `POST /api/v1/media/process-audios` → transcribe → chunk → optional analysis
  - `POST /api/v1/media/process-videos` → download/convert → transcribe → chunk → analysis
  - `POST /api/v1/media/process-pdfs` → parse → (OCR optional) → chunk → analysis
  - `POST /api/v1/media/process-ebooks` → parse chapters → chunk → analysis
  - `POST /api/v1/media/process-documents` → normalize → chunk → analysis

- Media ingestion (with DB writes and versioning):
  - `POST /api/v1/media/add` → dispatch by type, chunk if enabled, persist content, chunks, keywords, and analysis.
  - `POST /api/v1/media/mediawiki/ingest-dump` → stream-processed pages, chunk and store.

- Chunking-only API:
  - `POST /api/v1/chunking/chunk_text` → returns chunked “pieces” with metadata; supports templates and multiple strategies.

- Audio transcription and streaming:
  - `POST /api/v1/audio/transcriptions` → file-based transcription; can return plain text, JSON, SRT/VTT
  - `WS  /api/v1/audio/stream/transcribe` → real-time streaming transcription

## Default Piece Metadata

Chunk outputs commonly include:
- `text`: chunk content
- `method`: chunking method used (e.g., words, sentences, paragraphs, tokens, semantic, propositions, template name)
- `start_char` / `end_char`: offsets in the source content
- `chunk_index` and/or `chunk_id`: stable order and identifiers
- `language`: when available or inferred
- `metadata`: strategy-specific details (e.g., paragraph type, template rules applied)

Transcript segment outputs commonly include:
- `Start`/`End` or timestamps, `Text`, diarization info (speaker), confidence scores (when model provides them)

Claims include:
- `media_id`, `chunk_index`, span offsets, `claim_text`, `confidence`, `extractor`, `extractor_version`, and a `chunk_hash` for traceability.

## Where to Look in the Code

- FastAPI entry point and router wiring: `tldw_Server_API/app/main.py`
- Media endpoints (ingestion/processing): `tldw_Server_API/app/api/v1/endpoints/media.py`
- Chunking endpoints: `tldw_Server_API/app/api/v1/endpoints/chunking.py`
- Audio APIs (file + WebSocket): `tldw_Server_API/app/api/v1/endpoints/audio.py`
- Chunking module (strategies, templates, helpers): `tldw_Server_API/app/core/Chunking/`
- LLM analysis/summarization: `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`
- Media DB package (schema incl. chunks, versions, claims): `tldw_Server_API/app/core/DB_Management/media_db/native_class.py`

## Related Documentation

- Chunking Module - Developer Guide
- Chunking Templates - Developer Guide
- Ingestion Pipeline docs (Audio, Video, PDF, Ebooks, Documents, MediaWiki)
- Embeddings Module and RAG API Guides
- Evaluations (Proposition extraction and scoring)

## Legacy Notes (Deprecated)

- The former Gradio UI and `summarize.py` flow are deprecated. Some installer scripts may still reference them for convenience, but the supported interface is the FastAPI server with an integrated Web UI served by `main.py`.
