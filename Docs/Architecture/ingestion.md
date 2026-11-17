# Media ingestion pipeline (server)

This document describes the server's ingestion surface: ad-hoc `/media/add`, the process-only siblings, persistent ingestion sources, OAuth external connectors, the per-type processors, the single persistence seam, and the deferred embeddings dispatch.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Orchestration | `app/core/Ingestion_Media_Processing/persistence.py` | `add_media_orchestrate` / `add_media_persist`, per-item processing, persistence, embeddings dispatch |
| Upload gate | `Ingestion_Media_Processing/Upload_Sink.py`, `input_sourcing.py`, `download_utils.py` | `FileValidator` (magic/MIME/extension/size/YARA/archive safety), temp-dir staging, URL fetch with caps |
| Chunk options | `Ingestion_Media_Processing/chunking_options.py` | form → chunk-options resolution, template auto-apply |
| Processors | `PDF/…`, `Plaintext/…`, `Audio/…`, `Video/…`, `Books/…`, `MediaWiki/…` | `process_pdf`, `process_document_content`, `process_audio_files`, `process_videos`, `import_mediawiki_dump` |
| Batch glue | `Ingestion_Media_Processing/pipeline.py` | `ProcessItem`, `run_batch_processor` |
| Persistence seam | `app/core/DB_Management/media_db/repositories/media_repository.py` | `add_media_with_keywords`, `_persist_chunks` — the only Media/chunks writer |
| Endpoints | `app/api/v1/endpoints/media/add.py`, `process_*.py` | `POST /media/add`; process-only siblings (no DB writes) |
| Persistent sources | `app/core/Ingestion_Sources/` + `app/services/ingestion_sources_{worker,scheduler}.py` | local dirs / archive snapshots / Git repos → media or notes sinks |
| External connectors | `app/core/External_Sources/` + `app/services/connectors_{worker,sync_scheduler}.py` | OAuth connectors (Drive/OneDrive/Notion/Gmail/Zotero) — the chatbook-synced subscriptions |
| Research handoff | `Ingestion_Media_Processing/research_discovery_handoff.py` | intercepts `/media/add` for discovered PDFs |

## `/media/add` (dataflow)

1. Validate inputs; enforce the LLM quota when analysis is requested.
2. Temp-dir staging; per-type extension allowlists; `FileValidator` on uploads (per-item error results; size-class errors 413; nonfatal validation → 207 multi-status).
3. Storage quota (413 on exceed) and the daily media-bytes ledger.
4. URL expansion for video/audio playlists.
5. Chunking-plan resolution (explicit options, `chunking_mode=auto` planner, or template classifier).
6. Dispatch: audio/video → batch media processing; doc-like → semaphore-bounded processing (concurrency default 10) with the SSRF guard `assert_url_safe` before every URL fetch, then download validation, then the type processor in an executor thread.
7. Persist: safe-metadata subset + normalization; re-chunk at the persist seam; `add_media_with_keywords` on a worker-thread DB session. AV items persist the transcript as content plus a normalized Transcripts upsert.
8. Optional original-file storage (sha256 checksum, superseded-original cleanup).
9. Collections dual-write; embeddings dispatch (below).
10. Final status: 200 / 207 partial / 400 / 413 (email containers with children force 200).

## Persistent and external sources

- **Ingestion_Sources**: CRUD at `/api/v1/ingestion-sources` (types `local_directory | archive_snapshot | git_repository`; sinks `media | notes`; policies `canonical | import_only`). A scheduler enqueues jobs (domain `ingestion_sources`, `max_retries=0`); the worker loads a source snapshot, diffs, and applies changes to the media or notes sink (canonical deletes honored). **Git sources support the notes sink only.**
- **External_Sources**: the `FileSyncAdapter` protocol (list/changes/metadata/download/shared-link/webhooks); reconciliation bootstraps into Media and deltas into DocumentVersions, archiving on upstream delete. Connector workers handle `import`, `incremental_sync`, `subscription_renewal`, `repair_rescan`. **Webhooks only validate, dedupe, and enqueue — never ingest inline.** These are the server side of the chatbook's watchlist/server sources.

## Server vs chatbook pipeline

The chatbook stamps every chunk with `chunk_engine_version` (`parity-1@…`); **the server has no such stamp anywhere** — parity is by shared `UnvectorizedMediaChunks` schema (version/prev_version/merge_parent_uuid/sync log), not version identity. The server parses in executor threads inside the request (chatbook uses parse worker processes), persists through `DB_Management` repository seams, and defers embeddings to Jobs/Redis stages (ADR-022).

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Upload validation failure | Per-item Error result; nonfatal classes → 207; size classes → 413 |
| SSRF block | HTTPException from the guard; counted in `security_ssrf_block_total` (test mode relaxes only DNS resolution failures) |
| Per-item processing exception | Gathered with `return_exceptions` — the item errors, others continue |
| Path escape from staging | Item skipped via the safe-path resolver |
| Embeddings enqueue failure | Warning appended; `auto` mode falls back to BackgroundTasks; `jobs` mode does not |
| Background embedding failure | `mark_embeddings_error` on the media row |
| Chunk count drift | A post-persist consistency check reconciles unvectorized chunk counts |

## Config keys (selection)

Env: `MEDIA_ADD_EMBEDDINGS_MODE` (`auto|jobs|background` + aliases), `DOCUMENT_LIKE_CONCURRENCY`, `MAGIC_FILE_PATH`, `YARA_RULES_PATH`, `OCR_PAGE_CONCURRENCY`, `INGESTION_SOURCES_JOBS_QUEUE`, connector keys (`CONNECTORS_SYNC_SCHEDULER_ENABLED/_SCAN_SEC`, `CONNECTOR_OAUTH_STATE_TTL_MINUTES`, …). config.txt `[media_processing]`: per-type max sizes, the archive family, `yara_fail_open` (false), `media_add_embeddings_mode`, `document_like_concurrency`. Form fields: `embedding_dispatch_mode`, `generate_embeddings`, chunk options, `keep_original_file`, `overwrite_existing`, email flags.

## Verified gotchas

1. Doc-like persist-time re-chunking uses its own defaults (`sentences`/500/50) when chunk options are present — persisted chunks can differ from the ephemeral `process-*` result chunks (see [chunking-embeddings.md](./chunking-embeddings.md) for the three-passes trap).
2. Bodyless email parents (attachment-only) persist as valid empty content.
3. Research-discovery handoff bypasses the normal persist shape entirely.
4. Process-only endpoints keep a legacy `urls=['']` sentinel and emit legacy-signal headers.
5. Archive uploads reject traversal/symlinks/encrypted ZIPs; local-directory snapshots skip symlinks and non-configured suffixes.

## Governing decisions

ADR-022 (`Docs/ADR/022-embeddings-api-and-media-pipeline.md`), ADR-026 (outbound egress/SSRF policy), ADR-064 (ingestion quotas), ADR-034 (durable sync for server-origin media). Current references: `Docs/Code_Documentation/Ingestion_Media_Processing.md`, the per-type pipeline docs (video/audio current; documents/ebooks/MediaWiki older but accurate on signatures).

## Related docs

- [chunking-embeddings.md](./chunking-embeddings.md), [rag.md](./rag.md)
- [database-layer.md](./database-layer.md) — the media store
- [jobs-background.md](./jobs-background.md) — source workers and schedulers
