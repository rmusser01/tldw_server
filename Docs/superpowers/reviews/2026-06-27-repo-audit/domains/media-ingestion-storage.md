# Media, Ingestion, And Storage Domain Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: Media, Ingestion, and Storage
- In scope: media endpoints, ingestion pipelines, upload/download flows, file/path handling, archive/document parsing, generated artifacts, quotas, and storage tests.
- Out of scope: remediation implementation and unrelated UI polish.

## Findings Table

| ID | Candidate ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUDIT-2026-06-27-MEDIA-001 | CANDIDATE-media-ingestion-storage-001 | confirmed_issue | static_confirmed | high | high | security | Processing-only media endpoints bypass the media.create permission gate | open | validated |
| AUDIT-2026-06-27-MEDIA-002 | CANDIDATE-media-ingestion-storage-002 | confirmed_issue | static_confirmed | high | high | security | MediaWiki ingest persists into shared single-user content and vector namespaces | open | validated |
| AUDIT-2026-06-27-MEDIA-003 | CANDIDATE-media-ingestion-storage-003 | likely_risk | static_confirmed | medium | high | data_durability | Original file storage can orphan permanent files when MediaFiles row insertion fails | open | needs_reproduction |
| AUDIT-2026-06-27-MEDIA-004 | CANDIDATE-media-ingestion-storage-004 | improvement_opportunity | static_confirmed | low | high | test_gap | Header-declared oversized audio downloads are not covered because the regression test is a no-op | open | validated |

## Index Mapping

Use finding IDs like `AUDIT-2026-06-27-MEDIA-001`. Set `evidence_tier` from the report section bucket (`confirmed_issue`, `likely_risk`, or `improvement_opportunity`) and `evidence_strength` from the schema allowed values. Set `source_report` to `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md`, set `owner_domain` to this report owner, and include `affected_paths`, `recommendation`, `status`, and `validation_status` in each detailed finding.

## Confirmed Issues

### AUDIT-2026-06-27-MEDIA-001 / CANDIDATE-media-ingestion-storage-001

Processing-only media endpoints bypass the `media.create` permission gate.

- severity: high
- confidence: high
- category: security
- evidence_tier: confirmed_issue
- evidence_strength: static_confirmed
- status: open
- validation_status: validated
- owner_domain: Media, Ingestion, and Storage
- source_report: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md`
- affected_paths: `tldw_Server_API/app/api/v1/endpoints/media/process_audios.py`, `process_documents.py`, `process_pdfs.py`, `process_ebooks.py`, `process_code.py`, `process_emails.py`, `process_mediawiki.py`, relevant endpoint permission tests.

The DB dependency authenticates callers through `get_request_user` (`DB_Deps.py:266-279`), but authentication alone is not the same as the media creation entitlement used by other write-like media routes. The persistent `/media/add` route requires `RequirePermission(MEDIA_CREATE)` and `rbac_rate_limit("media.create")` (`add.py:26-35`), and comparable ingestion routes do the same (`process_videos.py:60-70`, `process_web_scraping.py:48-55`, `ingest_jobs.py:567-581`).

Several sibling processing endpoints omit that permission and RBAC rate-limit dependency while still accepting uploads, downloaded URLs, parsing, transcription, chunking, and analysis work: `/process-audios` (`process_audios.py:62-81`), `/process-documents` (`process_documents.py:71-89`), `/process-pdfs` (`process_pdfs.py:73-102`), `/process-ebooks` (`process_ebooks.py:136-151`), `/process-code` (`process_code.py:43-61`), `/process-emails` (`process_emails.py:40-49`), and both MediaWiki dump endpoints (`process_mediawiki.py:238-305`). `process_emails.py` also lacks the storage/API billing pre-checks present on the other processing endpoints (`process_emails.py:40-45`).

This creates an authorization boundary mismatch in multi-user/RBAC mode: a principal with a valid session but without `MEDIA_CREATE` can still invoke expensive media processing surfaces. The current tests include permission-denial coverage for media ingest jobs (`test_media_ingest_jobs_endpoint.py:354-381`), but searches over the relevant media processing tests did not find equivalent no-permission denial coverage for these processing-only endpoints.

Recommendation: introduce a shared dependency bundle for all media ingestion and processing routes, or define a separate explicit `media.process` permission if these endpoints intentionally require a different entitlement. Add regression tests that override the principal with no media permissions and assert 403 for each processing endpoint that accepts user media or remote input.

### AUDIT-2026-06-27-MEDIA-002 / CANDIDATE-media-ingestion-storage-002

MediaWiki ingest persists into shared single-user content and vector namespaces.

- severity: high
- confidence: high
- category: security
- evidence_tier: confirmed_issue
- evidence_strength: static_confirmed
- status: open
- validation_status: validated
- owner_domain: Media, Ingestion, and Storage
- source_report: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md`
- affected_paths: `tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py`, `tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py`, `tldw_Server_API/app/core/DB_Management/media_db/api.py`, `tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py`, `tldw_Server_API/app/core/DB_Management/media_db/runtime/factory.py`, `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_db_persistence.py`.

`/mediawiki/ingest-dump` streams uploaded dumps and calls `core_import_mediawiki_dump` with `store_to_db=True` and `store_to_vector_db=True` (`process_mediawiki.py:238-270`). The endpoint does not request `get_media_db_for_user`, does not take `current_user`, and does not pass a user-scoped DB/session into the core importer (`process_mediawiki.py:251-270`).

The core importer creates a DB handle with `managed_media_database(client_id="mediawiki_import")` (`Media_Wiki.py:977-983`), and item-level fallback does the same (`Media_Wiki.py:750-763`). `managed_media_database` calls `create_media_database` without a `db_path` (`api.py:110-127`), and the runtime factory falls back to `runtime.default_db_path` (`factory.py:52-78`). That default is `DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id())` when no configured SQLite content path exists (`defaults.py:41-47`, `defaults.py:169-177`). The MediaWiki vector path similarly uses `SINGLE_USER_FIXED_ID` instead of the request user when constructing `ChromaDBManager` (`Media_Wiki.py:625-630`).

The existing MediaWiki persistence tests assert the current behavior by expecting `managed_media_database(client_id="mediawiki_import")` with empty kwargs (`test_mediawiki_db_persistence.py:72-78`, `test_mediawiki_db_persistence.py:183-189`), but they do not verify multi-user isolation. In multi-user mode, this can place one user's ingested pages and embeddings into a shared/single-user namespace rather than the caller's content DB and vector collection, which is both a data separation and data discoverability issue.

Recommendation: refactor the MediaWiki endpoint to require `current_user` and `get_media_db_for_user`, then thread a request-scoped media writer and vector user ID into `import_mediawiki_dump` and `_store_mediawiki_chunks_in_vector_db`. Add multi-user tests proving user A's MediaWiki ingest is not visible in user B's media DB or vector namespace.

## Likely Risks

### AUDIT-2026-06-27-MEDIA-003 / CANDIDATE-media-ingestion-storage-003

Original file storage can orphan permanent files when MediaFiles row insertion fails.

- severity: medium
- confidence: high
- category: data_durability
- evidence_tier: likely_risk
- evidence_strength: static_confirmed
- status: open
- validation_status: needs_reproduction
- owner_domain: Media, Ingestion, and Storage
- source_report: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md`
- affected_paths: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`, `tldw_Server_API/app/core/Storage/filesystem_storage.py`, `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py`.

When `keep_original_file=True` for PDFs, documents, or ebooks, persistence stores the uploaded original into permanent storage first (`persistence.py:3207-3214`) and then inserts the `MediaFiles` row (`persistence.py:3221-3230`). Database errors are included in `_PERSISTENCE_NONCRITICAL_EXCEPTIONS` (`persistence.py:77-99`), and the catch block marks the item as not stored without compensating deletion (`persistence.py:3235-3239`). The filesystem backend already exposes `delete()` (`filesystem_storage.py:233-260`), but this path does not call it if DB registration fails after `storage.store()` succeeds.

The current regression test covers the happy path and asserts that two uploaded originals are stored and two DB rows are inserted (`test_persistence_original_storage.py:173-239`). It does not simulate `db.insert_media_file` raising after storage succeeds. If that failure happens in production, the API response reports original storage failure but the permanent blob remains without a DB row, so normal file APIs cannot retrieve or clean it up. It also consumes disk/quota outside the media metadata lifecycle.

Recommendation: make the storage-plus-registration sequence compensating. If `insert_media_file` fails after `storage.store()` returns, call `await storage.delete(storage_path)` and include cleanup failures in the warning/log path. Add a unit test with a fake storage backend and a DB double whose `insert_media_file` raises, asserting the stored path is deleted.

## Improvement Opportunities

### AUDIT-2026-06-27-MEDIA-004 / CANDIDATE-media-ingestion-storage-004

Header-declared oversized audio downloads are not covered because the regression test is a no-op.

- severity: low
- confidence: high
- category: test_gap
- evidence_tier: improvement_opportunity
- evidence_strength: static_confirmed
- status: open
- validation_status: validated
- owner_domain: Media, Ingestion, and Storage
- source_report: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md`
- affected_paths: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_download_limits.py`, `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py`.

`download_audio_file` has a fail-fast `Content-Length` size check (`Audio_Files.py:585-591`) and a separate streaming size guard (`Audio_Files.py:594-608`). The stream-over-limit test calls the function and asserts cleanup (`test_audio_download_limits.py:22-41`). The header-over-limit test sets up `MAX_FILE_SIZE`, a fake UUID, and a fake response with `content-length: 2048`, but it never calls `download_audio_file` and has no assertion (`test_audio_download_limits.py:44-56`). The test therefore passes even if the header guard regresses.

Recommendation: complete the test by invoking `download_audio_file(..., downloader=lambda *_, **__: faux_response)`, asserting `AudioFileSizeError`, and asserting the expected target path was not created.

## Coverage And Evidence

### Files Inspected

- Required audit context: `inventory.md`, `findings-index.json`, `endpoint-inventory.txt`, `backend-test-inventory.txt`, `db-migration-inventory.txt`, `bandit-app-summary.txt`, and the scaffold for this report.
- Media endpoint files: `media/add.py`, `media/process_videos.py`, `media/process_audios.py`, `media/process_documents.py`, `media/process_pdfs.py`, `media/process_ebooks.py`, `media/process_code.py`, `media/process_emails.py`, `media/process_mediawiki.py`, `media/process_web_scraping.py`, `media/ingest_jobs.py`, `media/file.py`, `media/item.py`, `media/listing.py`, `media/reprocess.py`, `media/playlist_preflight.py`, `media/document_figures.py`, `media/document_references.py`, and `media/__init__.py`.
- Core ingestion/storage files: `input_sourcing.py`, `persistence.py`, `Download_Utils.py`, `Upload_Sink.py`, `Audio/Audio_Files.py`, `Video/Video_DL_Ingestion_Lib.py`, `MediaWiki/Media_Wiki.py`, `Storage/filesystem_storage.py`, `Storage/generated_file_helpers.py`, `Storage/quota_enforcement.py`, `DB_Management/media_db/api.py`, `DB_Management/media_db/runtime/defaults.py`, `DB_Management/media_db/runtime/factory.py`, and selected media repository files.
- Tests inspected: `tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py`, `test_mediawiki_db_persistence.py`, `test_mediawiki_vector_storage.py`, `test_persistence_original_storage.py`, `test_audio_download_limits.py`, `test_process_batch_media_ssrf.py`, `test_media_upload_failures.py`, `tests/Media/test_process_code_and_uploads.py`, `test_media_usage_events.py`, `test_auto_chunking_process_endpoints.py`, `test_upload_sink_security.py`, selected `tests/Storage/*`, selected `tests/MediaFiles/*`, and selected `tests/Ingestion_Sources/*`.

### Tests Or Scans Run

Local focused tests:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Media/test_upload_sink_security.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_process_batch_media_ssrf.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py -q
```

Result: `32 passed, 176 warnings in 22.60s`.

Existing audit evidence consumed:

```bash
cat Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt
```

Result summary from existing evidence: Bandit baseline covered `tldw_Server_API/app`; 4,818 results, 26 medium, 0 high. I did not rerun Bandit because the domain task forbids environment-changing setup and source changes were not made.

Review and inventory commands run locally included:

```bash
cat Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md
cat Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json
cat Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/endpoint-inventory.txt
cat Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt
cat Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/db-migration-inventory.txt
sed -n '1,240p' Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md
rg --files tldw_Server_API/app/api/v1/endpoints | rg '(^|/)media|process_(videos|code|web_scraping)|chunking_templates'
rg --files tldw_Server_API/app/core/Ingestion_Media_Processing
rg --files tldw_Server_API/app/core/DB_Management/media_db
rg --files tldw_Server_API/tests | rg 'Media|MediaDB2|MediaFiles|Storage|MediaIngestion_NEW|Ingestion_Sources|WebScraping|mediawiki|upload|archive'
find tldw_Server_API/app/api/v1/endpoints/media -maxdepth 1 -type f -print | sort
rg -n 'RequirePermission|MEDIA_CREATE|rbac_rate_limit|guard_storage_quota|process-audios|process-documents|process-pdfs|process-ebooks|process-code|process-emails|mediawiki/ingest-dump|managed_media_database|SINGLE_USER_FIXED_ID|insert_media_file|Content-Length|content-length' tldw_Server_API/app tldw_Server_API/tests
nl -ba <selected media endpoint/core/test files> | sed -n '<line ranges>'
git status --short
```

### Blocked Or Unverified Areas

- No production/source code was edited.
- No Backlog tasks were created or updated, per coordinator instruction.
- No dependencies were installed, no services were started, no Docker commands were run, and no network access was used.
- I did not run the full media/storage test suite; only focused tests were run. Residual risk remains in long-running integrations, external media providers, optional OCR/VLM dependencies, and provider-backed transcription/embedding paths.
- I did not dynamically reproduce multi-user MediaWiki cross-tenant visibility or permission denial for every processing-only endpoint. The findings are static-confirmed from endpoint dependency wiring and runtime factory behavior.
- I did not update `findings-index.json`; this report records candidates for coordinator/index consolidation.

### Evidence Notes

- Positive controls observed: upload saving strips path components, blocks dangerous extensions, and applies allowed extension validation (`input_sourcing.py:83-174`); URL download handling validates target paths, applies egress policy, and enforces streaming size caps (`Download_Utils.py:68-84`, `Download_Utils.py:221-273`); file serving uses the request-scoped media DB before streaming storage paths (`file.py:136-210`); video URL ingestion validates egress, declared size, yt-dlp output paths, and download quota (`Video_DL_Ingestion_Lib.py:177-185`, `Video_DL_Ingestion_Lib.py:521-628`, `Video_DL_Ingestion_Lib.py:1368-1420`).
- Existing unrelated worktree changes were present before this report edit: `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/command-log.md` was modified, and two watchlist templates were untracked. They were not touched by this domain report.
