# Quick Ingest File Type Support Implementation Plan

## Stage 1: Contract Mapping
**Goal**: Compare the current Quick Ingest advertised accept list against backend media validation and document conversion support.
**Success Criteria**: Each advertised extension has an explicit backend validation/processing route or a documented reason for rejection.
**Tests**: Focused existing unit tests inspected; new regression tests planned for uncovered gaps.
**Status**: Complete

## Stage 2: Backend Validation Red Tests
**Goal**: Add failing tests for advertised extensions currently rejected by backend upload validation.
**Success Criteria**: Tests fail for `.markdown`, `.xhtml`, `application/ogg`, `video/avi`, and the legacy `.doc` contract before implementation.
**Tests**: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_file_validation.py`.
**Status**: Complete

## Stage 3: Document Conversion Red Tests
**Goal**: Add failing tests for advertised document extensions that cannot currently be converted.
**Success Criteria**: Tests fail for `.markdown` and `.xhtml` conversion before implementation.
**Tests**: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_plaintext_conversion.py`.
**Status**: Complete

## Stage 4: Minimal Implementation
**Goal**: Align validation and document conversion with the advertised Quick Ingest contract.
**Success Criteria**: `.markdown`, `.xhtml`, `.ogg` as `application/ogg`, and `.avi` as `video/avi` pass focused regression tests; legacy `.doc` is removed from advertised upload support and rejected by backend document validation.
**Tests**: Focused backend unit tests; frontend contract tests if UI constants need adjustment.
**Status**: Complete

## Stage 5: Verification
**Goal**: Run focused tests, security scan on touched backend code, and record remaining limitations.
**Success Criteria**: Targeted tests pass; Bandit reports no new findings in touched backend paths.
**Tests**: `python -m pytest ...`; `python -m bandit -r <touched backend paths>`.
**Status**: Complete

## Stage 6: Browser Extension Upload Coverage
**Goal**: Add browser-extension-specific Quick Ingest coverage for the accepted file-type contract.
**Success Criteria**: Extension e2e test asserts every advertised explicit upload extension is present, legacy `.doc` support is excluded, representative accepted files queue successfully, and submitted file names reach the mock ingest-job path.
**Tests**: `npx playwright test tests/e2e/quick-ingest-file-upload.spec.ts --project=chromium-extension`; local execution recorded as environment-blocked because Chromium does not load MV3 extension targets in this automation environment.
**Status**: Complete

## Stage 7: Direct CDP UAT
**Goal**: Exercise the built browser extension against the real FastAPI server, media-ingest worker, and SQLite media library without mocks.
**Success Criteria**: CDP-driven file selection queues every advertised accepted upload extension, excludes `.doc`, real job/media persistence is verified for processable file types, and persisted items render in the Media Library view.
**Tests**: Direct Chrome DevTools Protocol harness using `DOM.setFileInputFiles`, real `/api/v1/media/ingest/jobs`, real worker, Jobs/Media SQLite queries, and direct CDP rendered `/media` verification.
**Status**: Complete

Notes:
- Direct CDP UAT run `cdp-uat-1782443002442` queued all advertised accepted extensions and excluded `.doc`.
- Real document/ebook persistence passed; `.docx` persisted as media ID 19.
- Direct CDP exposed `.xhtml` MIME detection as `application/xml`; fixed in `Upload_Sink.py` and verified by real upload job 49/media ID 27.
- Direct CDP also exposed a real default audio/video processing blocker in the configured Parakeet ONNX batch STT path. A real audio upload with explicit `transcription_model=whisper-large-v3` completed as job 50/media ID 28, confirming upload support when STT is usable. The default Parakeet ONNX multi-graph runtime issue is outside this file-type support patch.
- Direct CDP rendered Media Library verification passed after materializing workspace frontend dependencies with `bun install` at `apps/`; `/media` showed persisted UAT items including `tldw-cdp-uat-1782443002442`, and screenshot evidence was captured at `/tmp/tldw-media-library-cdp.png`.
