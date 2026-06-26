---
id: TASK-12014
title: Fix Quick Ingest accepted file type support gaps
status: Done
labels:
- quick-ingest
- upload
- media
- webui
- browser-extension
- bugfix
priority: high
modified_files:
- tldw_Server_API/app/core/Ingestion_Media_Processing/Upload_Sink.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/input_sourcing.py
- tldw_Server_API/app/api/v1/endpoints/media/process_documents.py
- apps/packages/ui/src/components/Common/QuickIngest/constants.ts
- apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx
- apps/packages/ui/src/services/tldw/media-routing.ts
- apps/extension/tests/e2e/quick-ingest-file-upload.spec.ts
- Docs/superpowers/plans/2026-06-25-quick-ingest-file-type-support.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address UAT findings where WebUI and browser-extension Quick Ingest advertise accepted file types that are rejected or unsupported by backend validation/processing. Scope includes .markdown, .ogg, .avi, .doc, and robust verification of advertised accepted upload types where feasible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend upload validation accepts all file extensions advertised by Quick Ingest when the file content is valid for that type.
- [x] #2 .markdown uploads are accepted and routed as markdown documents.
- [x] #3 .ogg uploads are accepted for common Ogg MIME detections including application/ogg where appropriate.
- [x] #4 .avi uploads are accepted for common AVI MIME detections including video/avi where appropriate.
- [x] #5 .doc support is aligned across UI and backend: either legacy .doc is processed successfully or it is removed from advertised accepted types with tests documenting the contract.
- [x] #6 Focused regression tests cover the fixed validation/routing behavior.
- [x] #7 Verification results and any remaining environment-dependent limitations are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented in clean worktree `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/upload-file-type-support` on branch `codex/upload-file-type-support`, based on `origin/dev` commit `8b0726f2e3a9069a7da5cbfc57f459de3640076f`.

Changes:
- Backend upload validation now accepts `.markdown`, `.html`, `.htm`, `.xhtml`, `.xml`, `.json` as document uploads; `application/ogg` for Ogg audio; and `video/avi` for AVI video.
- Document processing now accepts `.markdown` as text and `.xhtml` through the existing sanitized HTML conversion path.
- `/process-documents` upload allow-list now includes `.markdown`.
- Shared WebUI/extension routing now treats HTML-like filenames as document uploads while keeping HTML URLs on the web-scraping route, and maps `application/ogg` to audio.
- Legacy binary `.doc` and `application/msword` were removed from Quick Ingest/source/workflow/attachment accept lists and backend document validation because no real `.doc` parser/converter exists in the current processing path.
- Extension Quick Ingest e2e coverage now includes a matrix test that asserts every advertised explicit upload extension is present in the bundled file input, confirms `.doc`/`application/msword` are excluded, selects representative files for all advertised extensions, and verifies each selected file reaches the mock ingest-job submission path.

Verification:
- `python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_file_validation.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_plaintext_conversion.py -q` -> 43 passed, 1 xfailed.
- `python -m pytest tldw_Server_API/tests/Media_Ingestion_Modification/test_url_acceptance_endpoints.py -q` -> 30 passed.
- `bunx vitest run src/components/Common/QuickIngest/__tests__/constants.test.ts src/services/tldw/__tests__/media-routing.test.ts` -> 2 files passed, 15 tests passed.
- `bunx vitest run src/components/Common/QuickIngest/__tests__/constants.test.ts src/services/tldw/__tests__/media-routing.test.ts src/components/Common/QuickIngest/__tests__/FileDropZone.acceptance.test.tsx` -> 3 files passed, 17 tests passed.
- `CI=1 npx playwright test tests/e2e/quick-ingest-file-upload.spec.ts --project=chromium-extension --reporter=line` built Chrome MV3 successfully, then failed in local sandbox before UI assertions because the mock server could not bind `127.0.0.1` (`listen EPERM`).
- Rerunning outside the sandbox with `TLDW_E2E_SKIP_EXTENSION_BUILD=1` allowed the mock server but local headless Chromium did not load the MV3 extension (`Could not determine extension id from [no extension targets]`). A minimal probe confirmed zero `chrome-extension://` targets in headless Chromium/Chrome for this build. Headed local probe/test timed out at browser startup/target setup in this automation environment.
- `TLDW_E2E_SKIP_EXTENSION_BUILD=1 npx playwright test tests/e2e/quick-ingest-file-upload.spec.ts --project=chromium-extension --list` -> lists all 3 tests, including the new accepted-extension matrix.
- `python -m bandit -r tldw_Server_API/app/core/Ingestion_Media_Processing/Upload_Sink.py tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py tldw_Server_API/app/core/Ingestion_Media_Processing/input_sourcing.py tldw_Server_API/app/api/v1/endpoints/media/process_documents.py -f json -o /tmp/bandit_quick_ingest_file_types.json` -> 0 findings.

Remaining environment-dependent limitations from the UAT task still apply: audio/video files may validate and submit but fail downstream STT/video processing when the local environment lacks the required models or when generated video fixtures have no audio stream. Browser-extension e2e execution also requires a local/CI browser environment that can load the built Chrome MV3 extension.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Quick Ingest accepted-file-type mismatches from UAT. Backend validation now accepts .markdown, HTML/XHTML/XML/JSON document uploads through the document path, application/ogg for Ogg audio, and video/avi for AVI. Document conversion now handles .markdown and .xhtml. Legacy binary .doc is removed from advertised UI upload support and backend document validation until a real parser is added. Added backend, shared UI, and browser-extension e2e regression coverage for the accepted-file contract; targeted backend/shared UI tests, Chrome MV3 build, Playwright test discovery, and Bandit passed. Local extension e2e execution is blocked in this automation environment because Chromium does not load MV3 extension targets.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
