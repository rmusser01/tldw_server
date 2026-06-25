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

Verification:
- `python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_file_validation.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_plaintext_conversion.py -q` -> 43 passed, 1 xfailed.
- `python -m pytest tldw_Server_API/tests/Media_Ingestion_Modification/test_url_acceptance_endpoints.py -q` -> 30 passed.
- `bunx vitest run src/components/Common/QuickIngest/__tests__/constants.test.ts src/services/tldw/__tests__/media-routing.test.ts` -> 2 files passed, 15 tests passed.
- `python -m bandit -r tldw_Server_API/app/core/Ingestion_Media_Processing/Upload_Sink.py tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py tldw_Server_API/app/core/Ingestion_Media_Processing/input_sourcing.py tldw_Server_API/app/api/v1/endpoints/media/process_documents.py -f json -o /tmp/bandit_quick_ingest_file_types.json` -> 0 findings.

Remaining environment-dependent limitations from the UAT task still apply: audio/video files may validate and submit but fail downstream STT/video processing when the local environment lacks the required models or when generated video fixtures have no audio stream.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Quick Ingest accepted-file-type mismatches from UAT. Backend validation now accepts .markdown, HTML/XHTML/XML/JSON document uploads through the document path, application/ogg for Ogg audio, and video/avi for AVI. Document conversion now handles .markdown and .xhtml. Legacy binary .doc is removed from advertised UI upload support and backend document validation until a real parser is added. Added focused backend and shared UI regression tests; targeted backend/UI tests and Bandit passed.
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
