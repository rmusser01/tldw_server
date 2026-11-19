---
id: TASK-15551
title: Support large native file uploads in Quick Ingest
status: Done
labels:
- quick-ingest
- media-ingestion
references:
- https://github.com/rmusser01/tldw_server/pull/3484
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Use a configurable 250 MiB document limit in Quick Ingest, allow 1 GiB audio/video and use a 1 GiB per-file API ceiling. Preserve existing Quick Ingest jobs, upload validation and owner-scoped routes. ADR assessment: no new ADR because this reuses the existing multipart uploads and Quick Ingest Jobs architecture. ADR-006 governs the touched Python security scan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Documents default to 250 MiB at the frontend, and audio/video accept 1 GiB.
- [x] #2 Preserve native File uploads and a one GiB per-file API ceiling with existing security checks.
- [x] #3 Verify boundary behavior and open a PR against dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
432 focused frontend tests and 121 backend tests passed, with one expected backend failure. The emitted Next.js client showed a 384 MiB override. No new runtime Bandit findings. Hosted browser proof used the same upload behavior. Exact-size Chrome controls were verified on staging with generated files and AI analysis disabled. Media jobs were cancelled after upload. A lost acknowledgement can leave a just-created job running on Cancel All.
Backend CI caught the upload settings exceeding the existing config loader size ceiling. Moved the parsing into a helper with the same defaults and override policies. The size ratchet, upload validation, config precedence, and config boundary checks passed locally with 92 tests and one expected failure. Bandit reported no findings on the new lines. The earlier app container timeout occurred during image transfer after the build and was retried on the same source.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chrome accepted the exact 250 MiB PDF, EPUB, and DOCX and exact 1 GiB WAV and MP4. The 500-page PDF preserved text from its first and last pages. All corresponding one-byte-over controls were rejected before upload. Live API limits were 1 GiB per standard file. The PR includes the verification results.
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
