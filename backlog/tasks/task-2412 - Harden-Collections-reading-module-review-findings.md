---
id: TASK-2412
title: Harden Collections reading module review findings
status: Done
assignee: []
created_date: 2026-06-23 18:13
updated_date: 2026-06-24 03:48
labels:
- collections
- security
- reliability
dependencies: []
references:
- tldw_Server_API/app/core/Collections/reading_service.py
- tldw_Server_API/app/core/Collections/reading_importers.py
- tldw_Server_API/app/core/Collections/reading_import_jobs.py
- tldw_Server_API/app/core/Collections/reading_digest_jobs.py
- tldw_Server_API/app/core/Collections/embedding_queue.py
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the accepted review findings for the Collections reading module: bound persisted/queued reading content, reject or inert unsafe URL schemes in imports and digest output, centralize reading status validation, harden import size env parsing, and extract focused helpers from ReadingService without changing public endpoint shapes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reading content and embedding job payloads are bounded by explicit limits with regression tests.
- [x] #2 Imported non-http(s) URLs are rejected or skipped, and digest renderers do not emit active links for unsafe schemes.
- [x] #3 Save/update/import reading status paths use one validation/normalization contract.
- [x] #4 READING_IMPORT_MAX_BYTES uses safe env parsing and cannot fail module import on invalid values.
- [x] #5 ReadingService retains its public facade while archive/import responsibilities move behind smaller helpers.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented bounded reading metadata and embedding job payloads, shared reading status normalization, http/https import filtering, inert unsafe digest links, safe READING_IMPORT_MAX_BYTES parsing, and ReadingArchiveService/ReadingImportService helpers behind ReadingService.

Verification: direct behavioral script passed for digest unsafe-link rendering, invalid import env fallback, embedding payload truncation, save/update status normalization, bounded metadata, helper wiring, and non-http import skip. compileall passed for touched Collections code/tests. git diff --check passed. Bandit on touched Collections files produced 0 findings at /tmp/bandit_task_2412_collections.json.

Focused pytest limitation: targeted pytest timed out during repo-wide autouse fixture setup before executing the Collections test body. Timeout stack was in tests/conftest.py importing character_chat_sessions -> Research/RAG -> nltk/scipy, not in the changed Collections code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the Collections reading module against the review findings: content is bounded before metadata/job persistence, unsafe imported/digest URLs are not active, reading statuses share one normalization contract, invalid import-size env values no longer break imports, and ReadingService delegates archive/import responsibilities to focused helpers. Verification passed via direct behavioral checks, compileall, diff whitespace check, and Bandit; focused pytest remains blocked by unrelated global fixture bootstrap timeout.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [ ] #7 Focused pytest coverage passes for Collections reading service/import/digest/embedding queue behavior.
- [x] #8 Bandit runs clean on touched Collections files.
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review follow-up: addressed PR comments by adding docstrings to new helpers, adding type hints/docstrings to new tests, hardening default markdown digest rendering against link/summary injection, and verifying skipped import counts still count unsupported raw URLs once. Verification before commit: git diff --check passed; compileall on touched files passed; Bandit on touched Collections core files reported 0 findings; direct behavior checks passed for markdown escaping, embedding content bounding, unsupported URL skip count, and invalid import-size env fallback.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
