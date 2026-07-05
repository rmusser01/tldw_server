---
id: TASK-12884
title: Fix audit oversized audio download regression test
status: Done
created_date: 2026-07-04 02:01
labels:
- audit
- remediation
- media
- tests
priority: low
references:
- AUDIT-2026-06-27-MEDIA-004
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_download_limits.py
updated_date: 2026-07-04 02:04
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate AUDIT-2026-06-27-MEDIA-004 by making the header-declared oversized audio download regression actually invoke the downloader, assert AudioFileSizeError, and verify no target file is created.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The oversized header-declared audio download test invokes download_audio_file with a faux downloader response.
- [x] #2 The regression asserts AudioFileSizeError for content-length above the configured max size.
- [x] #3 The regression asserts the expected target path is not created.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Corrected the AUDIT-2026-06-27-MEDIA-004 regression from a no-op setup into an active invocation of `download_audio_file` with a faux downloader response.
- The test now asserts `AudioFileSizeError`, verifies the deterministic target path is not created, and confirms the faux downloader was called once.
- No production code was touched; Bandit has no touched production scope for this test-only task.
- Verification: `python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_download_limits.py -q` passed with 3 tests; `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the oversized audio header regression coverage by making the unit test actually exercise `download_audio_file` with an oversized faux response. The test now proves the downloader rejects the payload before writing the expected destination file. No production code changed; focused tests and whitespace checks passed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused audio download limit tests pass.
- [x] #2 Bandit runs clean over touched production code if production code changes; otherwise the task notes record that no production code was touched.
- [x] #3 git diff --check passes.
- [x] #4 AUDIT-2026-06-27-MEDIA-004 closure evidence is recorded in task notes.
<!-- DOD:END -->
