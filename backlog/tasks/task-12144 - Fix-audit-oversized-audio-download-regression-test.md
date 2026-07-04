---
id: TASK-12144
title: Fix audit oversized audio download regression test
status: Done
created_date: 2026-07-04 18:30
labels:
- audit
- remediation
- media
- tests
priority: low
references:
- AUDIT-2026-06-27-MEDIA-004
- https://github.com/rmusser01/tldw_server/pull/2613
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_download_limits.py
updated_date: 2026-07-05 00:25
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
- The test now asserts `AudioFileSizeError`, verifies the deterministic target path is not created, confirms the faux downloader was called once, and checks the exact injected downloader call arguments.
- No production code was touched; Bandit has no touched production scope for this test-only task.
- Tracking hygiene: moved this audio-size audit record from duplicate `TASK-12140` to `TASK-12144` because latest dev already contains a different `TASK-12140`.
- Review follow-up: kept the Backlog task filename convention with spaces because this repository's Backlog.md task files use that standard format; tightened the oversized download regression to assert exact injected downloader call arguments.
- Current-dev refresh: rebased `codex/audit-audio-size-test-2026-07-04` onto `origin/dev` `09d9ec901e1d4548f7924f1c6bcefa963fadd9bd`; merge-base matches `origin/dev`.
- Current-dev validation: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_download_limits.py -q` passed with 3 tests; `git diff --check HEAD~1..HEAD` passed; `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_download_limits.py -f json -o /tmp/bandit_audio_size_test_origin_dev_09d9ec.json` reported 4 LOW B101 findings on pytest `assert` statements in test code only. No production code changed.
2026-07-04 latest-dev refresh: rebased and validated PR #2613 on origin/dev 6b727b221e55646eba663a03571e38302f7fafc2. Tested head 21eabda5e401. Verification: python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_download_limits.py -q => 3 passed, 15 warnings; bandit -r tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_download_limits.py => 4 LOW B101 pytest assert findings in test code only; git diff --check HEAD~1..HEAD => clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added regression coverage for audio download size-limit behavior. Final refresh validated against origin/dev 6b727b221e55646eba663a03571e38302f7fafc2 with focused tests passing; Bandit findings are limited to pytest assert usage in the touched test file; whitespace check clean.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused audio download limit tests pass.
- [x] #2 Bandit runs clean over touched production code if production code changes; otherwise the task notes record that no production code was touched.
- [x] #3 git diff --check passes.
- [x] #4 AUDIT-2026-06-27-MEDIA-004 closure evidence is recorded in task notes.
<!-- DOD:END -->
