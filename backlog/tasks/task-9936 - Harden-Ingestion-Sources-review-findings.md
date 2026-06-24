---
id: TASK-9936
title: Harden Ingestion Sources review findings
status: Done
assignee: []
created_date: '2026-06-23 14:41'
updated_date: '2026-06-23 20:43'
labels:
  - ingestion-sources
  - hardening
  - security
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-06-23-ingestion-sources-review-hardening.md
references:
  - https://github.com/rmusser01/tldw_server/pull/2457
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the current Ingestion_Sources module issues found during review: archive path collision mapping, archive expansion limits, local source scan/read bounds, fenced job completion verification, schema indexes, and strict bool normalization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Archive snapshots preserve content for suffix-colliding paths like `a.md` and `dir/a.md`.
- [x] #2 ZIP/TAR validation rejects archives exceeding member count, per-member byte, or total uncompressed byte limits.
- [x] #3 Local directory and local git snapshots mark oversized files as item failures and avoid unbounded reads.
- [x] #4 Local git enumeration has a bounded subprocess timeout.
- [x] #5 Source sync job completion detects `active_job_id` fence mismatches.
- [x] #6 Ingestion source schema includes indexes for list, scheduler, snapshot, artifact, and item-event access patterns.
- [x] #7 Core source payload bool parsing rejects string truthiness surprises.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Backlog MCP resources were unavailable and the official Backlog CLI hung on search, list, and create. The human requester approved a direct task-file fallback before repository edits. Work is isolated in `.worktrees/ingestion-sources-review-hardening` on branch `codex/ingestion-sources-review-hardening`.

RED verification before implementation: focused Ingestion Sources pytest command collected 31 tests, with 11 expected failures covering archive suffix collision, archive limit enforcement, oversized local files, git timeout wrapping, missing indexes, job fence mismatch handling, and string boolean rejection.

GREEN verification after implementation:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD python -m pytest -p pytest_asyncio.plugin tldw_Server_API/tests/Ingestion_Sources/test_archive_snapshot_adapter.py tldw_Server_API/tests/Ingestion_Sources/test_local_directory_adapter.py tldw_Server_API/tests/Ingestion_Sources/test_git_repository_adapter.py tldw_Server_API/tests/Ingestion_Sources/test_service_sqlite_state.py tldw_Server_API/tests/Ingestion_Sources/test_models_and_service_contract.py -v` -> 31 passed, 76 warnings in 33.19s.
- `python -m py_compile tldw_Server_API/app/core/Ingestion_Sources/archive_snapshot.py tldw_Server_API/app/core/Ingestion_Sources/diffing.py tldw_Server_API/app/core/Ingestion_Sources/local_directory.py tldw_Server_API/app/core/Ingestion_Sources/git_repository.py tldw_Server_API/app/core/Ingestion_Sources/service.py` -> exit 0.
- `python -m bandit -r tldw_Server_API/app/core/Ingestion_Sources -f json -o /tmp/bandit_ingestion_sources_review.json` -> exit 0; JSON summary had 0 issues.
- `git diff --check` -> exit 0.

Note: the default pytest run with all auto-loaded plugins was interrupted after stalling in broader plugin/app import cleanup. The same focused tests passed with plugin autoload disabled and `pytest_asyncio` explicitly enabled.

PR review response on 2026-06-23:
- Rebasing onto latest `origin/dev` completed cleanly.
- Dropped an unrelated Claims_Extraction design commit from the PR branch so the PR diff only contains Ingestion Sources changes and this task record.
- Addressed Gemini comments by validating archive total uncompressed limits against actual bytes read, while keeping conservative header prechecks, and by using a defensive archive member path lookup.
- Verified the `os` import comment was stale; `archive_snapshot.py` imports `os`.
- Addressed Qodo SLF001 feedback by removing the unqualified `# noqa: SLF001` from the git timeout test.
- Added ZIP and TAR tests that simulate under-reported member sizes and assert actual bytes over the total limit are rejected.

Review-response verification:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD python -m pytest -p pytest_asyncio.plugin tldw_Server_API/tests/Ingestion_Sources/test_archive_snapshot_adapter.py tldw_Server_API/tests/Ingestion_Sources/test_local_directory_adapter.py tldw_Server_API/tests/Ingestion_Sources/test_git_repository_adapter.py tldw_Server_API/tests/Ingestion_Sources/test_service_sqlite_state.py tldw_Server_API/tests/Ingestion_Sources/test_models_and_service_contract.py -v` -> 33 passed, 80 warnings in 32.29s.
- `python -m py_compile` on touched Ingestion Sources implementation files -> exit 0.
- `python -m bandit -r tldw_Server_API/app/core/Ingestion_Sources -f json -o /tmp/bandit_ingestion_sources_review.json` -> exit 0; JSON summary had 0 issues.
- `git diff --check` -> exit 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented archive member limit enforcement and explicit archive member path mapping, added local directory/local git file size fences, wrapped local git enumeration with a timeout, added ingestion source query indexes, rejected non-bool boolean payload values, and made sync job completion fail on active-job fence mismatches. Focused regression tests, py_compile, Bandit, and diff whitespace checks passed as recorded above.

PR: https://github.com/rmusser01/tldw_server/pull/2457
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
