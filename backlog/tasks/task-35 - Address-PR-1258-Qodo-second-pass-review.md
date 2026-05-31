---
id: TASK-35
title: Address PR 1258 Qodo second-pass review
status: Done
assignee:
  - Codex
created_date: '2026-05-04 05:12'
updated_date: '2026-05-04 05:24'
labels:
  - codegraph
  - mcp
  - review
dependencies:
  - TASK-34
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1258'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the new Qodo review comments on PR #1258 after the initial follow-up push. Scope is limited to typed test helpers in test_codegraph_indexer.py, per-candidate OSError resilience in CodeGraphIndexer indexing, and avoiding duplicate file opens in the indexer hot loop. Preserve existing CodeGraph behavior and keep PR #1258 focused on review remediation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New test helpers in test_codegraph_indexer.py use explicit type hints without ANN002 or ANN003 ignores.
- [x] #2 A per-file OSError during binary probe or content read or hashing is recorded as a per-file skipped or failed item and does not abort the whole indexing run.
- [x] #3 Indexer file processing avoids duplicate opens by reusing the binary probe bytes for extraction source or streaming hash.
- [x] #4 Focused CodeGraph and MCP tests pass after the review fixes.
- [x] #5 Ruff Bandit and git diff whitespace checks pass on touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify Qodo findings against current code and existing tests. 2. Add RED tests for per-candidate OSError handling and single-open indexing behavior. 3. Update test helper type hints to remove ANN002 and ANN003 ignores. 4. Refactor CodeGraphIndexer per-file I/O so each candidate is opened once and probe bytes are reused for binary detection plus extraction or streaming hash. 5. Run focused CodeGraph/MCP tests, Ruff, Bandit, and git diff --check before updating PR #1258.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented second-pass Qodo fixes. Added RED tests for unreadable per-candidate file handling and single-open file processing. CodeGraphIndexer now reads each candidate from one stream, reuses the binary probe for extraction source or streaming hash, and records per-file OSError as extraction_failed without aborting the run. Verification: focused CodeGraph/MCP suite passed with 53 passed and 5 warnings; Ruff touched files passed; Bandit /tmp/bandit_codegraph_qodo_second_pass.json reported 0 results and 0 errors; git diff --check passed.

No product documentation changes were needed for this PR-review-only resilience and test cleanup. Remaining external state: PR #1258 stays draft pending the required human-authored Change summary, and GitHub CI is queued after the latest push.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1258 second-pass Qodo review findings. Test helpers now use explicit type hints without ANN002 or ANN003 ignores, CodeGraphIndexer handles per-candidate file I/O OSError as extraction_failed without aborting the run, and candidate file processing now opens each file once by reusing the binary probe for extraction source or streaming hash. Verification passed locally: focused CodeGraph/MCP tests, Ruff touched files, Bandit touched scope, and git diff --check.
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
