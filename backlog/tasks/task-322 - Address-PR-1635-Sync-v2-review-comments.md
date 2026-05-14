---
id: TASK-322
title: Address PR 1635 Sync v2 review comments
status: Done
assignee: []
created_date: '2026-05-14 01:11'
updated_date: '2026-05-14 01:32'
labels:
  - sync
  - review
  - pr-1635
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1635'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review feedback on PR #1635 for the Sync v2 server substrate branch. Scope is limited to the isolated sync worktree and inline findings: SQL-side pull filtering, timezone-aware timestamp generation, direct conflict lookup, restore manifest aggregation, and focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All open actionable inline review threads on PR #1635 are inspected and either fixed or documented with a technical reason.
- [x] #2 Sync v2 pull filtering avoids Python-side scanning of large same-device echo windows by using database query predicates where applicable.
- [x] #3 Sync v2 conflict resolution uses direct conflict lookup with dataset ownership validation instead of scanning every dataset conflict list.
- [x] #4 Sync v2 restore manifest counts and byte estimates are produced with database aggregation rather than scanning up to the manifest limit in Python.
- [x] #5 Touched code avoids deprecated naive UTC datetime generation and is covered by focused tests.
- [x] #6 Focused Sync v2 tests, git diff whitespace check, and Bandit on touched production scope are run and recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect live PR inline threads, bot comments, and check failures. 2. Add tests for the concrete Sync v2 review fixes. 3. Patch store/service/database helpers for SQL filtering, direct conflict lookup, manifest aggregation, and timezone-aware timestamps. 4. Run focused tests, Bandit, and diff checks. 5. Push branch and resolve/reply to PR threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review-fix implementation pass: added SQL-side accepted/status/excluded-device pull filtering, timezone-aware UTC timestamps, direct conflict lookup with ownership validation, SQL restore-manifest aggregation, cached Sync v2 service wiring, bounded push batches, safer SyncStoreError mapping/log context, canonical legacy media payload hashing, workspace source-ref token matching, recursive nested-list redaction, and Backlog cleanup for reviewer-noted task files.

Verification so far: python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py tldw_Server_API/tests/Sync/test_sync_v2_security.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_service.py -q -> 142 passed; python -m pytest tldw_Server_API/tests/Sync tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py -q -> 145 passed; Bandit wrote /tmp/bandit_sync_v2_pr1635_review.json with empty errors/results; git diff --check clean.

Live PR thread refresh before staging found 21 open review threads: 5 Gemini, 5 Qodo, and 11 CodeRabbit. All mapped to the patch set or task-file cleanup. gh pr checks before this push still showed older Full Suite / UX Smoke / Onboarding E2E failures alongside passing required gates and passing CodeRabbit; previous log review indicated timeout/install or broad-suite boundary behavior rather than a Sync v2-specific failing assertion. The branch push should trigger a fresh check run.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the actionable PR #1635 Sync v2 review feedback in the isolated sync worktree. The patch moves Sync v2 HTTP composition into a cached core factory, adds safe contextual error logging and correct store-error mapping, bounds push batches, advances cursors across accepted and conflict envelopes, pushes pull filtering into SQL, uses direct conflict lookup with ownership checks, aggregates restore manifest metadata in SQL, canonicalizes legacy media payload hashing, tightens workspace source-ref detection, recursively redacts nested private lists, and cleans up reviewer-flagged Backlog task records. Focused Sync tests and the restore e2e pass, Bandit on touched production scope reports no findings, and git diff whitespace checks are clean.
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
