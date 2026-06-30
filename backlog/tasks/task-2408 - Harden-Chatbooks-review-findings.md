---
id: TASK-2408
title: Harden Chatbooks review findings
status: Done
assignee: []
created_date: 2026-06-23 18:11
updated_date: 2026-06-24 03:32
labels:
- chatbooks
- security
- review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address validated Chatbooks module review findings: v1.1 manifest path integrity, cancellation handling, quota admission races, unreachable Prompt Studio paths, service decomposition cleanup where practical, and OpenWebUI import bounds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated high-severity findings have regression tests.
- [x] #2 Chatbook v1.1 imports consume only verified manifest-resolved payload paths.
- [x] #3 Async cancellation is not swallowed by generic recoverable exception handling.
- [x] #4 Quota/concurrency admission is enforced atomically or documented as deferred with a concrete task if broader Jobs changes are required.
- [x] #5 Unreachable legacy code and unused helpers are removed where safe.
- [x] #6 Verification includes focused pytest and Bandit on touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Chatbooks fixes for v1.1 explicit import paths, cancellation propagation, OpenWebUI JSON/SQLite import bounds, async quota admission, and safe legacy cleanup. Async Chatbooks export/import now checks tier quota and inserts the pending Chatbooks job row inside one database transaction; endpoint preflight remains for fast feedback. Updated stale v1.1 async test from the removed Prompt Studio path to the current core Jobs payload path.

Documentation update not required: changes harden existing Chatbooks behavior and are covered by regression tests/task notes without changing public request or response schemas.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated and addressed the Chatbooks review findings and follow-up PR comments. Fixed v1.1 manifest path usage, cancellation propagation, OpenWebUI JSON/SQLite import bounds, async and sync quota admission, structured quota errors, PostgreSQL admission locking, and manifest path indexing. Full Chatbooks pytest suite and Bandit on touched code passed.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Post-PR review comments addressed on rebased dev: synchronous export/import now run service-level Chatbooks quota admission; quota rejections surface as structured QuotaExceededError instead of endpoint string matching; PostgreSQL count-and-insert admission takes a per-user advisory transaction lock; manifest import file resolution uses a single explicit-path index for the import; async export accepts content_selections=None as an empty selection.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
