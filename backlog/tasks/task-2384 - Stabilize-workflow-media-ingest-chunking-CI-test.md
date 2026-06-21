---
id: TASK-2384
title: Stabilize workflow media ingest chunking CI test
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-21 03:05'
labels:
  - ci
  - tests
  - workflows
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix PR #2258 CI failures in workflow test shards: media ingest polling should fail with explicit diagnostics instead of KeyError, and approval permission tests should set up waiting approval runs deterministically instead of depending on background scheduler timing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The media ingest chunking test handles workflow run polling with explicit response assertions and useful timeout diagnostics.
- [x] #2 The test fixture grants the permissions required by the workflow run status endpoint.
- [x] #3 The failed test passes locally and touched files pass syntax/security checks.
- [x] #4 Approval permission tests set up waiting approval runs deterministically without depending on background scheduler timing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CI run 27886920141 direct failures addressed:
- Job 82523799578, product-workflows-storage: test_media_ingest_local_text_chunking crashed with KeyError: status while polling a non-validated run payload.
- Job 82523800213, product-workflows-api: test_reject_allows_admin_override reused the engine run path for permission setup and the second helper call observed succeeded instead of waiting_approval.

Fixes:
- Added explicit status-code checks and timeout diagnostics to media-ingest run polling, and granted WORKFLOWS_RUNS_READ in the test auth principal.
- Changed approval-permission setup to create the waiting run/step rows directly in the test database, keeping the test focused on approve/reject authorization and removing background scheduler timing from setup.

Verification:
- product-workflows-api shard command: 77 passed, 2 skipped.
- product-workflows-storage shard command: 29 passed, 6 skipped.
- Focused media-ingest pair: 2 passed.
- compileall on touched tests: passed.
- git diff --check on touched tests/task: passed.
- Bandit on touched tests wrote /tmp/bandit_ci2258_workflow_tests.json; findings were existing low-severity pytest assert/test-token patterns only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stabilized both direct PR #2258 workflow shard failures. Media ingest polling now validates run-status responses and reports useful diagnostics. Approval permission tests now create waiting approval state directly in the workflow test database, avoiding a scheduler race while preserving approve/reject authorization coverage. Local shard verification passed for both affected CI shards.
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
