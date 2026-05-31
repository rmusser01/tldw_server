---
id: TASK-490.10
title: 'Sync v2 M1: Add replay and repair'
status: Done
assignee:
- '@Codex'
created_date: ''
updated_date: 2026-05-23 15:03
labels:
- sync
- sync-v2
- m1
- repair
- backend
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
parent_task_id: TASK-490
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add replay/repair support that rebuilds materialized Notes and Chat projections from accepted envelopes, retries failed applies, preserves tombstones, excludes conflict envelopes, and reports repair status.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Replay can rebuild Notes and Chat projections from accepted envelopes.
- [x] #2 Failed applies can be retried after the underlying projection issue is fixed.
- [x] #3 Tombstones are preserved and conflict envelopes are not replayed as accepted changes.
- [x] #4 Profile/status exposes failed apply counts and repair results.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-10-add-replay-and-repair
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Sync v2 replay/repair with a cursor-ordered accepted-envelope repair runner, authenticated dataset-scoped service method, /sync/repair endpoint, repair response schemas, and per-domain profile repair_status.

Verification:
- RED: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py -q failed with missing SyncV2Service.repair.
- GREEN: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py -q passed.
- Regression: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py tldw_Server_API/tests/Sync/test_sync_v2_models.py -q passed: 121 passed, 5 warnings.
- Focused Ruff on new/reworked replay/profile/schema/test files passed. Broader touched-file Ruff remains blocked by existing sync.py/service.py baseline issues outside this task scope.
- Bandit touched production paths wrote /tmp/bandit_task_490_10_replay_repair.json with no errors and no findings.
- git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Sync v2 M1 replay/repair for server-side projections. Repair can rebuild Notes and Chat projections from accepted envelopes, retry failed applies after projection recovery, preserve tombstones, skip conflict envelopes, expose /sync/repair results, and report repair health in profile domain status.

Known residual: broad Ruff over all touched legacy files still reports pre-existing sync.py/service.py baseline issues; focused lint over new/reworked files passed.
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
