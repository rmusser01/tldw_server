---
id: TASK-199
title: Implement VN Play branch navigation API
status: In Progress
assignee: []
created_date: '2026-05-09 22:22'
updated_date: '2026-05-09 22:24'
labels:
  - vn-play
  - api
  - implementation
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1463'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
documentation:
  - Docs/superpowers/specs/2026-05-09-vn-play-branch-navigation-api-design.md
  - >-
    Docs/superpowers/plans/2026-05-09-vn-play-branch-navigation-api-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1463 from the reviewed design spec and saved implementation plan. Scope covers backend branch navigation read model, session action idempotency and shared turn/restore mutation gate, branch-aware event filtering, guarded branch/checkpoint restore, API schemas/endpoints, docs, tests, Bandit, and PR-ready verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pure branch navigation read model derives active path, parent branch ids, direct/subtree event ranges, and stable warning payloads.
- [ ] #2 Repository persists session restore actions, enforces restore idempotency, and shares a session mutation gate with turn requests.
- [ ] #3 Service exposes branch navigation, branch-aware event filtering, branch restore, and checkpoint restore idempotency while preserving existing Freeform and Story turn behavior.
- [ ] #4 API exposes branch-navigation and branch restore endpoints, extends events filtering compatibly, maps stable errors, and documents the contract.
- [ ] #5 Focused VN Play tests pass, Bandit is run for touched backend scope, and final diff hygiene is clean.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use Docs/superpowers/plans/2026-05-09-vn-play-branch-navigation-api-implementation-plan.md as the implementation plan of record.

Execution mode: subagent-driven development, one implementation subagent per plan task, with spec compliance review followed by code quality review before moving to the next task.

Initial sequence:
1. Run preflight and focused VN Play baseline.
2. Dispatch Task 1 implementer for the pure branch navigation read model only.
3. Review Task 1 for spec compliance and code quality.
4. Integrate and commit Task 1 before dispatching Task 2.
5. Continue task-by-task through repository session actions, service integration, restore semantics, API endpoints, docs, and final verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preflight completed on rebased branch codex/vn-play-branch-navigation-api at origin/dev 8e52700d0 plus local planning commits. Focused VN Play baseline run: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q => 71 passed, 5 warnings in 19.26s.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
