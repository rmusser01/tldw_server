---
id: TASK-399
title: Risk Gate 4 prototype API contract and error semantics freeze
status: In Progress
assignee: []
created_date: '2026-05-16 01:30'
updated_date: '2026-05-16 01:44'
labels:
  - prototype-workspaces
  - risk-gate
  - backend
  - contract
dependencies:
  - TASK-324
  - TASK-389
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1456'
  - 'https://github.com/rmusser01/tldw_server/issues/1440'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Burn down Risk Gate 4 for prototype workspace collaboration by freezing owner/collaborator endpoint response models, stable error categories, OpenAPI response metadata, frontend contract fixtures, lifecycle examples, and migration/rollback notes. This tracks GitHub issue #1456 and should remain scoped to prototype workspace backend/API contract semantics plus the minimal fixture/docs updates needed by later frontend gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 OpenAPI and API docs match implemented prototype workspace behavior.
- [x] #2 Contract matrix covers stable error categories, retryability, frontend state buckets, and suggested handling.
- [x] #3 Lifecycle examples cover owner and collaborator flows through promotion review.
- [x] #4 Migration and rollback notes document configuration and deployment requirements.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-05-15-prototype-workspace-risk-gate-4-contract-freeze.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-15: Created isolated worktree .worktrees/prototype-risk-gate-4-contract-freeze on branch codex/prototype-risk-gate-4-contract-freeze from origin/dev 2df371fbe after Risk Gate 3 PR #1729 was merged and GitHub issue #1455 was closed.

2026-05-15: Baseline verification before implementation: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q passed with 103 passed, 5 warnings in 8.73s.

2026-05-15: Implemented structured PrototypeErrorResponse detail contract for prototype workspace endpoints and public prototype-session exchange, added OpenAPI response model metadata, froze the v2 frontend contract fixture, and updated Risk Gate 4 API/matrix docs with lifecycle examples plus migration/rollback notes.

2026-05-15: Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q passed with 107 passed, 5 warnings in 8.74s. Bandit touched backend paths wrote /tmp/bandit_prototype_risk_gate_4.json with 0 findings. git diff --check was clean.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
