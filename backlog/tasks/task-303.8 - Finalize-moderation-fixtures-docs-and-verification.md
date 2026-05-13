---
id: TASK-303.8
title: Finalize moderation fixtures docs and verification
status: To Do
assignee: []
created_date: '2026-05-13 00:58'
labels:
  - moderation
  - webui
  - docs
  - tests
dependencies:
  - TASK-303.7
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-moderation-review-rules-remediation-implementation-plan.md
parent_task_id: TASK-303
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 8 of the moderation remediation plan. Stabilize the route split and review workflow through fixtures, documentation, regression coverage, and final focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 E2E fixtures cover populated queue, empty queue, permission denied, backend error, partial data, expired undo, and redacted content states.
- [ ] #2 Smoke inventory uses canonical /moderation and /moderation/rules routes, with /moderation-playground retained only as legacy redirect coverage.
- [ ] #3 Moderation docs distinguish review from content rules, document permissions, sanitized data, retention/minimization, and review env vars.
- [ ] #4 Known unsupported producer states are documented if review producers remain incomplete.
- [ ] #5 Focused backend, frontend, E2E, design-state, Bandit, and diff hygiene verification results are recorded.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
