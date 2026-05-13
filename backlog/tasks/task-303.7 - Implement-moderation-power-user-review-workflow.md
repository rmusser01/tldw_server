---
id: TASK-303.7
title: Implement moderation power-user review workflow
status: To Do
assignee: []
created_date: '2026-05-13 00:58'
labels:
  - moderation
  - webui
  - frontend
dependencies:
  - TASK-303.6
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-moderation-review-rules-remediation-implementation-plan.md
parent_task_id: TASK-303
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 7 of the moderation remediation plan. Add efficient repeat-review workflows after audit and undo semantics are stable, while keeping destructive bulk decisions explicit and auditable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Queue rows support multi-select with selected count and clear selection controls.
- [ ] #2 Bulk decision bar supports approve, dismiss, block, redact, and escalate with confirmation and required reasons for high-risk actions.
- [ ] #3 Bulk endpoint integration renders partial failures inline without losing successful decisions.
- [ ] #4 Saved local filter presets and scoped keyboard shortcuts improve repeat-review speed without firing while typing.
- [ ] #5 Completion state appears when needs_review reaches zero and links to audit and content rules.
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
