---
id: TASK-252
title: Run scripted VN generation through backend runtime
status: In Progress
assignee: []
created_date: '2026-05-10 23:51'
labels:
  - vn
  - scripted-generation
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1535'
documentation:
  - Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md
  - Docs/superpowers/specs/2026-05-10-vn-scripted-model-generation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 5 from Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md: integrate model-backed generate opcodes into scripted VN Play runtime while preserving literal generation behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Automatic generation creates request/revision/generation rows and advances scene
- [ ] #2 Confirmation-gated generation pauses without model call
- [ ] #3 Batch cap of one pauses before a second automatic generation
- [ ] #4 scene_update persists applied and rejected visual resolver outcomes on the active revision
- [ ] #5 Model failure persists failed revision/request and does not advance cursor
- [ ] #6 Existing literal generation tests still pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing scripted-runtime tests for automatic generation, confirmation gating, batch cap, model failure, and literal preservation. 2. Refactor script execution so model-backed generate opcodes return generation descriptors instead of requiring literal text. 3. Resolve pinned profile snapshots from published script-version maps and call the Task 4 generation orchestration helper. 4. Persist generation events and scene updates for successful outputs while leaving failed attempts at the generation point for retry. 5. Verify focused VN Play/VN Scripts tests, compile, Bandit, and diff checks.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
