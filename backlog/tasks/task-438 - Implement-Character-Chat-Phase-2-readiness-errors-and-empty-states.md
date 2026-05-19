---
id: TASK-438
title: Implement Character Chat Phase 2 readiness errors and empty states
status: In Progress
assignee: []
created_date: 2026-05-19 04:18
labels:
- chat
- characters
- role-play
- phase-2
- frontend
- accessibility
dependencies: []
references:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
- TASK-426
- TASK-428
- TASK-429
- TASK-431
documentation:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Phase 2 from the first-class Character Chat PRD: make incomplete Character Chat setup, loading, no-provider, deleted/missing character, prompt/assistant catalog failures, and persistence state local and actionable on /chat. Preserve selected character intent through model/settings recovery and expose setup status changes to assistive tech.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Missing server, missing character, missing model, unavailable model, prompt load failure, and character catalog failure all have visible local Character Chat states.
- [ ] #2 Selecting a character before model setup preserves character intent through settings handoff and retry.
- [ ] #3 Character Chat setup/readiness status changes are exposed through appropriate live-region semantics.
- [ ] #4 Focused unit/integration tests and real-backend browser smoke coverage are recorded; Bandit is run or explicitly skipped for non-Python scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Follow `Docs/superpowers/plans/2026-05-19-character-chat-phase2-readiness-plan.md`. 2. Add failing tests for readiness panel, selector failure states, persistence labels, and restored missing character recovery. 3. Implement minimal existing-surface UI changes. 4. Verify with focused Vitest, real-backend browser smoke where available, Bandit skip note for frontend-only scope, and closeout evidence.
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
