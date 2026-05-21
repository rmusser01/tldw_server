---
id: TASK-457
title: Implement Character Chat Phase 7 model usability and send gating
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-21 01:33'
labels:
  - character-chat
  - roleplay
  - frontend
  - implementation
dependencies: []
references:
  - TASK-456
  - TASK-455
  - TASK-426
  - TASK-454
documentation:
  - Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
  - >-
    Docs/superpowers/plans/2026-05-20-character-chat-phase7-model-usability-send-gating-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Phase 7 from the Character Chat first-class PRD and implementation plan: model usability contract, readiness truth, status-surface alignment, SEND gating, provider/model failure recovery, and real-backend verification. Scope must stay on /chat Character Chat role-play readiness and send behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Model usability helper classifies loading, no_server, no_selection, no_models, selected_missing, provider_unconfigured, model_unavailable, degraded, and ready states with focused unit coverage.
- [ ] #2 Character Chat readiness panel, status strip, runtime inspector, composition preview, model selector copy, and SEND action consume one shared model-usability result and do not show positive health copy for unusable models.
- [ ] #3 Character selected plus no usable model blocks or converts SEND into a setup action without invoking submit, without calling /complete-v2, and without losing draft/character/session state.
- [ ] #4 Provider/model setup failures show actionable model/provider recovery copy instead of generic retry-only guidance when the failure is configuration-specific.
- [ ] #5 Real-backend Playwright verification covers no-provider/send-gating without simulated frontend responses; successful-send is verified only through a real callable provider or explicitly marked blocked by environment.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Execution started with superpowers:subagent-driven-development and superpowers:test-driven-development. Controller ran baseline before production edits: bunx vitest run ../packages/ui/src/utils/__tests__/chat-model-availability.test.ts ../packages/ui/src/components/Option/Playground/__tests__/CharacterChatReadinessPanel.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx --reporter=verbose passed with 3 files / 39 tests. Two Task 1 workers were shut down after no edits/progress reports; controller implemented Task 1 locally under TDD. Red test: focused utility test failed with 11 failures because buildChatModelUsability was not a function. Green tests: focused utility test passed with 33 tests; baseline trio passed with 3 files / 50 tests. Task 1 added the pure model usability contract/helper and unit coverage only; UI wiring remains pending.

Task 1 final: addressed spec-review gaps for provider aliases, colon-bearing local model IDs, and provider-qualified duplicate matching. Verification: focused utility suite passed with 41/41 tests; baseline trio passed with 3 files / 58 tests; git diff --check clean. Review gates: Task 1 spec review approved and code-quality review approved for HEAD ebda82474. Task 1 remains UI-unwired by design; Task 2 will map Character Chat readiness to the usability contract.

Task 2 local TDD: added failing readiness and panel tests for models-loading, selected-model-missing, provider-unconfigured, model-unavailable, no-models-available copy, ready, and send-disabled ordering. Implemented buildCharacterChatReadiness on top of buildChatModelUsability and added precise blocker copy preserving character/draft context. Verification: focused readiness/panel suite passed with 2 files / 50 tests; baseline trio passed with 3 files / 63 tests; git diff --check clean. AC #2 is not checked yet because status strip, runtime inspector, composition preview, model selector, and SEND controls are later tasks.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
