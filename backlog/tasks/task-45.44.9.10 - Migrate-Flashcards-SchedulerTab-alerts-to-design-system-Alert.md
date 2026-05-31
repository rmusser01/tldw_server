---
id: TASK-45.44.9.10
title: Migrate Flashcards SchedulerTab alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- flashcards
- product-state
priority: medium
parent_task_id: TASK-45.44.9
references:
- https://github.com/rmusser01/tldw_server/issues/1666
- apps/packages/ui/src/components/Flashcards/tabs/SchedulerTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/SchedulerTab.editor.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
modified_files:
- apps/packages/ui/src/components/Flashcards/tabs/SchedulerTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/SchedulerTab.editor.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- backlog/tasks/task-45.44.9.10 - Migrate-Flashcards-SchedulerTab-alerts-to-design-system-Alert.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the Flashcards SchedulerTab save-error and FSRS switch callouts off AntD Alert and onto the canonical design-system Alert primitive while preserving scheduler editing behavior and copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SchedulerTab save-error UI renders the design-system Alert primitive instead of AntD Alert.
- [x] #2 SchedulerTab FSRS switch information callout renders the design-system Alert primitive instead of AntD Alert.
- [x] #3 Focused SchedulerTab editor coverage proves both callouts remain visible and expose the design-system Alert marker.
- [x] #4 Design-system product-state baseline no longer contains SchedulerTab Alert exceptions and the product-state verifier passes or records unrelated baseline failures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing SchedulerTab editor assertions requiring the save-error and FSRS switch callouts to render with data-ds-component="Alert".
2. Replace SchedulerTab AntD Alert usage with the canonical design-system Alert primitive, preserving visible copy.
3. Remove SchedulerTab Alert exceptions from the product-state baseline.
4. Run focused SchedulerTab tests, design-system verifier, git diff --check, and frontend typecheck scope if practical.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD red/green completed. Added SchedulerTab editor assertions for the save-error and FSRS switch guidance callouts requiring a `data-ds-component="Alert"` ancestor. The red run failed because the existing AntD Alert rendered the copy without the design-system marker. Production code now imports the canonical design-system Alert primitive and renders the save-error as an error title and the FSRS switch guidance as an info Alert with title/body. Removed the two stale SchedulerTab Alert baseline exceptions. PR #2090 review follow-up replaced the brittle `"SM-2+"` visible-text dropdown opener in the FSRS guidance test with the stable `deck-scheduler-editor-field-scheduler-type` test id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated SchedulerTab's save-error and FSRS switch guidance callouts from AntD Alert to the design-system Alert primitive while preserving scheduler editing behavior and visible copy. Focused SchedulerTab editor coverage now verifies both callouts render with data-ds-component="Alert", and the product-state baseline no longer contains SchedulerTab Alert exceptions. PR #2090 review feedback was addressed by opening the scheduler-type Select through its stable test id instead of visible `"SM-2+"` copy. Verification: red SchedulerTab editor test failed on missing design-system markers; green SchedulerTab editor test passed 12/12; git diff --check passed; design-system verifier was run and still fails on unrelated Integrations/Writing/Research baseline drift plus stale Integrations baseline entries, with no SchedulerTab findings; full UI typecheck still fails on existing CharacterListContent.design-system.test.tsx density typing outside this slice. Bandit skipped because this slice touched frontend TSX/test/JSON and Backlog markdown only.
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
