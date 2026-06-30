---
id: TASK-303.3
title: Fix moderation content rules accessibility and responsive blockers
status: Done
assignee: []
created_date: '2026-05-12 20:42'
updated_date: '2026-05-12 22:33'
labels:
  - moderation
  - webui
  - accessibility
dependencies:
  - TASK-303.2
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-moderation-review-rules-remediation-implementation-plan.md
  - >-
    Docs/superpowers/specs/2026-05-12-moderation-review-rules-remediation-design.md
parent_task_id: TASK-303
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 3 of the moderation review/rules remediation plan. The /moderation/rules Content Rules surface should have programmatic labels, keyboard-friendly controls, responsive table containment, associated inline errors, polite dynamic status regions, and browser-verified behavior at mobile, tablet, and desktop widths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Icon-only buttons in moderation context and inline tools have stable accessible names.
- [x] #2 Inputs, textareas, selects, AntD selects, and hidden file input triggers have programmatic labels or labelled-by relationships.
- [x] #3 Phase segmented controls expose radio semantics and remain keyboard-operable.
- [x] #4 Tab bars and undo/confirmation flows preserve focus and keyboard accessibility.
- [x] #5 Rules, history, and overrides tables contain horizontal overflow internally without page-level horizontal scroll at 390px.
- [x] #6 Inline validation and dynamic result/error areas are associated with controls and announced politely where appropriate.
- [x] #7 Focused Vitest accessibility coverage and Playwright/CDP responsive checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 3 accessibility/responsive remediation for /moderation/rules: accessible names and labels, radio semantics, keyboard tab handling, focus return paths, live regions, internal table overflow containment, and layout min-width fixes for WebLayout and shared Layout.
Verification: Vitest moderation suite passed 20 files / 217 tests; Playwright/CDP moderation responsive + route suite passed 7 tests; git diff --check passed.
Known blocker: full tsc remains blocked by unrelated baseline errors in EmbeddingsModelSelectionConfig.tsx, persona-visuals.ts, and lib/api/vnPlay.ts. Bandit skipped because touched files are TS/TSX/tests/docs/backlog only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 3 complete. Moderation content rules now has programmatic labels for controls and upload inputs, keyboard-operable radio/tab patterns, focus return after quick-test and destructive confirmation flows, polite status/error announcement, and browser-verified mobile/tablet/desktop overflow containment. Added Testing Library accessibility assertions and Playwright/CDP responsive coverage. Full TypeScript still fails on unrelated pre-existing baseline errors outside this slice.
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
