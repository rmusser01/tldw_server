---
id: TASK-45.24
title: Normalize PersonaGarden loading labels for product-state guard
status: Done
assignee: []
created_date: '2026-05-09 04:37'
updated_date: '2026-05-09 04:41'
labels:
  - design-system
  - webui
  - guard
dependencies: []
references:
  - apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx
  - apps/packages/ui/src/design-system/states.ts
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the new PersonaGarden hardcoded Loading fallbacks flagged by the product-state verifier with design-system state registry labels so fresh dev guard output returns to the accepted baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PersonaGarden loading button labels use getDesignSystemState("loading").label instead of hardcoded Loading strings.
- [x] #2 The design-system verifier has no blocked PersonaGarden canonical-state-label findings after the change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Initial verifier run failed on fresh origin/dev plus the StatusTag slice with two blocked canonical-state-label findings for hardcoded Loading fallbacks in VisualPackEditor.tsx. Using that verifier output as the red guard evidence for this small cleanup.

Implementation: VisualPackEditor now uses getDesignSystemState("loading").label for its loading button fallbacks instead of new hardcoded Loading strings.

Verification: bun run verify:design-system-state passed with no blocked PersonaGarden canonical-state-label findings; the final verifier summary reported baseline exceptions 512 and local-status-badge 6. The focused StatusTag test, product-state guard test suite, and git diff --check also passed on the combined slice.

TypeScript caveat: bunx tsc --noEmit --pretty false still fails on unrelated repo-wide frontend baseline errors; a filtered tsc pass for the touched files produced no VisualPackEditor errors.

Bandit: skipped because this cleanup only changes TypeScript and Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Normalized PersonaGarden loading labels to the design-system state registry so the product-state verifier returns to the accepted baseline while the Watchlists StatusTag slice removes its own baseline exception.
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
