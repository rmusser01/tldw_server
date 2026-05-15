---
id: TASK-45.46
title: Migrate Persona Buddy diagnostics labels to design-system registry
status: Done
assignee: []
created_date: '2026-05-15 06:41'
updated_date: '2026-05-15 07:01'
labels:
  - design-system
  - webui
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/PersonaGarden/personaBuddyDiagnostics.ts
  - apps/packages/ui/src/design-system/states.ts
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - https://github.com/rmusser01/tldw_server/pull/1722
documentation:
  - Docs/Design/tldw_web_design_system_inventory.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by routing the remaining Persona Buddy diagnostics canonical Ready and Loading row labels through the shared design-system state registry instead of local string literals. This is a narrow baseline-reduction slice for apps/packages/ui/src/components/PersonaGarden/personaBuddyDiagnostics.ts and the design-system state label exports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona Buddy diagnostics use design-system registry label exports for Ready and Loading diagnostics values without changing diagnostic state ranking or copy.
- [x] #2 Focused Persona Buddy diagnostics coverage proves the labels come from the registry contract rather than local literals.
- [x] #3 The design-system product-state baseline no longer contains Persona Buddy diagnostics canonical-state-label exceptions.
- [x] #4 Focused tests and design-system verifier results are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation completed in /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/design-system-next-slice-6 on branch codex/design-system-next-slice-6.

Persona Buddy diagnostics now consume READY_STATE_LABEL and LOADING_STATE_LABEL from the design-system registry exports, preserving the existing diagnostic ordering and state logic.

Focused coverage mocks the design-system registry labels to prove Persona Buddy diagnostics read canonical labels instead of local literals.

Removed Persona Buddy diagnostics canonical-state-label exceptions from apps/packages/ui/scripts/design-system-product-state-baseline.json.

PR opened against dev: https://github.com/rmusser01/tldw_server/pull/1722.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated Persona Buddy diagnostics Ready and Loading labels to design-system registry exports, added focused registry-label coverage, exported LOADING_STATE_LABEL from the state registry helper, and removed the Persona Buddy canonical-state-label baseline exceptions. Verification recorded: focused Persona Buddy/design-system tests passed, product-state verifier passed, combined guard suite passed, baseline JSON parse passed, git diff --check passed, and package TypeScript still has existing unrelated diagnostics with no touched-file matches. Bandit is not applicable because this slice touched TypeScript, test, and JSON frontend files only.
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
