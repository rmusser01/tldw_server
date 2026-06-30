---
id: TASK-210
title: Migrate WorkspaceStatusStrip ready label to design-system state registry
status: Done
assignee: []
created_date: '2026-05-10 02:28'
updated_date: '2026-05-10 02:34'
labels:
  - design-system
  - frontend
  - product-state
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the WorkspaceStatusStrip runtime ready label with the canonical design-system ready state label and remove the matching product-state baseline exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WorkspaceStatusStrip uses getDesignSystemState('ready').label for the backend-available idle runtime label.
- [x] #2 Focused tests prove the ready fallback reads from the design-system state registry rather than a hardcoded literal.
- [x] #3 The canonical-state-label baseline entry for WorkspaceStatusStrip is removed and the design-system product-state verifier passes.
- [x] #4 Pre-existing unbaselined PersonaGarden visual-library availability status is migrated off AntD Tag so the product-state verifier has no blocked findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused RED test proving WorkspaceStatusStrip ready status reads from getDesignSystemState('ready').label. 2. Replace the WorkspaceStatusStrip hardcoded Ready runtime label with the design-system state registry label. 3. Remove the matching canonical-state-label baseline exception. 4. Clear the pre-existing VisualPackEditor product-state verifier blocker by replacing availability AntD Tags with token-backed spans. 5. Run focused tests, product-state guard, verifier, diff check, and broad tsc touched-file filtering.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented WorkspaceStatusStrip ready status through getDesignSystemState('ready').label and added a partial design-system mock in its focused test. Removed canonical-state-label baseline entry for src/components/Option/ChatWorkspace/WorkspaceStatusStrip.tsx:Ready. While running verify:design-system-state, found a pre-existing blocked finding on src/components/PersonaGarden/VisualPackEditor.tsx caused by AntD Tag availability badges. Replaced those availability badges with token-backed spans preserving the visible lowercase copy. Bandit skipped because touched code is UI TypeScript/JSON/Backlog markdown with no Python execution surface.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated WorkspaceStatusStrip ready label to the design-system state registry and removed its product-state baseline exception. Also cleared a pre-existing unbaselined VisualPackEditor availability Tag blocker so the product-state verifier is green. Verification: RED WorkspaceStatusStrip Vitest failed on mocked registry label before implementation; GREEN WorkspaceStatusStrip Vitest passed 3/3; VisualPackEditor focused test passed 20/20; product-state guard passed 52/52; verify:design-system-state passed with baseline exceptions now 507 total and 39 canonical-state-label; git diff --check passed; broad UI tsc still fails on existing unrelated repo-wide debt with no touched-file matches.

PR: https://github.com/rmusser01/tldw_server/pull/1491
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
