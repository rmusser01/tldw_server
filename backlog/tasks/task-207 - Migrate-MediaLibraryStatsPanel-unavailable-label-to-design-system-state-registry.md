---
id: TASK-207
title: >-
  Migrate MediaLibraryStatsPanel unavailable label to design-system state
  registry
status: Done
assignee: []
created_date: '2026-05-10 01:47'
updated_date: '2026-05-10 01:58'
labels:
  - design-system
  - frontend
  - product-state
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the MediaLibraryStatsPanel storage unavailable fallback label with the canonical design-system unavailable state label and remove the matching product-state baseline exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MediaLibraryStatsPanel storage unavailable fallback uses getDesignSystemState('unavailable').label as its default translation value.
- [x] #2 Focused tests prove the unavailable fallback reads from the design-system state registry rather than a hardcoded literal.
- [x] #3 The canonical-state-label baseline entry for MediaLibraryStatsPanel is removed and the design-system product-state verifier passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a regression test that mocks getDesignSystemState('unavailable') to a distinct label and asserts the storage unavailable fallback renders it. 2. Replace the MediaLibraryStatsPanel hardcoded unavailable default with the design-system state registry label. 3. Remove the matching canonical-state-label baseline exception and run focused verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented MediaLibraryStatsPanel unavailable fallback through getDesignSystemState('unavailable').label. Added focused test coverage using a partial design-system mock. Removed canonical-state-label baseline entry for src/components/Media/MediaLibraryStatsPanel.tsx:Unavailable. Bandit skipped because touched code is UI-only TypeScript/JSON/Backlog markdown with no Python execution surface.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the MediaLibraryStatsPanel storage unavailable label to the design-system state registry and removed its product-state baseline exception. Verification: RED focused Vitest failed on the mocked registry label before implementation; GREEN focused Vitest passed 4/4; product-state guard passed 52/52; verify:design-system-state passed with baseline exceptions now 508 total and 40 canonical-state-label; git diff --check passed; broad UI tsc still fails on existing unrelated repo-wide debt with no touched-file matches.

PR: https://github.com/rmusser01/tldw_server/pull/1489
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
