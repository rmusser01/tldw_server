---
id: TASK-45.38
title: Migrate AgentRegistry status labels to design-system registry
status: Done
assignee:
  - codex
created_date: '2026-05-13 14:23'
updated_date: '2026-05-13 14:29'
labels:
  - design-system
  - webui
  - extension
dependencies: []
references:
  - apps/packages/ui/src/components/Option/AgentRegistry/index.tsx
  - >-
    apps/packages/ui/src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the shared product-state design-system migration by routing AgentRegistry runtime setup/availability status labels through the canonical design-system state registry. Keep scope limited to AgentRegistry status label resolution, focused test coverage, and removal of the migrated canonical-state-label baseline entries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AgentRegistry status labels for available, setup-required, and unavailable states resolve through getDesignSystemState with safe fallbacks.
- [x] #2 Focused AgentRegistry tests cover the design-system registry calls without brittle full-module mocking.
- [x] #3 The product-state guard baseline no longer contains AgentRegistry canonical-state-label exceptions for Ready or Unavailable.
- [x] #4 Focused tests and the design-system product-state verifier pass or any unrelated baseline failures are documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused AgentRegistry coverage that requires runtime status labels to use getDesignSystemState for available/setup-required states while preserving the existing module exports.
2. Run the focused AgentRegistry test to confirm the design-system expectation fails against the current hardcoded labels.
3. Update AgentRegistry status label resolution to map available -> ready, requires_setup -> setup_required, unavailable -> unavailable via getDesignSystemState with safe fallback labels.
4. Remove only the migrated AgentRegistry canonical-state-label baseline entries for Ready and Unavailable.
5. Verify with focused AgentRegistry tests, product-state guard tests, bun run verify:design-system-state, git diff --check, and document frontend-only Bandit skip if applicable.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline verifier on fresh origin/dev passed with 504 allowed legacy exceptions: 480 antd-product-state-import and 24 canonical-state-label. AgentRegistry has two canonical-state-label exceptions and existing focused tests, making it a narrow next slice.

RED: bunx vitest run src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx --reporter=dot failed because Setup required was not rendered from the design-system registry; the component still rendered the local hardcoded Setup Required label.

Implementation: AgentRegistry now maps available/requires_setup/unavailable to ready/setup_required/unavailable design-system state keys, resolves labels through getDesignSystemState with DESIGN_SYSTEM_STATES fallback definitions, and preserves the existing status color behavior.

Verification: focused AgentRegistry test passed with 5 tests; combined AgentRegistry plus product-state guard tests passed with 57 tests; bun run verify:design-system-state passed with baseline exceptions reduced from 504 to 502 and canonical-state-label reduced from 24 to 22; git diff --check passed.

TypeScript caveat: bunx tsc --noEmit --pretty false --project tsconfig.json still fails on existing repo-wide baseline errors in unrelated tests/components; no reported errors referenced AgentRegistry or the touched baseline file. Bandit skipped because this is a frontend-only TypeScript/JSON slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated AgentRegistry runtime status labels to resolve through the canonical design-system state registry. Available, setup-required, and unavailable statuses now map to ready, setup_required, and unavailable registry states with DESIGN_SYSTEM_STATES fallbacks, preserving the existing status color behavior while centralizing user-facing state copy.

Added focused AgentRegistry coverage that preserves the real design-system module and spies only on getDesignSystemState. Removed the two obsolete AgentRegistry canonical-state-label baseline exceptions for Ready and Unavailable.

Verification passed for focused AgentRegistry tests, AgentRegistry plus product-state guard tests, bun run verify:design-system-state, and git diff --check. Package-wide TypeScript remains blocked by unrelated existing baseline errors; no errors referenced AgentRegistry or the touched baseline file. Bandit was skipped because this is a frontend-only TypeScript/JSON change.
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
