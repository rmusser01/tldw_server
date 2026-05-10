---
id: TASK-242
title: Migrate demo media ready labels to design-system registry
status: Done
assignee: []
created_date: '2026-05-10 19:39'
updated_date: '2026-05-10 19:56'
labels:
  - design-system
  - frontend
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1547'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the frontend design-system product-state cleanup by replacing Review demo media hardcoded Ready status labels in src/utils/demo-content.ts with the canonical design-system ready state label while preserving demo media item shape and existing Processing status behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 getDemoMediaItems supplies ready demo media status labels from getDesignSystemState('ready').label rather than hardcoded Ready literals.
- [x] #2 Existing demo media titles, metadata, item order, and Processing status behavior remain unchanged.
- [x] #3 Focused utility coverage proves ready labels come through the design-system registry.
- [x] #4 The matching demo-content Ready canonical-state-label baseline exceptions are removed and the design-system verifier passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused utility test that mocks getDesignSystemState('ready') and proves getDemoMediaItems uses the registry label while preserving Processing and item order. 2. Replace the hardcoded demo Ready statuses with the ready state registry label. 3. Remove the three demo-content Ready baseline exceptions. 4. Verify focused test, product-state guard test, design-system verifier, diff check, and touched-scope typecheck output.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red: demo-content.design-system.test.ts mocked getDesignSystemState('ready') to return 'Ready via registry' and failed while getDemoMediaItems still returned literal Ready statuses. Green: getDemoMediaItems now uses getDesignSystemState('ready').label for the two ready demo media records and preserves Processing/item order. Removed the three demo-content Ready baseline exceptions.

Verification: bunx vitest run src/utils/__tests__/demo-content.design-system.test.ts --reporter=dot passed 1 test; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 52 tests; bun run verify:design-system-state exited 0 and reports baseline exceptions 510 with canonical-state-label 37; git diff --check exited 0. Broad bunx tsc --noEmit --pretty false still exits 2 with the existing 239-line repo baseline and no touched-scope matches for demo-content, baseline, task-242, getDemoMediaItems, or getDesignSystemState. Bandit skipped because this slice only touches UI TypeScript, JSON baseline data, and Backlog metadata.

PR review fix: Qodo flagged that widening DemoMediaPreview.status to string made ReviewPage styling depend on display-label comparison. Added typed statusKey ('ready' | 'processing') plus statusLabel display text, updated ReviewPage to branch on statusKey, and updated the ReviewPage demo-content mock.

Review-fix verification: bunx vitest run src/utils/__tests__/demo-content.design-system.test.ts --reporter=dot passed 1 test; bunx vitest run src/components/Review/__tests__/ReviewPage.connection.test.tsx --reporter=dot passed 3 tests; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 52 tests; bun run verify:design-system-state exited 0; git diff --check exited 0. Broad bunx tsc --noEmit --pretty false still exits 2 with the existing 239-line repo baseline and no touched-scope matches for demo-content, ReviewPage, ReviewPage.connection, baseline, task-242, statusKey/statusLabel, getDemoMediaItems, or getDesignSystemState.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
getDemoMediaItems now resolves ready demo media status labels through getDesignSystemState('ready').label while leaving Processing and demo item order unchanged. Added focused mocked-registry utility coverage and removed the three src/utils/demo-content.ts Ready canonical-state-label baseline exceptions, reducing design-system product-state baseline debt from 513 to 510 entries.

PR: https://github.com/rmusser01/tldw_server/pull/1547

PR review follow-up: demo media now separates typed status identity from display text via statusKey/statusLabel, and ReviewPage styles demo media chips from statusKey rather than comparing localized display labels.
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
