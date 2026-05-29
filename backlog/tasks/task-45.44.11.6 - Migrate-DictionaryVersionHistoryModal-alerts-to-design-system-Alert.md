---
id: TASK-45.44.11.6
title: Migrate DictionaryVersionHistoryModal alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- dictionaries
- product-state
priority: medium
parent_task_id: TASK-45.44.11
references:
- apps/packages/ui/src/components/Option/Dictionaries/DictionaryVersionHistoryModal.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
modified_files:
- apps/packages/ui/src/components/Option/Dictionaries/DictionaryVersionHistoryModal.tsx
- apps/packages/ui/src/components/Option/Dictionaries/__tests__/DictionariesWorkspace.layout.test.tsx
- apps/packages/ui/src/components/Option/Dictionaries/__tests__/DictionaryVersionHistoryModal.design-system.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate DictionaryVersionHistoryModal's error and success product-state AntD Alerts to the canonical design-system Alert primitive while preserving version-history error copy and revision-restored messaging.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DictionaryVersionHistoryModal error message renders through the design-system Alert primitive with the existing title and details.
- [x] #2 DictionaryVersionHistoryModal revision-restored message renders through the design-system Alert primitive with the existing title and details.
- [x] #3 The matching DictionaryVersionHistoryModal Alert product-state baseline exceptions are removed and focused guard coverage has no DictionaryVersionHistoryModal findings.
- [x] #4 Focused regression coverage proves both migrated alert states use design-system primitives.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- [x] Add focused failing tests that render the version-history load failure and revision-restored states and assert each alert uses the shared design-system Alert marker.
- [x] Replace the DictionaryVersionHistoryModal AntD Alert usages with the shared design-system Alert primitive while preserving copy.
- [x] Remove the migrated DictionaryVersionHistoryModal Alert rows from the product-state baseline.
- [x] Run focused tests, product-state guard/unit verifier checks, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `DictionaryVersionHistoryModal.design-system.test.tsx`; red state failed because the existing AntD Alerts did not provide the `data-ds-component="Alert"` marker.
- Replaced the version-history error and revision-restored AntD `Alert` usages with the shared design-system `Alert` using `variant="error"` and `variant="success"`. Existing title/detail copy is unchanged.
- Removed both `DictionaryVersionHistoryModal` Alert baseline rows.
- Verification: `bunx vitest run src/components/Option/Dictionaries/__tests__/DictionaryVersionHistoryModal.design-system.test.tsx` passed.
- Review follow-up: added explicit null guards before narrowing the focused Alert container query results to `HTMLElement`, so missing design-system wrappers fail as assertion failures instead of later `within()`/property access errors.
- CI follow-up: fixed `DictionariesWorkspace.layout.test.tsx`'s local `react-router-dom` mock to provide `useLocation`, matching `WorkspaceConnectionGate`'s current dependency.
- Verification: `bunx vitest run src/components/Option/Dictionaries/__tests__/DictionariesWorkspace.layout.test.tsx --maxWorkers=1 --no-file-parallelism` passed after reproducing the CI failure locally.
- Verification: `bun run test:dictionaries` passed with 30 test files / 124 tests.
- Verification: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed.
- Verification: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed with an empty `/tmp/dictionary-version-alerts-tsc-after-ci-fix.log`.
- Verification: `git diff --check` passed.
- Product-state verifier: `bun run verify:design-system-state` still exits 1 on existing current-dev guard drift in IntegrationPolicyPanel, WritingActionBar, Notes, and ResearchWorkspace surfaces plus stale IntegrationPolicyPanel baseline entries; `/tmp/design-system-dictionary-version-alerts-post-rebase.log` contains no `DictionaryVersionHistoryModal` findings and reports 205 remaining baseline exceptions.
- Bandit: skipped because this slice only touches frontend TypeScript/TSX, JSON, and Backlog task metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the `DictionaryVersionHistoryModal` version-history error and revision-restored states to the shared design-system Alert primitive, added focused marker coverage for both states with explicit null guards, and removed the retired product-state baseline exceptions. Also fixed the dictionary layout test's stale router mock after the PR's dictionary CI job exposed its missing `useLocation` export. Focused regression coverage, the full dictionary Vitest script, product-state guard unit coverage, TypeScript, and diff whitespace checks passed; the full product-state verifier remains blocked by existing current-dev guard drift outside this slice and has no DictionaryVersionHistoryModal findings.
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
