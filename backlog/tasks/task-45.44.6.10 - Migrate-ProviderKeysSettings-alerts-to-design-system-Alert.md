---
id: TASK-45.44.6.10
title: Migrate ProviderKeysSettings alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-01 06:26'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Option/Settings/ProviderKeysSettings.tsx
  - >-
    apps/packages/ui/src/components/Option/Settings/__tests__/ProviderKeysSettings.design-system-alert.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - 'https://github.com/rmusser01/tldw_server/issues/1659'
parent_task_id: TASK-45.44.6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Settings/account-security product-state migration by replacing ProviderKeysSettings AntD Alert product-state callouts with the shared design-system Alert primitive, then remove the matching baseline exceptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ProviderKeysSettings product-state callouts render through the shared design-system Alert primitive while preserving user-facing copy and severity.
- [x] #2 The ProviderKeysSettings AntD Alert product-state baseline entries are removed without introducing new unbaselined findings for that path.
- [x] #3 Focused regression coverage verifies the design-system Alert marker for the migrated branches.
- [x] #4 Focused tests, scoped product-state guard verification, TypeScript/touched-scope check, diff whitespace check, and Bandit skip rationale are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added focused ProviderKeysSettings design-system Alert regression tests for BYOK-unavailable guidance and provider-key load failures. The RED Vitest run failed 2/2 because the current AntD Alert markup did not expose data-ds-component="Alert".
- Replaced the two ProviderKeysSettings AntD Alert callouts with the shared design-system Alert primitive while preserving info/error severity, title/body copy, and dismiss behavior for the error state.
- Removed the two matching ProviderKeysSettings baseline exceptions; baseline count moved from 79 to 77 and the touched path now has zero baseline entries.
- Verification: focused ProviderKeysSettings Vitest passed 2/2; product-state guard unit passed 54/54; bun run verify:design-system-state exited 0 with 77 baseline exceptions; baseline parse reported ProviderKeysSettings count 0; git diff --check exited 0.
- TypeScript caveat: node --max-old-space-size=8192 ./node_modules/typescript/bin/tsc --noEmit --pretty false exits 2 on unrelated existing frontend baseline diagnostics in QuickIngest, Layout shell overrides, setup onboarding, and quick-ingest-open; the output included no ProviderKeysSettings, ProviderKeysSettings.design-system-alert, baseline, or TASK-45.44.6.10 diagnostics.
- Bandit skipped because this slice touched frontend TSX/test/JSON/Backlog markdown only, with no Python code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated ProviderKeysSettings BYOK-unavailable and load-error product-state alerts from direct AntD Alert usage to the shared design-system Alert primitive, added focused DOM coverage for both branches, and removed the two matching ProviderKeysSettings product-state baseline exceptions. Baseline count moved from 79 to 77 total exceptions.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
