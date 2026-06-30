---
id: TASK-45.44.6.6
title: Migrate GuardianSettings alerts to design-system Alert
status: Done
assignee: []
created_date: '2026-05-30 16:00'
updated_date: '2026-05-30 16:06'
labels:
  - design-system
  - webui
  - product-state
  - settings
dependencies: []
references:
  - TASK-45.44.6
  - 'https://github.com/rmusser01/tldw_server/issues/1663'
  - apps/packages/ui/src/components/Option/Settings/GuardianSettings.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the nine GuardianSettings AntD Alert product-state callouts to the shared design-system Alert primitive while preserving self-monitoring, guardian controls, crisis resources, and server-availability copy. Remove the matching baseline exceptions and verify the scoped Settings/account-security guard state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GuardianSettings no longer imports AntD Alert or renders AntD Alert product-state callouts.
- [x] #2 Representative self-monitoring, guardian controls, crisis resources, online, and offline guidance renders inside the design-system Alert container.
- [x] #3 GuardianSettings product-state baseline exceptions are removed and the scoped product-state guard is clean.
- [x] #4 Verification is recorded, including any unrelated baseline guard blockers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect GuardianSettings alert branches and existing tests to identify focused render assertions.
2. Add failing tests that representative GuardianSettings guidance copy renders inside the design-system Alert container.
3. Replace AntD Alert with the shared Alert primitive while preserving title/body copy and conditional rendering.
4. Remove the nine matching GuardianSettings baseline entries and run focused tests, scoped product-state guard, TypeScript, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented via TDD: first added DS Alert ancestor assertions to GuardianSettings and GuardianSettings.connection tests and observed the expected red failures against AntD Alert. Migrated all GuardianSettings info/warning product-state callouts to DsAlert while preserving global unavailable, self-monitoring unavailable, guardian controls unavailable, crisis resources unavailable/disclaimer, auth/setup/unreachable/offline copy, and navigation actions. Removed the nine GuardianSettings allowed-legacy baseline entries. Verification: GuardianSettings focused Vitest files failed before implementation with eight DS ancestor failures, then passed with 16/16 tests; scoped product-state guard for src/components/Option/Settings/GuardianSettings.tsx reported no issues; baseline JSON parse passed with GuardianSettings baseline count 0, Settings path count 21, total baseline count 165; TypeScript tsc --noEmit exited 0; git diff --check exited 0. Full verify:design-system-state remains red on unrelated existing blocked findings in WritingPlayground, Notes, ResearchWorkspace, and WorkspaceCapabilityRemediation. Bandit skipped because this is frontend TS/JSON only with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated GuardianSettings product-state callouts from AntD Alert to the shared design-system Alert primitive, added regression assertions for DS alert rendering across unavailable/offline/crisis guidance states, and removed the obsolete GuardianSettings baseline exceptions.
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
