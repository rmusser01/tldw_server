---
id: TASK-45.44.6.2
title: Migrate TldwConnectionSettings auth alerts to design-system Alert
status: Done
assignee:
- Codex
created_date: ''
updated_date: 2026-05-16 16:20
labels:
- design-system
- webui
- extension
- product-state
dependencies: []
references:
- apps/packages/ui/src/components/Option/Settings/TldwConnectionSettings.tsx
- apps/packages/ui/src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/1781
documentation:
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
parent_task_id: TASK-45.44.6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the remaining TldwConnectionSettings authentication status alerts from AntD Alert to the shared design-system Alert primitive while preserving login-required and logged-in copy, logout behavior, and product-state guard coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TldwConnectionSettings renders login-required and logged-in auth status messages through the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 Existing auth mode, magic-link, and logout behavior remain covered by focused tests.
- [x] #3 The design-system product-state baseline no longer contains TldwConnectionSettings AntD Alert exceptions.
- [x] #4 Focused tests, design-system verifier, git diff check, and TypeScript/Bandit applicability are recorded before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused tests proving login-required and logged-in auth notices render inside the design-system Alert primitive and that logged-in logout action still calls onLogout.
2. Run the focused test before implementation to capture the expected failure while TldwConnectionSettings still uses AntD Alert.
3. Replace the two AntD Alert auth notices with the shared design-system Alert primitive, mapping info/success variants and converting the logout action to the primitive action contract.
4. Remove the two TldwConnectionSettings AntD Alert baseline entries.
5. Verify focused tests, product-state verifier, git diff check, and document TypeScript/Bandit applicability.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented via TDD. Added failing coverage for the login-required and logged-in auth notices to require the shared design-system Alert marker and preserve the logged-in Logout action. Migrated both TldwConnectionSettings auth notices from AntD Alert to DesignSystemAlert with info/success variants and polite status regions. Extended the Alert primitive action contract with an optional Button variant so the logout action can keep danger styling while using the primitive action API. Removed the two TldwConnectionSettings AntD Alert baseline exceptions.

Verification: focused red test failed on missing data-ds-component marker before implementation. After implementation, bunx vitest run src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 58 tests. bun run verify:design-system-state passed with baseline exceptions reduced from 400 to 398 and Settings exceptions at 47. Baseline JSON parse passed. git diff --check passed. bunx tsc --noEmit --pretty false still exits 2 on existing repo-wide TypeScript debt; filtered output for TldwConnectionSettings, tldw-review-comments, components/ui/primitives/Alert, and the baseline file had no matches after fixing the touched test TFunction import. Bandit not run because this slice touches frontend TypeScript/JSON/backlog files only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated TldwConnectionSettings login-required and logged-in auth notices to the shared design-system Alert primitive, preserved logout behavior and danger action styling through a small Alert action variant extension, and removed the two resolved product-state baseline exceptions.

PR #1781 review follow-up moved info/success polite status-region defaults into the Alert primitive, removed duplicate role/aria-live props from TldwConnectionSettings, renamed the local import to Alert, and verified that the logout action's danger variant is valid for the existing Button API.
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
