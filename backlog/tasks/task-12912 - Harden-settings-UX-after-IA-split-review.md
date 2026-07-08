---
id: TASK-12912
title: Harden settings UX after IA split review
status: In Progress
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the senior UX/HCI review findings on the merged settings IA split: reduce interruption from backend-down state, improve accessibility semantics, add progressive disclosure for dense UI customization settings, and clarify data management consequences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings server-unreachable state no longer blocks unrelated settings tasks with a modal.
- [x] #2 Settings pages have valid title/main landmark/form semantics for the reviewed routes.
- [x] #3 UI customization reduces first-screen cognitive load with collapsed or task-oriented sections for shortcut matrices and display settings.
- [x] #4 Data management explains export/import/reset consequences inline before action.
- [x] #5 Focused tests and visual/a11y verification cover the changed behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current merged settings implementation and existing tests.
2. Add failing tests for accessibility semantics, non-blocking error behavior, progressive disclosure, and data danger copy.
3. Implement minimal UI changes using existing components and patterns.
4. Run focused Vitest, Playwright/axe checks, and diff verification.
5. Update task, commit, and push branch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced settings-owned `main` landmarks with shell-safe `div`/`section` semantics so the hosted app keeps one page `main`.
- Added explicit document titles to the reviewed hosted settings pages.
- Converted `/settings/ui` shortcut, theme, and system display sections to collapsed native disclosures.
- Changed settings backend-unreachable handling from a modal to a non-blocking status toast on settings routes.
- Added inline export/import/reset consequence copy and tightened visible settings action button contrast.
- Added focused tests for landmark/title semantics, invalid `dl` prevention, collapsed disclosure defaults, inline unreachable status, and data consequence copy.
- Verification:
  - `bun run test:run ../packages/ui/src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/PreferencesSettings.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/UiCustomizationSettings.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/DataManagementSettings.test.tsx __tests__/components/layout/WebLayout.backend-unreachable.test.tsx ../packages/ui/src/routes/__tests__/option-settings-route-split.test.tsx`
  - `bun run typecheck`
  - `apps/tldw-frontend/node_modules/.bin/eslint -c apps/tldw-frontend/eslint.config.mjs <touched files>` exits 0; Next plugin emits a cwd notice because the config is invoked from repo root for cross-package files.
  - Playwright + axe on `/settings/preferences`, `/settings/ui`, and `/settings/data` at desktop and mobile: 0 axe violations, one `main`, no nested `main`, no invalid `dl`, reviewed titles present, UI disclosures closed by default, and no backend-unreachable dialog.
- Bandit: not applicable; touched code is frontend TypeScript/TSX plus Backlog metadata, with no Python files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the settings UX after the IA split review by removing modal interruption on settings routes, reducing `/settings/ui` density with collapsed native sections, improving route/page semantics, clarifying data action consequences, and fixing reviewed settings contrast/a11y issues.
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
