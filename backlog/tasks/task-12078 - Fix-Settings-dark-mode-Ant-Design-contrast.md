---
id: TASK-12078
title: Fix Settings dark mode Ant Design contrast
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-30 19:52
labels:
- webui
- accessibility
- dark-mode
dependencies: []
modified_files:
- apps/packages/ui/src/assets/tailwind-shared.css
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Settings pages render several Ant Design components with light-theme text and surfaces in dark mode, causing dark text on dark app backgrounds. Fix the shared AntD theme/override layer and validate the Settings page in the browser.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings form labels, tab labels, helper text, inputs, segmented controls, and default buttons are readable in dark mode.
- [x] #2 The fix uses shared theme/override infrastructure rather than one-off copy changes.
- [x] #3 Browser validation confirms key Settings controls meet readable contrast in dark mode.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: Ant Design generated component styles were still winning for Settings labels, tabs, helper text, inputs, segmented controls, and default buttons, so those controls kept light-theme black text/surfaces inside the app dark shell.

Implementation: extended the shared AntD bridge in apps/packages/ui/src/assets/tailwind-shared.css to map common AntD text, surface, border, input, tab, segmented, and default-button selectors to semantic app tokens with enough specificity to beat generated component CSS.

Verification: browser checked http://127.0.0.1:3000/settings/tldw in dark mode. Computed contrast after fix: heading 15.58, inactive tab 8.93, active tab 5.57, form label 15.58, helper text 9.69, input 13.29, password wrapper 13.29, segmented 13.29, selected segmented label 14.36, default button 14.36. Timeouts tab interaction selected successfully and browser console had no warnings/errors. Ran git diff --check and bun run test:run ../packages/ui/src/components/Option/Settings/__tests__/tldw-settings-tabs.test.tsx from apps/tldw-frontend: 4 tests passed. Earlier bunx vitest from repo root failed before test execution because temporary latest Vitest could not resolve jsdom and globbed nested worktrees; reran through package-local script successfully. Bandit skipped because only CSS changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Settings dark-mode readability by extending the shared Ant Design CSS bridge so Settings tabs, labels, helper text, inputs, segmented controls, cards/modals, and default buttons use the app semantic theme tokens. Follow-up PR review fixes preserve AntD danger/success/warning/disabled states, keep card surfaces distinct from elevated overlays, and preserve dangerous default-button styling while retaining dark-mode readability. Verified the live Settings route in dark mode with computed contrast checks and a tab interaction before review follow-up; package-local Settings tab tests passed. Bandit not applicable for CSS-only change.
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
