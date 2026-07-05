---
id: TASK-12172
title: Verify Characters page dark-theme visual fidelity
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-05 18:42
labels:
- frontend
- theme
- webui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Walk through the WebUI Characters page in dark mode, check menus/options for light-theme or low-contrast visual drift, apply the smallest shared-token fix if needed, and record focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Characters page dark-mode walkthrough has no large light-surface leaks.
- [x] #2 Characters page dark-mode walkthrough has no low-contrast dark text leaks.
- [x] #3 Characters page primary visible controls and at least one page option/menu state are covered by focused visual regression.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extended the existing dark-theme visual-fidelity Playwright smoke to cover /characters, including the filters panel, Display dropdown, and New character drawer. Fixed confirmed shared Ant theme leaks in dropdown item text, popover wrappers, and drawer wrapper surfaces through apps/packages/ui/src/assets/tailwind-shared.css. Tightened the visual scan to ignore hidden, aria-hidden, and sr-only subtrees so non-visible helper markup does not produce false positives.

PR review follow-up: added dark coverage for selected table rows, active dropdown items, and clipped Ant select wrappers to prevent rounded-corner leaks.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Characters dark-theme walkthrough has now been rerun against a live FastAPI backend on 127.0.0.1:8000 and Next WebUI on localhost:8080, with no API route mocks. The authenticated live pass covered the base Characters table, real character row/card selection, card More actions dropdown, filters panel, Manage tags modal, Display menu options (Comfortable/Compact/Dense/Keyboard shortcuts), and New character drawer. Live API calls included /api/v1/characters/query and /api/v1/characters/world-books with X-API-KEY present and returned 200. Fixed additional real-data light-theme leaks in Ant notification, table/header/body/sorted-column, and tag components through the shared CSS bridge. Verification: node /private/tmp/tldw-real-characters-dark-check.mjs passed with no visual leaks, no request failures, and no bad API statuses; npx playwright test e2e/smoke/dark-theme-visual-fidelity.spec.ts --reporter=line passed; git diff --check -- apps/packages/ui/src/assets/tailwind-shared.css passed. Whole-repo git diff --check remains blocked by unrelated Docs/Design/Tool-Calling.md trailing EOF blank line. Bandit skipped because touched repo code is frontend CSS only.

PR review follow-up added selected-row and active-dropdown dark-state selectors, clipped the Ant select wrapper, and kept the visual scan bounded to visible viewport content.
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
