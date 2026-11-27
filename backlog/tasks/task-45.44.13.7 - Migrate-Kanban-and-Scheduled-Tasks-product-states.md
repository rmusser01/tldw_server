---
id: TASK-45.44.13.7
title: Migrate Kanban and Scheduled Tasks product states
status: Done
labels:
- design-system
- webui
- kanban
- scheduled-tasks
references:
- TASK-45.44.13
- https://github.com/rmusser01/tldw_server/issues/1670
- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_visual_qa_checklist.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clear the Kanban and Scheduled Tasks product-state findings under GitHub #1670 and remove the stale Models baseline entry. Parent task TASK-45.44.13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Kanban checklist errors, draft messages, and import guidance use the shared Alert and preserve their content and actions.
- [x] #2 Scheduled task details use the canonical unavailable label, and the exact Kanban and stale Models exceptions are removed.
- [x] #3 Focused guard and behavior tests, type checking, and representative browser verification pass. Record before and after counts with other areas untouched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. The migration follows the existing shared product-state design contract without changing APIs or token policy. Base 2053ec5ad4.
Verified on latest remote dev 2053ec5ad4. The parent #1670 area had 26 blocked findings, 14 allowed exceptions, and one stale entry. After this slice it has 22 blocked findings, 13 allowed exceptions, and zero stale entries. Repository totals changed from 97 to 93 blocked and from 35 to 34 allowed. The exact finding IDs in other areas are identical. The full verifier exits 1 for the remaining blockers.
Red guards identified four affected components and the retired entries. A targeted test caught insertion of a new status region on Stop. The final UI reuses the same live region, keeps checklist/generation failures assertive, and uses the standard foreground token for readable alert text.
Passed 303 focused Kanban, Scheduled Tasks, and guard tests, full shared-UI typecheck, scoped ESLint, and git diff --check. Bandit is inapplicable to the TypeScript, JSON, and Markdown scope.
Chromium verified the real WebUI import modal at 1440x900 and 390x844 with a fixture backend. It preserved both format descriptions, the upload affordance, and the modal bounds. Desktop before and after captures were inspected. Rendered dark-theme guidance text measured 10.56:1. The Chrome extension build and isolated packaged-extension Kanban import probe passed using a loopback fixture server. Local browser evidence is in /private/tmp/tldw-tracker-1673.dVPeGP.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated Kanban checklist, generation, and import states to shared alerts, reused the canonical Scheduled Tasks unavailable label, and retired the two scoped baseline entries. Verified the focused slice and recorded the remaining parent-area findings.
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
