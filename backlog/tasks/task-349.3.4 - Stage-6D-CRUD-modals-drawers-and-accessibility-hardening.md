---
id: TASK-349.3.4
title: Stage 6D CRUD modals drawers and accessibility hardening
status: To Do
dependencies:
- TASK-349.3.3
labels:
- watchlists
- stage6
- frontend
- accessibility
priority: medium
parent_task_id: TASK-349.3
documentation:
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md
- Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden constrained CRUD modals, drawers, action focus, and keyboard behavior for Watchlists source forms, OPML import, monitor forms/previews, template editor, settings drawer, and cross-tab navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Source form, OPML import, monitor form, monitor preview, template editor, and settings drawer primary actions remain visible and usable at constrained width.
- [ ] #2 Modal/drawer footers do not clip at 420x760, and dense editors use full-width or stacked layouts where needed.
- [ ] #3 Keyboard navigation reaches constrained navigation, create/edit/delete actions, and primary drawer/modal actions with accessible names.
- [ ] #4 Escape/cancel closes constrained drawers/modals without leaving focus trapped in removed nodes.
- [ ] #5 Focused Vitest coverage records modal/drawer width, accessibility, and keyboard regression behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Depends on `TASK-349.3.3`. Follow `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md` Task 4. Prefer Ant Design Modal/Drawer APIs and local utility classes; do not introduce a custom modal system.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
