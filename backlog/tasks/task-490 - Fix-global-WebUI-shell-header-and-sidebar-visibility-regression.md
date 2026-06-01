---
id: TASK-490
title: Fix global WebUI shell header and sidebar visibility regression
status: In Progress
labels:
- webui
- extension
- layout
- shell
- regression
priority: High
documentation:
- Docs/superpowers/specs/2026-06-01-webui-shell-visibility-regression-design.md
modified_files:
- Docs/superpowers/specs/2026-06-01-webui-shell-visibility-regression-design.md
- backlog/tasks/task-490 - Fix-global-WebUI-shell-header-and-sidebar-visibility-regression.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the global WebUI/extension app shell contract so the top bar and sidebar load on normal pages. Scope includes /chat as a regression route, but the primary fix is the shared shell visibility behavior rather than route-by-route patches.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Brainstorming/design phase. Approved direction: fix the global shell contract in the WebUI root shell and nested OptionLayout handoff, then cover representative routes with regression tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design spec written after user approved the shell-first repair direction. Spec review loop completed on 2026-06-01 with status Approved. Reviewer had no blocking issues and recommended enumerating exact public/setup/settings/recovery routes during implementation planning.
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
