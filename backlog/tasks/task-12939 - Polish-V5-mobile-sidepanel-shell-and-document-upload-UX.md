---
id: TASK-12939
title: Polish V5 mobile sidepanel shell and document upload UX
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-09 14:39'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up polish for the V5 mobile sidepanel composer: remove the WebUI debug shell header/sidebar from the sidepanel debug route, reduce redundant healthy connection status, simplify nested composer chrome, and make document attachment explicit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sidepanel debug route renders only the sidepanel header, not the WebUI shell header/sidebar.
- [x] #2 Connected empty state omits the redundant healthy Connected body status and shows a shorter suggestion set.
- [x] #3 V5 compact composer removes the extra top border/frame and uses readable meta facets.
- [x] #4 V5 mobile action row exposes an explicit Attach document button wired to the existing context file upload flow.
- [x] #5 Regression tests cover the mobile sidepanel route and compact V5 composer chrome.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-09-v5-mobile-sidepanel-polish.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the WebUI shell chrome from the sidepanel debug route, trimmed redundant healthy connected empty-state copy, reduced compact V5 composer chrome, made V5 document attachment explicit via the existing context file picker, and added regression coverage for the sidepanel/mobile rendering behavior.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the V5 mobile sidepanel polish pass. Verification: targeted Vitest passed, Playwright mobile smoke passed, frontend TypeScript passed, git diff --check passed, Bandit skipped because no Python files changed. Package UI TypeScript still fails on existing unrelated Notes/background errors outside this task scope.
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
