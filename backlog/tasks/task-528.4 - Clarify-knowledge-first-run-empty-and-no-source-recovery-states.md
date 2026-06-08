---
id: TASK-528.4
title: Clarify /knowledge first-run empty and no-source recovery states
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 02:54'
labels:
  - webui
  - extension
  - knowledge
  - ux
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-06-07-knowledge-first-run-empty-recovery-plan.md
parent_task_id: TASK-528
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clarify the /knowledge first-run, empty, no indexed source, no selected source, offline, and failed-search recovery states for beginner users. The page remains Knowledge QA for searching a personal library and reviewing grounded answers with citations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First-run copy explains what /knowledge searches and how grounded answers use citations.
- [x] #2 No indexed sources is distinct from no selected sources.
- [x] #3 Recovery CTAs route to add/index sources, select sources, retry backend, or enable web fallback only when appropriate.
- [x] #4 No-source disabled search state has visible inline explanation, not hover-only tooltip.
- [x] #5 Beginner recovery states are covered by component and route tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-06-07-knowledge-first-run-empty-recovery-plan.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a Knowledge QA ready-state recovery classifier that distinguishes no indexed sources, no selected sources, web-only search, backend unavailable, and ready states. Updated first-run copy to describe searching selected personal-library sources and inspecting citations. Added no-indexed add/index CTAs to existing Media/Notes surfaces, no-selected source selection CTAs, visible inline disabled-search explanations, server-capability-aware web fallback controls, and conditional no-results actions. Added WebUI and extension route specs for empty recovery states; runtime browser execution is documented as blocked by local Chromium/WXT issues.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Clarified /knowledge beginner recovery states without adding flashcard behavior. The page now distinguishes no indexed library sources from no selected source categories, disables Ask with visible inline recovery copy when the library cannot be searched, gates web fallback controls on server capability, and hides nearest-match recovery unless backend candidate data exists. Verification: focused Knowledge QA Vitest suite passed 48 tests; extension TypeScript compile passed; WebUI Playwright route execution was blocked by Chromium MachPort permission denial; extension route execution was blocked by the known WXT production build hang (TASK-306); Bandit was not applicable because no Python files were touched.
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
