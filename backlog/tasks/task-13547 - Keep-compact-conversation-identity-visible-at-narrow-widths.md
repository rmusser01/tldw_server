---
id: TASK-13547
title: Keep compact conversation identity visible at narrow widths
status: Done
labels:
- chat
- ux
- accessibility
dependencies:
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The shared compact header hides the active conversation title below 640px while retaining redundant branding. Restore readable truncated identity and existing full-title editing without changing owner authority or removing controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The current title remains visible and fits the compact header at 320px, 390px and desktop widths.
- [x] #2 Existing full-title editing, Escape cancellation, handoff readiness and header controls continue to work.
- [x] #3 Native wrapper evidence, both client type checks, existing regressions and independent reviews document the bounded fix and its limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Origin/dev 2c5f19d0328360d45bce99b8bae75ce252cc312f is included in HEAD 3fa7c77138. Duplicate compact-title search returned none; remote/open/local maxima 13530/13526/13546 checked before ID allocation. Accepted master design Stage4 owns compact title priority. ADR required: no; CSS layout repair retains existing title loading, selection authority, routes, persistence and controls. ADR049 remains governing selection authority; ADR059 governs task editing. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_compact_title_13547.md.
All four bounded stages complete. CSS-only shared fix, four callers. Actual compact wrapper 320/390/1280 with five actions, visible56/126/192px truncated title and no header overflow; full title edit and changed-then-Escape preserved saved name. Final15/4 existing behavior tests, both types0, effective lint1/1/0added, two owned JSX regions and diff pass; two independent final source reviewers clear. All5 original saved message hashes unchanged, full-chat/default viewport restored. Exact artifacts and excluded stale bundle/temporary formatter/localNode26 failure evidence in master design. TSX-only Bandit N/A. Existing packaged/physical/AT/zoom gates remain; broader stages stay open. Latestdev2c5f19d freshly fetched/included. Owned plan retired after recording completion.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The compact header retains visible truncated identity at small widths and gives it space before redundant branding, preserving all controls and existing full-title editing. Native wrapper and existing behavior/type/static evidence are recorded in the master design; no new loader, state authority or dependency. All three acceptance criteria complete within this bounded scope.
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
