---
id: TASK-15556
title: Keep Media and Research Workspace visible in the collapsed sidebar
status: Done
labels:
- webui
references:
- https://github.com/rmusser01/tldw_server/pull/3490
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The collapsed sidebar puts both selected shortcuts at the end of its scrolling list, where shorter browser windows clip them. Keep both visible above that list while preserving selections and expanded navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected Media and Research Workspace shortcuts are visible in the collapsed sidebar without scrolling its shortcut list.
- [x] #2 Other selected shortcuts stay reachable and omitted shortcuts stay hidden.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The placement assertion failed on dev before the fix. The independent follow-up review reproduced clipping with classic scrollbar gutters and a zero-height remaining shortcut list at a very short height when both priority icons were omitted. The follow-up removes the outer scrollbar gutter, uses a thin inner scrollbar and inset shortcut focus treatment, keeps room for any remaining shortcuts, and tests the order from a saved nondefault selection. All 180 sidebar tests, scoped ESLint, and compiled CSS verification passed. Chromium reproduced the old clipping with classic gutters, then verified both priority icons and a focused ordinary shortcut at 1200 × 320, plus keyboard access to Chat and Settings at 900 × 220 with both priority icons omitted. Native Windows, Firefox, and Safari were not exercised. No Python production code changed. The owner added the human-written change summary to PR #3490. Rebased onto dev with the merged Playwright upgrade after the old runner failed before browser assertions. The repository exposes no public dev WebUI deployment target.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Keep selected Media and Research Workspace buttons above the collapsed sidebar shortcut scroller. The rest of the list continues to scroll, and very short windows can scroll the whole rail.
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
