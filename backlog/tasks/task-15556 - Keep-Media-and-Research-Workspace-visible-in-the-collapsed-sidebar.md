---
id: TASK-15556
title: Keep Media and Research Workspace visible in the collapsed sidebar
status: Done
labels:
- webui
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
New placement assertion failed on dev before the fix. 54 focused sidebar tests and scoped ESLint passed. Chromium hit-tested both buttons with the rendered sidebar and compiled shared stylesheet before scrolling at 1200 × 440 and 1200 × 320; the last shortcut remained reachable by scrolling.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Keep selected Media and Research Workspace buttons above the collapsed sidebar shortcut scroller. The rest of the list continues to scroll, and very short windows can scroll the whole rail.
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
