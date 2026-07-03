---
id: TASK-12095
title: Remove chat cockpit top rail and solidify character picker menus
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-02 05:52'
labels:
  - webui
  - chat
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up fixes for the chat page/webui cockpit: remove the top rail, place the focus/cockpit control beside Shortcuts, and ensure character picker menus render with solid backing instead of transparent overlays.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No standalone top cockpit rail is rendered in chat.
- [x] #2 The focus/cockpit button is in the same row as the Shortcuts button and immediately to its left.
- [x] #3 Character picker dropdown/menu popups have an opaque themed surface and readable contrast.
- [x] #4 Focused tests cover top-rail removal/control placement and picker menu surface opacity.
- [x] #5 Browser QA verifies desktop and mobile chat layout after the change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the shell-level cockpit header, moved the focus/cockpit control into the chat utility row immediately before Shortcuts, made assistant/character picker popups use opaque themed surfaces, and split the mobile composer so the send control no longer steals textarea width.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented chat cockpit follow-up fixes: no top cockpit rail, focus/cockpit toggle placed directly left of Shortcuts, solid picker popup surfaces, and improved mobile composer width. Verification: focused Vitest suite passed (9 files, 69 tests); browser QA at 1440x960 and 390x844 confirmed no top rail, correct control order, opaque picker background rgb(23, 26, 31), no mobile horizontal overflow, and 260px mobile textarea width; git diff --check passed; Bandit on touched frontend scope reported 0 findings/0 Python LOC. TypeScript full UI pass still fails on unrelated baseline errors in Notes, AudioStudio, ScheduledTasks, background, MCP hub, voice cloning, and useChatActions.character.integration test.
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
