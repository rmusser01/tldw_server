---
id: TASK-13271
title: Consolidate the three chat shortcuts surfaces and correct the documented binding
status: In Progress
assignee: []
created_date: 2026-09-19 14:47
updated_date: 2026-09-19 14:47
labels:
- chat
- webui
- consistency
- ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three controls whose names are variants of "shortcuts" open three different panels: a left-rail control opens the page navigator, a top-bar control opens the keyboard reference, and a chat toolbar control opens a third chat-specific panel. The chat panel documents Shift+/ as "Open keyboard shortcuts"; pressing Shift+/ actually opens the page navigator. The keyboard reference lists its own binding as a third value. Verified live by probing each control by accessible name.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One concept has one name; navigation is not labelled as shortcuts.
- [x] #2 One keyboard-shortcuts surface exists, on one binding.
- [ ] #3 Every advertised binding performs the action it documents.
- [x] #4 The chat-specific bindings appear as a section of the single surface.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-09: User authorized all identified chat issues and enhancements. Investigation during TASK-13270 found normal WebLayout mounts legacy KeyboardShortcutsModal while PageHelpModalHost and PageHelpModal also listen to the same shortcuts event, opening two dialogs. Consolidate under existing demand-loaded PageHelpModalHost, preserve explicit shortcuts/tutorial intent, remove the competing page-navigator ? binding and chat-specific help panel, label navigation by its purpose, and document the effective configured chat bindings in the unified surface. Test real events/keyboard/rendered dialogs, close/focus/draft and warm/cold opening. ADR required: no; presentation ownership and existing optional chunk behavior only; ADR049/050 conversation ownership unchanged, ADR059 tracking. Paired plan Docs/Plans/IMPLEMENTATION_PLAN_chat_tour_host_13270.md.
Visible shared role=dialog and role=alertdialog guards now govern both question-mark opening and Escape dismissal; actual Joyride and real stacked Command Palette controls qualify ownership. Hidden retained ancestor visibility reuses the existing helper. One existing global Help host owns tutorial/shortcut intent and configured binding documentation; page navigation has purpose-specific labels, full and compact scopes exclude unavailable commands. Both independent source reviews clear current ownership and target lifecycle delta; final lazy resource-load review/build/native gates still pending.
Independent source follow-up confirms remaining capability mismatches: header-hidden/independently hidden-sidebar full shells still register Sidebar/Quick shortcuts without their UI hosts, palette Switch Model targets absent shell settings; compact wide Pro forced docked sidebar cannot honor collapse, although overlay toggle works. Keep AC3 open until handlers/palette/docs share actual availability and Pro collapse is resolved. Normal CmdK works after waiting for real lazy dialog. This is distinct from unified reference ownership, which is qualified; do not mark whole shortcut task Done from the current matrix.
Unified ownership/reference core is verified with13270 and committed incrementally;AC3 stays open for host capability mismatch and compact docked Pro collapse. Own continuation plan Docs/Plans/IMPLEMENTATION_PLAN_chat_shortcut_capabilities_13271.md; source reviewers checking media hydration/mobile host edges before implementation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
