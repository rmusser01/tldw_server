---
id: TASK-13263
title: Stop swallowing every question mark typed in the chat composer
status: Done
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - keyboard
  - p0
  - ux-audit
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The global key handler on /chat intercepted a bare "?" before its editable-target guard ran, so no question mark could be typed into the chat composer. In a chat client this blocks the most common way users phrase a message. The defect predates the 2026-09-15 audit and shipped untested because synthetic key events omit the shift flag that a physical "?" carries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A "?" typed into the chat composer appears in the message text.
- [x] #2 A "?" pressed outside any editable target still opens the shortcuts help panel.
- [x] #3 Cmd/Ctrl+F from the composer still opens in-thread search.
- [x] #4 Escape from the thread-search input still closes it.
- [x] #5 A regression test fails when the editable-target guard is removed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce with a real keystroke against a live stack and rule out a test artifact with a control element.
2. Read the handler and identify why the existing guard does not apply to this branch.
3. Reject the blanket early return: confirm it would break modifier chords and Escape handling.
4. Guard the affected branch only, reusing the helper already present in the same directory.
5. Extract the branch condition so the defect can be pinned, and add regression tests.
6. Prove the tests fail against the unfixed condition, then verify all four behaviours live.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Guarded the "?" branch itself rather than hoisting a blanket early return above the handler body.

The obvious fix, moving the existing editable-target return to the top of the handler, was rejected after checking what else sits above it. Cmd/Ctrl+F and the two Escape-to-close branches deliberately run while focus is inside an input. The composer is focused on page load, so an early return would have made in-thread search unreachable in the default state, and Escape would have stopped closing the thread-search bar from its own input. Both were verified as regressions before choosing the narrower fix. A modifier chord cannot be produced by ordinary typing, so intercepting it inside a text field is a product decision, not this bug.

Reused the isEditableTarget helper already defined in playground-shortcuts.ts, which also covers select elements and does an instanceof check, and deleted the weaker inline duplicate in Playground.tsx. Extracted the branch condition as shouldOpenShortcutsHelp so it can be tested without mounting a 4,400-line component.

The defect shipped untested because synthetic typing does not set the shift flag the condition requires, so the branch never fired in tests. The new cases set it explicitly. Verified the two regression tests fail when the guard is removed and pass with it.

Checked every other plain-character keyboard handler in the shared UI package. All of them guard correctly, including the extension shell, so this was a single-site bug rather than a defect class. The underlying duplication is filed separately.

Verified live against llama.cpp, backend and WebUI: "?" types in the composer, "?" outside an input still opens the panel, Cmd+F still opens thread search, Escape still closes it.

Modified: Playground.tsx, playground-shortcuts.ts, __tests__/playground-shortcuts.test.ts
<!-- SECTION:NOTES:END -->
