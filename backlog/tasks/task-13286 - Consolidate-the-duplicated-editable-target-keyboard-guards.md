---
id: TASK-13286
title: Consolidate the duplicated editable-target keyboard guards
status: Done
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-24 00:24'
labels:
  - webui
  - keyboard
  - refactor
  - ux-audit
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The predicate answering "is the user typing right now" is reimplemented at least fourteen times across the shared UI package under at least five different names, including isEditableTarget, isInputFocused, shouldIgnoreShortcut, isInputField and isTypingTarget. Every other screen applies it correctly; the chat page was the one that computed it and applied it too late, which produced the question-mark defect. The duplication is why one site could drift without anyone noticing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One shared helper covers input, textarea, select and contenteditable targets.
- [x] #2 Every global keyboard handler in the shared UI package uses it.
- [x] #3 The duplicate local definitions are removed.
- [x] #4 A test covers each element kind the helper must treat as editable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Commit 0a99b2a8c6. AC1: apps/packages/ui/src/utils/editable-target.ts exports isEditableTarget(target). It covers input, textarea and select; contenteditable including descendants, the empty-value and plaintext-only forms, and excluding contenteditable=false; and role textbox/combobox/searchbox. AC2/AC3: 30+ local copies were removed and every global keyboard handler found in packages/ui now calls the shared helper. The removed copies were: isEditableShortcutTarget (hooks/keyboard), isInputElement (hooks/useKeyboardShortcuts), isEditableTarget (Playground, message shortcuts, FamilyGuardrailsWizard, KnowledgeQA SearchBar, ItemsTab), isInputFocused (Kanban, Watchlists), shouldIgnoreShortcut (ModerationReview), shouldIgnoreGlobalShortcut (Notes utils, both Dictionaries hooks), shouldIgnoreNotesDockShortcutTarget, isEditableKeyboardTarget (ResearchWorkspace), isEditableEventTarget (WorkflowEditor; the file and its 4-case test were deleted and are covered by the new test), plus inline copies in Layout, DocumentWorkspacePage, DocumentViewer, EpubViewer, useTranscriptDisplay, Flashcards x3, Prompt, Skills Manager, STT, QuickIngestTabs, KnowledgeTabs, SourceList, useMediaKeyboardShortcuts, Review interaction-context, useCharacterShortcuts, useSelectionKeyboard and OmniSearchBar. Deliberate per-site extras were kept and commented: useSelectionKeyboard lets checkboxes through, ItemsTab also blocks .ant-select-dropdown, and STT also skips buttons for Space. Divergences found and unified (bugs by the task's own framing): most copies skipped <select>; many missed descendants of contenteditable roots; only 2 honoured ARIA textbox roles; OmniSearchBar checked only an exact contenteditable='true' on the target. AC4: src/utils/__tests__/editable-target.test.ts has 18 cases. Against the old Playground isEditableTarget it gets 7 failed / 11 passed, because in jsdom every contenteditable and role case fails. With the new helper it gets 18/18. Before/after 'vitest related' over all 34 touched sources: before, 28 failed files / 76 failed tests (pre-existing, mostly NotesManagerPage/MediaReviewPage/Kanban suites). After, 27 / 75, and the FAILED list only lost Playground.search.integration 'Cmd/Ctrl+F', probably a load flake; there are no new failures. +18 passing tests from the new file. Typecheck: tsc -p apps/packages/ui/tsconfig.json has the same 367 pre-existing errors before and after, as an identical set. Lint: packages/ui has no eslint config or lint script, and tldw-frontend's eslint ignores files outside its base path, so tsc is the only static gate for these files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced 30+ duplicated 'is the user typing' guards across packages/ui with one shared utils/editable-target.ts isEditableTarget. Every global keyboard handler now uses it. Per-site extras are kept only where they are deliberate: checkbox navigation, the antd dropdown portal, and Space on buttons. An 18-case test covers each element kind; the old Playground copy fails 7 of those cases. Vitest over all related suites and tsc show no new failures.
<!-- SECTION:FINAL_SUMMARY:END -->
