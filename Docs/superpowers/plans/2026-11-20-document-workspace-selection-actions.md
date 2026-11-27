# Document Workspace Selection Actions Implementation Plan

**Goal** Make the selected-text actions usable in PDF and EPUB readers and reliably open Chat with the selected passage.

**ADR assessment** No new ADR. The selection and prompt handoff are transient workspace state under the existing account boundary. Document chat continuation follows [ADR-049](../../ADR/049-chat-history-selection-ownership.md).

## Stage 1 — Reproduce the action failures

**Status** Complete

**Success criteria** Tests expose loss of the Ask AI prompt before Chat mounts, and loss of the parent selection when a portal or a dialog is used. Cover PDF text and the EPUB CFI.

**Tests** Extend `TextSelectionPopover.test.tsx`, the Document Chat sidebar tests and the existing workspace page tests.

## Stage 2 — Repair the selection and handoff

**Status** Complete

**Success criteria** Highlight saves at the selected location. Copy and Listen operate on the selected text. Translate keeps the passage, handles rapid language changes, and allows retry. Ask AI opens Chat at every breakpoint and adds the prompt to the active conversation's unsent draft.

**Tests** Verify selection boundary events and stale document or account changes. Verify that nested controls can be used, Escape closes the current layer, and shortcut focus returns to its origin.

## Stage 3 — Label, verify and publish

**Status** In Progress

**Success criteria** Concise action labels appear on hover and keyboard focus and remain visible on touch. Browser and test results accompany both fixes on the existing Dev PR, followed by a Hosted PR.

**Tests** Exercise selectors, tab navigation and drawer width. Run focused Vitest, frontend typecheck, scoped lint, and a Hosted patch replay. Review and record the actual results in TASK-15585.

The request contract also covers the selected text sent to Translate and Listen. These routes reject a stale expected account before dispatch. The public EPUB reader reuses the Hosted archive-opening fix so the selection actions can run on uploaded EPUB files.
