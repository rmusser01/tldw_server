---
id: TASK-13512
title: Qualify structured formatting in native editor undo history
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-07 02:16'
updated_date: '2026-10-07 02:53'
labels:
  - notes
  - structured-editing
dependencies: []
documentation:
  - >-
    /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editors.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove explicit formatting uses the existing editor history before integrating parser output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Real native typing, formatting, typing, undo, undo and redo preserve the correct document states and earlier history.
- [ ] #2 Identity and editability guards refuse stale/read-only results; no-op output creates no edit.
- [ ] #3 Selection, caret, focus, scroll and composition behavior are qualified without a custom snapshot undo stack.
- [ ] #4 Qualification evidence names runtime versions, test commands and unsupported environments; temporary production wiring is removed before commit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: Docs/ADR/031-notes-capability-sync-domains.md; approved cross-app design ADR-194 linked from external spec
Reason: native history qualification implements the approved reversible-edit rule without changing editor ownership.
1. Read I1 in the approved editors plan.
2. Add real native history tests and minimal guarded bridges.
3. Temporarily wire the actual Notes page for fixed replacement qualification, record the trace, then remove temporary wiring.
4. Run targeted bridge/browser checks, record limitations, review and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
I1 browser adoption blocked: controlled Notes history passes Chromium145/Firefox146 but WebKit26 first Undo returns A instead of C. Original variant reproduced twice. One authorized synchronous blur/focus boundary on both sides of native insertText also fails exactly D -> A on WebKit; further experiments stopped. Product wiring removed; candidate retained only as opt-in failure fixture. Six guard tests and targeted tsc pass. Evidence/proposal: Docs/superpowers/reviews/2026-09-27-document-editor-history.md. No new ADR; existing ADR031/ADR194 govern. In Progress pending revised editor approval.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
