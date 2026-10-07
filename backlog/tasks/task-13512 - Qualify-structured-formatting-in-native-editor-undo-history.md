---
id: TASK-13512
title: Qualify structured formatting in native editor undo history
status: In Progress
assignee:
- '@codex'
created_date: 2026-10-07 02:16
updated_date: 2026-10-07 09:32
labels:
- notes
- structured-editing
dependencies: []
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editing.md
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
Monaco qualification approved by human option1 on 2026-10-07. ADR required: yes; existing governing ADR194(C)/ADR065(S), amended before any new runtime/asset contract if needed. Continue I1 using installed Monaco only for structured source modes with bundled/local assets in Next and extension, unchanged Markdown ownership. Read task-11-monaco-context.md and exact review constraints. Qualify actual temporary Notes-page integration/native keyboard history in all three engines plus source representation/currentness/composition/focus/caret/model isolation; no custom undo or source CDN. Remove temporary product wiring before explicit commits. Append dated candidate evidence/report and self-review, keep task In Progress for independent review. Do not rerun unchanged C bridge tests.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Installed Monaco test-only candidate independently approved for exact LF/CRLF native history and guarded source editing: 36 all-browser controls plus 3 strengthened viewport controls, units2+2, frontend typecheck exit0 with12GB heap and actual MV3 local-assets/CSP smoke. Temporary product wiring/assets removed; Markdown ownership retained. BOM/mixed/lone-CR sources are refused read-only. Existing textarea remains unqualified. ADR194(C)/ADR065(S) apply. Overall I1 remains In Progress: I3 owns shared loader ordering, production packaging and refresh/currentness; I5 owns real OS IME/accessibility/narrow-layout release evidence. Missing attachment paths and unbaselined warning/heap limits are retained for final review. See durable editor-history report and scoped review/scope-resolution artifacts.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Candidate qualification independently approved; overall I1 remains In Progress until explicit production/IME/accessibility gates are verified. Native LF/CRLF history and source preservation are evidenced; no production formatter adoption claim.
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
