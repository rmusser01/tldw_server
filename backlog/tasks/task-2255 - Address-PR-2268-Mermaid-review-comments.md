---
id: TASK-2255
title: Address PR 2268 Mermaid review comments
status: Done
labels:
- frontend
- chat
- mermaid
- pr-review
references:
- https://github.com/rmusser01/tldw_server/pull/2268
modified_files:
- apps/packages/ui/src/components/Common/Markdown.tsx
- apps/packages/ui/src/components/Common/MermaidDiagramBlock.tsx
- apps/packages/ui/src/components/Common/MermaidPreviewDialog.tsx
- apps/packages/ui/src/components/Common/Playground/CompactMessage.tsx
- apps/packages/ui/src/components/Common/__tests__/Markdown.mermaid.test.tsx
- apps/packages/ui/src/components/Common/__tests__/MermaidDiagramBlock.test.tsx
- apps/packages/ui/src/components/Common/__tests__/MermaidPreviewDialog.test.tsx
- apps/packages/ui/src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx
- backlog/tasks/task-2255 - Address-PR-2268-Mermaid-review-comments.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2268 on latest dev, inspect and address PR review comments and GitHub checks, verify the Mermaid chat rendering changes, and update the PR branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased `codex/chat-mermaid-diagrams-pr` on `origin/dev` and inspected the unresolved PR review threads before applying fixes. The review comments mapped to four frontend risks: closed-fence detection drift, duplicate diagram labels, redundant SVG sanitization in wrappers, and over-broad streaming suppression.

The Markdown renderer now records closed Mermaid fences with normalized line endings, case-insensitive language matching, source line ranges, and a fenced-block fallback. Diagram wrappers rely on the already-sanitized Mermaid output instead of re-sanitizing in `MermaidDiagramBlock` and `MermaidPreviewDialog`. Diagram block labels include a React component ID to avoid duplicate `aria-labelledby` targets when message-local block indexes repeat. Compact message rows only disable Mermaid rendering for the currently streaming final row.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2268 branch rebased on latest origin/dev. Review feedback addressed in Markdown Mermaid matching, diagram/preview SVG handling, source reset behavior, header ID uniqueness, and compact-message streaming gating. Verification recorded: focused Mermaid-related Vitest suite passed (8 files, 63 tests); git diff --check passed; escalated Next Turbopack frontend build passed with pre-existing broad-pattern warnings and token-sync OK. Bandit not run because this task touched only TypeScript/TSX test/frontend code plus Backlog metadata, no Python code.
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
