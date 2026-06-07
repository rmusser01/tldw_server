---
id: TASK-2276
title: Address PR 2284 Mermaid artifact panel review comments
status: Done
assignee:
- '@codex'
labels:
- frontend
- tests
- review
priority: medium
references:
- https://github.com/rmusser01/tldw_server/pull/2284
- https://github.com/rmusser01/tldw_server/pull/2284#discussion_r3368815174
modified_files:
- apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx
- backlog/tasks/task-2276 - Address-PR-2284-Mermaid-artifact-panel-review-comments.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address review feedback on PR #2284 for Mermaid artifact panel rendering coverage, specifically ensuring global window event listeners in the jump-to-source test are always cleaned up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rebase PR branch onto latest origin/dev.
- [x] #2 Wrap event listener cleanup in the Mermaid artifact jump-to-source test with try/finally.
- [x] #3 Resolve or respond to the GitHub review thread.
- [x] #4 Run focused frontend verification for the touched Mermaid artifact test scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Rebased `codex/chat-mermaid-artifact-panel-tests` onto `origin/dev` at `9b36cbf232`.
- Wrapped the Mermaid artifact jump-to-source test body in `try...finally` so `tldw:focus-artifacts-trigger` and `tldw:scroll-to-latest` listeners are always removed.
- Verification:
  - `bunx vitest run src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx` passed: 1 file, 2 tests.
  - `bunx vitest run src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.jump-source.guard.test.ts src/components/Common/__tests__/MermaidDiagramBlock.test.tsx src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx` passed: 4 files, 27 tests.
  - `git diff --check` passed.
  - `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --project tsconfig.json` failed in latest `origin/dev` KnowledgeQA test fixtures outside this PR diff: `KnowledgeQALayout.behavior.test.tsx` and `knowledgeQaStateFixtures.ts`.
- Bandit not run: touched scope is frontend test code plus Backlog task metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2284 onto latest origin/dev, wrapped Mermaid artifact panel jump-to-source listener cleanup in try/finally, pushed the rebased review-fix branch, and resolved the inline Gemini review thread. Focused Mermaid/chat tests and git diff --check passed. Full UI type-check currently fails in unrelated latest-dev KnowledgeQA test fixtures outside this PR diff; Bandit was skipped because the touched scope is frontend test code plus Backlog metadata.
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
