---
id: TASK-2274
title: Address Mermaid chat card PR review comments
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 05:25'
labels:
- webui
- chat
- mermaid
- review
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/2276
- 'PR #2276 review pass'
modified_files:
- apps/packages/ui/src/components/Common/Playground/Message.tsx
- apps/packages/ui/src/components/Common/Playground/CompactMessage.tsx
- apps/packages/ui/src/components/Common/MermaidDiagramBlock.tsx
- apps/packages/ui/src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx
- apps/packages/ui/src/components/Knowledge/__tests__/KnowledgePanelQAPreview.test.tsx
- apps/packages/ui/src/utils/stable-hash.ts
- backlog/tasks/task-2266 - Fix-TaskActivityNotice-i18next-count-typing.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2276 onto latest dev and address validated review feedback for Mermaid chat card artifacts, compact message hashing, context-id fallbacks, task timestamp serialization, and rebased UI TypeScript verification blockers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2276 is rebased onto latest origin/dev.
- [x] #2 Validated review comments are addressed with scoped code or task metadata changes.
- [x] #3 Regression tests cover safe fallback contexts, per-segment Mermaid namespaces, and disabled compact artifact contexts.
- [x] #4 Local focused tests and UI TypeScript verification pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased branch onto origin/dev at 6c2b2b7c8f with no conflicts. Verified review feedback before fixing: duplicate Mermaid artifact contexts across message Markdown segments, unsafe undefined/null fallback context, CompactMessage hashing while Mermaid actions are disabled, duplicated branch-local hash helper, and unquoted TASK-2266 updated_date. Added failing Message Mermaid tests for the three behavioral review issues, then fixed them. The PR's Wizard coverage check failure was inspected and failed before tests due Docker Hub postgres image pull timeouts; sampled full-suite failed job had no failed test output. A rebased TypeScript baseline issue in KnowledgePanelQAPreview.test.tsx was fixed by removing an unnecessary third vi.mock option for installed antd.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2276 onto latest origin/dev and addressed validated review comments. Mermaid artifact contexts now use safe local fallbacks plus per-render greeting/segment namespaces, CompactMessage only computes/passes artifact contexts when Mermaid actions are enabled, the branch-local hash helper is shared through stableHashString, and TASK-2266 updated_date is quoted. Also fixed the rebased UI TypeScript blocker in KnowledgePanelQAPreview.test.tsx. Verification: Message Mermaid red tests failed on the intended review regressions, then passed; focused Vitest suite passed (5 files, 49 tests); Knowledge QA preview test passed (3 tests); UI TypeScript check passed; git diff --check and locale JSON parse passed. Bandit is not applicable because only TypeScript, tests, JSON/task metadata, and docs/task files were touched.
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
