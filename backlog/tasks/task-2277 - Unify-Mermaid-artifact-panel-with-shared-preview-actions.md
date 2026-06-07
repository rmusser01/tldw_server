---
id: TASK-2277
title: Unify Mermaid artifact panel with shared preview actions
status: Done
assignee:
- '@codex'
labels:
- frontend
- chat
- mermaid
- artifacts
priority: medium
references:
- Docs/superpowers/specs/2026-06-04-chat-mermaid-diagrams-design.md
- Docs/superpowers/plans/2026-06-04-chat-mermaid-diagrams-implementation-plan.md
- https://github.com/rmusser01/tldw_server/pull/2284
- https://github.com/rmusser01/tldw_server/pull/2292
modified_files:
- Docs/superpowers/plans/2026-06-07-mermaid-artifact-preview-unification-plan.md
- apps/packages/ui/src/components/Sidepanel/Chat/ArtifactsPanel.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx
- backlog/tasks/task-2277 - Unify-Mermaid-artifact-panel-with-shared-preview-actions.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Unify diagram artifacts in the chat artifact panel with the shared Mermaid preview/action surface used by assistant markdown Mermaid blocks. Keep user messages unchanged and avoid broad non-chat markdown behavior changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Artifact panel diagram rendering reuses the shared Mermaid action/preview surface or a shared wrapper rather than bespoke action behavior.
- [x] #2 Diagram artifacts keep copy/download/open behavior consistent with assistant Mermaid blocks.
- [x] #3 Focused tests cover artifact-panel diagram rendering and preview actions.
- [x] #4 No changes are made to user-message Mermaid rendering behavior.
- [x] #5 Verification results and any known skips are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Docs/superpowers/plans/2026-06-07-mermaid-artifact-preview-unification-plan.md` for the three-stage implementation checklist.
- Updated `ArtifactsPanel` diagram rendering to use `MermaidDiagramBlock` with `enableArtifactAction={false}` instead of bare `Mermaid`.
- Updated `ArtifactsPanel.mermaid.test.tsx` with a shared-block contract test for preview/copy/download controls and absence of the recursive `View Mermaid diagram` action.
- TDD red run: `bunx vitest run src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx` failed because `shared-mermaid-diagram-block` was absent while the panel still rendered bare `Mermaid`.
- Verification:
  - `bunx vitest run src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx` passed: 1 file, 2 tests.
  - `bunx vitest run src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx src/components/Common/__tests__/MermaidDiagramBlock.test.tsx src/components/Common/__tests__/MermaidPreviewDialog.test.tsx` passed: 3 files, 25 tests.
  - `git diff --check` passed.
  - `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --project tsconfig.json` failed in existing latest-`dev` KnowledgeQA test fixtures outside this change: `KnowledgeQALayout.behavior.test.tsx` and `knowledgeQaStateFixtures.ts`.
- Bandit not run: touched scope is frontend TypeScript tests/components plus Markdown task/plan metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Mermaid diagram artifacts in the chat artifact panel now reuse the shared `MermaidDiagramBlock` action/preview surface instead of rendering bare `Mermaid`. The panel keeps its existing footer actions while the embedded diagram block supplies consistent Mermaid preview, source-copy, and SVG-download controls without exposing the recursive artifact-open action. Focused Mermaid tests and whitespace checks passed; full UI type-check remains blocked by existing latest-`dev` KnowledgeQA fixture errors outside this change.
Draft PR: https://github.com/rmusser01/tldw_server/pull/2292
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
