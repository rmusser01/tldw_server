---
id: TASK-2278
title: Unify Research Workspace Mermaid artifact modal preview
status: Done
assignee:
- '@codex'
created_date: ''
updated_date: 2026-06-07 16:57
labels:
- frontend
- mermaid
- research-workspace
- artifacts
dependencies: []
references:
- Docs/superpowers/specs/2026-06-04-chat-mermaid-diagrams-design.md
- https://github.com/rmusser01/tldw_server/pull/2292
- https://github.com/rmusser01/tldw_server/pull/2293
- https://github.com/rmusser01/tldw_server/pull/2293#pullrequestreview-3332723163
priority: medium
modified_files:
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/ArtifactModalContent.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx
- Docs/superpowers/plans/2026-06-07-research-mermaid-artifact-preview-unification-plan.md
- backlog/tasks/task-2278 - Unify-Research-Workspace-Mermaid-artifact-modal-preview.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Unify the Research Workspace Mermaid artifact modal path with the shared Mermaid preview/action components used by chat Mermaid blocks and chat artifact cards, without changing user-message rendering behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Research Workspace Mermaid artifact modal reuses the shared Mermaid preview/action component path or a shared wrapper instead of maintaining separate Mermaid preview controls.
- [x] #2 Existing raw-output fallback for non-Mermaid or invalid mind-map output remains intact.
- [x] #3 Focused tests cover the shared preview controls for Mermaid artifact modal content.
- [x] #4 No user-message Mermaid rendering behavior changes are introduced.
- [x] #5 Verification results and any known skips are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-07-research-mermaid-artifact-preview-unification-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented via the existing shared MermaidDiagramBlock path in MindMapArtifactViewer. Valid Research Workspace Mermaid mind-map artifact modal content delegates preview, copy, and SVG-download controls to the shared block with enableArtifactAction=false. The raw fallback branch for invalid or non-Mermaid output is unchanged, and no user-message markdown rendering paths were changed. Verification recorded: RED focused Vitest failed before implementation on missing research-shared-mermaid-block, Research Workspace mind map focused Vitest passed 4 tests, full StudioPane.stage2 passed 26 tests, shared Mermaid tests passed 27 tests, git diff --check passed, higher-heap tsc failed only on existing unrelated KnowledgeQA fixture errors, and Bandit is not applicable for frontend-only TypeScript docs and Backlog changes.

PR review follow-up: rebased PR #2293 onto latest origin/dev after PR #2294 merged. Addressed CodeRabbit's prop-contract comment by removing the now-unused title prop from MindMapArtifactViewer and the corresponding title prop at the modal call site. Verification after the review fix: Research Workspace mind map focused Vitest passed 4 tests, shared Mermaid tests passed 27 tests, git diff --check passed, and higher-heap UI type-check still fails only on existing unrelated KnowledgeQA fixture errors with no diagnostics in touched files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Research Workspace Mermaid mind-map artifact modals now reuse the shared Mermaid diagram block used by assistant-facing Markdown surfaces. The modal keeps its raw fallback for invalid/non-Mermaid output, disables nested artifact actions, and removes the previous modal-local zoom/PNG/export control path in favor of shared preview/copy/SVG-download behavior.
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
