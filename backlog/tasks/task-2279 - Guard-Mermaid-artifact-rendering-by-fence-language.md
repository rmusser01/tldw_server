---
id: TASK-2279
title: Guard Mermaid artifact rendering by fence language
status: Done
references:
- Docs/superpowers/specs/2026-06-04-chat-mermaid-diagrams-design.md
- Docs/superpowers/plans/2026-06-04-chat-mermaid-diagrams-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Common/CodeBlock.tsx
- apps/packages/ui/src/components/Common/Markdown.tsx
- apps/packages/ui/src/components/Common/__tests__/CodeBlock.artifacts.test.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/ArtifactsPanel.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx
documentation:
- Docs/superpowers/specs/2026-06-04-chat-mermaid-diagrams-design.md
- Docs/superpowers/plans/2026-06-04-chat-mermaid-diagrams-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent non-Mermaid diagram-like code fences such as graphviz, dot, and generic diagram from opening as Mermaid diagram artifacts after the shared chat artifact renderer landed. Mermaid fences should still render through the shared MermaidDiagramBlock path; non-Mermaid diagram languages should remain code artifacts until dedicated renderers exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented language guards in CodeBlock and ArtifactsPanel so only normalized Mermaid fences/artifacts route to MermaidDiagramBlock. Graphviz, DOT, and generic diagram fences now open as code artifacts, and non-Mermaid diagram-kind artifacts fall back to highlighted code display. Added focused regression coverage for CodeBlock artifact classification and ArtifactsPanel language-gated rendering, and removed Prism key-prop spreading in touched highlighted code paths.

Verification after rebasing onto origin/dev: RED tests first failed for graphviz/dot/diagram opening as diagram artifacts and for graphviz diagram-kind artifacts rendering via MermaidDiagramBlock. GREEN verification passed with `bunx vitest run src/components/Common/__tests__/CodeBlock.artifacts.test.tsx src/components/Common/__tests__/Markdown.mermaid.test.tsx src/components/Common/__tests__/MermaidDiagramBlock.test.tsx src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx` (54 tests). `git diff --check origin/dev...HEAD` passed. `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit` still fails in unrelated KnowledgeQA test fixtures (`KnowledgeQALayout.behavior.test.tsx`, `knowledgeQaStateFixtures.ts`) with the existing missing/optional id/sourceStatus/sourceHealth errors. Bandit skipped because the touched implementation is frontend TypeScript/Markdown task metadata, not Python.
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
