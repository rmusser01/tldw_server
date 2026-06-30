---
id: TASK-512
title: Implement Mermaid diagram rendering in assistant chat markdown
status: Done
labels:
- frontend
- chat
- mermaid
- implementation
references:
- Docs/superpowers/specs/2026-06-04-chat-mermaid-diagrams-design.md
- Docs/superpowers/plans/2026-06-04-chat-mermaid-diagrams-implementation-plan.md
modified_files:
- apps/packages/ui/src/types/chat-settings.ts
- apps/packages/ui/src/hooks/useChatSettings.ts
- apps/packages/ui/src/components/Option/Settings/ChatSettings.tsx
- apps/packages/ui/src/components/Option/Settings/__tests__/ChatSettings.test.tsx
- apps/packages/ui/src/components/Common/Mermaid.tsx
- apps/packages/ui/src/components/Common/__tests__/Mermaid.test.tsx
- apps/packages/ui/src/components/Common/MermaidDiagramBlock.tsx
- apps/packages/ui/src/components/Common/MermaidPreviewDialog.tsx
- apps/packages/ui/src/components/Common/__tests__/MermaidDiagramBlock.test.tsx
- apps/packages/ui/src/components/Common/__tests__/MermaidPreviewDialog.test.tsx
- apps/packages/ui/src/components/Common/Markdown.tsx
- apps/packages/ui/src/components/Common/__tests__/Markdown.mermaid.test.tsx
- apps/packages/ui/src/components/Common/Playground/Message.tsx
- apps/packages/ui/src/components/Common/Playground/MessageContent.tsx
- apps/packages/ui/src/components/Common/Playground/ReasoningBlock.tsx
- apps/packages/ui/src/components/Common/Playground/CompactMessage.tsx
- apps/packages/ui/src/components/Common/QuickChatHelper/QuickChatMessage.tsx
- apps/packages/ui/src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx
- apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the reviewed Mermaid chat PRD using the plan in Docs/superpowers/plans/2026-06-04-chat-mermaid-diagrams-implementation-plan.md. Scope: frontend/shared UI code only unless verification reveals a necessary adjacent change. Follow TDD task slices, keep user messages unchanged, and verify Mermaid resolution/build behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Mermaid diagram rendering for assistant-facing chat markdown surfaces using the shared Markdown renderer while leaving user messages unchanged. Added the persisted renderMermaidDiagrams chat setting, exposed sanitized Mermaid render state, introduced inline diagram chrome plus preview/download/copy controls, routed closed ```mermaid and ~~~mermaid fences through MermaidDiagramBlock, and enabled Mermaid only for completed assistant rows in Playground, CompactMessage, extracted MessageContent, and QuickChat. Reasoning blocks suppress Mermaid while actively streaming, and completed assistant rows remain renderable while a later assistant message streams.

Final review follow-up: the core Mermaid renderer now sanitizes generated SVG before inserting it into the inline DOM and before reporting success state; the Markdown scanner now supports tilde fences and requires matching fence marker type/length.

Verification:
- Red: bunx vitest run src/components/Common/__tests__/Mermaid.test.tsx src/components/Common/__tests__/Markdown.mermaid.test.tsx failed before the follow-up fix on unsafe inline SVG and ~~~mermaid routing.
- Green: bunx vitest run src/components/Common/__tests__/Mermaid.test.tsx src/components/Common/__tests__/Markdown.mermaid.test.tsx passed (2 files, 16 tests).
- Focused suite: bunx vitest run src/components/Common/__tests__/Mermaid.test.tsx src/components/Common/__tests__/MermaidDiagramBlock.test.tsx src/components/Common/__tests__/MermaidPreviewDialog.test.tsx src/components/Common/__tests__/Markdown.mermaid.test.tsx src/components/Common/__tests__/Markdown.github-code-blocks.test.tsx src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx src/components/Option/Settings/__tests__/ChatSettings.test.tsx passed (8 files, 57 tests).
- Build: NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run build passed from apps/tldw-frontend; token-sync check passed. Existing Turbopack broad glob warnings remain in apps/tldw-frontend/lib/documentation.ts.
- Diff check: git diff --check passed.
- Lint: local targeted ESLint on the final follow-up files exited 0 with one existing @next/next/no-img-element warning in Markdown.tsx. Wider touched-file lint is blocked by existing baseline issues in Message.tsx/CompactMessage/tests.
- Bandit skipped: frontend TypeScript/React-only change; no Python touched.
Known unchanged dirty entry: apps/packages/ui/node_modules/antd symlink is pre-existing and intentionally not staged.
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
