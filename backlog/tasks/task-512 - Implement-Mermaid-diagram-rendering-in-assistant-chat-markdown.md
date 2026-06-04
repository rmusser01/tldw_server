---
id: TASK-512
title: Implement Mermaid diagram rendering in assistant chat markdown
status: In Progress
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
- apps/packages/ui/src/components/Common/Mermaid.tsx
- apps/packages/ui/src/components/Common/MermaidDiagramBlock.tsx
- apps/packages/ui/src/components/Common/MermaidPreviewDialog.tsx
- apps/packages/ui/src/components/Common/Markdown.tsx
- apps/packages/ui/src/components/Common/Playground/Message.tsx
- apps/packages/ui/src/components/Common/Playground/MessageContent.tsx
- apps/packages/ui/src/components/Common/Playground/ReasoningBlock.tsx
- apps/packages/ui/src/components/Common/Playground/CompactMessage.tsx
- apps/packages/ui/src/components/Common/QuickChatHelper/QuickChatMessage.tsx
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
