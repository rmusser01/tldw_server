---
id: TASK-113
title: Implement MCP chat personal tool availability filter
status: In Progress
assignee:
  - Codex
created_date: '2026-05-07 14:20'
updated_date: '2026-05-07 14:22'
labels:
  - mcp
  - chat
  - webui
  - extension
  - implementation
dependencies:
  - TASK-112
  - TASK-112.1
references:
  - apps/packages/ui/src/hooks/useMcpTools.tsx
  - apps/packages/ui/src/store/mcp-tools.ts
  - apps/packages/ui/src/models/index.ts
  - apps/packages/ui/src/services/tldw/TldwChat.ts
  - >-
    apps/packages/ui/src/components/Option/Playground/PlaygroundMcpSettingsModal.tsx
  - apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx
documentation:
  - Docs/superpowers/specs/2026-05-06-mcp-chat-personal-tool-filter-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved MCP chat personal tool availability filter for the shared WebUI/extension UI layer. Chat must expose only personally enabled executable MCP tools while preserving server-side MCP Hub/RBAC authority. The first implementation slice should cover shared normalization/filtering state, request construction/raw preview parity, and minimal shared selector coverage for WebUI /chat and extension sidepanel chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 MCP tool discovery exposes discoveredTools, availableTools, chatTools, disabled preferences, collision state, and counts without removing canExecute=false tools from selector visibility.
- [ ] #2 Disabled MCP tool preferences persist per connection/user scope and newly discovered tools remain enabled by default.
- [ ] #3 Chat request construction and raw preview use the same normalized chatTools list and omit tools/tool_choice when no chat tools remain.
- [ ] #4 WebUI /chat and extension sidepanel chat provide per-tool enable/disable controls through a shared selector or shared selector logic.
- [ ] #5 Focused tests cover scoped persistence, normalization/collisions, hook filtering, request/raw-preview parity, and visible selector toggle behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a shared chat tool utility layer for normalization, identity, collision detection, disabled filtering, grouping fallback, and no-tools behavior.
2. Extend MCP settings/store/hook contract with scoped disabled preferences, discoveredTools, availableTools, chatTools, collision state, counts, and toggle helpers.
3. Migrate pageAssistModel, ChatTldw normalization, and raw preview to the shared chatTools/no-tools contract.
4. Add a shared MCP tool selector and wire it into Playground settings plus extension sidepanel ControlRow.
5. Verify with focused Vitest coverage, OpenAPI client path checks when feasible, git diff checks, and Bandit skip documentation because this slice is frontend TypeScript-only unless Python changes appear.

Plan file: Docs/superpowers/plans/2026-05-07-mcp-chat-personal-tool-filter-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 1 complete: added `apps/packages/ui/src/utils/chat-tools.ts` and `apps/packages/ui/src/utils/__tests__/chat-tools.test.ts`. Verified RED with module-not-found failure, then GREEN with `bunx vitest run apps/packages/ui/src/utils/__tests__/chat-tools.test.ts` reporting 1 file / 6 tests passed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
