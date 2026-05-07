---
id: TASK-113
title: Implement MCP chat personal tool availability filter
status: Done
assignee:
  - Codex
created_date: '2026-05-07 14:20'
updated_date: '2026-05-07 14:42'
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
- [x] #1 MCP tool discovery exposes discoveredTools, availableTools, chatTools, disabled preferences, collision state, and counts without removing canExecute=false tools from selector visibility.
- [x] #2 Disabled MCP tool preferences persist per connection/user scope and newly discovered tools remain enabled by default.
- [x] #3 Chat request construction and raw preview use the same normalized chatTools list and omit tools/tool_choice when no chat tools remain.
- [x] #4 WebUI /chat and extension sidepanel chat provide per-tool enable/disable controls through a shared selector or shared selector logic.
- [x] #5 Focused tests cover scoped persistence, normalization/collisions, hook filtering, request/raw-preview parity, and visible selector toggle behavior.
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

Stage 2 complete: extended MCP disabled-tool setting/store/useMcpTools contract with scoped disabled preferences, discoveredTools, availableTools, chatTools, collision names, counts, and toggle helpers. Verified RED with hook tests failing on missing discoveredTools/chatTools fields, then GREEN with `bunx vitest run src/hooks/__tests__/useMcpTools.gating.test.tsx src/utils/__tests__/chat-tools.test.ts` in `apps/packages/ui` reporting 2 files / 11 tests passed.

Stage 3 complete: pageAssistModel now uses stored chatTools by default; ChatTldw and raw preview use the shared request normalizer; no-tools requests omit tools/tool_choice and loop-compat headers. Verified RED on new request/raw-preview/helper tests, then GREEN with `bunx vitest run src/utils/__tests__/chat-tools.test.ts src/models/__tests__/pageAssistModel.mcp-tools.test.ts src/components/Option/Playground/__tests__/usePlaygroundRawPreview.mcp-tools.test.tsx` in `apps/packages/ui` reporting 3 files / 11 tests passed.

Stage 4 complete: added shared `McpToolSelector`, wired it into Playground MCP settings and sidepanel ControlRow, and switched Playground MCP control/raw-preview counts to chatTools. Verified RED with the selector test failing on missing component, then GREEN with `bunx vitest run src/components/Common/__tests__/McpToolSelector.test.tsx src/hooks/playground/__tests__/useMcpToolsControl.test.tsx src/components/Option/Playground/__tests__/Playground.request-budget.test.tsx src/hooks/__tests__/useMcpTools.gating.test.tsx` in `apps/packages/ui` reporting 4 files / 10 tests passed.

Stage 5 complete: final focused Vitest suite passed with 7 files / 21 tests; `bun run verify:openapi` passed in `apps/packages/ui`; `bun run verify:openapi` passed in `apps/extension`; `git diff --check` passed. Bandit skipped because no Python files changed in this frontend TypeScript/docs/task slice.
<!-- SECTION:NOTES:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the personal MCP chat tool filter through the shared WebUI/extension UI layer. The hook now exposes discovered, available, and chat-enabled tools with scoped disabled preferences and counts; request construction and raw preview use the same normalized chatTools contract; Playground and sidepanel chat now render a shared per-tool selector for enabling or disabling tools in the active session scope.
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
