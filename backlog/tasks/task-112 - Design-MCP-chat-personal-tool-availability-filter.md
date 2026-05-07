---
id: TASK-112
title: Design MCP chat personal tool availability filter
status: Done
assignee:
  - Codex
created_date: '2026-05-07 06:17'
updated_date: '2026-05-07 06:19'
labels:
  - mcp
  - chat
  - webui
  - extension
  - design
dependencies: []
references:
  - apps/packages/ui/src/hooks/useMcpTools.tsx
  - apps/packages/ui/src/store/mcp-tools.ts
  - apps/packages/ui/src/models/index.ts
  - apps/packages/ui/src/components/Option/Playground/PlaygroundMcpControl.tsx
  - >-
    apps/packages/ui/src/components/Option/Playground/PlaygroundMcpSettingsModal.tsx
  - apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx
documentation:
  - Docs/MCP/mcp_hub_management.md
  - Docs/Plans/2026-03-04-chat-tool-calling-convergence-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a reviewed design spec for making MCP server/tool availability from chat a personal preference filter across WebUI /chat, extension options /chat, and extension sidepanel chat. Server-side MCP Hub configuration, RBAC, catalogs, and external server connection management remain authoritative and out of scope for chat-surface mutation; chat only chooses which already-available executable tools are exposed to chat requests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec documents the approved personal availability filter behavior for WebUI /chat, extension options /chat, and extension sidepanel chat.
- [x] #2 Spec distinguishes server authority from local personal chat preferences and keeps MCP Hub as the management surface for server/admin changes.
- [x] #3 Spec covers data flow from MCP discovery through RBAC/catalog/module filters and personal per-tool toggles into chat request construction.
- [x] #4 Spec covers UX/error states for unavailable MCP, unhealthy MCP, no executable tools, and all tools disabled by preference.
- [x] #5 Spec covers testing expectations for hooks, components, request construction, raw preview, and mocked WebUI/extension chat smoke coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write `Docs/superpowers/specs/2026-05-06-mcp-chat-personal-tool-filter-design.md` capturing the approved design for a shared personal MCP tool availability filter across WebUI `/chat`, extension options `/chat`, and extension sidepanel chat.
2. Ground the spec in current repo seams: `useMcpTools`, `mcp-tools` store, `pageAssistModel`, Playground MCP controls, sidepanel `ControlRow`, and MCP Hub docs.
3. Include architecture, components, data flow, UX/error handling, testing strategy, non-goals, and implementation notes without making code changes in this step.
4. Verify the spec file exists and review the diff for only the intended spec plus Backlog tracking changes.
5. Finalize the Backlog task with acceptance criteria, non-code verification/Bandit skip, and final summary, then commit the spec/tracking work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created `Docs/superpowers/specs/2026-05-06-mcp-chat-personal-tool-filter-design.md` with the approved personal MCP tool availability filter design. Verification: reviewed the written spec and ran `git diff --check` scoped to the new spec and task path; no whitespace errors reported. Bandit skipped because this task only adds documentation/Backlog tracking and touches no executable code. Spec-review subagent was not dispatched because current tool policy requires explicit user authorization for subagent delegation; user review remains the next gate before implementation planning.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a design spec for a shared personal MCP tool availability filter across WebUI `/chat`, extension options `/chat`, and extension sidepanel chat. The design keeps MCP Hub and server RBAC as the authority for availability and permissions, then applies a local persistent disabled-tool preference before chat request construction. It also documents request degradation when no chat tools remain, UI states for unavailable/unhealthy/empty/disabled cases, and focused test coverage for hooks, selector components, request building, raw preview, and mocked chat smoke tests.

Verification: reviewed the spec content and ran scoped `git diff --check` with no whitespace errors. Bandit is documented as skipped because the change is documentation-only.
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
