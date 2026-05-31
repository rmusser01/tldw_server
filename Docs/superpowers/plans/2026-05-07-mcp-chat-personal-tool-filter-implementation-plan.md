# MCP Chat Personal Tool Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement per-user MCP tool enable/disable controls for chat, shared by WebUI `/chat`, extension options `/chat`, and extension sidepanel chat.

**Architecture:** Keep backend MCP discovery/RBAC unchanged. Add shared client-side normalization, scoped disabled-tool preferences, filtered `chatTools`, and a reusable selector that both Playground and sidepanel surfaces can render. Request construction and raw preview must consume the same filtered/normalized tools and omit `tools`/`tool_choice` when no chat tools remain.

**Tech Stack:** React, TypeScript, Zustand, Plasmo storage-backed settings registry, TanStack Query, Vitest, Testing Library.

---

## File Map

- Create `apps/packages/ui/src/utils/chat-tools.ts`
  - Shared tool-name extraction, sanitization, normalized identity, collision detection, grouping helpers, and no-tools filtering helpers.
- Create `apps/packages/ui/src/utils/__tests__/chat-tools.test.ts`
  - Unit tests for normalization, disabled filtering, collision exclusion, and grouping fallback.
- Modify `apps/packages/ui/src/services/settings/ui-settings.ts`
  - Add `MCP_DISABLED_TOOLS_SETTING` with a versioned scoped preference shape.
- Modify `apps/packages/ui/src/store/mcp-tools.ts`
  - Track discovered/raw tools, chat-filtered tools, scoped disabled preferences, active preference scope, collision names, and counts.
- Modify `apps/packages/ui/src/hooks/useMcpTools.tsx`
  - Hydrate scoped preferences, compute `discoveredTools`, `availableTools`, `chatTools`, collisions, counts, and expose toggle helpers.
- Modify `apps/packages/ui/src/hooks/__tests__/useMcpTools.gating.test.tsx`
  - Add hook tests for canExecute visibility, scoped persistence, default enabled new tools, and disabled filtering.
- Modify `apps/packages/ui/src/models/index.ts`
  - Use stored `chatTools` by default and shared executable/no-tools behavior.
- Create `apps/packages/ui/src/models/__tests__/pageAssistModel.mcp-tools.test.ts`
  - Request-building tests for disabled tools, no-tools omission, and collision filtering.
- Modify `apps/packages/ui/src/services/tldw/TldwChat.ts`
  - Use the shared normalization helper so `ChatTldw` and selector identity match.
- Modify `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundRawPreview.ts`
  - Use the same `chatTools`/normalization/no-tools behavior as actual request construction.
- Create `apps/packages/ui/src/components/Option/Playground/__tests__/usePlaygroundRawPreview.mcp-tools.test.tsx`
  - Raw-preview parity tests.
- Create `apps/packages/ui/src/components/Common/McpToolSelector.tsx`
  - Shared selector UI for tool search, grouping, per-tool toggles, counts, and degraded state labels.
- Create `apps/packages/ui/src/components/Common/__tests__/McpToolSelector.test.tsx`
  - Selector toggle and state-rendering tests.
- Modify `apps/packages/ui/src/components/Option/Playground/PlaygroundMcpSettingsModal.tsx`
  - Render the shared selector alongside existing catalog/module controls.
- Modify `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx`
  - Render the shared selector in the sidepanel tools area.
- Modify `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
  - Pass `chatTools` counts to controls and raw preview.
- Add `backlog/tasks/task-113 - Implement-MCP-chat-personal-tool-availability-filter.md`
  - Mirror the Backlog task record into this worktree before committing.

## Stage 1: Shared Tool Normalization And Filtering

**Goal:** Build the pure utility layer first so all later code uses one identity contract.

**Success Criteria:** Utility tests fail before implementation, then pass; helpers return normalized names, collision sets, filtered chat tools, and grouping labels.

**Tests:** `bunx vitest run apps/packages/ui/src/utils/__tests__/chat-tools.test.ts`

**Status:** Complete

- [x] Write failing tests for `normalizeChatToolName`, `resolveMcpToolIdentity`, `buildChatToolFilterState`, and `getMcpToolGroupLabel`.
- [x] Run the utility test and verify it fails because the helper module does not exist.
- [x] Implement `apps/packages/ui/src/utils/chat-tools.ts` with shared normalization and collision-aware filtering.
- [x] Run the utility test and verify it passes.

## Stage 2: Store, Settings, And Hook Contract

**Goal:** Expose discovered, available, and chat-filtered tools plus scoped disabled preferences from `useMcpTools`.

**Success Criteria:** The hook keeps `canExecute: false` tools visible in `discoveredTools`, filters them from `availableTools`, filters disabled/colliding tools from `chatTools`, and persists disabled preferences per active scope.

**Tests:** `bunx vitest run apps/packages/ui/src/hooks/__tests__/useMcpTools.gating.test.tsx apps/packages/ui/src/utils/__tests__/chat-tools.test.ts`

**Status:** Complete

- [x] Add failing hook tests for discovered-vs-available-vs-chat tools, disabled persistence, scoped isolation, and newly discovered tools defaulting enabled.
- [x] Run the hook tests and verify the expected failures.
- [x] Add `MCP_DISABLED_TOOLS_SETTING` and extend `mcp-tools` store state/actions.
- [x] Update `useMcpTools` to hydrate preferences, compute scope, store discovered/chat tools, and expose toggle helpers/counts.
- [x] Run focused hook and utility tests until green.

## Stage 3: Request Construction And Raw Preview Parity

**Goal:** Ensure actual chat requests and raw preview use the same chat-filtered tool list.

**Success Criteria:** `pageAssistModel`, `ChatTldw`, and raw preview all use the shared normalization helper and omit `tools`/`tool_choice` when no chat tools remain.

**Tests:** `bunx vitest run apps/packages/ui/src/models/__tests__/pageAssistModel.mcp-tools.test.ts apps/packages/ui/src/components/Option/Playground/__tests__/usePlaygroundRawPreview.mcp-tools.test.tsx`

**Status:** Complete

- [x] Write failing request tests showing disabled tools are excluded and no-tools requests omit both fields.
- [x] Write failing raw-preview parity tests for the same behavior.
- [x] Run the request/raw-preview tests and verify expected failures.
- [x] Refactor `ChatTldw` to import the shared normalization helper.
- [x] Update `pageAssistModel` to use `chatTools` from the MCP store by default.
- [x] Update raw preview dependencies and request building to use `chatTools` and shared no-tools behavior.
- [x] Run request/raw-preview tests until green.

## Stage 4: Shared Selector UI In Playground And Sidepanel

**Goal:** Add per-tool toggles through shared selector UI on both chat surfaces.

**Success Criteria:** Users can toggle individual tools on/off, see counts and degraded states, and the same selector logic renders in Playground settings and sidepanel tools UI.

**Tests:** `bunx vitest run apps/packages/ui/src/components/Common/__tests__/McpToolSelector.test.tsx apps/packages/ui/src/hooks/playground/__tests__/useMcpToolsControl.test.tsx`

**Status:** Complete

- [x] Write failing selector tests for toggle behavior, unexecutable display, disabled display, collision display, and empty/degraded state labels.
- [x] Run selector tests and verify expected failures.
- [x] Implement `McpToolSelector`.
- [x] Wire selector into `PlaygroundMcpSettingsModal`.
- [x] Wire selector into sidepanel `ControlRow`.
- [x] Update Playground count props to use `chatTools.length` for chat-enabled counts.
- [x] Run selector and focused existing MCP control tests until green.

## Stage 5: Focused Verification And Cleanup

**Goal:** Verify the implementation slice and record completion state.

**Success Criteria:** Focused tests pass, WebUI/extension OpenAPI client path checks still pass when feasible, and docs/task records are current.

**Tests:**
- `bunx vitest run apps/packages/ui/src/utils/__tests__/chat-tools.test.ts apps/packages/ui/src/hooks/__tests__/useMcpTools.gating.test.tsx apps/packages/ui/src/models/__tests__/pageAssistModel.mcp-tools.test.ts apps/packages/ui/src/components/Option/Playground/__tests__/usePlaygroundRawPreview.mcp-tools.test.tsx apps/packages/ui/src/components/Common/__tests__/McpToolSelector.test.tsx`
- `bun run verify:openapi` in `apps/packages/ui`
- `bun run verify:openapi` in `apps/extension`

**Status:** Complete

- [x] Run all focused Vitest coverage for the implementation slice.
- [x] Run OpenAPI verification in `apps/packages/ui`.
- [x] Run OpenAPI verification in `apps/extension`.
- [x] Run `git diff --check`.
- [x] Run Bandit if executable Python changed; otherwise document the non-Python skip.
- [x] Update `TASK-113` notes/final summary and commit.
