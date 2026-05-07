# MCP Chat Personal Tool Filter Design

Date: 2026-05-06
Status: Approved in session
Backlog: TASK-112

## Goal

Let users choose which already-available MCP tools are exposed to chat from:

- WebUI `/chat`
- extension options `/chat`
- extension sidepanel chat

The choice is a personal preference that persists across chats until changed. It
does not mutate MCP Hub configuration, server policy, external server state, or
RBAC.

## Problem

The current chat MCP controls mostly expose `tool_choice` plus catalog/module
filters. That is useful for narrowing the tool list, but it does not give users
a direct way to turn individual tools on or off for normal chat use.

The implementation seams already exist:

- `apps/packages/ui/src/hooks/useMcpTools.tsx` fetches MCP health, catalogs,
  modules, and tools.
- `apps/packages/ui/src/store/mcp-tools.ts` holds the shared MCP tool state.
- `apps/packages/ui/src/models/index.ts` reads that store and injects tools into
  `pageAssistModel`.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundMcpControl.tsx`
  and `PlaygroundMcpSettingsModal.tsx` expose chat-page MCP controls.
- `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx` has a
  separate extension sidepanel MCP control.
- `Docs/MCP/mcp_hub_management.md` defines MCP Hub as the shared management
  surface for ACP profiles, catalogs, external servers, and policy.

The missing piece is a shared personal availability layer between server
discovery and chat request construction.

## Decisions

- Use a personal availability filter, not server-side MCP Hub mutation.
- Apply the same filter to WebUI `/chat`, extension options `/chat`, and
  extension sidepanel chat.
- Show MCP servers/modules as grouping and filtering concepts in chat.
- Keep start/stop/connect/disconnect, credentials, catalogs, and policy changes
  in MCP Hub.
- Store preferences as local/personal state that persists across browser
  sessions and chats.
- Prefer storing disabled tool names so newly discovered tools are enabled by
  default.

## Non-Goals

- No backend MCP Hub policy changes.
- No external MCP server lifecycle controls in the chat composer.
- No server-side persistence of the personal filter in this slice.
- No changes to RBAC, `canExecute`, approval policies, or tool execution
  semantics.
- No redesign of the chat loop or tool approval protocol.

## Architecture

The chat MCP control becomes a shared personal tool availability layer in
`apps/packages/ui`.

Server truth remains unchanged:

- `/api/v1/mcp/tools` decides what tools exist for the authenticated user.
- Catalog and module filters shape discovery.
- RBAC and policy determine `canExecute`.
- MCP Hub manages external servers, credentials, catalogs, ACP profiles, and
  governance.

The client adds one local decision:

`MCP discovery -> RBAC/canExecute filter -> personal disabled-tool filter -> chat tools payload`

The same filtered result feeds:

- WebUI `/chat`, which renders the shared Playground surface.
- extension options `/chat`, which also renders the shared Playground surface.
- extension sidepanel chat, which currently has separate control markup.

## Data Model

Extend the shared MCP tool store with durable personal filter state:

- `disabledToolNames: string[]`
- `setToolEnabled(toolName, enabled)`
- `enableTools(toolNames)`
- `disableTools(toolNames)`
- `resetToolFilter()`

The stable identity is the effective function/tool name after client
normalization. That is the name the chat request sends and the name tool calls
refer back to.

Store disabled names rather than enabled names:

- Existing behavior remains permissive by default.
- Newly discovered tools become usable unless the user disables them.
- Removed tools can remain in local disabled state harmlessly and take effect
  again if they return.

Preferences should use the same local settings registry pattern as existing MCP
catalog/module settings, with an explicit setting key such as
`tldw:mcp:disabledTools`.

## Hook Contract

`useMcpTools` should continue fetching health, catalogs, modules, and tool
definitions. It should return both server-filtered and chat-filtered views:

- `availableTools`: executable tools after MCP health, server discovery,
  catalog/module filters, and `canExecute`.
- `chatTools`: `availableTools` after the personal disabled-tool filter.
- `disabledToolNames`: persisted disabled tool names.
- toggle helpers for one tool, visible groups, all tools, and reset.

Existing `tools` consumers should migrate deliberately. A compatibility alias is
acceptable during migration, but request construction must be explicit about
using `chatTools`.

## UI Design

Use one reusable MCP tool selector component for Playground and sidepanel. The
component should support:

- current MCP status and chat-enabled count
- search by tool name or description
- grouping by external server/source when available, then module, then `MCP`
- per-tool on/off toggles
- enable all/disable all for the current visible group
- reset personal filter
- visible distinction between unavailable, unexecutable, disabled, and enabled
  tools
- link to MCP Hub for server/catalog/policy/credential management

The existing `tool_choice` control remains:

- `none`: do not send tools.
- `auto`: send `chatTools` and let the model choose.
- `required`: require a tool only when at least one `chatTool` is available.

Catalog/module filters remain advanced narrowing controls. They should not be
presented as the only way to enable/disable tools.

## Request Construction

`pageAssistModel` should use the personal-filtered `chatTools` when it builds
`ChatTldw`.

Request preview should use the same filtered list so the raw payload matches the
actual request behavior.

If the user has `tool_choice` set to `auto` or `required` but no `chatTools`
remain, request construction should defensively degrade to no tools and
`tool_choice: none`.

The UI should also prevent or clearly downgrade `required` when no tools remain,
so the composer reflects what will actually be sent.

## Error And Empty States

The chat control should distinguish these states:

### MCP unavailable

The server does not expose MCP capabilities. Disable the control and link to
setup or health information.

### MCP unhealthy

MCP endpoints exist but the health probe fails. Do not send tools. Show the
health state and route the user toward health/MCP Hub.

### No executable tools

Tools may exist, but none are executable after RBAC, policy, catalog/module
filters, and `canExecute`. Show that this is a permission or filter outcome, not
a personal toggle outcome.

### All chat tools disabled

Executable tools exist, but the personal filter disables them all. Keep MCP
available, show an "all disabled" state, and let the user re-enable tools from
the same control.

## Testing Strategy

### Unit Tests

- `useMcpTools` returns `availableTools` and `chatTools`.
- Disabled tool names persist through the settings registry.
- Newly discovered tools are enabled by default.
- Disabled tools are excluded from `chatTools` but remain visible as available
  tools in the selector.
- Missing or vanished disabled tool names do not break filtering.

### Component Tests

- Playground MCP selector shows the same chat-enabled count used by requests.
- Sidepanel MCP selector shows the same chat-enabled count used by requests.
- Per-tool toggles update the shared filter.
- Group enable/disable works for visible tools only.
- The UI distinguishes:
  - MCP unavailable
  - MCP unhealthy
  - no executable tools
  - all tools disabled by preference

### Request-Building Tests

- `pageAssistModel` sends only personal-enabled MCP tools.
- `pageAssistModel` coerces tool choice to `none` when filtered tools are empty.
- Raw request preview mirrors the actual filtered tool payload.
- Existing `canExecute: false` filtering is preserved.

### E2E Smoke

Use a mocked MCP server with two tools.

- In WebUI `/chat`, disable one tool, send a chat request, and assert the chat
  request payload contains only the enabled tool.
- In extension sidepanel chat, repeat the same assertion.
- Re-enable the disabled tool and verify both tools are sent.

## Rollout Notes

This can ship incrementally:

1. Add store/settings support and hook-level filtering behind existing MCP
   controls.
2. Migrate request construction and raw preview to `chatTools`.
3. Replace Playground and sidepanel UI with the shared selector.
4. Add e2e smoke coverage for WebUI `/chat` and sidepanel parity.

The first implementation slice should avoid backend changes unless current tool
metadata is insufficient to derive useful group labels.

## Acceptance Criteria

- Chat exposes per-tool personal enable/disable controls for MCP tools.
- Personal choices persist across chats until changed.
- WebUI `/chat`, extension options `/chat`, and extension sidepanel chat use the
  same filter.
- Chat request payloads include only MCP tools enabled by the personal filter.
- MCP Hub remains the management surface for server/admin configuration.
- UI and request construction handle empty and degraded states consistently.
