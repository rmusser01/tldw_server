# Research Workspace MCP Hub Deep-Link Design

Date: 2026-05-26
Backlog: TASK-478.21

## Decision

Use approach C: MCP Hub owns workspace-set, shared-workspace, and path-trust
state. Research Workspace must not grow a parallel MCP binding projection or
management panel. It should only pass the active canonical workspace context to
MCP Hub through a route-level deep link.

The first implementation slice is frontend-only unless existing MCP Hub APIs
cannot truthfully represent match, no-match, and unavailable states.

## User Goal

From Research Workspace, a user who wants tools or agents should be able to open
the canonical MCP Hub workspace management view for the active workspace and
immediately understand whether that workspace is already included in an MCP
workspace set or needs MCP Hub setup.

## Scope

- Build a contextual Research Workspace link to:
  `/mcp-hub?workflow=workspaces&view=workspace-sets&workspace_id=<id>&source=research-workspace`.
- Keep Research Workspace's existing compact readiness disclosure as the only
  entry point for this slice.
- Teach MCP Hub Workspace Sets to read `workspace_id` query context and surface:
  existing workspace-set membership, no matching workspace set, and load error.
- Preserve MCP Hub as the place where users create or edit workspace sets.
- Update the live UAT matrix only as far as live CDP/backend evidence supports.

## Non-Goals

- No `/workspace-playground` aliases, redirects, route metadata, or active
  labels.
- No Research Workspace-owned MCP binding endpoint or projection in this slice.
- No duplicate Research Workspace source membership inside MCP Hub.
- No new persistent banner or top-level trust bar.
- No ACP or sandbox implementation in this task.

## UX Contract

Research Workspace should use bounded, contextual copy:

- Link label: `Open MCP Hub`
- Destination: MCP Hub Workspaces workflow, Workspace Sets view, carrying the
  active canonical workspace ID.

MCP Hub Workspace Sets should own the interpretation:

- Existing binding: show that one or more workspace sets include the workspace.
- No binding: explain that no workspace set includes the workspace yet and offer
  the existing MCP Hub create/edit controls.
- Error/unavailable: show the existing MCP Hub load error state; do not make
  Research Workspace guess.

## Testing Contract

- Unit tests should cover the generated Research Workspace MCP Hub URL.
- MCP Hub tests should cover route query context, existing membership, no
  membership, and preserving route state.
- The live UAT matrix row `RW-UAT-021` should remain `Partial` unless a live
  backend plus WebUI run proves the handoff.

