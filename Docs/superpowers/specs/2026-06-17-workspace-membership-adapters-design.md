# Workspace Membership Adapters Design

## Goal

Complete the next backend slice for GitHub issue #2378 by adding stable Workspace membership adapters for remaining ACP-adjacent resource domains without making membership a trust source for execution, file access, or runtime admission.

## Scope

Implement concrete adapters for:

- `prompt`
- `workflow`
- `watchlist`
- `acp_session`
- `sandbox_session`

Explicitly defer:

- `note`, because the global note ownership/lifecycle boundary is not clear enough from the current membership adapter surface.
- `acp_run`, because #2378 asks for ACP sessions and sandbox sessions, and no stable ACP run descriptor writer is assumed for this slice.

Unsupported/deferred types must continue to fail closed with `unsupported_resource_type`.

## Architecture

The existing Workspace membership service stays registry-driven. New adapters validate through their owning domain stores before a membership row is written, then return a canonical `WorkspaceResourceRef` for persistence and summary rendering.

`WorkspaceMembershipContext` will carry optional handles for prompt, workflow, and watchlist databases, plus request metadata needed for workflow ownership checks. Endpoint dependencies should follow the existing optional `media_db` pattern: unrelated membership calls must not fail when one optional subsystem DB is unavailable.

ACP and Sandbox session adapters validate against `workspace_runtime_bindings` rows in the ChaChaNotes DB. Runtime binding descriptors are metadata-only references. They do not grant trust, path access, ACP execution permission, Sandbox admission, or MCP file access.

## Resource Contracts

### Prompt

Validate through `PromptsDatabase.fetch_prompt_details()`. Accept ID, UUID, or name on input, but store the canonical numeric prompt ID. Summaries may expose name, author, timestamps, and compact format metadata, but must not expose prompt text, details, structured definition bodies, or other prompt content.

### Workflow

Validate through `WorkflowsDatabase.get_definition()`. Require same tenant and owner, unless the request context says the current user is a workflows admin. Store canonical numeric workflow ID. Summaries may expose name, version, description, tags, and active/deleted state, but not `definition_json` or run inputs/outputs.

### Watchlist

Validate through the current user's `WatchlistsDatabase.get_watchlist()`. Store canonical numeric watchlist ID. Summaries may expose name, domain, status, priority, tags, timestamps, and deleted/archived state, but not long objective/body-like fields.

### ACP And Sandbox Sessions

Validate against active `workspace_runtime_bindings` for the current workspace. `acp_session` requires `binding_kind="acp_session"` and `owner_domain="acp"`. `sandbox_session` requires `binding_kind="sandbox_session"` and `owner_domain="sandbox"`. Summaries use the already-normalized/redacted descriptor fields only.

Reverse lookup for runtime-bound resources remains row-based. Runtime binding IDs are workspace-scoped, so resolving summaries must use each membership row's workspace context instead of treating the resource ID as globally authoritative.

## Error Handling

Missing optional domain DB handles return adapter-specific 503 errors only when that adapter is used.

Missing, deleted, cross-workspace, wrong-kind, or wrong-owner resources return `resource_not_found` style failures and must not disclose that a resource exists outside the caller's valid scope.

Summary resolution failures remain non-fatal for list/read paths and produce unresolved summaries with safe messages.

## Testing

Add tests before implementation for:

- Registry support for the new resource types and fail-closed behavior for `note` and `acp_run`.
- Prompt ID/UUID/name canonicalization and deleted/missing resource handling.
- Workflow tenant/owner/admin validation and inactive summaries.
- Watchlist active/deleted handling.
- ACP/Sandbox session validation through runtime bindings, including wrong kind/domain and archived bindings.
- Existing membership API behavior for pilot adapters remains stable.

Run focused Workspace tests and Bandit on touched backend paths before completion.
