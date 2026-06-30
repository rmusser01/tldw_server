# MCP Unified Stage 4K Gateway Profile Management Design

## Summary

Stage 4K adds the first profile-management surface for the standalone MCP
gateway. It turns the existing built-in profile preset catalog into usable
front-end modes by exposing stored profile listing, stored profile inspection,
preset duplication, and default profile get/set through both the gateway CLI and
FastAPI router.

The slice must stay narrow. It should not add arbitrary profile editing,
workspace binding management, approval policy editing, credential grant editing,
external server lifecycle management, or UI changes. Those are later governance
and lifecycle slices.

## Current Context

The merged Stage 4 gateway work already provides:

- `mcp_unified.gateway.cli` with `validate-config`, `list-presets`, and
  `show-preset`.
- `mcp_unified.gateway.fastapi` with status, JSON-RPC HTTP, and WebSocket
  transport.
- `mcp_unified.gateway.config` with profile bootstrap config and memory/sqlite
  profile-store selection.
- `mcp_unified.gateway.bootstrap` with default preset seeding into a
  `ProfileStore`.
- `mcp_unified.gateway.profile_runtime.ProfileAwareGatewayRuntime`, which
  enforces profile policy on tool discovery and execution.
- `ProfileStore` and `ProfileAssignmentStore` storage protocols.
- `SQLiteMCPStore`, which implements profile and assignment persistence.
- Built-in immutable profile presets with deterministic version/timestamp
  provenance.

The missing product step is a management API that lets a front-end create a
stored profile from a built-in preset and choose that profile as the gateway
default mode.

## Goals

- Add one package-owned profile management service reused by CLI and FastAPI.
- List and inspect stored profiles.
- Duplicate one built-in preset into the configured profile store.
- Set and read the gateway default profile.
- Make no-profile JSON-RPC requests resolve through the stored default profile
  when configured.
- Keep explicit transport profile selectors higher precedence than the default.
- Return deterministic JSON payloads and machine-readable reason codes.
- Preserve package boundaries: new package code must not import
  `tldw_Server_API`.

## Non-Goals

- Arbitrary profile editing.
- Profile deletion or disable APIs.
- Workspace/path binding management.
- Approval policy CRUD.
- Credential grant CRUD.
- External MCP server lifecycle management.
- Audit viewer APIs.
- Front-end UI changes.
- Changing `tldw_server` legacy no-profile MCP behavior.

## Architecture

Introduce `mcp_unified.gateway.profiles` with a small
`GatewayProfileManager`. The manager is the only new business-logic surface for
this slice. CLI commands and FastAPI endpoints call the manager rather than
duplicating profile-store behavior.

The manager depends on:

- `ProfileStore` for stored profile documents.
- `ProfileAssignmentStore` for default-profile assignment. SQLite-backed
  gateways use the persisted assignment store; tests and memory-backed
  development configurations use a small in-memory assignment store that
  implements the same protocol.
- Optional `AuditStore` for profile lifecycle events when the configured store
  provides one.
- Built-in preset helpers for read-only preset lookup and duplication.

The manager should be created during gateway bootstrap. `GatewayProfileBootstrap`
should expose the profile manager alongside the profile-aware runtime, profile
store, assignment store, audit store when available, default profile id, and
seeded profile ids. FastAPI app creation should accept either a manager or
bootstrap object for management endpoints; JSON-RPC transport behavior should
remain runtime-driven.

Management endpoints are control-plane routes. They must be mounted only when
profile management is explicitly enabled or when a manager/bootstrap object is
provided to the app factory. A gateway deployment that exposes only MCP
JSON-RPC traffic should be able to omit these routes entirely.

The default resolver used by `ProfileAwareGatewayRuntime` should resolve in this
order:

1. Explicit transport profile id from header/query metadata.
2. Stored gateway default profile assignment.
3. Constructor/bootstrap default id for compatibility.
4. Structured `profile_required` denial when no default exists.

This keeps explicit front-end selections authoritative while allowing the
gateway to behave like a configured MCP server for clients that do not send a
profile id on every request.

The running FastAPI manager and runtime resolver must share the same assignment
store instance or assignment-resolution abstraction. After `PUT
/profiles/default` succeeds, subsequent no-profile JSON-RPC requests in that
same running gateway process must observe the new default on the next resolver
read without requiring a process restart.

## CLI Contract

Add these commands to `mcp-unified-gateway`:

- `list-profiles`
- `show-profile <profile_id>`
- `duplicate-preset <preset_id> [--profile-id <profile_id>] [--name <name>]`
- `get-default-profile`
- `set-default-profile <profile_id>`

All commands emit one JSON object. Expected domain failures emit JSON on stderr
without tracebacks.

Store-backed profile-management commands must take an explicit gateway config
source. Stage 4K should use `--config <path>` on each profile-management
command, with `MCP_UNIFIED_GATEWAY_CONFIG` as an optional environment fallback.
The config is loaded through the existing gateway config loader, and its
`store` section selects the profile and assignment stores. `list-presets` and
`show-preset` remain catalog-only commands and do not require a config file.

The CLI is an offline store-management tool, not an RPC client for a running
gateway process. When the CLI and a running gateway point at the same SQLite
store, changes are visible to the running gateway on the next store read. The
CLI does not push an in-memory notification to a running process in Stage 4K.

Mutating profile-management commands must reject nonpersistent memory-store
configs unless an implementation-plan-approved test/development flag is added.
The default production path is a SQLite store selected by config. Read-only
commands may inspect config-seeded memory profiles for tests, but their payloads
must clearly reflect that the store is memory-backed and nonpersistent.

All profile-management CLI success payloads include store metadata:

```json
{"kind": "sqlite", "persistent": true}
{"kind": "memory", "persistent": false}
```

For read-only memory-store commands, the payload must include
`"store": {"kind": "memory", "persistent": false}` exactly. Mutating
memory-store commands return a domain error unless the later implementation plan
adds and tests an explicit development-only override flag.

Success payloads:

```json
{"ok": true, "profiles": [], "store": {"kind": "sqlite", "persistent": true}}
{"ok": true, "profile": {}, "store": {"kind": "sqlite", "persistent": true}}
{"ok": true, "preset_id": "project-researcher", "preset_version": "2026.05.27", "profile": {}, "store": {"kind": "sqlite", "persistent": true}}
{"ok": true, "profile": {}, "assignment": {}, "store": {"kind": "sqlite", "persistent": true}}
{"ok": true, "profile_id": "project-researcher", "assignment": {}, "store": {"kind": "sqlite", "persistent": true}}
```

CLI exit codes:

- `0`: success
- `1`: domain error, including not found, collision, disabled, or unavailable
  store
- `2`: argument parsing or request-shape error

## FastAPI Contract

Add package-owned management endpoints under the existing gateway router prefix:

- `GET /profiles`
- `GET /profiles/{profile_id}`
- `POST /profiles/from-preset`
- `GET /profiles/default`
- `PUT /profiles/default`

`POST /profiles/from-preset` accepts:

```json
{
  "preset_id": "project-researcher",
  "profile_id": "workspace-researcher",
  "name": "Workspace Researcher"
}
```

Only `preset_id` is required. `profile_id` and `name` are optional; omitting
`profile_id` uses the preset id as the stored profile id, and omitting `name`
keeps the preset's display name.

`PUT /profiles/default` accepts:

```json
{
  "profile_id": "workspace-researcher"
}
```

HTTP status mapping:

- `200`: successful read or write
- `404`: profile or preset not found, including no configured default profile
- `409`: profile id collision
- `422`: malformed request body
- `503`: required profile or assignment store unavailable

Reason-code mapping should be deterministic:

- `profile_not_found`: `404`
- `preset_not_found`: `404`
- `default_profile_not_configured`: `404`
- `profile_disabled`: `409`
- `profile_already_exists`: `409`
- `invalid_profile_request`: `422`
- `profile_store_unavailable`: `503`
- `assignment_store_unavailable`: `503`

Expected error payload fields:

```json
{
  "ok": false,
  "error": "human-readable message",
  "reason_code": "profile_not_found",
  "profile_id": "workspace-researcher"
}
```

Initial reason codes:

- `profile_not_found`
- `profile_disabled`
- `profile_already_exists`
- `preset_not_found`
- `default_profile_not_configured`
- `profile_store_unavailable`
- `assignment_store_unavailable`
- `invalid_profile_request`

## Profile Duplication Behavior

Duplicating a preset must:

1. Look up the built-in preset by id.
2. Return `preset_not_found` if the preset is absent.
3. Choose the requested profile id, or use the preset id when no profile id is
   provided.
4. Reject existing stored profile ids with `profile_already_exists`.
5. Duplicate the preset with caller-owned profile data.
6. Apply an optional display name without changing preset id/version
   provenance.
7. Store the duplicated profile through `ProfileStore.upsert_profile`.
8. Return the stored profile in deterministic JSON.

Built-in presets remain immutable templates. Stored profiles record
`preset_id`, `preset_version`, and provenance from the duplicated preset.

## Default Profile Behavior

Setting the default profile must:

1. Load the profile by id.
2. Return `profile_not_found` if absent.
3. Return `profile_disabled` if present but disabled.
4. Persist a default `ProfileAssignment` with `is_default=True` through the
   configured assignment store.
5. Return the selected profile id and assignment/default provenance.

Reading the default profile must:

1. Prefer enabled default assignments from `ProfileAssignmentStore`.
2. Fall back to bootstrap/default id if no assignment exists.
3. Return `default_profile_not_configured` when neither exists.
4. Return `profile_not_found` or `profile_disabled` if the configured default no
   longer points at an executable profile.

Runtime resolution must fail closed. Tool discovery returns an empty tool list
when no profile resolves. Tool execution raises structured policy denial with
the resolver reason code.

## Data And Store Notes

Stage 4K must not introduce another default-profile persistence model. Use
`ProfileAssignment` with `is_default=True` through a
`ProfileAssignmentStore`-compatible abstraction for all default selection,
including memory-backed tests. The assignment id should be stable and
gateway-local, for example `gateway-default`, so repeated
`set-default-profile` updates the same default record instead of accumulating
competing defaults.

If multiple legacy/default assignments are present, Stage 4K should choose the
enabled default assignment with the greatest `updated_at` timestamp, using
assignment id as a stable ascending tie-breaker. Cleanup or migration of older
default assignments belongs to a later slice.

Memory-store tests should use a small in-memory assignment store rather than
special-casing default state in the runtime. This keeps the runtime and manager
on the same default-resolution path across SQLite and memory-backed tests.

## Audit Events

The manager should accept an optional `AuditStore`. When present, Stage 4K emits
append-only audit events for:

- profile duplication from a built-in preset
- default profile assignment changes
- failed management attempts caused by missing profiles, disabled profiles,
  preset misses, or id collisions

Audit event payloads must contain ids, reason codes, and provenance metadata,
not secrets or full profile documents. If no audit store is configured, profile
management still works and response payloads should indicate only normal
operation results. Audit viewer APIs remain out of scope for this slice.

## Security And Boundaries

- Profile management is local gateway administration, not public untrusted
  profile editing.
- The first slice does not create a new auth layer. Profile management routes
  are local/admin control-plane routes and must be mounted only when explicitly
  enabled or when a caller supplies the profile manager/bootstrap object to the
  app factory.
- No secrets may be accepted or returned by the profile-management payloads.
- Stored profile documents must remain copy-isolated on read and write.
- Error handling must not leak tracebacks for expected domain failures.
- New package files must not import `tldw_Server_API`.
- Existing `tldw_server` MCP route behavior must not change.

## Testing Strategy

Add focused package-local tests for:

- `GatewayProfileManager`
  - list/show stored profiles
  - duplicate preset success
  - duplicate unknown preset
  - duplicate profile id collision
  - set/get default profile
  - missing/disabled default profile behavior
  - copy isolation for returned profiles

- CLI
  - each new command returns deterministic JSON
  - domain failures return JSON on stderr without tracebacks
  - parse failures return exit code `2`

- FastAPI
  - profile endpoints mirror manager semantics
  - HTTP status mapping matches reason codes
  - `PUT /profiles/default` affects no-profile JSON-RPC requests
  - management endpoints are absent when profile management is not enabled

- Runtime/default resolution
  - explicit transport profile overrides stored default
  - no default still fails closed
  - missing/disabled default fails closed
  - running gateway default changes affect subsequent no-profile JSON-RPC
    requests without restart

- Boundary and compatibility
  - no `tldw_Server_API` imports in new package files
  - memory store still works in tests
  - read-only memory-store CLI payloads include
    `{"store": {"kind": "memory", "persistent": false}}`
  - memory-store mutating CLI commands fail without the explicit dev override
  - SQLite-backed manager uses persisted assignment/profile stores
  - optional audit store receives profile lifecycle events when configured
  - existing Stage 4A-J tests continue passing

Verification should include focused pytest for gateway/profile tests, ruff on
touched files, Bandit on touched production package files, and `git diff
--check`.

## Implementation Slicing Guidance

Stage 4K should be one PR if it stays inside this boundary. If the work starts
to require arbitrary profile mutation, workspace binding, or approval policy
management, split those into Stage 4L+ rather than expanding Stage 4K.

Suggested implementation order for the later implementation plan:

1. Add manager-level RED tests and manager models/errors.
2. Implement `GatewayProfileManager`.
3. Wire CLI commands to the manager.
4. Wire FastAPI profile endpoints to the manager.
5. Wire stored default resolution into profile-aware runtime bootstrap.
6. Add SQLite/memory default assignment coverage.
7. Run compatibility, lint, Bandit, and diff checks.
