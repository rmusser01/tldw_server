# Workspace Primary Root Attach API Design

## Status

Draft for TASK-2235.

## Purpose

The previous Workspace Core slice made `workspace_id` the canonical identity,
persisted `workspace_profile`, added Workspace-owned primary root storage, and
exposed read-only `/roots`, `/capabilities`, and `/context` surfaces. The next
slice should let clients attach or replace a Workspace primary project root
through a stable API without coupling future Sandbox, MCP, ACP, and file
inventory work to a large endpoint handler.

This design adds a reusable Workspace Core root-binding service and a thin write
endpoint for the single-primary-root Project Workspace model.

## Goals

- Attach the first primary root to a research Workspace and upgrade it to
  `workspace_profile: project`.
- Replace the current primary root only when the request explicitly asks to
  replace it.
- Support `host_local` and `sandbox_volume` root backends in the same API
  contract.
- Validate `host_local` paths against Workspace/ACP allowed root configuration,
  containment, traversal, and symlink escape rules before persistence.
- Validate `sandbox_volume` bindings through a bounded Workspace-owned wrapper
  contract without implementing full volume create/delete lifecycle yet.
- Keep public responses redacted: never expose `absolute_root` through ordinary
  Workspace API responses.
- Preserve existing read-only roots, context, and capability response shapes.

## Non-Goals

- No secondary roots.
- No root detach/archive/delete endpoint.
- No file inventory Jobs worker.
- No Git operations.
- No MCP trusted-root mutation.
- No ACP/harness launch changes.
- No Sandbox volume create/delete/mount lifecycle.
- No UI changes.

## API Contract

Add one write endpoint:

```http
PUT /api/v1/workspaces/{workspace_id}/roots/primary
```

Request schema:

```json
{
  "backend": "host_local",
  "root_id": "primary",
  "absolute_root": "/allowed/project/path",
  "display_name": "Project",
  "replace_existing": false,
  "expected_workspace_version": 3
}
```

Shared request rules:

- `root_id` is optional. When omitted, the service resolves to the current
  primary root id for same-binding retries, otherwise to the stable default
  `primary`. This keeps retry behavior deterministic without forcing
  first-time clients to invent ids.
- If supplied, `root_id` must be log-safe ASCII using letters, digits,
  underscore, dash, dot, or colon; 1-128 characters.
- `display_name` is optional, trimmed, and capped at 120 characters. If omitted,
  derive a conservative display name from the directory basename or Sandbox
  volume binding display name.
- `expected_workspace_version` is optional, but when supplied it is a strict
  optimistic-lock token that must be checked in the DB transaction that writes
  the root binding.

For `host_local`:

- `absolute_root` is required.
- `sandbox_volume_id` is rejected.
- `absolute_root` must resolve to an existing directory.
- The directory must be inside a configured allowed root.
- The root itself must not be a symlink.
- The response must include only `path_hint`, not `absolute_root`.

For `sandbox_volume`:

```json
{
  "backend": "sandbox_volume",
  "root_id": "primary",
  "sandbox_volume_id": "volume-123",
  "display_name": "Website build",
  "replace_existing": false,
  "expected_workspace_version": 3
}
```

- `sandbox_volume_id` is required.
- `absolute_root` is rejected.
- The id must be bounded and safe for logging: ASCII letters, digits,
  underscore, dash, dot, and colon; 1-128 characters.
- A Workspace-owned Sandbox volume resolver interface validates ownership or
  returns a conservative `not_configured`/`unavailable` state. The default first
  slice can be syntax-only when no Sandbox volume registry exists, but it must
  keep the validation call site explicit so real Sandbox lifecycle can replace
  it without changing endpoint behavior.

Response:

- Return `WorkspaceRootsResponse`.
- Use status `200` for first attach, idempotent same-root replay, and explicit
  replacement.
- Include `workspace_profile: project`.
- Include `primary_root`.
- Continue to include `roots`.

## Replacement And Idempotency Rules

- If no primary root exists, attach the requested root.
- If the current primary root matches the requested backend and resolved binding
  target, replay as idempotent even when `root_id` was omitted. Same-binding
  comparison uses backend plus resolved `absolute_root` for `host_local`, or
  backend plus `sandbox_volume_id` for `sandbox_volume`; it does not rely on
  `root_id` alone.
- Same-binding replay may update mutable binding metadata when provided and
  should repair operational state from the latest validation result. For
  example, a previously unavailable Sandbox binding can move from
  `sandbox_mount_state: unavailable` to `ready` when the resolver later confirms
  it.
- If a different primary root exists and `replace_existing` is false, return
  `409 Conflict` with `code: workspace_primary_root_exists`.
- If a different primary root exists and `replace_existing` is true, replace it
  through the existing one-primary-root DB semantics.
- If `expected_workspace_version` is provided and does not match the current
  workspace version inside the write transaction, return `409 Conflict`.
- The DB layer remains responsible for final uniqueness/race handling. A
  service-only precheck is useful for diagnostics, but not sufficient for the
  optimistic-lock guarantee.

This avoids accidental root replacement while keeping the API usable for
retries and migration flows.

## Configuration

Host-local project roots should not reuse broad ingestion allowlists by default.
The Workspace service should resolve allowed roots from:

1. `[WORKSPACES].project_root_allowed_base_paths`
2. `WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS`
3. `TLDW_WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS`
4. Compatibility fallback: `[ACP-WORKSPACE].allowed_base_paths` and
   `ACP_WORKSPACE_ALLOWED_BASE_PATHS`

If no roots are configured, attaching a `host_local` root returns `503` with
`code: workspace_project_roots_not_configured`.

The compatibility fallback exists because ACP workspace roots already gate
agentic project work. Ingestion allowed roots are intentionally excluded because
permission to ingest from a directory is not the same as permission to treat it
as a writable project workspace.

The implementation must add matching config examples and operator-facing docs
for the new `[WORKSPACES].project_root_allowed_base_paths` key and both env var
names. Tests should cover precedence, de-duplication, compatibility fallback,
and the no-configured-roots failure path.

## Service Boundary

Create a Workspace Core service, tentatively:

```text
tldw_Server_API/app/core/Workspaces/root_binding_service.py
```

Responsibilities:

- Normalize and validate attach requests.
- Resolve allowed project-root base paths.
- Validate host-local root containment and symlink constraints.
- Validate sandbox volume id shape and call a resolver interface.
- Enforce explicit replacement intent before calling the DB method.
- Call a DB method that performs root binding persistence and optional
  `expected_workspace_version` comparison in the same transaction. This can be
  an extended `CharactersRAGDB.upsert_workspace_primary_root` or a narrowly
  named wrapper that delegates to it after the version check.
- Return the persisted root row plus enough metadata for the endpoint to render
  `WorkspaceRootsResponse`.

The service should not:

- import FastAPI
- raise `HTTPException`
- start Sandbox sessions or Jobs
- mutate MCP or ACP records
- expose absolute paths in returned public response helpers

Expected service-level exceptions:

- `WorkspaceRootInputError`
- `WorkspaceRootConflictError`
- `WorkspaceRootConfigurationError`
- `WorkspaceRootValidationError`

The endpoint maps these to existing HTTP error patterns.

## Host-Local Validation

The host-local validation path should:

1. Expand `~` but require an absolute path after expansion.
2. Resolve the candidate with `strict=False` first for stable diagnostics.
3. Require the final candidate to exist and be a directory.
4. Reject if the candidate itself is a symlink.
5. For each configured allowed root, use the existing
   `resolve_safe_local_path(candidate, allowed_root)` containment primitive.
6. Reject if no allowed root contains the candidate.
7. Store the resolved absolute root in the DB, but only return a basename or
   display name through public responses.

The service should not recursively scan the tree in this slice. Recursive
symlink handling belongs to the future file inventory Jobs worker.

## Sandbox Volume Wrapper

The first slice makes sandbox-managed roots first-class at the Workspace API
boundary without claiming full Sandbox lifecycle support.

Introduce a small resolver protocol:

```python
class SandboxVolumeResolver(Protocol):
    def validate_workspace_volume(
        self,
        *,
        workspace_id: str,
        user_id: str,
        sandbox_volume_id: str,
    ) -> SandboxVolumeBinding:
        ...
```

`SandboxVolumeBinding` should be bounded:

- `sandbox_volume_id`
- `state`: `ready`, `not_configured`, `unavailable`, or `failed`
- `display_name`
- `reason_code`

The default resolver can be conservative and syntax-only until a persistent
Sandbox volume registry exists, but it must not claim ownership or readiness.
In this first slice, an unconfigured resolver may either reject strict attach
requests with `503`, or persist a fail-closed binding with
`sandbox_mount_state: not_configured` and all Sandbox/MCP/ACP/action capability
flags disabled until a real resolver validates ownership. Tests should inject a
fake resolver so the service boundary is real from the beginning.

Persisted root defaults for a successful sandbox attach:

- `backend: sandbox_volume`
- `root_state: attached` to represent the Workspace binding, not proof that the
  volume is mounted or action-ready
- `sandbox_mount_state`: resolver state, or `not_configured`
- `absolute_root`: omitted
- `sandbox_volume_id`: requested volume id

When `sandbox_mount_state` is not `ready`, response capabilities must remain
fail-closed for write, preview, indexing, MCP, ACP, and agent launch actions.

## Error Mapping

Suggested API status mapping:

- `400`: malformed payload, wrong fields for backend, non-absolute path,
  nonexistent path, not a directory, symlink root.
- `403`: host-local path outside configured allowed roots.
- `404`: workspace not found.
- `409`: primary root exists without `replace_existing`, backend mismatch,
  expected workspace version mismatch, DB uniqueness conflict.
- `503`: project-root allowed roots are not configured, or the Sandbox volume
  resolver is unavailable when strict validation is required.

Error details should include stable `code` and `message` fields. They must not
include local absolute paths except in server logs.

## Security And Privacy

- Do not expose `absolute_root` in attach responses, context responses,
  capability responses, or `/roots`.
- Log rejected paths only at warning/debug level and avoid echoing raw paths in
  client-facing errors.
- Treat host-local roots as write-capable project surfaces, not just readable
  ingestion sources.
- Keep sandbox volume ids bounded and log-safe.
- Fail closed for write, Sandbox, MCP, ACP, preview, and indexing actions when
  root validation or resolver state is incomplete.

## Testing Strategy

Unit tests:

- Allowed-root parsing and de-duplication.
- `root_id` omission resolves to a deterministic `primary` binding and remains
  idempotent on retry.
- Supplied `root_id` and `display_name` bounds are enforced.
- Host-local happy path.
- Host-local rejects non-absolute, nonexistent, file, symlink root, traversal,
  and outside-allowlist candidates.
- Sandbox volume happy path with injected fake resolver.
- Sandbox volume rejects missing/unsafe ids.
- Unconfigured/default Sandbox resolver behavior is fail-closed and does not
  expose action-ready capabilities.
- Existing root without `replace_existing` returns conflict.
- Same-root replay is idempotent.
- Same-root replay repairs operational state from the latest validation result.
- `expected_workspace_version` mismatch is enforced by the DB write transaction,
  not only by service precheck.
- Workspace project-root config precedence, fallback, docs examples, and
  no-configured-roots handling.

API tests:

- `PUT /roots/primary` attaches a host-local root and returns redacted
  `WorkspaceRootsResponse`.
- A research Workspace becomes `workspace_profile: project`.
- Replacement requires `replace_existing: true`.
- `sandbox_volume` attach returns `sandbox_volume_id` as `path_hint`.
- Existing `/roots`, `/context`, and `/capabilities` responses remain
  compatible and redacted after attach.
- DB/service errors map to contextual HTTP responses.

Verification:

- Focused Workspace tests.
- Compile smoke for Workspace Core and workspace endpoint/schema files.
- Bandit on touched backend scope.
- `git diff --check`.

## Open Follow-Ups

- Public detach/archive root API.
- Sandbox volume create/delete/mount lifecycle and orphan recovery.
- File inventory Jobs worker.
- Explicit file-content indexing policy.
- MCP trusted-root binding mutation.
- ACP/harness runtime envelope consumption.
- Project Workspace UI root attach flow.
