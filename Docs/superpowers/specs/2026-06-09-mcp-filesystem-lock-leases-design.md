# MCP Filesystem Lock Leases Design

## Goal

Add advisory filesystem lock leases for MCP safe file tools so agents can reduce edit races before calling `fs.patch` or `fs.write`. This first slice is process-local and in-memory, matching the approved short-term scope. It must not pretend to coordinate across multiple server processes, hosts, or persisted restarts.

## Scope

This slice adds two tools to the existing filesystem MCP module:

- `fs.lock_acquire`: acquire or renew an exclusive advisory lease for one workspace-relative path.
- `fs.lock_release`: release a lease by path and lease token.

The tools use the same trusted workspace-root resolution and path normalization as `fs.write` / `fs.patch`. Lock paths are returned only as normalized workspace-relative paths. Responses must not include absolute host paths.

Lock acquisition may target a nonexisting file for create flows, but the parent path must resolve inside the workspace. If the target already exists as a symlink, acquisition fails with `file_not_regular` because the mutation tools would reject the same target.

## Non-Goals

- No shared DB-backed lock table.
- No filesystem lock files.
- No mandatory locks for every deployment.
- No delete, rename, move, share, chmod, or admin file actions.
- No bypass of hash/read-receipt preimage checks. Locks reduce races; they do not replace preimage validation.

## Lease Model

Each lease is keyed by a normalized workspace identity plus workspace-relative path. The initial workspace identity is the resolved workspace-root path string. This is process-local metadata, not a cross-process guarantee.

A lease contains:

- `lease_id`: opaque server-generated token.
- `path`: normalized workspace-relative path.
- `owner`: caller-supplied owner label, sanitized and bounded.
- `expires_at`: UTC ISO timestamp.
- `ttl_seconds`: bounded TTL actually granted.
- `workspace_id` and `session_id`: optional context metadata when available.

Acquire behavior:

- If no active lease exists, create one.
- If an active lease exists with the same `lease_id`, renew it.
- If an active lease exists with a different `lease_id`, fail with `lock_conflict`.
- Expired leases are cleaned opportunistically before decisions.

Release behavior:

- Release succeeds only when the caller supplies the matching path and `lease_id`.
- Releasing a missing or expired lease is idempotent and returns `released: false`.
- Releasing with a wrong active `lease_id` fails with `lock_conflict`.

## Mutation Validation

`fs.patch`, `fs.write`, and `fs.edit` get an optional `lock_lease_id` argument. The default is advisory-only: mutations do not require a lock unless module setting `require_lock_for_mutation` is true.

When a `lock_lease_id` is supplied, the mutation validates that every affected path has an active matching lease before writing. When the setting requires locks, missing `lock_lease_id` fails with `lock_required`.

Validation happens after path resolution and before final write commit. Existing preimage checks and immediate preimage rechecks remain authoritative.

## Safety And Policy

Lock acquire/release are management tools but not write tools. They require path-scope action `lock`, with file-policy metadata for the `lock` action. If the current policy implementation does not know `lock`, this slice must fail closed or add the action metadata in the approved file-policy action registry. If the taxonomy already knows `lock` but marks it unimplemented, this slice should flip that metadata to implemented once the tools land.

The tools should be path-boundable through `path_argument_hints: ["path"]`. For `fs.patch`, lock validation must use parsed patch paths, not caller-supplied path hints.

## Result Shape

Acquire success:

```json
{
  "acquired": true,
  "renewed": false,
  "path": "docs/story.txt",
  "lease_id": "opaque-token",
  "owner": "agent-1",
  "expires_at": "2026-06-09T16:30:00Z",
  "ttl_seconds": 300
}
```

Conflict:

```json
{
  "reason_code": "lock_conflict",
  "path": "docs/story.txt",
  "held": true,
  "held_owner": "agent-1",
  "expires_at": "2026-06-09T16:30:00Z"
}
```

Conflict payloads are actionable but safe: no absolute paths, no process IDs, no raw context metadata beyond the caller-supplied owner label.

## Testing

Focused tests should cover:

- Tool descriptors and strict argument validation.
- Acquire, conflict, renew, TTL expiry, and release behavior.
- Path escape rejection through existing workspace resolution.
- Optional lock validation for `fs.write replace`.
- Optional lock validation for `fs.patch` with module-derived patch paths.
- Existing hash/read-receipt preimage tests stay green.

## Follow-Up

A later slice can replace the in-memory manager with a filesystem- or DB-backed implementation through a small lock-manager interface. That slice should define cross-process ownership, cleanup, and admin inspection semantics explicitly.
