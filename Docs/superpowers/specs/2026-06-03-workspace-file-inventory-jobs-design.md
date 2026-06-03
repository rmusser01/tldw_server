# Workspace File Inventory Jobs Design

## Status

Draft for TASK-2240.

## Purpose

Project Workspaces need a safe way to understand the shape of an attached
primary root before agents, sandboxes, MCP tools, file indexing, previews, or
Git views act on it. The current Workspace Core contract stores the primary
root and exposes `file_inventory_state`, but there is no first-class scan
worker, inventory read model, status projection, or bounded diagnostic trail.

This design adds a metadata-only file inventory scanner owned by Workspace
Core and executed through Jobs. It deliberately does not index file contents.
It answers: "What files exist under this project root, what was skipped, is the
scan current, and why can the system trust or not trust this root for later
actions?"

## Goals

- Scan the attached primary root metadata through a user-visible Jobs worker.
- Persist durable scan status, bounded counts, diagnostics, and file metadata.
- Preserve the Workspace Core rule that `workspace_id` is the canonical
  identity for research and project workspaces.
- Keep absolute root paths private. Public API responses expose relative paths
  and redacted root hints only.
- Support `host_local` roots in the first implementation slice.
- Treat `sandbox_volume` roots as first-class but fail closed until a mounted
  path resolver is available.
- Honor Workspace ignore policy and a conservative `.gitignore` subset without
  automatic content indexing.
- Report partial success when some files cannot be inspected.
- Expose enough status for the WebUI to distinguish queued, scanning, current,
  partial, stale, failed, disabled, and unready roots.

## Non-Goals

- No file-content indexing.
- No source/RAG ingestion from files.
- No Git operation beyond optional metadata placeholders.
- No file open/download/preview endpoint.
- No Workspace file tree UI implementation.
- No secondary roots.
- No Sandbox volume lifecycle or mount creation.
- No MCP trusted-root mutation.
- No ACP/harness launch integration.
- No route aliases or redirects.

## Core Design Decisions

### Jobs Own Execution, Workspace Owns Durable State

The scan is a Jobs task because it is user-visible, can take time, needs
progress, and should be inspectable by existing Jobs tooling. Jobs owns worker
leasing, retries, cancellation, and progress heartbeats. Workspace Core owns the
durable scan projection so the page can render status even after Jobs rows are
archived or unavailable.

### Job Payloads Do Not Carry Absolute Root Paths

The enqueue payload should include only:

- `workspace_id`
- `root_id`
- `root_version`
- `scan_id`
- `ignore_policy_fingerprint`
- `requested_by`

The worker resolves the root from `CharactersRAGDB`, revalidates the binding,
and then scans. This keeps absolute host paths out of job payloads and prevents
stale or forged path payloads from becoming authority.

### Metadata-Only Means No File Body Reads

The scanner may read directory entries, stat metadata, symlink metadata, and
bounded ignore-policy files such as `.gitignore`. It must not read ordinary file
contents, compute content hashes, extract text, chunk files, embed files, or send
file content to models. Later explicit indexing can consume the inventory as an
allowlist candidate source.

### Public Paths Are Relative

Inventory item APIs return project-root-relative POSIX paths. They never return
`absolute_root`, user home paths, sandbox host mount paths, or symlink targets.
Diagnostic path hints follow the same rule and are length-bounded.

### Partial Success Is A Valid Outcome

Permission errors, disappearing files, stat failures, oversized paths, malformed
ignore files, and scan limits should not make the whole job fail when the root
itself is valid. The scan completes as `partial`, stores bounded diagnostics,
and updates counts. Worker setup failures, invalid payloads, missing workspace
identity, missing root binding, unready sandbox mounts, and unsafe root
validation failures produce `failed`.

## State Model

Inventory state values:

- `not_started`: no scan has been requested for the root.
- `queued`: a scan record exists and a Jobs row was enqueued.
- `scanning`: a worker acquired the job and started traversal.
- `current`: scan completed without diagnostics that affect coverage.
- `partial`: scan completed with bounded diagnostics or scan limits.
- `stale`: root binding version or policy fingerprint changed after the latest
  completed scan.
- `failed`: scan could not run or root validation failed.
- `disabled`: inventory scanning is intentionally unavailable.

Root state and inventory state are related but separate. A root can be
`attached` while inventory is `failed` or `stale`.

## Persistence Contract

Add Workspace-owned tables in `CharactersRAGDB`.

### `workspace_file_inventory_scans`

One row per scan attempt:

- `scan_id TEXT PRIMARY KEY`
- `workspace_id TEXT NOT NULL`
- `root_id TEXT NOT NULL`
- `root_version INTEGER NOT NULL`
- `job_id INTEGER`
- `job_uuid TEXT`
- `state TEXT NOT NULL`
- `requested_by TEXT`
- `ignore_policy_fingerprint TEXT NOT NULL`
- `root_snapshot_token TEXT`
- `started_at DATETIME`
- `completed_at DATETIME`
- `counts_json TEXT NOT NULL DEFAULT '{}'`
- `diagnostics_json TEXT NOT NULL DEFAULT '[]'`
- `created_at DATETIME NOT NULL`
- `updated_at DATETIME NOT NULL`
- `version INTEGER NOT NULL DEFAULT 1`

Recommended indexes:

- `(workspace_id, root_id, created_at DESC)`
- `(workspace_id, root_id, state)` for active queued/scanning lookup
- `(job_id)` when a Jobs row exists

### `workspace_file_inventory_items`

Current item projection keyed by root-relative path:

- `workspace_id TEXT NOT NULL`
- `root_id TEXT NOT NULL`
- `relative_path TEXT NOT NULL`
- `scan_id TEXT NOT NULL`
- `entry_kind TEXT NOT NULL`
- `size_bytes INTEGER`
- `mtime_ns INTEGER`
- `mode_bits INTEGER`
- `extension TEXT`
- `mime_hint TEXT`
- `language_hint TEXT`
- `ignored BOOLEAN NOT NULL DEFAULT 0`
- `ignore_reason TEXT`
- `indexing_candidate BOOLEAN NOT NULL DEFAULT 0`
- `last_seen_at DATETIME NOT NULL`
- `deleted BOOLEAN NOT NULL DEFAULT 0`
- `metadata_json TEXT NOT NULL DEFAULT '{}'`
- primary key `(workspace_id, root_id, relative_path)`

The first slice should not persist content hashes. A metadata fingerprint can be
derived from relative path, entry kind, size, mtime, and mode if needed for
change detection, but it must not require opening ordinary files.

### Root Projection

The existing `workspace_project_roots.file_inventory_state` remains the fast
root-level status flag. Detailed status comes from the latest scan row. The
read model computes `stale` when the latest scan `root_version` or
`ignore_policy_fingerprint` no longer matches the current root/policy.

## DB Method Contract

Add focused DB methods rather than ad hoc SQL in endpoints or workers:

- `begin_workspace_file_inventory_scan(workspace_id, root_id, root_version, policy_fingerprint, requested_by)`
  - returns an existing active scan for the same root when queued/scanning
  - otherwise creates a new scan with state `queued`
  - updates the root `file_inventory_state` to `queued`
- `attach_workspace_file_inventory_job(scan_id, job_row)`
  - stores Jobs id/uuid idempotently
- `mark_workspace_file_inventory_scanning(scan_id)`
  - transitions scan and root state to `scanning`
- `complete_workspace_file_inventory_scan(scan_id, state, counts, diagnostics, root_snapshot_token)`
  - accepts only `current`, `partial`, or `failed`
  - updates root `file_inventory_state`
  - stores bounded JSON
- `replace_workspace_file_inventory_items(workspace_id, root_id, scan_id, items)`
  - batch-upserts current rows and marks unseen previous rows deleted
- `get_workspace_file_inventory_status(workspace_id)`
- `list_workspace_file_inventory_items(workspace_id, prefix, cursor, limit, include_ignored)`

These methods must catch SQLite and backend abstraction errors consistently with
existing Workspace root methods.

## Jobs Contract

Create Workspace Jobs helpers:

```text
tldw_Server_API/app/core/Workspaces/file_inventory_jobs.py
```

Constants:

- `WORKSPACE_JOBS_DOMAIN = "workspaces"`
- `WORKSPACE_FILE_INVENTORY_JOB_TYPE = "workspace_file_inventory_scan"`
- Queue from `WORKSPACE_FILE_INVENTORY_JOBS_QUEUE`, default `default`.

Enqueue helper:

- Creates or reuses a DB scan record first.
- Creates a Jobs row with idempotency key
  `workspace-file-inventory-scan:{scan_id}`.
- Stores the Jobs id/uuid back on the scan row.
- Returns a response-ready scan status object.

The helper must not swallow enqueue failures silently. If Jobs is unavailable,
the API should return a clear `503` and keep the scan row in a recoverable
`failed` or unqueued state with bounded diagnostics.

## Worker Contract

Create a worker entrypoint:

```text
tldw_Server_API/app/services/workspace_file_inventory_jobs_worker.py
```

The worker:

1. Validates `job_type`.
2. Coerces and validates payload.
3. Loads the workspace primary root by `workspace_id` and `root_id`.
4. Verifies `root_version` still matches, or marks scan `stale`/`failed`.
5. Resolves a local scan path through Workspace root binding validation.
6. Fails closed for missing roots, symlink root escapes, unconfigured allowed
   roots, or unready sandbox volumes.
7. Marks scan `scanning`.
8. Runs metadata traversal in a worker thread.
9. Writes item projection in bounded batches.
10. Completes scan as `current`, `partial`, or `failed`.
11. Returns counts and diagnostics to Jobs completion result.

Worker progress should use existing JobManager progress updates or WorkerSDK
heartbeats. Durable scan state remains in Workspace tables.

## Root Resolution

`host_local` scan resolution must reuse the root-binding containment policy:

- expand and resolve the stored root
- require it to exist and be a directory
- reject the root itself when it is a symlink
- require containment inside configured Workspace project-root allowed bases
- never traverse symlinked directories

`sandbox_volume` resolution is first-class but fail-closed in this slice:

- if a mounted local path resolver exists and reports `ready`, scan that path
- otherwise mark scan `failed` with code `sandbox_mount_not_ready`
- do not invent a mount path from `sandbox_volume_id`

## Ignore Policy

The first implementation should define a Workspace-owned ignore policy module:

```text
tldw_Server_API/app/core/Workspaces/file_inventory_ignore.py
```

Policy inputs:

- built-in generated and dependency directories
- built-in secret-like file patterns
- optional workspace-level ignore patterns from root metadata or future settings
- bounded `.gitignore` files when present

Built-in directory skips should include:

- `.git`
- `node_modules`
- `.venv`
- `venv`
- `__pycache__`
- `.pytest_cache`
- `.mypy_cache`
- `.ruff_cache`
- `.next`
- `.turbo`
- `dist`
- `build`
- `coverage`
- `target`

Built-in secret-like file skips should include:

- `.env`
- `.env.*`
- `*.pem`
- `*.key`
- `id_rsa`
- `id_ed25519`
- `.netrc`

Because the repo does not currently depend on a pathspec library, the first
slice should either:

1. implement and test a conservative `.gitignore` subset with `fnmatch`, or
2. add a small dependency such as `pathspec` only after dependency review.

The safer first implementation is option 1. It should clearly document that the
policy is conservative and may skip more rather than less when unsure.

The ignore policy must produce a stable fingerprint from policy version,
built-in rules, workspace rules, and bounded ignore-file metadata. This
fingerprint drives stale detection.

## Traversal Bounds

Default bounds should be configurable but conservative:

- max files recorded: 25,000
- max directories visited: 10,000
- max depth: 32
- max single relative path length: 512 characters
- max diagnostics retained: 50
- max diagnostics JSON: 16 KiB
- max scan seconds: 120
- `.gitignore` max file bytes: 64 KiB each

When a bound is hit, the scan completes as `partial` with a diagnostic code such
as `scan_limit_reached`, `path_too_long`, or `ignore_file_too_large`.

## Diagnostics

Diagnostics are bounded, redacted, and structured:

```json
{
  "code": "permission_denied",
  "path_hint": "src/private",
  "message": "A path could not be inspected."
}
```

Rules:

- maximum 50 diagnostics
- no absolute paths
- `path_hint` is project-relative and capped at 240 characters
- `message` is generic and capped at 200 characters
- raw exception strings are not returned to public API responses
- backend logs may include more detail only when safe and never for secrets

## API Contract

Add three Workspace routes:

```http
POST /api/v1/workspaces/{workspace_id}/file-inventory/scan
GET  /api/v1/workspaces/{workspace_id}/file-inventory/status
GET  /api/v1/workspaces/{workspace_id}/file-inventory/items
```

### Scan Request

```json
{
  "force": false,
  "expected_root_version": 3
}
```

Request rules:

- `force` defaults to false.
- `expected_root_version` is optional. When supplied, it must match the current
  primary root version before a scan is queued.
- If a scan is already queued or scanning for the same root, return that scan
  instead of creating another.
- If `force` is false and the latest scan is current for the same root version
  and policy fingerprint, return the current status without enqueuing.
- If `force` is true and no scan is active, create a new scan even when the
  latest scan is current. The newly created scan gets a fresh `scan_id`, so the
  Jobs idempotency key remains retry-safe.

Response:

```json
{
  "workspace_id": "workspace-1",
  "root_id": "primary",
  "state": "queued",
  "job": {
    "id": 123,
    "uuid": "job-uuid",
    "status": "pending",
    "progress_percent": 0,
    "progress_message": "queued"
  },
  "counts": {
    "files": 0,
    "directories": 0,
    "ignored": 0,
    "diagnostics": 0
  },
  "updated_at": "2026-06-03T12:00:00Z"
}
```

### Status Response

Status returns latest durable scan information and, when available, live Jobs
progress:

- `workspace_id`
- `root_id`
- `state`
- `stale`
- `last_scan_id`
- `last_scan_started_at`
- `last_scan_completed_at`
- `ignore_policy_fingerprint`
- `root_snapshot_token`
- `counts`
- `diagnostics`
- `job`

### Items Response

Items are paginated:

Query parameters:

- `prefix`
- `limit` default 100, max 500
- `cursor`
- `include_ignored` default false
- `entry_kind`

Items expose:

- `relative_path`
- `entry_kind`
- `size_bytes`
- `mtime`
- `extension`
- `mime_hint`
- `language_hint`
- `ignored`
- `ignore_reason`
- `indexing_candidate`

No absolute paths or file contents are returned.

## Context And Capabilities Integration

Extend `WorkspaceProjectRoot` or the root projection with a nested
`file_inventory` object using the existing `WorkspaceFileInventory` schema name
as the compatibility anchor:

```json
{
  "state": "attached",
  "file_inventory_state": "current",
  "file_inventory": {
    "state": "current",
    "total_file_count": 143,
    "indexed_file_count": 0,
    "updated_at": "2026-06-03T12:00:00Z"
  }
}
```

`indexed_file_count` remains `0` until explicit file-content indexing exists.
The context response must not imply that inventoried files are queryable.

Allowed actions:

- `scan_files`: allowed only when a project root is attached and not known
  unready from persisted Workspace state. The worker still revalidates the
  filesystem or sandbox mount before scanning.
- `view_file_inventory`: allowed when a project root exists, even if last scan
  failed, so users can see diagnostics.
- `index_file_content`: remains disabled unless explicit indexing policy is
  enabled in a later slice.

## Error Mapping

- `404`: workspace not found.
- `409`: no primary root, root version mismatch, root unready for requested
  operation, conflicting active scan when force semantics cannot be honored.
- `422`: invalid request fields.
- `503`: Jobs service unavailable.

Worker-discovered root validation failures should be represented in scan status
as `failed`, even if the enqueue request originally succeeded. This includes
missing host paths, removed allowed-root configuration, unsafe symlink state,
and unready sandbox mounts.

## Security And Privacy

- No file content reads except bounded ignore-policy files.
- No absolute paths in public API responses, diagnostics, or job payloads.
- Symlink traversal disabled.
- Secret-like files are ignored by default.
- Scan limits prevent accidental unbounded traversal.
- Sandbox volume ids are not treated as mount paths.
- Item listing is scoped by Workspace ownership through the existing per-user
  `CharactersRAGDB` dependency.

## Open Questions Resolved For First Slice

- **Should source status be read-computed, event-updated, or both?** Use both:
  Jobs progress is event-updated, Workspace scan/root state is event-updated,
  and `stale` is computed on read from root version and policy fingerprint.
- **Which primitive owns scan progress?** Jobs owns execution and live progress;
  Workspace tables own durable inventory state.
- **Should file content hashes be stored?** No. Store metadata only.
- **Should `sandbox_volume` scan now?** Only if a real mounted-path resolver is
  available. Otherwise fail closed with diagnostics.
- **Should full `.gitignore` parity be required?** No. Start with a documented,
  tested conservative subset unless dependency review approves `pathspec`.

## Risks And Mitigations

- **Risk: inventory is mistaken for indexed/queryable content.**
  - Mitigation: keep `indexing_state: disabled`, `indexed_file_count: 0`, and
    disabled `index_file_content` action.
- **Risk: public API leaks host paths.**
  - Mitigation: never put absolute paths in job payloads or public responses;
    test redaction.
- **Risk: scanner overloads large repositories.**
  - Mitigation: hard bounds, partial state, progress updates, and cancellation.
- **Risk: incomplete `.gitignore` behavior surprises users.**
  - Mitigation: document conservative support and expose ignored counts/reasons.
- **Risk: stale scan appears current after root replacement.**
  - Mitigation: compare root version and policy fingerprint on read.

## Follow-Up Slices

- Explicit file-content indexing policy and indexing Jobs.
- Full Git status projection.
- Sandbox mounted-path resolver and volume lifecycle integration.
- MCP trusted-root mutation from Workspace root state.
- Project Workspace UI file tree, scan controls, and diagnostics drawer.
- Agent harness runtime envelope consumption.
