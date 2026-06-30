# Workspace Cross-Resource Membership Design

Date: 2026-06-07

Task: `TASK-2315`

Epic alignment: GitHub issue #1990, Workspace Phase 2 cross-resource membership service.

## Purpose

Workspace needs one canonical way to say "this resource belongs in this workspace" without turning `/workspaces` into a global browse/search filter and without collapsing Research Workspace, Project Workspace, MCP trusted roots, ACP execution sessions, and Sandbox roots into the same persistence concept.

This design adds a generic Workspace membership layer over existing domain storage. It lets a workspace gather notes, media/sources, artifacts, chats, prompts, workflows, watchlists, ACP sessions, Sandbox sessions, and future resource types through validated links. The first implementation slice should support existing Workspace notes, media/sources, artifacts, and chats, then expose adapter contracts for reusable global notes and the remaining domains.

## Verified Current State

- Canonical Workspace identity already lives in `workspaces` in `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`.
- Workspace profile is persisted as `workspace_profile` with current values `research` and `project`.
- Current Research Workspace sub-resources are scoped tables:
  - `workspace_sources`
  - `workspace_artifacts`
  - `workspace_artifact_versions`
  - `workspace_notes`
- Conversations already have `scope_type` and `workspace_id`; workspace-scoped chat is not the same thing as generic membership.
- Project Workspace roots live in `workspace_project_roots`, with `host_local` and `sandbox_volume` backends.
- Root state, source status, service capability, and file inventory are projected on read through `tldw_Server_API/app/core/Workspaces/`.
- `tldw_Server_API/app/core/Workspaces/README.md` explicitly says source status is a projection, `workspace_id` is canonical identity, and file inventory is metadata-only.
- `GET /api/v1/workspaces/{workspace_id}/context` is the current read envelope for Workspace UI shell state.
- Current `origin/dev` includes MCP Hub effective permission preview for path-scope decisions. That preview consumes MCP policy/root bindings and should remain separate from generic Workspace resource membership.

## Concepts To Keep Separate

| Concept | Owns | Does not own |
| --- | --- | --- |
| Workspace identity | `workspace_id`, name, profile, lifecycle, owner scope | Domain-specific content internals |
| Generic workspace membership | Cross-resource links, roles, labels, provenance, transfer policy | Source ingestion/indexing state, runtime execution state |
| Research Workspace sources | Selected source set, source ordering, source readiness, grounded research context | Global Library visibility or all media ownership |
| Project Workspace root | One primary local/sandbox root, file inventory metadata, Git/root health | Arbitrary workspace membership for every file by default |
| MCP trusted root binding | Tool trust and root exposure for MCP | Canonical workspace identity |
| ACP/Sandbox runtime session | Execution lineage, runtime state, generated outputs | Workspace membership unless explicitly linked/promoted |

The practical rule: membership is association, not relocation. Adding a note or media item to a workspace must not hide it from global Notes/Library surfaces.

## Recommended Architecture

Use a server-backed generic membership table plus a typed resource adapter registry.

The membership service should live in Workspace Core and use domain adapters to validate each resource before writing a membership row. The database should store only stable association metadata; the richer title/status/preview for each resource should be resolved through adapters at read time or cached as bounded labels for list performance.

This is preferable to adding `workspace_id` columns to every domain table because most workspace relationships are many-to-many over time: a source can appear in multiple research projects, a note can be reused, and an accepted artifact can be copied or linked across projects. It is also simpler than an event/projection-first model for the first slice.

## Data Model

Add `workspace_resource_memberships` to ChaChaNotes.

```sql
CREATE TABLE IF NOT EXISTS workspace_resource_memberships (
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    label TEXT,
    transfer_policy TEXT NOT NULL DEFAULT 'link',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_by_user_id TEXT,
    updated_by_user_id TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    client_id TEXT NOT NULL DEFAULT 'unknown',
    version INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (workspace_id, resource_type, resource_id)
);
```

Recommended indexes:

- `(workspace_id, deleted, resource_type, role)`
- `(resource_type, resource_id, deleted)`
- `(workspace_id, updated_at)`

PostgreSQL should use the same shape with `BOOLEAN` defaults and `TIMESTAMP`.

### Resource Types

Use stable snake-case string resource types. The registry should fail closed for unknown types.

Initial supported adapters:

- `workspace_note`
- `media`
- `workspace_source`
- `workspace_artifact`
- `chat`

Documented adapter contracts for later slices:

- `note`
- `prompt`
- `workflow`
- `watchlist`
- `acp_session`
- `sandbox_session`
- `project_file`
- `study_deck`
- `quiz`
- `study_pack`

`workspace_source` is included separately from `media` because a source has Research Workspace-specific selection, position, and source readiness. `media` membership means "this Library item is associated with the workspace"; `workspace_source` means "this source row participates in the Research Workspace source set."

`workspace_note` is the first note adapter because current `workspace_notes` are scoped Workspace sub-resources, not rows in the global `notes` table. A future `note` adapter should point to reusable global notes after access validation is stable. The implementation should not map one to the other unless a domain flow explicitly creates a copied or promoted note.

### Lifecycle

Membership rows are soft-deleted by default so link history and future audit trails remain recoverable.

`DELETE /memberships/{resource_type}/{resource_id}` should set `deleted=1`, update `updated_at`, update `updated_by_user_id`, and bump `version`. It should be idempotent for an already-deleted row.

`POST /memberships` should handle matching rows as follows:

- Active row with matching meaningful fields: return the existing row.
- Active row with conflicting role, transfer policy, label, or canonical resource ID: return `409`.
- Deleted row: validate the backing resource again, restore the same row with `deleted=0`, apply the new request fields, update actor/provenance metadata, and bump `version`.

If the backing resource cannot be validated during restore, the restore must fail with the same not-found, forbidden, or service-unavailable error that a fresh link would return.

### Roles

Use a small generic role set first:

- `member`: Default association.
- `source`: Resource is evidence or context for research.
- `artifact`: Resource is a generated or accepted work product.
- `conversation`: Resource is a chat/session associated with the workspace.
- `runtime`: Resource is an ACP/Sandbox/MCP runtime record associated with the workspace.
- `reference`: Resource is related but not part of the active research/query context.

Adapters can expose display grouping independent of role. Do not introduce domain-specific roles like `primary_research_pdf` in the first schema; put that in metadata if needed.

### Transfer Policy

Use explicit policy values:

- `link`: Membership points to an existing resource. Global record remains intact.
- `copy`: Membership references a workspace-owned copy created by a domain flow.
- `promote`: Runtime/generated output has been promoted into a workspace-owned resource.
- `import`: Resource came from migration or external import.

The membership API should not perform copy/import/promotion itself in the first slice. It should record the policy used by the calling domain flow.

### Provenance

`provenance_json` is bounded and safe to expose. It may include:

- `created_by`
- `source_surface`
- `source_event_id`
- `origin_workspace_id`
- `operation_id`
- `job_id`
- `migration_id`
- `adapter_version`

Do not store file contents, prompt transcripts, model outputs, secrets, absolute root paths, request headers, API keys, or unbounded diagnostics in membership provenance.

## Adapter Contract

Create a Workspace membership adapter interface in `tldw_Server_API/app/core/Workspaces/membership_adapters.py`.

Each adapter should provide:

- `resource_type`: stable string.
- `validate_access(resource_id, context) -> ResourceRef`: proves existence, owner/user scope, non-deleted state, and visibility.
- `summarize(resource_id, context) -> ResourceSummary`: label, href, subtype, updated_at, deleted/archived state if available.
- `on_link(...)`: optional hook for domain side effects after a membership row is created.
- `on_unlink(...)`: optional hook for domain side effects after a membership row is soft-deleted.

The service context should include:

- `workspace_id`
- `user_id`
- `chacha_db`
- `media_db` when available
- request metadata for audit/provenance

Adapters own resource ID canonicalization. The membership service should store the canonical `resource_id` returned by the adapter, so clients can pass compatible integer or string IDs without creating duplicate rows for the same resource.

The service must reject:

- Unsupported `resource_type`.
- Missing or deleted resource.
- Cross-user/cross-owner resource.
- Resource types whose backing service is unavailable when validation is required.
- Attempts to link hidden runtime records before the runtime adapter marks them linkable.

## API Design

Add membership routes under the existing Workspaces API:

```text
GET    /api/v1/workspaces/{workspace_id}/memberships
POST   /api/v1/workspaces/{workspace_id}/memberships
GET    /api/v1/workspaces/{workspace_id}/memberships/{resource_type}/{resource_id}
DELETE /api/v1/workspaces/{workspace_id}/memberships/{resource_type}/{resource_id}
GET    /api/v1/workspace-memberships/resources/{resource_type}/{resource_id}
```

### Request And Response Shape

`POST /workspaces/{workspace_id}/memberships`:

```json
{
  "resource_type": "media",
  "resource_id": "123",
  "role": "source",
  "label": "Optional user-facing label",
  "transfer_policy": "link",
  "provenance": {
    "source_surface": "library",
    "operation_id": "optional"
  }
}
```

Response:

```json
{
  "workspace_id": "ws-123",
  "resource_type": "media",
  "resource_id": "123",
  "role": "source",
  "label": "Paper title",
  "transfer_policy": "link",
  "provenance": {},
  "summary": {
    "title": "Paper title",
    "subtitle": "PDF",
    "href": "/media/123",
    "updated_at": "2026-06-07T12:00:00Z",
    "state": "available"
  },
  "created_at": "2026-06-07T12:00:00Z",
  "updated_at": "2026-06-07T12:00:00Z",
  "version": 1
}
```

List response should support:

- `resource_type`
- `role`
- `include_deleted=false`
- `resolve=true`
- `limit`
- `cursor`

Return grouped totals by `resource_type` and `role` for manager and UI badges.

Default list ordering must be deterministic: `updated_at DESC`, then `resource_type ASC`, then `resource_id ASC`. Cursor pagination should encode that ordering tuple, not just an offset, so concurrent inserts do not produce duplicate or skipped records in normal use.

### Error Contract

Use stable machine-readable errors:

- `workspace_not_found`
- `workspace_membership_resource_type_unsupported`
- `workspace_membership_resource_not_found`
- `workspace_membership_resource_forbidden`
- `workspace_membership_backing_service_unavailable`
- `workspace_membership_conflict`
- `workspace_membership_version_mismatch`
- `workspace_archived`

Duplicate create should be idempotent when all meaningful fields match. If the existing membership has conflicting role, transfer policy, or label, return `409` unless the request explicitly opts into update semantics.

## Read Model

The membership list should be a dedicated read model, not a replacement for `/context`.

`GET /api/v1/workspaces/{workspace_id}/context` may later include a compact membership summary:

```json
{
  "memberships": {
    "total": 14,
    "by_resource_type": {"media": 6, "workspace_note": 4, "chat": 2, "workspace_artifact": 2},
    "by_role": {"source": 6, "reference": 4, "conversation": 2, "artifact": 2}
  }
}
```

Do not include the full membership list in `/context` by default. That would make the page shell too heavy for power users with many resources.

Membership summaries may help the UI explain what belongs to a Workspace, but they must not drive MCP path-permission decisions. MCP effective permission preview and runtime tool admission should continue to use MCP policy assignments, Workspace Sets, trusted-root bindings, and path-scope enforcement. Generic membership can link an MCP- or runtime-related record for discoverability only after the MCP/ACP/Sandbox adapter exposes a safe summary.

## Relationship To Existing Workspace Tables

Existing scoped tables stay in place.

- `workspace_sources`: continues to represent Research Workspace source set, ordering, selected state, and ingestion/indexing status projection.
- `workspace_artifacts`: continues to represent workspace-owned traceable work products and versions.
- `workspace_notes`: continues to represent workspace-scoped notes created inside the workspace.
- `conversations.workspace_id`: continues to represent workspace-scoped conversations.

Membership rows can point to these records, but they do not replace them in the first implementation slice.

Recommended initial backfill behavior:

- For each `workspace_sources` row, create a `workspace_source` membership with role `source`.
- Optionally also create a `media` membership with role `source` when `media_id > 0`, marked with provenance `{"source_surface":"workspace_sources_backfill"}`.
- For each `workspace_artifacts` row, create a `workspace_artifact` membership with role `artifact`.
- For each `workspace_notes` row, create a `workspace_note` membership with role `reference`. Do not pretend scoped notes are global notes.
- For each `conversations` row with `scope_type='workspace'`, create a `chat` membership with role `conversation`.

Backfill must be non-destructive and idempotent. If a resource cannot be resolved, record a bounded diagnostic and skip it; do not rewrite ownership or delete existing rows.

## Ownership And Privacy

In current single-user mode, the per-user database boundary is the main owner boundary. The membership service should still accept `user_id` and treat it as required context so multi-user support does not need an API redesign.

Rules:

- Workspace reads/writes use the existing `get_chacha_db_for_user` and `get_request_user` dependencies.
- Media validation must use the current user's Media DB.
- The API never accepts `owner_id` from clients.
- Membership ownership is derived from the Workspace and adapter-validated backing resource, not from client-supplied membership fields.
- `created_by_user_id` and `updated_by_user_id` are actor/audit fields; they are not authorization boundaries.
- Public membership summaries must not expose absolute local paths, sandbox internal mount paths, secrets, or hidden runtime payloads.
- Project file membership should use project-root-relative paths only, once implemented.

Archived Workspaces remain readable for recovery and review. Membership write operations for archived Workspaces should fail with `workspace_archived` unless the request is part of an explicit unarchive/restore flow.

## Audit And Activity

Membership writes should emit an audit/activity event with:

- `workspace_id`
- `resource_type`
- `resource_id`
- `action`: `link`, `unlink`, `restore`, `update`
- `role`
- `transfer_policy`
- `user_id`
- bounded provenance

If a general audit/activity table is not ready, the first implementation can add a Workspace-local hook interface and log events through existing sync/log primitives. The important part is to keep the service boundary ready for team workspace governance.

## Migration And Failure States

Schema migration:

- Add SQLite and PostgreSQL table creation in the workspace schema ensure path.
- Add DB methods for create, get, list-by-workspace, list-by-resource, soft-delete, restore/update.
- Catch SQLite `IntegrityError` and backend `BackendDatabaseError` consistently for duplicate races.

Operational behavior:

- Unknown resource type fails closed.
- Archived workspaces reject membership writes with `workspace_archived`.
- Missing backing DB returns `503` or mapped service-unavailable error only when validation requires it.
- Partial list resolution should not fail the whole list. Return the membership row with `summary.state='unresolved'` and a bounded reason when one adapter fails during read.
- Link/write validation must fail if the adapter cannot prove access.
- Soft-deleted resources should remain as membership rows with `summary.state='deleted'` when `resolve=true`; default list should hide deleted memberships unless requested.

## Implementation Roadmap

### Stage 1: Persistence And Schemas

Goal: Add generic membership persistence and response schemas.

Deliverables:

- DB schema for SQLite/PostgreSQL.
- DB methods with idempotent create, update, soft-delete, list-by-workspace, list-by-resource.
- Pydantic schemas for request/response/list filters.
- Unit tests for duplicate create, conflict create, soft-delete, restore, archived workspace write rejection, and list indexes.

Parallelizable: DB tests and schema model tests can be written independently.

### Stage 2: Adapter Registry And Pilot Adapters

Goal: Validate resources through domain adapters before membership writes.

Deliverables:

- Registry with fail-closed lookup.
- Pilot adapters for `workspace_note`, `media`, `workspace_source`, `workspace_artifact`, and `chat`.
- Defer global `note` membership behind the documented adapter contract unless implementation confirms stable global-note access validation without broadening the first slice.
- Adapter tests for missing/deleted/cross-owner cases.

Parallelizable: Adapter implementation can be split by resource type.

### Stage 3: API Routes

Goal: Expose membership management through Workspaces API.

Deliverables:

- `GET/POST/GET/DELETE /workspaces/{workspace_id}/memberships...`
- `GET /workspace-memberships/resources/{resource_type}/{resource_id}`
- Stable error mapping.
- Integration tests for supported type, unsupported type, idempotent duplicate, conflict duplicate, list grouping, and list-by-resource.

Parallelizable: Endpoint tests can be prepared while adapters are implemented.

### Stage 4: Backfill And Read Summary

Goal: Seed memberships from existing Workspace sub-resource data without destructive reassignment.

Deliverables:

- Idempotent backfill helper for existing scoped Workspace data.
- Bounded diagnostics for skipped/unresolved rows.
- Compact membership summary for Workspace context, if needed by UI.
- Tests proving global browse/search is unaffected.

Parallelizable: Backfill tests and context-summary tests can run independently after Stage 1.

### Stage 5: Future Resource Adapter Contracts

Goal: Make later ACP/Sandbox/MCP/project-file support predictable.

Deliverables:

- Document adapter contracts for `prompt`, `workflow`, `watchlist`, `acp_session`, `sandbox_session`, and `project_file`.
- Define project-file ID format as `{root_id}:{relative_path_hash}` or equivalent stable root-relative identifier; do not use absolute paths.
- Define runtime-link policy: ACP/Sandbox sessions are membership-linkable only after their runtime service exposes safe summaries.

Parallelizable: Documentation and future adapter stubs can happen after the registry exists.

## Tests

Focused tests:

- `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_resource_memberships_db.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_context_membership_summary.py`

Behavior to cover:

- Create membership for supported resources.
- Reject unsupported resource type.
- Reject missing/deleted resource.
- Reject cross-owner/cross-user resource where adapter can detect it.
- Duplicate same create is idempotent.
- Duplicate with conflicting role/policy returns `409`.
- Re-linking a soft-deleted membership restores the row after re-validating the backing resource.
- Archived workspaces reject membership writes with `workspace_archived`.
- List by workspace returns grouped counts and stable ordering.
- List by resource returns all memberships for that resource in the current user's scope.
- Soft-delete hides memberships by default.
- Backfill is idempotent and non-destructive.
- Existing `/workspaces/{workspace_id}/sources`, `/artifacts`, `/notes`, `/context`, and global browse/search tests still pass.

Security validation:

- Run Bandit on touched Python implementation paths once runtime code exists.
- For this docs-only design slice, Bandit is not applicable.

## Design Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Membership table becomes a dumping ground | Keep resource adapters strict and fail closed; document supported types. |
| UI treats workspace membership as global filtering | API copy and tests must state membership does not hide global records. |
| `workspace_sources` and `media` membership duplicate meaning | Keep both types explicit and document their difference. |
| Adapter reads make list endpoints slow | Support `resolve=false`, pagination, compact labels, and grouped counts. |
| Multi-user ownership gets bolted on later | Require `user_id` in service context and forbid client-supplied owner fields from day one. |
| Runtime records expose unsafe metadata | ACP/Sandbox adapters are contract-only until safe summaries exist. |
| Project file membership leaks local paths | Use root-relative IDs and redacted path hints only. |
| MCP trust becomes confused with membership | Keep MCP permission preview and path admission on MCP policy/root bindings; use membership only for discoverability. |

## Open Decisions For Implementation Plan

1. Whether global `note` follows immediately after scoped `workspace_note` or waits for broader Notes ownership/access cleanup.
2. Whether membership update should be `PATCH /memberships/{type}/{id}` in Stage 3 or deferred until create/delete/list are stable.
3. Whether backfill runs automatically during schema ensure or as an explicit maintenance endpoint/job. Recommended: explicit helper/job first to avoid surprising startup work.

## Recommendation

Proceed with the generic membership registry plus adapters. It gives Workspaces a single reusable association model, keeps existing Research Workspace and Project Workspace behavior intact, and leaves enough structure for future MCP/ACP/Sandbox/team governance without overbuilding the first slice.
