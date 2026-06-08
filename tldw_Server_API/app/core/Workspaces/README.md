# Workspaces

Workspaces contains helper services that support the canonical Workspace Core
contract, workspace capability reporting, source-ingest job coordination, source
status projection, and traceable artifact export. The broader workspace CRUD API
and persistence live in endpoint and ChaChaNotes database layers; this package
holds focused core helpers used by those flows.

## Start Here

- `models.py` defines canonical Workspace Core literals and fail-closed helpers.
- `context.py` builds read-only workspace context envelopes.
- `service_capabilities.py` derives workspace service readiness/capability information.
- `source_jobs.py` builds and enqueues workspace source ingestion jobs.
- `status_projection.py` projects source status from Jobs, Media, and RAG state.
- `file_inventory_models.py`, `file_inventory_ignore.py`, `file_inventory_scanner.py`,
  and `file_inventory_jobs.py` implement metadata-only project-root inventory
  scanning and Jobs enqueue helpers.
- `workspace_artifact_exports.py` prepares traceable workspace artifact exports.
- Related API surface: `app/api/v1/endpoints/workspaces.py` and `app/api/v1/endpoints/workspace_migrations.py`.
- Related tests: `tests/Workspaces/`.

## Responsibilities

- Normalize Workspace Core profile, kind, root, resolution, and allowed-action contracts.
- Report service capabilities and readiness for workspace-dependent features.
- Queue ingestion jobs for workspace sources.
- Project workspace source status by combining job, media, and retrieval/indexing state.
- Queue and project metadata-only file inventory scans for Workspace-owned
  primary project roots.
- Export promoted workspace artifacts with traceability metadata.
- Keep feature-specific helper logic out of the large workspace endpoint module.

## Module Map

- `models.py` - canonical profiles, kinds, root states, and allowed-action helpers.
- `context.py` - Workspace Core context and runtime capability envelope builder.
- `service_capabilities.py` - readiness and capability projection.
- `source_jobs.py` - workspace source job payload and enqueue helpers.
- `status_projection.py` - source status derivation.
- `file_inventory_*` - ignore policy, metadata scanner, durable status models,
  and Jobs enqueue helpers for project-root file inventory.
- `workspace_artifact_exports.py` - artifact export helpers and traceability checks.

## Membership

`workspace_resource_memberships` stores Workspace-to-resource associations. It
does not transfer ownership, hide or move global records, or make `/workspaces`
the global filter for Library, Notes, Chat, or search surfaces. The first slice
supports explicit links for `workspace_note`, `media`, `workspace_source`,
`workspace_artifact`, and `chat`.

`workspace_sources` remains the Research Workspace source-selection and
readiness table. A `workspace_source` membership can associate that source row
with the generic membership read model, but it does not replace source
selection, source ordering, ingest readiness, or status projection behavior.

The explicit backfill helper links existing Workspace-scoped source, artifact,
note, and chat rows into `workspace_resource_memberships` on demand. Backfill is
idempotent and intentionally not automatic at startup; callers must opt in when
they want existing rows represented in the generic membership table.

MCP effective permission preview and path admission continue to use MCP policy
and root bindings. Generic Workspace membership is not a trust source for MCP
tool execution, file access, ACP execution, or Sandbox path admission.

Future membership adapters must validate access through the owning domain
adapter before linking or resolving a resource. Unsupported resource types must
fail closed, and summaries/provenance should avoid exposing secrets, absolute
paths, sandbox mount paths, prompts, model output contents, or file contents.

## How It Connects

- `app/api/v1/endpoints/workspaces.py` exposes workspace CRUD, sources, status, preview, sub-resource routes, and read-only Workspace Core contract surfaces.
- `app/api/v1/endpoints/workspace_migrations.py` exposes workspace migration routes.
- `app/api/v1/schemas/workspace_schemas.py` defines workspace request and response models.
- ChaChaNotes workspace tables store workspace metadata, persisted `workspace_profile`, project roots, sources, notes, and artifact references.
- Jobs, Media DB, and RAG/indexing state feed `status_projection.py`.
- Sync domain adapters and artifact promotion flows consume workspace export/status behavior.

## Workspace Core Contract

- `workspace_id` is the canonical identifier for research and project workspaces.
- `workspace_profile` is persisted workspace intent. Current values are `research`
  and `project`.
- `workspace_kind` is a compatibility/display alias derived from
  `workspace_profile`; it is not a second source of truth.
- A research workspace may be upgraded to a project workspace by attaching a
  Workspace-owned primary root.
- A project root may use `host_local` or `sandbox_volume` as its persisted
  backend. Public responses expose only redacted `path_hint` values, never
  `absolute_root`.
- Root creation and mutation are not public API in this slice. The first public
  root surface is the read-only `GET /api/v1/workspaces/{workspace_id}/roots`
  contract.
- `GET /api/v1/workspaces/{workspace_id}/capabilities`,
  `GET /api/v1/workspaces/{workspace_id}/context`, and
  `GET /api/v1/workspaces/{workspace_id}/roots` are read contract surfaces for
  workspace profile, root state, resolution status, service capability state,
  and fail-closed allowed actions.

## Architecture Notes

### Core Flow

- Workspace CRUD and subresource routing live in `workspaces.py`; this package
  provides focused helpers used by that endpoint rather than owning the full
  persistence model.
- `context.py` consumes the persisted workspace row, optional primary root,
  source summary, service capabilities, and partial dependency errors to build
  schema-versioned read envelopes.
- `source_jobs.py` creates stable workspace-source ingest Jobs after a source
  row exists, using idempotency keys based on workspace id, source id, and media
  id.
- `status_projection.py` builds read-computed source status by merging
  workspace source rows, active/failed Jobs, Media DB chunk/vector state, and
  readiness summaries.
- File inventory scans are Jobs-backed and metadata-only. They enumerate
  project-root-relative paths, entry kinds, size/mtime/mode metadata, extension
  and MIME/language hints, ignore decisions, bounded diagnostics, and aggregate
  counts. They do not read ordinary file contents, extract symbols, create
  embeddings, or add files to source/RAG indexes.
- Public inventory responses and Workspace Core context use redacted
  `path_hint` and project-root-relative paths. They must not expose
  `absolute_root` through normal workspace read contracts.
- The primary in-process worker is controlled by
  `WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ENABLED` and the `workspaces` route
  gate. The worker consumes the workspace file inventory Jobs queue; disabling
  it leaves scan requests queued or inactive depending on Jobs backend behavior.
- `workspace_artifact_exports.py` renders accepted artifact versions to export
  payloads with identity, lineage, review metadata, and redaction metadata.

### State And Operations

- ChaChaNotes remains the workspace metadata store; Jobs, Media DB, RAG, and
  Sync provide adjacent state that is projected into workspace responses.
- Workspace profile and project-root binding are persisted. Runtime readiness,
  resolution, and allowed actions are computed on read and fail closed when a
  required subsystem state is unknown or unavailable.
- Source status is intentionally derived at read time. Avoid writing projected
  readiness back as source truth unless the database contract changes.
- Artifact export requires an accepted review state and preserves traceability
  through `root_artifact_id`, `artifact_version_id`, `source_lineage`, and
  embedded metadata.
- Generic membership rows are a read-model association layer over domain-owned
  resources. Domain tables and adapters remain responsible for ownership,
  visibility, and access validation.

### Follow-Up Slices

- Public root attach/update API with host-local validation and Sandbox-managed
  root wrappers.
- Sandbox volume creation and mount lifecycle.
- Explicit file-content indexing policy and indexing Jobs. This is intentionally
  separate from metadata-only inventory so users can inspect files before any
  content extraction or embedding work is permitted.
- MCP trusted-root binding and ACP/harness runtime consumption.
- Project Workspace UI for root health, file tree metadata, Git state, and
  remediation actions.

### Extension Checklist

- New source lifecycle state: update `status_projection.py`,
  `workspace_schemas.py`, and source status API tests.
- New Workspace Core state: update `models.py`, `context.py`,
  `workspace_schemas.py`, and the roots/capabilities/context API tests.
- New workspace job type: update `source_jobs.py`, Jobs queue expectations, and
  idempotency tests.
- New artifact export format: update `workspace_artifact_exports.py`, schema
  allowlists, and artifact promotion/export contract tests.
- New membership resource type: add a fail-closed domain adapter, validate
  resource access through the owning domain API, update schemas/tests, and keep
  MCP/root/path trust decisions outside generic membership.

## Extension Points

- For new capability fields, update `service_capabilities.py`, workspace schemas, and capability tests.
- For new project-root behavior, update `models.py`, `context.py`, ChaChaNotes
  root persistence methods, workspace schemas, and workspace API/DB tests.
- For new source-ingest behavior, update `source_jobs.py` and workspace source API tests.
- For status changes, inspect `status_projection.py` and tests that combine job/media/RAG state.
- For artifact export changes, update `workspace_artifact_exports.py` and artifact promotion contract tests.

## Testing

- `tests/Workspaces/test_workspace_core_models.py`
- `tests/Workspaces/test_workspace_project_roots_db.py`
- `tests/Workspaces/test_workspace_core_context.py`
- `tests/Workspaces/test_workspaces_api.py`
- `tests/Workspaces/test_workspace_service_capabilities.py`
- `tests/Workspaces/test_workspace_source_status_api.py`
- `tests/Workspaces/test_workspace_source_preview_context_api.py`
- `tests/Workspaces/test_workspace_sub_resources_api.py`
- `tests/Workspaces/test_workspace_membership_adapters.py`
- `tests/Workspaces/test_workspace_memberships_api.py`
- `tests/Workspaces/test_workspace_context_membership_summary.py`
- `tests/Workspaces/test_workspace_migration_api.py`
- `tests/Workspaces/test_workspace_rate_limit_contract.py`
- `tests/Agent_Orchestration/test_artifact_promotion_contract.py`
- `tests/Sync/test_sync_v2_domain_adapters.py`

## Gotchas

- Source status is a projection over adjacent systems, not the primary source of truth for jobs, media rows, or embeddings.
- `workspace_profile` is the persisted source of truth; do not branch on
  `workspace_kind` except for compatibility/display behavior.
- Do not expose `absolute_root` through ordinary workspace responses. Use
  redacted `path_hint` values unless a future privileged admin endpoint defines
  a stricter access contract.
- Artifact exports should preserve traceability to promoted artifacts; avoid exporting untracked runtime output through this helper.
