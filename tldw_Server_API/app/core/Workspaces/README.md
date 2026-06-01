# Workspaces

Workspaces contains helper services that support workspace capability reporting, source-ingest job coordination, source status projection, and traceable artifact export. The broader workspace CRUD API and persistence live in endpoint and ChaChaNotes database layers; this package holds focused core helpers used by those flows.

## Start Here

- `service_capabilities.py` derives workspace service readiness/capability information.
- `source_jobs.py` builds and enqueues workspace source ingestion jobs.
- `status_projection.py` projects source status from Jobs, Media, and RAG state.
- `workspace_artifact_exports.py` prepares traceable workspace artifact exports.
- Related API surface: `app/api/v1/endpoints/workspaces.py` and `app/api/v1/endpoints/workspace_migrations.py`.
- Related tests: `tests/Workspaces/`.

## Responsibilities

- Report service capabilities and readiness for workspace-dependent features.
- Queue ingestion jobs for workspace sources.
- Project workspace source status by combining job, media, and retrieval/indexing state.
- Export promoted workspace artifacts with traceability metadata.
- Keep feature-specific helper logic out of the large workspace endpoint module.

## Module Map

- `service_capabilities.py` - readiness and capability projection.
- `source_jobs.py` - workspace source job payload and enqueue helpers.
- `status_projection.py` - source status derivation.
- `workspace_artifact_exports.py` - artifact export helpers and traceability checks.

## How It Connects

- `app/api/v1/endpoints/workspaces.py` exposes workspace CRUD, sources, status, preview, and sub-resource routes.
- `app/api/v1/endpoints/workspace_migrations.py` exposes workspace migration routes.
- `app/api/v1/schemas/workspace_schemas.py` defines workspace request and response models.
- ChaChaNotes workspace tables store workspace metadata, sources, notes, and artifact references.
- Jobs, Media DB, and RAG/indexing state feed `status_projection.py`.
- Sync domain adapters and artifact promotion flows consume workspace export/status behavior.

## Extension Points

- For new capability fields, update `service_capabilities.py`, workspace schemas, and capability tests.
- For new source-ingest behavior, update `source_jobs.py` and workspace source API tests.
- For status changes, inspect `status_projection.py` and tests that combine job/media/RAG state.
- For artifact export changes, update `workspace_artifact_exports.py` and artifact promotion contract tests.

## Testing

- `tests/Workspaces/test_workspaces_api.py`
- `tests/Workspaces/test_workspace_service_capabilities.py`
- `tests/Workspaces/test_workspace_source_status_api.py`
- `tests/Workspaces/test_workspace_source_preview_context_api.py`
- `tests/Workspaces/test_workspace_sub_resources_api.py`
- `tests/Workspaces/test_workspace_migration_api.py`
- `tests/Workspaces/test_workspace_rate_limit_contract.py`
- `tests/Agent_Orchestration/test_artifact_promotion_contract.py`
- `tests/Sync/test_sync_v2_domain_adapters.py`

## Gotchas

- Source status is a projection over adjacent systems, not the primary source of truth for jobs, media rows, or embeddings.
- Artifact exports should preserve traceability to promoted artifacts; avoid exporting untracked runtime output through this helper.
