# Ingestion Sources

Ingestion_Sources manages persistent sync sources that feed Media DB or Notes from local directories, archives, and Git repositories. It owns source metadata, snapshots, diffing, artifact handling, sink application, local-directory entitlement checks, and Jobs-backed sync enqueueing while the worker and scheduler services run the actual background sync loop.

## Start Here

- `models.py` defines source types, sink types, policies, and shared dataclasses.
- `service.py` owns ingestion source schema creation, CRUD, snapshots, artifacts, items, and events.
- `local_directory.py`, `archive_snapshot.py`, and `git_repository.py` build source snapshots.
- `diffing.py` compares snapshots and normalizes archive roots.
- `sinks/` applies sync changes to Media DB or Notes.
- `jobs.py` enqueues source sync jobs.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/ingestion_sources.py`.
- Related tests: `tldw_Server_API/tests/Ingestion_Sources/`.

## Responsibilities

- Create, list, update, and read ingestion source records.
- Keep source identity immutable after the first successful sync.
- Build source snapshots from allowed local roots, safe archive uploads, or Git repositories.
- Diff snapshots into created, changed, unchanged, and deleted items.
- Apply changes to Media DB or Notes through sink implementations.
- Persist source artifacts, sync items, snapshots, and events.
- Enqueue user-visible sync jobs in the Jobs system.
- Gate local-directory creation through feature flags and AuthNZ context.

## Module Map

- `models.py`: enums, dataclasses, and source identity structures.
- `service.py`: DB schema, source lifecycle, snapshot records, artifacts, items, and events.
- `local_directory.py`: filesystem snapshot builder with allowed-root and suffix checks.
- `archive_snapshot.py`: ZIP/TAR validation, extraction safety, artifact persistence, and retention.
- `git_repository.py`: local and GitHub repository snapshot builder.
- `diffing.py`: snapshot comparison and archive root normalization.
- `jobs.py`: `ingestion_sources` sync job creation.
- `access_policy.py`: local-directory entitlement checks.
- `sinks/media_sink.py`: Media DB create, update, and canonical delete handling.
- `sinks/notes_sink.py`: Notes create, update, soft delete, and detached-conflict handling.

## How It Connects

- `ingestion_sources.py` exposes `/ingestion-sources` routes for CRUD, capabilities, directory browsing, archive upload, sync, items, and reattach flows.
- `app/services/ingestion_sources_worker.py` loads snapshots, diffs them, applies sinks, records degraded item state, and completes or fails Jobs.
- `app/services/ingestion_sources_scheduler.py` uses APScheduler to enqueue scheduled sync jobs.
- Media DB, ChaChaNotes DB, Storage, Jobs, AuthNZ, feature flags, and GitHub remote access are adjacent dependencies.
- API documentation lives in `Docs/API-related/Ingestion_Sources_API.md`.

## Extension Points

- Add a source type by extending `models.py`, adding a snapshot builder, updating endpoint validation, and teaching the worker to dispatch it.
- Add a sink by creating a new implementation under `sinks/` and extending source creation validation.
- Change local-directory allowed roots in the config helpers used by `local_directory.py` and endpoint directory browsing.
- Add Git provider support in `git_repository.py` after matching its URL and size-safety checks.
- Extend sync event or item state in `service.py` and worker tests together.

## Testing

- Direct coverage lives under `tldw_Server_API/tests/Ingestion_Sources/`.
- Tests cover API flows, service schema behavior, workers, scheduler behavior, local directories, archive validation, Git repositories, sinks, path browsing, and local-directory access policy.

## Gotchas

- Git repository sources currently support the Notes sink only.
- Archive uploads reject traversal, symlink, encrypted ZIP, and unsupported member cases.
- Local-directory snapshots only include configured suffixes and skip symlinks.
- There is no module-specific job status endpoint; sync progress is surfaced through the shared Jobs path and ingestion source item state.
