# File Artifacts

File_Artifacts creates, validates, stores, exports, and purges structured generated files. It centralizes adapter selection for file types such as tables, calendars, images, and research packages, records artifact metadata in Collections DB, writes generated export files into user storage, and runs asynchronous export work through the Jobs system.

## Start Here

- `file_artifacts_service.py` is the main service for creating artifacts, fetching records, deleting records, purging expired exports, selecting adapters, and running exports.
- `adapter_registry.py` defines the default adapter specs and loads file adapters.
- `adapters/` contains validation and export implementations for supported artifact types.
- `jobs_worker.py` consumes asynchronous file artifact export jobs.
- `metrics.py` registers optional file artifact metrics.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/files.py`, declared under `/files`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/file_artifacts_schemas.py`.
- Related tests: `tldw_Server_API/tests/FileArtifacts/`, `tldw_Server_API/tests/Files/`, and `tldw_Server_API/tests/Storage/test_file_artifacts_storage_integration.py`.

## Responsibilities

- Validate structured file artifact payloads with the adapter for the requested file type.
- Create and retrieve file artifact records in Collections DB.
- Export artifacts to requested formats such as ICS, Markdown, HTML, XLSX, CSV, JSON, PNG, JPG, or WebP when supported by the adapter.
- Write persisted export files to the user's generated output storage.
- Enqueue and consume Jobs for asynchronous artifact exports.
- Delete individual artifacts and purge expired export files.
- Expose picker-safe reference images for image-related workflows through the files endpoint.

## Module Map

- `file_artifacts_service.py`: service entrypoint, artifact creation, adapter lookup, export orchestration, deletion, and purge behavior.
- `adapter_registry.py`: default adapter registry and dynamic adapter loading.
- `adapters/base.py`: adapter protocol and validation/export result contracts.
- `adapters/table_adapter_base.py`: shared table adapter helpers.
- `adapters/data_table_adapter.py`, `markdown_table_adapter.py`, `html_table_adapter.py`, `xlsx_adapter.py`: table-oriented adapters.
- `adapters/ical_adapter.py`: calendar artifact adapter.
- `adapters/image_adapter.py`: image artifact adapter.
- `adapters/research_package_adapter.py`: research package adapter.
- `jobs_worker.py`: `file_artifact_export` worker in the `files` Jobs domain.
- `metrics.py`: optional metrics registration.

## How It Connects

- `files.py` exposes create, reference image list, get, export, delete, and purge routes.
- The endpoint uses AuthNZ user dependencies, Collections DB, Media DB for reference image candidates, and storage path helpers from `DatabasePaths`.
- `file_artifacts_schemas.py` defines create, response, delete, purge, and reference image models.
- Data Tables uses File Artifacts for export flows.
- Image Generation connects through reference image listing and the image adapter.
- Jobs integration uses `FILES_JOBS_QUEUE`, `FILES_JOBS_WORKER_ID`, and the `file_artifact_export` job type.
- Export files are written under user-scoped generated output storage and are served back by the files endpoint when ready.

## Architecture Notes

### Core Flow

- The files endpoint builds a user-scoped service with Collections DB and generated-output storage paths, then delegates artifact validation and export work to `file_artifacts_service.py`.
- Artifact creation selects an adapter from `adapter_registry.py`, validates the structured payload, and persists artifact metadata before any export file is served.
- Synchronous exports write the generated output and return ready metadata; asynchronous exports enqueue `file_artifact_export` Jobs that `jobs_worker.py` consumes.
- Download and cleanup routes consume ready export metadata, clear stale export state, and purge expired generated files.

### State And Data

- Collections DB stores artifact records, structured payload metadata, export status, MIME details, expiration, and consumption state.
- User generated-output storage holds rendered files; adapter code owns the in-memory payload and export result shape for each artifact type.
- Reference image listing reads Media DB candidates but keeps File Artifacts focused on generated-file metadata and export validation.

### Security And Operations

- Export path resolution rejects absolute paths, nested unsafe paths, and paths outside the user's generated output directory.
- Adapter validation is the trust boundary for artifact payloads. New artifact types should validate before persisting or rendering output.
- Failed async exports must reset export state so stale job ids are not treated as downloadable files.
- Keep reference-image and image-export allowlists in sync with endpoint tests before exposing new media types.

### Extension Checklist

- New artifact type: add an adapter, register it, update schemas or endpoint maps if exposed publicly, and add adapter tests.
- New export format: update the adapter, endpoint MIME map, FileArtifacts tests, and Files endpoint tests.
- New cleanup behavior: update `file_artifacts_service.py`, purge routes, storage integration tests, and async export tests together.

## Extension Points

- Add a file type by creating an adapter under `adapters/`, registering it in `adapter_registry.py`, and adding schema or endpoint support if needed.
- Add an export format by extending the relevant adapter and the endpoint MIME type map in `files.py`.
- Change asynchronous export behavior in `jobs_worker.py` and `file_artifacts_service.py` together.
- Change cleanup policy by inspecting purge behavior in `file_artifacts_service.py` and `files.py`.
- Extend reference image support by inspecting `files.py` and `Image_Generation/reference_images.py`.

## Testing

- Direct service and adapter tests live under `tldw_Server_API/tests/FileArtifacts/`.
- Endpoint tests live under `tldw_Server_API/tests/Files/`.
- Storage integration coverage lives in `tldw_Server_API/tests/Storage/test_file_artifacts_storage_integration.py`.
- Main shutdown behavior for job pollers is covered in `tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py`.

## Gotchas

- The files endpoint rejects export paths that are absolute, nested, unsafe, or outside the user generated output directory.
- Failed async exports reset export state so stale job metadata is not treated as a ready artifact.
- Adapter validation is the boundary for structured payload shape; do not bypass it when creating artifacts.
- Export files may expire or be consumed depending on stored export metadata and purge behavior.
