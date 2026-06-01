# Storage

Storage defines shared storage abstractions and local filesystem helpers used by generated files, user files, media-adjacent artifacts, TTS outputs, voice uploads, and quota checks. The package provides the common backend interface, a path-safe filesystem backend, generated-file registration helpers, and quota enforcement against AuthNZ storage policy.

## Start Here

- `storage_interface.py` defines the async `StorageBackend` protocol and storage exceptions.
- `filesystem_storage.py` implements local filesystem persistence with path sanitization and streaming reads.
- `generated_file_helpers.py` coordinates generated-file storage and database registration.
- `quota_enforcement.py` checks user/org/team storage quotas before writes.
- Related API surface: `app/api/v1/endpoints/storage*.py` and `app/api/v1/endpoints/admin/admin_storage_quotas.py`.
- Related tests: `tests/Storage/`.

## Responsibilities

- Provide a common async storage backend interface for storing, retrieving, streaming, deleting, and sizing stored objects.
- Store filesystem-backed objects under user/media-scoped paths while preventing path traversal outside the configured base path.
- Register generated files and their metadata with the database layer after bytes are written.
- Enforce AuthNZ-backed hard and soft storage quota checks before accepting new data.
- Support cleanup flows used by file artifacts, image generation, voice/TTS storage, and VN assets.
- Provide backup scheduling helpers for storage-related background work.

## Module Map

- `storage_interface.py` - backend protocol and `StorageError`.
- `filesystem_storage.py` - local backend implementation and safe path construction.
- `generated_file_helpers.py` - generated-file write and metadata registration helpers.
- `quota_enforcement.py` - storage quota lookup and enforcement.
- `backup_schedule_jobs.py` - job helpers for backup scheduling.

## How It Connects

- Storage endpoints in `app/api/v1/endpoints/storage.py`, `storage_user_files.py`, `storage_user_folders.py`, `storage_download.py`, `storage_usage.py`, `storage_trash.py`, and `storage_helpers.py` expose user-facing storage operations.
- `app/api/v1/endpoints/storage_admin_quotas.py` and `app/api/v1/endpoints/admin/admin_storage_quotas.py` expose storage quota administration.
- AuthNZ quota repositories provide quota state used by `quota_enforcement.py`.
- `app/core/File_Artifacts/`, `app/core/Image_Generation/`, `app/core/TTS/`, `app/core/VoiceAssistant/`, and `app/core/VN_Assets/` use storage helpers for file persistence and cleanup.

## Extension Points

- For a new storage backend, implement `StorageBackend` in `storage_interface.py` and add backend-specific tests.
- For new generated-file metadata, inspect `generated_file_helpers.py` and the relevant DB registration path first.
- For quota policy changes, update `quota_enforcement.py` and admin quota endpoint tests together.
- For cleanup behavior, add tests under `tests/Storage/` or the consuming module's cleanup tests.

## Testing

- `tests/Storage/`
- `tests/Files/`
- `tests/Admin/test_admin_storage_quotas.py`
- `tests/Image_Generation/test_reference_images.py`
- `tests/VN_Assets/test_storage_cleanup.py`

## Gotchas

- Filesystem paths must be resolved under the configured base path; do not assemble stored paths manually in callers.
- Quota enforcement intentionally blocks hard-limit and remaining-space violations, but logs and allows soft-limit cases.
- Generated-file helpers usually need to write bytes and register database metadata; doing only one side leaves orphaned state.
