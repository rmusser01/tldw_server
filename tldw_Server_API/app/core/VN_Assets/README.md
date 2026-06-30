# VN_Assets

VN_Assets manages visual novel asset packs, asset-slot state, generated asset jobs, storage cleanup, prompt previews, manifests, and portability import/export. It is the core asset layer used by the VN APIs and by VN play/script setup flows that need ready character, background, and style assets.

## Start Here

- `service.py` is the primary asset-pack service.
- `models.py` defines asset-pack, slot, item, and readiness models.
- `jobs.py` and `worker.py` enqueue and process asset generation jobs.
- `storage.py` handles generated asset storage helpers.
- `portability/` contains export, import, preview, conflict, archive, and fingerprint helpers.
- Related API surface: `app/api/v1/endpoints/vn_assets.py`.
- Related tests: `tests/VN_Assets/`.

## Responsibilities

- Create, read, update, and delete visual novel asset packs and slots.
- Expand starter asset matrices and derive readiness/state for asset packs.
- Build generation prompt previews and enqueue idempotent asset generation jobs.
- Persist generated files safely and clean up blocked or unused generated assets.
- Build asset manifests for runtime consumers.
- Export, preview, and import portable asset-pack archives with conflict detection.
- Coordinate concurrency controls for backend asset generation.

## Module Map

- `service.py` - asset-pack service layer.
- `models.py` - domain models for packs, slots, items, readiness, and generation state.
- `jobs.py` and `worker.py` - generation job payloads, enqueue logic, and worker execution.
- `storage.py` - generated-file storage and path helpers.
- `manifest.py` - runtime manifest builder.
- `matrix.py` - starter matrix expansion.
- `state.py` - readiness and slot-status derivation.
- `prompts.py` - prompt preview generation.
- `cleanup_blockers.py`, `concurrency.py`, and `constants.py` - cleanup, locking, and shared constants.
- `portability/` - portable archive export/import/preview implementation.

## How It Connects

- `app/api/v1/endpoints/vn_assets.py` exposes asset-pack, generation, manifest, and portability routes.
- `app/api/v1/schemas/vn_asset_schemas.py` defines request and response models.
- `app/core/DB_Management/VNAssetPacks_DB.py` stores asset-pack state.
- `app/services/vn_asset_jobs_worker.py` runs queued asset generation work.
- `app/core/VN_Play/` and `app/core/VN_Scripts/` use VN asset readiness and manifests for runtime setup.
- `app/core/Image_Generation/` and generated-file storage helpers supply provider output and file persistence.

## Extension Points

- For a new asset slot or readiness rule, update `models.py`, `state.py`, `manifest.py`, schemas, and state-machine tests.
- For generation behavior, inspect `jobs.py`, `worker.py`, `prompts.py`, and generation-job tests.
- For storage cleanup changes, update `storage.py`, `cleanup_blockers.py`, and `tests/VN_Assets/test_storage_cleanup.py`.
- For archive compatibility, update the relevant `portability/` helpers and portability tests together.

## Testing

- `tests/VN_Assets/test_vn_assets_api.py`
- `tests/VN_Assets/test_vn_asset_packs_db.py`
- `tests/VN_Assets/test_generation_jobs.py`
- `tests/VN_Assets/test_matrix_expansion.py`
- `tests/VN_Assets/test_state_machine.py`
- `tests/VN_Assets/test_manifest_builder.py`
- `tests/VN_Assets/test_prompt_preview.py`
- `tests/VN_Assets/test_storage_cleanup.py`
- `tests/VN_Assets/test_backend_concurrency.py`
- `tests/VN_Assets/test_portability_*.py`

## Gotchas

- Asset generation is asynchronous and idempotency-sensitive; keep job payload and idempotency key behavior stable.
- Generated asset files and database rows must stay in sync, especially during cleanup and portability import.
- Portability archives need validation and conflict handling before writing into a user's asset-pack database.
