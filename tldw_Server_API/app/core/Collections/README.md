# Collections

Collections manages reading-list content, saved outputs, feed/digest jobs,
reading import/export, embeddings enqueue, and collection utility helpers. It is
the core service layer behind reading, items, output templates, generated
outputs, reading highlights, and reading digest/import sidecar workers.

## Start Here

- Reading service: `reading_service.py`.
- Imports and jobs: `reading_importers.py`, `reading_import_jobs.py`,
  `reading_import_jobs_worker.py`, `reading_digest_jobs.py`, and
  `reading_digest_jobs_worker.py`.
- Embeddings queue: `embedding_queue.py`.
- Helpers: `utils.py`.
- API endpoints and schemas: `app/api/v1/endpoints/reading.py`,
  `reading_highlights.py`, `items.py`, `outputs.py`, `outputs_templates.py`,
  and matching schemas in `app/api/v1/schemas/`.
- Tests: `tests/Collections/`.

## Responsibilities

- Save, list, update, import, and export reading-list items.
- Queue embeddings work for new or changed reading content.
- Manage reading import/digest jobs and sidecar worker entry points.
- Support generated output/template flows through endpoint/service layers.
- Provide small collection helpers such as deterministic text hashing.

## Module Map

- `reading_service.py` fetches, normalizes, deduplicates, persists, and updates
  reading items.
- `reading_importers.py` parses import formats.
- `reading_import_jobs.py` and `reading_digest_jobs.py` define Jobs-backed work.
- `reading_import_jobs_worker.py` and `reading_digest_jobs_worker.py` are worker
  entry points.
- `embedding_queue.py` enqueues collection item embeddings.
- `utils.py` contains shared hashing and helper functions.

## How It Connects

- DB adapters live in `app/core/DB_Management/Collections_DB.py`.
- Jobs tracks reading import/digest/embedding work.
- Web scraping/article extraction can hydrate reading content.
- Outputs and templates are exposed through API endpoints and may create files or
  notification-ready artifacts.

## Extension Points

- Add collection item origins through DB adapter/schema changes, then update API
  and tests.
- Add import formats in `reading_importers.py` with deterministic fixture tests.
- Add background workflows using Jobs definitions plus a worker entry point.

## Testing

- Reading service/API: `tests/Collections/test_reading_service.py` and
  `tests/Collections/test_reading_api.py`.
- Items and outputs: `tests/Collections/test_items_and_outputs_api.py`,
  `tests/Collections/test_outputs_templates_api.py`, and
  `tests/Collections/test_output_templates_seeding.py`.
- Import/export and workers: `tests/Collections/test_reading_import_export.py`
  and `tests/Collections/test_reading_digests.py`.
- Embeddings queue: `tests/Collections/test_embedding_queue.py`.

## Gotchas

- Some endpoint paths still keep compatibility fallbacks to legacy Media DB
  shapes. Preserve compatibility tests when changing listing behavior.
- Use DB adapters rather than raw SQL in endpoints and services.
