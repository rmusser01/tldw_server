# WebClipper

WebClipper saves captured web content into notes, workspace placements, attachments, enrichment records, and related source status. The package currently centers on `WebClipperService`, which validates clip input, persists note content, handles attachments, and maps enrichment/status updates for the web clipper API.

## Start Here

- `service.py` contains `WebClipperService` and its save/status/enrichment operations.
- Related API surface: `app/api/v1/endpoints/web_clipper.py`.
- Related schemas: `app/api/v1/schemas/web_clipper_schemas.py`.
- Related tests: `tests/Notes_NEW/unit/test_web_clipper_service.py` and web clipper API tests.

## Responsibilities

- Save captured page content as notes with preserved title, body, metadata, and source URL data.
- Place clips into workspaces and workspace sources when requested.
- Persist and reference attachments associated with a clip.
- Track clip status and enrichment state after the initial save.
- Map service errors into endpoint-level responses through the web clipper API.
- Keep note, workspace, media/source, and attachment updates coordinated.

## Module Map

- `service.py` - clip save, status lookup, enrichment persistence, and cross-domain persistence helpers.

## How It Connects

- `app/api/v1/endpoints/web_clipper.py` exposes clip creation, status, and enrichment routes.
- `app/api/v1/schemas/web_clipper_schemas.py` defines clip request/response contracts.
- Notes and ChaChaNotes DB methods provide note persistence.
- Workspace APIs and database methods provide optional workspace placement/source tracking.
- Media/source and attachment helpers are used when clips create source records or include files.

## Extension Points

- For a new captured content field, update `service.py`, `web_clipper_schemas.py`, and service/API tests together.
- For attachment handling, inspect the save path in `service.py` and tests that assert attachment persistence.
- For workspace placement or source-status changes, update `service.py` and workspace-aware web clipper tests.
- For enrichment status changes, start with `persist_enrichment` and endpoint error-mapping tests.

## Testing

- `tests/Notes_NEW/unit/test_web_clipper_service.py`
- `tests/Notes_NEW/unit/test_web_clipper_endpoint_error_mapping.py`
- `tests/Notes_NEW/integration/test_web_clipper_api.py`
- `tests/ChaChaNotesDB/test_web_clipper_db.py`

## Gotchas

- Clip body fidelity matters; avoid transformations that change the saved user-visible page content without tests.
- A clip can touch notes, workspaces, source status, and attachments in one operation, so partial failure behavior should be explicit.
- Workspace/source status in the clip response should come from persisted state, not recomputed optimistic assumptions.
