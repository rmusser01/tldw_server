# Phase 4.4 Endpoint Hotspot Inventory

**Date:** 2026-04-25

**Status:** Inventory complete; route-family implementation plans pending.

## Purpose

Rank large endpoint modules and choose a safe first decomposition target. This is a planning artifact only. It avoids runtime route movement until Phase 2/3 closeout and Phase 3 helper contracts are stable.

## Method

Static line-count snapshot from selected endpoint modules, plus route/dependency scans for `storage.py`, `slides.py`, and `data_tables.py`.

No OpenAPI diff, frontend caller audit, or endpoint tests were run in this pass.

## Top Endpoint Hotspots

| File | Lines | Risk | First-pass recommendation |
| --- | ---: | --- | --- |
| `persona.py` | 7395 | Very high | Defer. Large user-facing surface and likely frontend coupling. |
| `character_chat_sessions.py` | 6787 | Very high | Defer. Chat/session/provider behavior is too broad for a first slice. |
| `chat.py` | 6269 | Very high | Defer. OpenAI-compatible and streaming semantics are contract-sensitive. |
| `watchlists.py` | 5889 | High | Defer until scheduler/jobs boundaries are mapped. |
| `paper_search.py` | 5201 | High | Defer because provider integrations and external failures are broad. |
| `embeddings_v5_production_enhanced.py` | 4749 | High | Defer because ML/provider health behavior is sensitive. |
| `workflows.py` | 4651 | Very high | Defer. Workflow execution and adapter coupling are broad. |
| `notes.py` | 4287 | Medium-high | Later candidate after Phase 3 response and auth helpers stabilize. |
| `mcp_hub_management.py` | 3921 | High | Later candidate with MCP-specific auth tests. |
| `auth.py` | 3760 | Very high | Defer until Phase 3.4 auth standardization has landed. |

## Candidate Route Families

### `storage.py`

Static signal:

- 767 lines.
- Router prefix: `/storage`.
- User-owned routes use `User = Depends(get_request_user)`.
- Admin quota routes use `require_storage_admin`.

Route groups:

- file list/detail/download/delete/update
- bulk delete and bulk move
- folders
- usage and usage breakdown
- least-accessed files
- trash list/restore/permanent delete
- admin quotas for user, team, and org

Relevant test signals:

- `tldw_Server_API/tests/Storage/`
- `tldw_Server_API/tests/AuthNZ/unit/test_storage_quota_service_backend_selection.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_storage_admin_claims.py`
- `tldw_Server_API/tests/Admin/test_admin_storage_quotas.py`
- frontend storage quota banner tests

Recommended first Phase 4.4 target:

- `storage.py` user-owned JSON routes only.

Suggested first boundary:

- Keep `/files/{file_id}/download` in the original module for the first slice because it returns a file response.
- Keep `/admin/quotas/...` routes together and out of the first user-owned slice because they have a different auth dependency.
- Extract user-owned file/folder/usage/trash route registration only after an OpenAPI diff baseline exists.

### `slides.py`

Static signal:

- 2399 lines.
- Router prefix: `/slides`.
- Route groups are already visible: deck CRUD, templates, styles, versions, render jobs, generation, export, health.
- Uses permission dependencies, RBAC rate limits, `SlidesDatabase`, `get_request_user`, media DB, Collections DB, and Jobs.

Relevant test signals:

- `tldw_Server_API/tests/Slides/test_slides_api.py`
- `tldw_Server_API/tests/Slides/test_slides_db.py`
- `tldw_Server_API/tests/Slides/test_slides_assets.py`
- `tldw_Server_API/tests/Slides/test_slides_export.py`
- `tldw_Server_API/tests/Slides/test_presentation_render_jobs.py`
- `tldw_Server_API/tests/Slides/test_presentation_rendering.py`
- Presentation Studio frontend tests

Recommendation:

- Good second candidate after Phase 3 helper contracts land.
- Start with templates/styles/versions before render jobs or export.
- Leave export and render job routes until file response, Jobs, and Collections behavior are covered by an OpenAPI diff and focused tests.

### `data_tables.py`

Static signal:

- 1195 lines.
- Router prefix: `/data-tables`.
- Route groups include generate, list, detail, export, update, metadata update, delete, regenerate, job status, and job cancel.
- Uses both `current_user: User = Depends(get_request_user)` and `principal: AuthPrincipal = Depends(get_auth_principal)`.
- Uses permission dependencies, RBAC rate limits, Media DB, Collections DB, and Jobs.

Relevant test signals:

- `tldw_Server_API/tests/DataTables/test_data_tables_api.py`
- `tldw_Server_API/tests/DataTables/test_data_tables_export.py`
- `tldw_Server_API/tests/DataTables/test_data_tables_jobs_integration.py`
- `tldw_Server_API/tests/DataTables/test_data_tables_worker.py`
- `tldw_Server_API/tests/DB_Management/test_data_tables_crud.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_data_table_*`
- frontend DataTables page tests

Recommendation:

- Defer as a first decomposition target.
- It is smaller than the top hotspots, but the route dependencies cross Jobs, Media DB, Collections DB, and mixed auth principal/user behavior.
- Revisit after `storage.py` or `slides.py` proves the route-splitting pattern.

## Recommended First Target

Start with `storage.py` user-owned JSON routes after Phase 2/3 closeout is stable and after a route-splitting plan exists.

Rationale:

- It has the clearest route-family boundaries.
- It is small enough to validate the extraction pattern without touching a massive endpoint.
- It has focused backend and frontend-adjacent test signals.
- Admin quota routes and download routes can be isolated from the first slice.

Draft route-family plan:

- `Docs/superpowers/plans/2026-04-25-phase4-4-storage-endpoint-decomposition-plan.md`

## Required Route-Family Plan Before Code Movement

For the selected endpoint, create a dedicated plan with:

- current route list and response models
- auth dependency map
- rate-limit dependency map
- file/streaming/non-JSON response exemptions
- frontend caller map
- OpenAPI baseline and diff command
- focused backend and frontend test commands
- rollback plan

## Do Not Do Yet

- Do not split `chat.py`, `auth.py`, `persona.py`, `character_chat_sessions.py`, or `workflows.py` first.
- Do not split file download, streaming, webhook, or `204 No Content` routes into generic JSON helpers.
- Do not change response shapes while doing endpoint decomposition.
- Do not change auth dependencies while doing endpoint decomposition unless Phase 3.4 has already standardized the target route family.

## Handoff Checklist

- [ ] Maintainers accept `storage.py` user-owned JSON routes as the first Phase 4.4 target, or choose an alternate.
- [x] Route-family plan is created before runtime edits.
- [x] OpenAPI baseline command is identified before route movement.
- [x] Focused tests are listed before code movement.
- [x] Admin quota and file download routes are explicitly excluded from the first slice or given their own plan.
