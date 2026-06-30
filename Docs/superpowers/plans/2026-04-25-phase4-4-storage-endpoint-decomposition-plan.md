# Phase 4.4 Storage Endpoint Decomposition Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. This is a Phase 4 plan. Do not implement it until Phase 2/3 closeout is stable and maintainers accept `storage.py` user-owned JSON routes as the first endpoint decomposition target.

**Goal:** Decompose `tldw_Server_API/app/api/v1/endpoints/storage.py` without changing public route paths, response bodies, auth behavior, file download behavior, or admin quota behavior.

**Architecture:** Keep `/api/v1/storage` as the public prefix. Preserve the existing `storage.router` import used by `main.py`. Extract pure helpers first, then user-owned JSON route groups. Keep file-download and admin quota routes out of the first route movement.

**Tech Stack:** FastAPI, OpenAPI, pytest, Bandit

---

## Current Route Map

Public router:

- `router = APIRouter(prefix="/storage", tags=["storage"])`
- included from `tldw_Server_API/app/main.py`

User-owned JSON routes:

- `GET /files`
- `GET /files/{file_id}`
- `DELETE /files/{file_id}`
- `PATCH /files/{file_id}`
- `POST /files/bulk-delete`
- `POST /files/bulk-move`
- `GET /folders`
- `POST /folders`
- `GET /files/least-accessed`
- `GET /usage`
- `GET /usage/breakdown`
- `GET /trash`
- `POST /trash/restore/{file_id}`
- `DELETE /trash/{file_id}`

Non-JSON or special routes:

- `GET /files/{file_id}/download` returns `FileResponse`.

Admin quota routes:

- `PUT /admin/quotas/user/{user_id}`
- `PUT /admin/quotas/team/{team_id}`
- `PUT /admin/quotas/org/{org_id}`
- `GET /admin/quotas/team/{team_id}`
- `GET /admin/quotas/org/{org_id}`

## Dependency Map

User-owned routes:

- `User = Depends(get_request_user)`
- `_get_service()`
- generated files repo from the storage quota service

Admin routes:

- `AuthPrincipal = Depends(require_storage_admin)`
- `require_storage_admin` preserves role, permission, and legacy `is_admin` compatibility.

Special behavior:

- Download route resolves a filesystem path under the user's outputs or voices directory.
- Download route must preserve path traversal rejection and `FileResponse`.
- Usage and restore routes update storage quota counters.

## Stage 1: Baseline And OpenAPI Snapshot

**Goal:** Record current behavior before moving routes.
**Success Criteria:** Focused storage tests pass on the accepted base, and OpenAPI path set is captured.
**Tests:** Focused storage endpoint/admin tests and OpenAPI path guard.
**Status:** Complete

- [x] Create a clean worktree from the accepted base.
- [x] Confirm no active dirty work exists in `storage.py`.
- [x] Run focused backend tests:

```bash
source .venv/bin/activate
python3 -m pytest \
  tldw_Server_API/tests/Storage/test_storage_endpoints.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_storage_admin_claims.py \
  tldw_Server_API/tests/Admin/test_admin_storage_quotas.py \
  -v
```

- [x] Run OpenAPI guard from the UI package if dependencies are installed:

```bash
cd apps/extension
bun run verify:openapi
```

- [x] Record whether `/api/v1/storage/*` paths are currently part of any client guard.
- [x] Do not edit runtime code in this stage.

Notes:

- Focused storage/admin baseline passed after #1220 was merged into `dev`: `43 passed, 6 warnings`.
- `bun run verify:openapi` passed from `apps/extension`, verifying 256 client paths and 46 media fallback fields with the reviewed OSS exceptions unchanged.

## Stage 2: Extract Storage Endpoint Helpers

**Goal:** Move pure helpers out of `storage.py` before moving any routes.
**Success Criteria:** `storage.py` still exposes the same router and helper behavior; focused tests pass.
**Tests:** Focused storage endpoint/admin tests.
**Status:** Complete

Candidate helper module:

- `tldw_Server_API/app/api/v1/endpoints/storage_helpers.py`

Candidate helpers:

- `_principal_is_storage_admin`
- `_to_generated_file`
- `_resolve_storage_base_dir`
- `_parse_datetime`
- `_to_quota_status`

Implementation constraints:

- Preserve `require_storage_admin` behavior and status codes.
- Keep compatibility imports in `storage.py` if tests patch helper names directly.
- Do not move routes in this stage.
- Do not alter datetime parsing fallback behavior.

Notes:

- Extracted `_principal_is_storage_admin`, `_to_generated_file`, `_resolve_storage_base_dir`, `_parse_datetime`, and `_to_quota_status` to `storage_helpers.py`.
- Added direct helper coverage in `test_storage_helpers.py` for admin claim compatibility, datetime parsing, generated-file conversion, base directory selection, and quota conversion.
- Preserved the `storage.DatabasePaths` monkeypatch seam for existing download tests while keeping route behavior unchanged.
- Focused helper/storage/admin suite passed: `53 passed, 6 warnings`.

## Stage 3: Extract User-Owned JSON File And Folder Routes

**Goal:** Move only user-owned JSON routes into sidecar route modules while keeping public paths unchanged.
**Success Criteria:** `storage.router` still registers the same paths, methods, response models, and dependency behavior.
**Tests:** Focused storage endpoint tests and OpenAPI path guard.
**Status:** Complete

Candidate modules:

- `tldw_Server_API/app/api/v1/endpoints/storage_user_files.py`
- `tldw_Server_API/app/api/v1/endpoints/storage_user_folders.py`

Candidate moved routes:

- `GET /files`
- `GET /files/{file_id}`
- `DELETE /files/{file_id}`
- `PATCH /files/{file_id}`
- `POST /files/bulk-delete`
- `POST /files/bulk-move`
- `GET /folders`
- `POST /folders`
- `GET /files/least-accessed`

Implementation constraints:

- Preserve route registration order. In particular, verify `GET /files/least-accessed` still resolves as intended relative to `GET /files/{file_id}`.
- Keep `GET /files/{file_id}/download` in `storage.py` for this stage.
- Keep admin quota routes in `storage.py` for this stage.
- Preserve ownership checks and status codes.
- Do not change response shapes or introduce Phase 3 envelopes.

Notes:

- Moved user-owned file routes to `storage_user_files.py` and folder routes to `storage_user_folders.py`.
- Preserved the existing `storage._get_service` monkeypatch seam through a dynamic sidecar resolver and re-exported moved handler functions from `storage.py` for direct-test compatibility.
- Added route-level regression coverage proving `GET /api/v1/storage/files/least-accessed` is no longer captured by `GET /files/{file_id}`.
- Added canonical pagination metadata to the least-accessed response to satisfy the existing generated list response schema.
- Focused storage/admin suite passed after route movement: `44 passed, 6 warnings`.
- OpenAPI verifier passed from `apps/extension`.
- Touched-scope Bandit passed with zero findings.

## Stage 4: Extract Usage And Trash JSON Routes

**Goal:** Move usage and trash route groups after file/folder route extraction is stable.
**Success Criteria:** Usage counters, restore behavior, and trash response models remain unchanged.
**Tests:** Focused storage endpoint tests.
**Status:** Complete

Candidate modules:

- `tldw_Server_API/app/api/v1/endpoints/storage_usage.py`
- `tldw_Server_API/app/api/v1/endpoints/storage_trash.py`

Candidate moved routes:

- `GET /usage`
- `GET /usage/breakdown`
- `GET /trash`
- `POST /trash/restore/{file_id}`
- `DELETE /trash/{file_id}`

Implementation constraints:

- Preserve quota warning calculation.
- Preserve restore quota counter updates for user, org, and team usage.
- Preserve trash permanent-delete status behavior.

Notes:

- Moved usage routes to `storage_usage.py` and trash routes to `storage_trash.py`.
- Added route-level characterization coverage for `/usage`, `/usage/breakdown`, `/trash`, `/trash/restore/{file_id}`, and `/trash/{file_id}` before extraction.
- Preserved the existing `storage._get_service` monkeypatch seam through dynamic sidecar resolvers and re-exported moved handler functions from `storage.py`.
- Focused storage/admin suite passed after route movement: `49 passed, 6 warnings`.
- OpenAPI verifier passed from `apps/extension`.
- Touched-scope Bandit passed with zero findings.

## Stage 5: Plan Download And Admin Splits Separately

**Goal:** Avoid mixing special route behavior into the first decomposition PR.
**Success Criteria:** Download and admin route movement have their own accepted plan before any move.
**Tests:** Download path traversal tests and admin claim tests.
**Status:** Complete

Download route requirements:

- Preserve `FileResponse`.
- Preserve base directory selection for voice clone files.
- Preserve path traversal rejection.
- Preserve `accessed_at` update.

Admin quota route requirements:

- Preserve `require_storage_admin`.
- Preserve legacy `is_admin` compatibility.
- Preserve 401 behavior when principal dependency is unavailable in tests.
- Preserve 403 detail for non-admin principals.

Notes:

- Admin quota routes were split into `storage_admin_quotas.py` as a conservative JSON-only tranche.
- Admin quota tests now patch the sidecar service dependency directly, avoiding a circular import back into `storage.py`.
- Re-exported admin quota handlers and `require_storage_admin` from `storage.py` for direct import compatibility.
- Added direct compatibility coverage for storage/admin sidecar re-exports.
- Focused storage/admin suite passed after admin route movement: `57 passed, 6 warnings`.
- Download route was split into `storage_download.py` as a separate `FileResponse` tranche.
- Download tests now patch the sidecar service dependency directly, avoiding a circular import back into `storage.py`.
- Re-exported `download_file` from `storage.py` for direct import compatibility.
- Added direct compatibility coverage for the storage/download sidecar re-export.
- Focused storage/admin suite passed after download route movement: `58 passed, 6 warnings`.
- OpenAPI verifier passed from `apps/extension`.
- Touched-scope Bandit passed with zero findings.

## Verification

Focused backend tests:

```bash
source .venv/bin/activate
python3 -m pytest \
  tldw_Server_API/tests/Storage/test_storage_endpoints.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_storage_admin_claims.py \
  tldw_Server_API/tests/Admin/test_admin_storage_quotas.py \
  -v
```

OpenAPI guard:

```bash
cd apps/extension
bun run verify:openapi
```

Touched-scope Bandit:

```bash
source .venv/bin/activate
python3 -m bandit -r tldw_Server_API/app/api/v1/endpoints/storage.py tldw_Server_API/app/api/v1/endpoints/storage_admin_quotas.py tldw_Server_API/app/api/v1/endpoints/storage_download.py tldw_Server_API/app/api/v1/endpoints/storage_helpers.py tldw_Server_API/app/api/v1/endpoints/storage_user_files.py tldw_Server_API/app/api/v1/endpoints/storage_user_folders.py tldw_Server_API/app/api/v1/endpoints/storage_usage.py tldw_Server_API/app/api/v1/endpoints/storage_trash.py -f json -o /tmp/bandit_phase4_4_storage_endpoint.json
```

## Out Of Scope

- Changing any `/api/v1/storage` route path.
- Changing auth dependencies.
- Applying Phase 3 standard response envelopes.
- Applying Phase 3 pagination helpers.
- Moving `GET /files/{file_id}/download` in the first route split.
- Moving admin quota routes in the first route split.
- Raising coverage thresholds.

## Handoff Checklist

- [x] Maintainers accept `storage.py` user-owned JSON routes as the first Phase 4.4 endpoint target.
- [x] Clean worktree from accepted base exists.
- [x] Stage 1 focused tests pass before route movement.
- [x] OpenAPI path set is captured before route movement.
- [x] File download and admin quota routes remain excluded from the first route split.
- [x] Admin quota routes are extracted in their own conservative tranche.
- [x] File download route is extracted in its own conservative tranche.
- [x] Bandit is run on touched source before PR handoff.
