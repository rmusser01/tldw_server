# Storage API (Generated Files)

This API manages user-scoped generated files, storage quotas, trash/restore flows, and cleanup helpers.

Code anchors:
- Router: `tldw_Server_API/app/api/v1/endpoints/storage.py` mounts the per-area routers `storage_user_files.py`, `storage_download.py`, `storage_usage.py`, `storage_trash.py`, `storage_user_folders.py` and `storage_admin_quotas.py` (all in the same directory)
- Schemas: `tldw_Server_API/app/api/v1/schemas/storage_schemas.py:180`
- Quota service: `tldw_Server_API/app/services/storage_quota_service.py:71`
- Cleanup worker: `tldw_Server_API/app/services/storage_cleanup_service.py:224`

Base path: `/api/v1/storage`

## Security and Path Handling

Downloads and cleanup operations validate resolved filesystem paths stay inside the correct per-user base directory:
- Download traversal guard: `tldw_Server_API/app/api/v1/endpoints/storage_download.py:62`
- Category-based base dir resolution (outputs vs voices): `tldw_Server_API/app/api/v1/endpoints/storage_helpers.py:77`
- Cleanup safe resolver: `tldw_Server_API/app/services/storage_cleanup_service.py:56`

Voice clones (`file_category == voice_clone`) are resolved under the voices directory; other files use the outputs directory.

## Core File Endpoints

- `GET /files`
  - List generated files for the current user with pagination and filters.
  - Anchor: `tldw_Server_API/app/api/v1/endpoints/storage_user_files.py:62`
- `GET /files/{file_id}`
  - Returns metadata and updates `accessed_at`.
  - Anchor: `tldw_Server_API/app/api/v1/endpoints/storage_user_files.py:184`
- `GET /files/{file_id}/download`
  - Streams the file from disk after ownership and traversal checks.
  - Anchor: `tldw_Server_API/app/api/v1/endpoints/storage_download.py:24`
- `DELETE /files/{file_id}?hard_delete={bool}`
  - Soft delete (default) subtracts usage and moves to trash.
  - Hard delete removes the record; usage subtraction only occurs if the file was not already soft-deleted.
  - Anchors: `tldw_Server_API/app/api/v1/endpoints/storage_user_files.py:207`, `tldw_Server_API/app/services/storage_quota_service.py:986`
- `PATCH /files/{file_id}`
  - Update metadata such as tags, folder tag, or retention fields.
  - Anchor: `tldw_Server_API/app/api/v1/endpoints/storage_user_files.py:238`

## Bulk Operations

- `POST /files/bulk-delete`
  - Deletes each file through the quota service to ensure usage counters are updated correctly.
  - Anchor: `tldw_Server_API/app/api/v1/endpoints/storage_user_files.py:139`
- `POST /files/bulk-move`
  - Bulk folder-tag changes.
  - Anchor: `tldw_Server_API/app/api/v1/endpoints/storage_user_files.py:160`

Bulk delete examples (soft vs hard delete):

```bash
BASE_URL="http://127.0.0.1:8000"
API_KEY="${SINGLE_USER_API_KEY}"

# Soft delete (moves to trash, updates usage once)
curl -sS -X POST "$BASE_URL/api/v1/storage/files/bulk-delete" \
  -H "X-API-KEY: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"file_ids":[101,102,103],"hard_delete":false}' | jq

# Hard delete (immediate removal; safe for already-trashed files)
curl -sS -X POST "$BASE_URL/api/v1/storage/files/bulk-delete" \
  -H "X-API-KEY: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"file_ids":[201,202],"hard_delete":true}' | jq
```

## Cleanup Suggestions

- `GET /files/least-accessed?limit=20`
  - Returns least recently accessed files (oldest first), using `COALESCE(accessed_at, created_at)`.
  - Anchor: `tldw_Server_API/app/api/v1/endpoints/storage_user_files.py:106`
  - Repo implementation: `tldw_Server_API/app/core/AuthNZ/repos/generated_files_repo.py:939`

Least-accessed example:

```bash
curl -sS "$BASE_URL/api/v1/storage/files/least-accessed?limit=20" \
  -H "X-API-KEY: $API_KEY" | jq
```

## Usage and Quotas

- `GET /usage`
  - Returns total usage, category breakdown, and quota warnings.
  - Soft limit is 80% and hard limit is 100%.
  - `quota_mb` is `null` when no storage quota applies to the user (unlimited); `available_mb` and `usage_percentage` are then `null` and no warning is raised. A quota of `0` blocks uploads and reports both limits reached.
  - Anchors:
    - Endpoint logic: `tldw_Server_API/app/api/v1/endpoints/storage_usage.py:26`
    - Warning fields: `tldw_Server_API/app/api/v1/schemas/storage_schemas.py:180`
- `GET /usage/breakdown`
  - Returns folder and category breakdown plus quota numbers.
  - Anchor: `tldw_Server_API/app/api/v1/endpoints/storage_usage.py:92`

Quick usage warning check:

```bash
curl -sS "$BASE_URL/api/v1/storage/usage" \
  -H "X-API-KEY: $API_KEY" | jq '{
    quota_mb,
    quota_used_mb,
    usage_percentage,
    at_soft_limit,
    at_hard_limit,
    warning
  }'
```

Example `GET /usage` response fields that are new/important:

```json
{
  "usage_percentage": 83.4,
  "at_soft_limit": true,
  "at_hard_limit": false,
  "warning": "Approaching storage limit (80%+)"
}
```

Team/org quota checks now surface a soft-limit warning string when appropriate:
- `tldw_Server_API/app/core/AuthNZ/repos/storage_quotas_repo.py:517`

## Trash and Restore

- `GET /trash`
  - List trashed files.
  - Anchor: `tldw_Server_API/app/api/v1/endpoints/storage_trash.py:26`
- `POST /trash/restore/{file_id}`
  - Restores a soft-deleted file and re-adds its usage.
  - Anchor: `tldw_Server_API/app/api/v1/endpoints/storage_trash.py:56`
- `DELETE /trash/{file_id}`
  - Permanently deletes a file already in trash without re-subtracting usage.
  - Anchor: `tldw_Server_API/app/api/v1/endpoints/storage_trash.py:83`

Trash lifecycle examples:

```bash
# 1) List trash
curl -sS "$BASE_URL/api/v1/storage/trash?offset=0&limit=50" \
  -H "X-API-KEY: $API_KEY" | jq

# 2) Restore a trashed file (re-adds usage)
TRASHED_FILE_ID=101
curl -sS -X POST "$BASE_URL/api/v1/storage/trash/restore/$TRASHED_FILE_ID" \
  -H "X-API-KEY: $API_KEY" | jq

# 3) Permanently delete a trashed file (no double subtraction)
curl -sS -X DELETE "$BASE_URL/api/v1/storage/trash/$TRASHED_FILE_ID" \
  -H "X-API-KEY: $API_KEY" | jq
```

## Admin Quota Endpoints

These endpoints require admin privileges. There are two groups.

Per-user quota values are the user's own `limits.storage_quota_mb` UserProfiles value. A quota is enforced only when `USAGE_QUOTAS_ENABLED` is on. See `Docs/Operations/Usage_Quotas.md`.

Under `/api/v1/storage` (`tldw_Server_API/app/api/v1/endpoints/storage_admin_quotas.py`):
- User quota set: `PUT /admin/quotas/user/{user_id}` (`storage_admin_quotas.py:40`)
- Team quota set: `PUT /admin/quotas/team/{team_id}` (`storage_admin_quotas.py:73`)
- Org quota set: `PUT /admin/quotas/org/{org_id}` (`storage_admin_quotas.py:97`)
- Team quota get: `GET /admin/quotas/team/{team_id}` (`storage_admin_quotas.py:121`)
- Org quota get: `GET /admin/quotas/org/{org_id}` (`storage_admin_quotas.py:136`)

Under `/api/v1/admin/storage-quotas` (`tldw_Server_API/app/api/v1/endpoints/admin/admin_storage_quotas.py`):
- `GET /api/v1/admin/storage-quotas/users/{user_id}`: the user's quota and usage (`admin_storage_quotas.py:117`)
- `PUT /api/v1/admin/storage-quotas/users/{user_id}`: set or remove the user's own quota (`admin_storage_quotas.py:138`)
- `GET` / `PUT /api/v1/admin/storage-quotas/orgs/{org_id}` (`admin_storage_quotas.py:172`, `:191`) and `GET /api/v1/admin/storage-quotas/summary` (`:223`)

### User versus team and org request bodies

The user routes take `SetUserQuotaRequest` (`storage_schemas.py:254`); the admin-prefixed user route takes `UpdateUserQuotaRequest` (`admin_storage_quotas.py:79`):

| Field | Rule |
|---|---|
| `quota_mb` | Integer or `null`, `ge=0`. A value sets the user's own quota. `0` blocks uploads. `null` removes the user's own value, so a team or org value applies if one exists, otherwise the user is unlimited. |
| `soft_limit_pct`, `hard_limit_pct` | Only on `SetUserQuotaRequest`; used in the response flags (defaults 80 and 100). |

The team and org routes under `/storage/admin/quotas` take `SetQuotaRequest` (`storage_schemas.py:247`): `quota_mb` is an integer with a minimum of 100, and `null` is not accepted. Those are shared pools, separate from each member's own `limits.storage_quota_mb`. The admin-prefixed org route takes `quota_mb` greater than 0.

```bash
# Set a user's quota to 2 GB, then remove it
curl -sS -X PUT "$BASE_URL/api/v1/admin/storage-quotas/users/42" \
  -H "X-API-KEY: $API_KEY" -H "Content-Type: application/json" -d '{"quota_mb": 2048}'
curl -sS -X PUT "$BASE_URL/api/v1/admin/storage-quotas/users/42" \
  -H "X-API-KEY: $API_KEY" -H "Content-Type: application/json" -d '{"quota_mb": null}'
```

### Nullable quota fields

`quota_mb` and `storage_quota_mb` in storage responses (`/usage`, `/usage/breakdown`, `/users/storage`, the admin quota responses, the profile `quotas` section) are nullable. `null` means unlimited: no quota applies to that user. Example values such as `"storage_quota_mb": 5120` show a quota that was set; they are not a default.

## Background Cleanup Worker

A background cleanup worker can be enabled to purge expired files, old trash, and temp directories:
- Cleanup cycle: `tldw_Server_API/app/services/storage_cleanup_service.py:224`
- Worker start: `tldw_Server_API/app/services/startup_cleanup_workers.py:279`
- Worker stop: `tldw_Server_API/app/services/startup_cleanup_workers.py:345`

Environment variables:
- `STORAGE_CLEANUP_ENABLED` (default: true)
- `STORAGE_CLEANUP_INTERVAL_SEC` (default: 3600)
- `STORAGE_TRASH_RETENTION_DAYS` (default: 30)
- `STORAGE_CLEANUP_BATCH_SIZE` (default: 100)

Note: trash purge uses hard delete directly against the repo (no usage subtraction), which is correct for already-soft-deleted files.

## Related Integrations

- TTS download link headers: `tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py:1288`
- Voice clone registration: `tldw_Server_API/app/core/TTS/voice_manager.py:1078`
- File artifacts export registration: `tldw_Server_API/app/core/File_Artifacts/file_artifacts_service.py:440`

For developer-oriented flow details and integration patterns, see:
- `Docs/Code_Documentation/Guides/Generated_Files_Storage_Code_Guide.md:1`
