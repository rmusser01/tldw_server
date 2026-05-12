# Chatbook API Documentation

## Overview

The Chatbook API provides functionality for exporting and importing collections of content (conversations, notes, characters, etc.) in a portable archive format. It also supports importing OpenWebUI chat export JSON and uploaded OpenWebUI `webui.db` database files through the same import and preview endpoints. This enables users to backup, share, and migrate their data between instances or users.

Developer Code Guide: `Docs/Code_Documentation/Guides/Chatbooks_Code_Guide.md:1`

## Auth + Rate Limits
- Single-user: `X-API-KEY: <key>`
- Multi-user: `Authorization: Bearer <JWT>`
- Standard limits apply; export/import run as background jobs and may be constrained by per-user concurrency.

## Table of Contents

1. [Introduction](#introduction)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Data Models](#data-models)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Code Examples](#code-examples)
8. [Migration Guide](#migration-guide)

## Introduction

### What is a Chatbook?

A Chatbook is a portable archive format (`.chatbook` file) that contains:
- Conversations and chat histories
- Notes and documents
- Character definitions
- World books and lore
- Dictionaries for text replacement
- Generated documents
- Media files and attachments (optional; small binaries bundled per content-type limits, large media referenced via metadata)
- Embeddings (optional)

### Key Features

- **Selective Export**: Choose specific content types and items to include
- **Conflict Resolution**: Multiple strategies for handling duplicate content during import
- **Async Processing**: Large operations can run in the background
- **Job Management**: Track progress of export/import operations
- **Security**: File validation, size limits, and path traversal protection
- **User Isolation**: Content is automatically scoped to authenticated users

### Request Flow (ASCII)

```text
Export (sync)
  Client → POST /api/v1/chatbooks/export (async_mode=false)
         → Validate + Quotas → Service writes manifest+files → ZIP → ExportJob completed
         ← 200 { job_id, download_url }

Export (async)
  Client → POST /api/v1/chatbooks/export (async_mode=true)
         → ExportJob pending → enqueue core Jobs (domain=chatbooks) or Prompt Studio
         ← 200 { job_id }
  Worker → process → write ZIP → update job (download_url, expires_at, status)

Import (sync/async)
  Client → POST /api/v1/chatbooks/import (multipart)
         → Save temp → Branch by source_format
         → Chatbook archive: Validate ZIP → Secure extract → Import selections
         → OpenWebUI JSON: Parse export JSON → Import chat trees
         → OpenWebUI DB: Preview users → Require selected user → Import that user's chat trees and mirror folders
         ← Sync: { success, imported_items, warnings } or Async: { job_id }

OpenWebUI attachment hydration
  Client → POST /api/v1/chatbooks/openwebui/hydration/preview (JSON)
         → Validate owner/admin access → Resolve data root under allowed roots
         → Read imported OpenWebUI metadata + webui.db file rows → Return counts/warnings
  Client → POST /api/v1/chatbooks/openwebui/hydration/jobs (JSON)
         → Enqueue openwebui_attachment_hydration job
  Worker → Copy referenced images, register supported files, optionally process files
```

## Authentication

Supported modes (server decides based on `AUTH_MODE`):
- Single-user mode: `X-API-KEY: <key>`
- Multi-user mode: `Authorization: Bearer <JWT>`

Examples:
```http
# Single-user
X-API-KEY: <your-api-key>

# Multi-user
Authorization: Bearer <your-jwt-token>
```

## API Endpoints

### 1. Create Chatbook Export

**Endpoint**: `POST /api/v1/chatbooks/export`

**Description**: Create a new chatbook export with selected content.

**Request Body**:
```json
{
  "name": "Weekly Backup - January 2024",
  "description": "Complete backup of all content",
  "content_selections": {
    "conversation": ["conv_123", "conv_456"],
    "note": ["note_789"],
    "character": []  // Empty array means all characters
  },
  "author": "John Doe",
  "include_media": true,
  "media_quality": "compressed",
  "include_embeddings": false,
  "include_generated_content": true,
  "tags": ["backup", "weekly"],
  "categories": ["work"],
  "async_mode": false
}
```

**Response (Synchronous)**:
```json
{
  "success": true,
  "message": "Chatbook created successfully",
  "job_id": "0c9d9a3a-6d1c-4c8f-9c84-9a0c2c2d8f77",
  "download_url": "/api/v1/chatbooks/download/0c9d9a3a-6d1c-4c8f-9c84-9a0c2c2d8f77"
}
```

Implementation notes:
- Sync mode persists a completed export job and returns its `job_id` plus a `download_url` that uses this UUID.
- For robust automation, prefer async mode and then poll job status to obtain the canonical `download_url` by `job_id`.
- When evaluation exports exceed row caps, export job metadata can include continuation tokens and the manifest can include `truncation.evaluations.continuations` so clients can resume the same chatbook export.

**Response (Asynchronous)**:
```json
{
  "success": true,
  "message": "Export job started",
  "job_id": "0c9d9a3a-6d1c-4c8f-9c84-9a0c2c2d8f77"
}
```

### 2. Import Chatbook

**Endpoint**: `POST /api/v1/chatbooks/import`

**Description**: Import content from an uploaded chatbook archive, OpenWebUI chat export JSON, or OpenWebUI webui.db database.

**Request**: Multipart form data
- `file` (form field): The import file (required)
- `source_format` (form field): `chatbook` (default), `openwebui_json`, or `openwebui_db`
- `selected_openwebui_user_id` (form field): required when `source_format=openwebui_db`; use a source user id returned by preview
- `conflict_resolution` (form field): `skip` (default) or `rename` for current import flows
- `prefix_imported` (form field): boolean, default `false`
- `content_selections` (form field): optional JSON object encoded as a string; unsupported content types are rejected
- `import_media` and `import_embeddings` (form fields): must remain `false` in v1; true values are rejected

Supported multipart fields:
```json
{
  "source_format": "chatbook",
  "selected_openwebui_user_id": "user_abc123",
  "content_selections": "{\"conversation\":[\"conv_123\"],\"note\":[]}",
  "content_selections_json_shape": {
    "conversation": ["conv_123"],  // Only import specific items
    "note": []  // Import all notes if content selections are supported
  },
  "conflict_resolution": "skip",
  "prefix_imported": false,
  "import_media": false,
  "import_embeddings": false,
  "async_mode": false
}
```

For `source_format=chatbook`, the upload must be a `.zip` or `.chatbook` archive. For `source_format=openwebui_json`, the upload must be a safe `.json` filename and the server does not run ZIP validation or Chatbook manifest extraction. For `source_format=openwebui_db`, the upload must be a safe `.db` or `.sqlite` filename and a valid SQLite database.

OpenWebUI JSON import supports normal OpenWebUI "Export Chats" JSON files. It imports every valid chat as one tldw conversation and preserves all valid message branches through `parent_message_id`.

OpenWebUI webui.db database import supports uploaded SQLite databases copied from an OpenWebUI instance. Preview first, choose exactly one `selected_openwebui_user_id`, then import with `source_format=openwebui_db`. The import reads only chats owned by the selected source user. Source folders are mirrored into tldw folders under `OpenWebUI / <selected user> / ...`, and folder links are attached to imported conversations. Duplicate detection uses `source=openwebui` and a deterministic external reference for both JSON and DB imports. `skip` is the default duplicate behavior; `rename` creates an intentional second copy with a unique imported title and external reference. OpenWebUI v1 does not support `overwrite`, `merge`, admin export import, live OpenWebUI server import, embeddings import, content selections, or unreferenced attachment import. Files, images, and artifacts import as metadata references first; use the OpenWebUI attachment hydration endpoints after import to copy referenced images and register supported files from a server-local OpenWebUI data root.

**Response**:
```json
{
  "source_format": "chatbook",
  "success": true,
  "message": "Chatbook imported successfully",
  "imported_items": {
    "conversation": 5,
    "note": 10,
    "character": 2
  }
}
```

**OpenWebUI JSON Response**:
```json
{
  "source_format": "openwebui_json",
  "success": true,
  "message": "OpenWebUI chats imported successfully",
  "openwebui_result": {
    "imported_chats": 4,
    "skipped_chats": 1,
    "failed_chats": 0,
    "imported_messages": 128,
    "skipped_messages": 0,
    "duplicate_chats": 1,
    "warnings": []
  }
}
```

**OpenWebUI Database Response**:
```json
{
  "source_format": "openwebui_db",
  "success": true,
  "message": "OpenWebUI database chats imported successfully",
  "openwebui_db_result": {
    "imported_chats": 4,
    "skipped_chats": 1,
    "failed_chats": 0,
    "imported_messages": 128,
    "duplicate_chats": 1,
    "mirrored_folders": 3,
    "folder_links": 4,
    "warnings": []
  }
}
```

### 3. Preview Chatbook

**Endpoint**: `POST /api/v1/chatbooks/preview`

**Description**: Preview chatbook archive, OpenWebUI JSON contents, or OpenWebUI database users without importing.

**Request**: Multipart form data
- `file`: The file to preview
- `source_format`: `chatbook` (default), `openwebui_json`, or `openwebui_db`

**Response**:
```json
{
  "source_format": "chatbook",
  "manifest": {
    "version": "1.0.0",
    "name": "My Chatbook",
    "description": "Research collection",
    "author": "Jane Doe",
    "created_at": "2024-01-15T10:30:00Z",
    "total_conversations": 10,
    "total_notes": 25,
    "total_characters": 3,
    "total_size_bytes": 5242880,
    "content_items": []  // Preview omits detailed items for performance
  }
}
```

**OpenWebUI JSON Preview Response**:
```json
{
  "source_format": "openwebui_json",
  "openwebui_preview": {
    "chat_count": 5,
    "message_count": 132,
    "branched_chat_count": 2,
    "duplicate_chat_count": 1,
    "attachment_reference_count": 3,
    "malformed_chat_count": 0,
    "warnings": [],
    "items": [
      {
        "title": "Research thread",
        "external_ref": "chat_abc123",
        "message_count": 31,
        "branched": true,
        "duplicate": false,
        "warning_count": 0
      }
    ]
  }
}
```

**OpenWebUI Database Preview Response**:
```json
{
  "source_format": "openwebui_db",
  "openwebui_db_preview": {
    "user_count": 2,
    "users": [
      {
        "source_user_id": "user_abc123",
        "display_label": "Alice",
        "email": "alice@example.test",
        "chat_count": 12,
        "message_count": 344,
        "folder_count": 5,
        "branched_chat_count": 3,
        "duplicate_chat_count": 1,
        "archived_chat_count": 2,
        "pinned_chat_count": 4,
        "attachment_reference_count": 8,
        "warning_count": 0,
        "warnings": []
      }
    ],
    "warnings": []
  }
}
```

### 4. Preview OpenWebUI Attachment Hydration

**Endpoint**: `POST /api/v1/chatbooks/openwebui/hydration/preview`

**Description**: Preview hydration for files referenced by imported OpenWebUI conversations. This endpoint validates the server-local OpenWebUI data root, scans the selected imported tldw conversations for preserved OpenWebUI attachment metadata, resolves referenced files through `webui.db` and `uploads/`, and returns counts and warnings without copying files.

Hydration v1 is reference-driven. It does not scan every file under `uploads/` and does not connect to a live OpenWebUI server.

**Auth and path requirements**:
- Single-user mode is allowed.
- Multi-user mode requires an owner/admin principal authorized for the imported conversations.
- `openwebui_data_root` must be a server-local path containing `webui.db` and `uploads/`.
- The data root and resolved files must be under `Files.ingestion_source_allowed_roots`, `INGESTION_SOURCE_ALLOWED_ROOTS`, or `TLDW_INGESTION_SOURCE_ALLOWED_ROOTS`.

**Request**: JSON
```json
{
  "openwebui_data_root": "/srv/openwebui/data",
  "scope": {
    "conversation_ids": ["conv_123", "conv_456"],
    "source_user_id": "user_abc123"
  },
  "process_supported_files": false
}
```

Fields:
- `openwebui_data_root`: required server-local OpenWebUI data root.
- `scope.conversation_ids`: imported tldw conversation IDs to scan. Empty means no conversations are selected.
- `scope.source_user_id`: optional OpenWebUI user id used for database `chat_file` fallback lookups.
- `process_supported_files`: optional, defaults to `false`. When `true`, preview includes supported-file processing intent for non-image files.

**Response**:
```json
{
  "scope": {
    "conversation_ids": ["conv_123", "conv_456"],
    "source_user_id": "user_abc123"
  },
  "process_supported_files": false,
  "summary": {
    "referenced_files": 3,
    "returned_items": 3,
    "omitted_items": 0,
    "resolved_files": 2,
    "image_files": 1,
    "media_files": 1,
    "missing_files": 1,
    "unsupported_files": 0,
    "failed_files": 0,
    "hydrated_images": 0,
    "registered_media_files": 0,
    "already_hydrated": 0,
    "processed_files": 0,
    "warning_count": 1
  },
  "items": [
    {
      "conversation_id": "conv_123",
      "message_id": "msg_123",
      "file_id": "file_abc",
      "status": "preview_ready",
      "file_kind": "image",
      "mime_type": "image/png"
    }
  ],
  "warnings": ["1 referenced file was missing"]
}
```

`items` is capped to protect API clients from very large responses. Use `summary.referenced_files`, `summary.returned_items`, and `summary.omitted_items` to detect whether the response was truncated. Common warnings include `missing_file`, `unsupported_file_type`, `file_too_large`, `path_rejected`, and paths outside allowed roots.

### 5. Create OpenWebUI Attachment Hydration Job

**Endpoint**: `POST /api/v1/chatbooks/openwebui/hydration/jobs`

**Description**: Enqueue the same hydration operation as a background job. Use preview first; the WebUI gates job creation on a successful preview for the same payload.

Image files are copied into tldw-owned storage and message metadata is updated. Non-image files are registered in the Media DB when available. File processing remains opt-in through `process_supported_files=true`; with the default `false`, supported non-image files are registered but not processed.

Hydration output storage uses `OPENWEBUI_HYDRATION_MEDIA_STORAGE_PATH` when set, otherwise `MEDIA_STORAGE_PATH`, otherwise the server's default media storage directory.

**Request**: Same JSON shape as preview.

**Response**:
```json
{
  "job_id": "123",
  "job_uuid": "7d0d6a9a-4a48-4d35-bd4e-3e33b6f2dfb1",
  "status": "pending",
  "domain": "chatbooks",
  "queue": "default",
  "job_type": "openwebui_attachment_hydration",
  "owner_user_id": "1",
  "created_at": "2026-05-11T17:00:00Z",
  "updated_at": "2026-05-11T17:00:00Z",
  "result": null,
  "error": null
}
```

### 6. Get OpenWebUI Attachment Hydration Job

**Endpoint**: `GET /api/v1/chatbooks/openwebui/hydration/jobs/{job_id}`

**Description**: Return one OpenWebUI hydration job visible to the current caller. Non-admin users can only read their own jobs.

**Response**:
```json
{
  "job_id": "123",
  "status": "completed",
  "job_type": "openwebui_attachment_hydration",
  "result": {
    "summary": {
      "referenced_files": 3,
      "returned_items": 3,
      "omitted_items": 0,
      "resolved_files": 2,
      "hydrated_images": 1,
      "registered_media_files": 1,
      "processed_files": 0,
      "warning_count": 1
    },
    "warnings": ["1 referenced file was missing"]
  },
  "error": null
}
```

### 7. Download Chatbook

**Endpoint**: `GET /api/v1/chatbooks/download/{job_id}`

**Description**: Download a completed export.

**Response**: Binary file stream (application/zip)

Signed URLs (optional):
- If `CHATBOOKS_SIGNED_URLS=true` and `CHATBOOKS_SIGNING_SECRET` is set, the download link is signed and requires query params `exp` (Unix timestamp) and `token` (HMAC SHA256 of `"{job_id}:{exp}"`).
- If `CHATBOOKS_ENFORCE_EXPIRY=true`, the server enforces job-level `expires_at` and returns `410` when expired.
- Invalid or missing signature returns `403`; an expired `exp` parameter returns `410`.

### 8. List Export Jobs

**Endpoint**: `GET /api/v1/chatbooks/export/jobs`

**Query Parameters**:
- `limit`: Maximum results (1-1000, default: 100)
- `offset`: Skip results (default: 0)

**Response**:
```json
{
  "jobs": [
    {
      "job_id": "0c9d9a3a-6d1c-4c8f-9c84-9a0c2c2d8f77",
      "status": "completed",
      "chatbook_name": "Weekly Backup",
      "created_at": "2024-01-15T10:00:00Z",
      "completed_at": "2024-01-15T10:05:00Z",
      "progress_percentage": 100,
      "total_items": 50,
      "processed_items": 50,
      "file_size_bytes": 10485760,
  "download_url": "/api/v1/chatbooks/download/0c9d9a3a-6d1c-4c8f-9c84-9a0c2c2d8f77",
      "expires_at": "2024-02-14T10:05:00Z"
    }
  ],
  "total": 15
}
```

### 9. Get Export Job Status

**Endpoint**: `GET /api/v1/chatbooks/export/jobs/{job_id}`

**Response**: Same as individual job in list response.

### 10. List Import Jobs

**Endpoint**: `GET /api/v1/chatbooks/import/jobs`

**Query Parameters**:
- `limit`: Maximum results (1-1000, default: 100)
- `offset`: Skip results (default: 0)

**Response**:
```json
{
  "jobs": [
    {
      "job_id": "cb_import_20240115_def456",
      "status": "completed",
      "chatbook_path": "/tmp/uploads/chatbook.zip",
      "created_at": "2024-01-15T11:00:00Z",
      "completed_at": "2024-01-15T11:03:00Z",
      "progress_percentage": 100,
      "total_items": 45,
      "processed_items": 45,
      "successful_items": 43,
      "failed_items": 0,
      "skipped_items": 2,
      "conflicts": [],
      "warnings": []
    }
  ],
  "total": 8
}
```

### 11. Get Import Job Status

**Endpoint**: `GET /api/v1/chatbooks/import/jobs/{job_id}`

**Response**: Same as individual job in list response.

### 12. Cancel Export Job

**Endpoint**: `DELETE /api/v1/chatbooks/export/jobs/{job_id}`

**Response**:
```json
{
  "message": "Export job 0c9d9a3a-6d1c-4c8f-9c84-9a0c2c2d8f77 cancelled"
}
```

### 13. Cancel Import Job

**Endpoint**: `DELETE /api/v1/chatbooks/import/jobs/{job_id}`

**Response**:
```json
{
  "message": "Import job cb_import_20240115_def456 cancelled"
}
```

### 14. Remove Export Job

**Endpoint**: `DELETE /api/v1/chatbooks/export/jobs/{job_id}/remove`

Removes a completed or cancelled export job (and deletes the exported archive).

**Response**:
```json
{
  "success": true,
  "message": "Export job 0c9d9a3a-6d1c-4c8f-9c84-9a0c2c2d8f77 removed",
  "job_id": "0c9d9a3a-6d1c-4c8f-9c84-9a0c2c2d8f77"
}
```

### 15. Remove Import Job

**Endpoint**: `DELETE /api/v1/chatbooks/import/jobs/{job_id}/remove`

Removes a completed or cancelled import job.

**Response**:
```json
{
  "success": true,
  "message": "Import job cb_import_20240115_def456 removed",
  "job_id": "cb_import_20240115_def456"
}
```

### 16. Cleanup Expired Exports

**Endpoint**: `POST /api/v1/chatbooks/cleanup`

**Description**: Remove expired export files to free storage.
Scheduled cleanup runs via the Chatbooks worker; use this endpoint to trigger a manual sweep.

**Response**:
```json
{
  "deleted_count": 5
}
```

### 17. Service Health

Lightweight liveness check for the Chatbooks subsystem.

**Endpoint**: `GET /api/v1/chatbooks/health`

**Response**:
```json
{
  "service": "chatbooks",
  "status": "healthy|degraded|unhealthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "storage_base": {"path": "<USER_DB_BASE_DIR>", "exists": true, "writable": true}
  }
}
```

`USER_DB_BASE_DIR` is defined in `tldw_Server_API.app.core.config` (defaults to `Databases/user_databases/` under the project root). Override via environment variable or `Config_Files/config.txt` as needed.

## Data Models

### ContentType Enum
```
- conversation
- note
- character
- media
- embedding
- prompt
- evaluation
- world_book
- dictionary
- generated_document
```

### ConflictResolution Enum
```
- skip: Skip items that already exist
- overwrite: Replace existing items
- rename: Add with modified name
- merge: Combine with existing (future feature)
```

OpenWebUI JSON and database imports support `skip` and `rename` in v1.

### ExportStatus Enum
```
- pending: Job queued
- in_progress: Currently processing
- completed: Successfully finished
- failed: Error occurred
- cancelled: Manually stopped
- expired: File removed
```

### ImportStatus Enum
```
- pending: Job queued
- validating: Checking file integrity
- in_progress: Importing content
- completed: Successfully finished
- failed: Error occurred
- cancelled: Manually stopped
```

### Manifest Metadata (selected fields)
- `metadata.binary_limits`: Per content-type max bundled bytes applied during export.
- `truncation.evaluations.continuations`: Continuation tokens for resumable evaluation exports.

## Error Handling

### Error Response Format
```json
{
  "detail": "Detailed error message",
  "error_type": "ValidationError",
  "job_id": "0c9d9a3a-6d1c-4c8f-9c84-9a0c2c2d8f77",
  "suggestions": [
    "Check file format",
    "Ensure file size is under 100MB"
  ]
}
```

### Common Error Types
- `ValidationError`: Invalid input parameters
- `AuthenticationError`: Missing or invalid token
- `AuthorizationError`: Insufficient permissions
- `NotFoundError`: Resource doesn't exist
- `ConflictError`: Operation conflicts with current state
- `QuotaExceededError`: User quota limit reached
- `FileTooLargeError`: File exceeds size limit
- `RateLimitError`: Too many requests

### HTTP Status Codes
- `200`: Success
- `202`: Accepted (async operation started)
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `409`: Conflict
- `413`: Payload Too Large
- `429`: Too Many Requests
- `500`: Internal Server Error

## Rate Limiting

Per-user limits enforced at the endpoint level:
- Export: 5/minute
- Import: 5/minute
- Preview: 10/minute
- Download: 20/minute

List/status endpoints may be subject to global API rate limits but have no dedicated per-route limiter. Exceeded limits return HTTP 429.

## Code Examples

### Python Example

```python
import json
import time
import uuid
from urllib.request import Request, urlopen

# Configuration
API_BASE = "http://localhost:8000/api/v1"
TOKEN = "your-jwt-token"
headers = {"Authorization": f"Bearer {TOKEN}"}  # In single-user mode use: {"X-API-KEY": API_KEY}

def request_json(method, url, payload=None, headers=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = Request(url, data=data, headers=hdrs, method=method)
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def download_file(url, headers, dest_path, chunk_size=8192):
    req = Request(url, headers=headers, method="GET")
    with urlopen(req) as resp, open(dest_path, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)

def encode_multipart(fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()

    def add_line(line):
        body.extend(line.encode("utf-8"))
        body.extend(b"\r\n")

    for name, value in fields.items():
        add_line(f"--{boundary}")
        add_line(f'Content-Disposition: form-data; name="{name}"')
        add_line("")
        add_line(str(value))

    for name, filename, content, content_type in files:
        add_line(f"--{boundary}")
        add_line(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"')
        add_line(f"Content-Type: {content_type or 'application/octet-stream'}")
        add_line("")
        body.extend(content)
        body.extend(b"\r\n")

    add_line(f"--{boundary}--")
    return boundary, bytes(body)

# Create a chatbook export (async)
data = request_json(
    "POST",
    f"{API_BASE}/chatbooks/export",
    {
        "name": "Weekly Backup",
        "description": "Complete backup of all content",
        "content_selections": {},  # Empty = export everything
        "include_media": True,
        "async_mode": True,
    },
    headers=headers,
)
job_id = data.get("job_id")
if job_id:
    print(f"Export started: {job_id}")

    # Monitor export job
    while True:
        status_data = request_json(
            "GET",
            f"{API_BASE}/chatbooks/export/jobs/{job_id}",
            headers=headers,
        )

        print(f"Progress: {status_data['progress_percentage']}%")

        if status_data["status"] == "completed":
            # Download the chatbook
            download_file(
                f"{API_BASE}/chatbooks/download/{job_id}",
                headers=headers,
                dest_path="my_backup.chatbook",
            )
            print("Download complete!")
            break
        elif status_data["status"] == "failed":
            print(f"Export failed: {status_data['error_message']}")
            break

        time.sleep(5)  # Check every 5 seconds

# Import a chatbook (options as multipart form fields)
with open("my_backup.chatbook", "rb") as f:
    boundary, body = encode_multipart(
        {
            "source_format": "chatbook",
            "conflict_resolution": "skip",
            "prefix_imported": "false",
            "import_media": "false",
            "import_embeddings": "false",
        },
        [("file", "my_backup.chatbook", f.read(), "application/octet-stream")],
    )
    upload_headers = {
        **headers,
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    req = Request(
        f"{API_BASE}/chatbooks/import",
        data=body,
        headers=upload_headers,
        method="POST",
    )
    with urlopen(req) as resp:
        if resp.status != 200:
            print(f"Import failed with status {resp.status}")
        else:
            result = json.loads(resp.read().decode("utf-8"))
            print(f"Imported: {result['imported_items']}")
```

### JavaScript Example

```javascript
const API_BASE = 'http://localhost:8000/api/v1';
const TOKEN = 'your-jwt-token';

// Create chatbook
async function createChatbook() {
  const response = await fetch(`${API_BASE}/chatbooks/export`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${TOKEN}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      name: 'Weekly Backup',
      description: 'Complete backup',
      content_selections: {},
      include_media: true,
      async_mode: true
    })
  });

  const data = await response.json();
  return data.job_id;
}

// Monitor job progress
async function monitorJob(jobId) {
  while (true) {
    const response = await fetch(
      `${API_BASE}/chatbooks/export/jobs/${jobId}`,
      {
        headers: {
          'Authorization': `Bearer ${TOKEN}`
        }
      }
    );

    const status = await response.json();
    console.log(`Progress: ${status.progress_percentage}%`);

    if (status.status === 'completed') {
      return status.download_url;
    } else if (status.status === 'failed') {
      throw new Error(status.error_message);
    }

    // Wait 5 seconds before checking again
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
}

// Import chatbook (options as multipart form fields)
async function importChatbook(file) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('source_format', 'chatbook');
  formData.append('conflict_resolution', 'skip');
  formData.append('prefix_imported', 'false');
  formData.append('import_media', 'false');
  formData.append('import_embeddings', 'false');

  const response = await fetch(`${API_BASE}/chatbooks/import`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${TOKEN}`
    },
    body: formData
  });

  return await response.json();
}

// Usage
createChatbook()
  .then(jobId => monitorJob(jobId))
  .then(downloadUrl => {
    console.log(`Download at: ${downloadUrl}`);
  })
  .catch(error => {
    console.error('Export failed:', error);
  });
```

## Migration Guide

### Migrating from Single-User to Multi-User

When upgrading from the single-user version to multi-user with authentication:

1. **Export your data** before migration:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/chatbooks/export" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <your-jwt-token>" \
     -d '{"name": "Pre-migration backup", "description": "Complete backup before auth migration", "content_selections": {}, "include_media": true}'
   ```

2. **Enable authentication** in the new version

3. **Import your data** with your new user token:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/chatbooks/import" \
     -H "Authorization: Bearer <new-token>" \
     -F "file=@backup.chatbook"
   ```

### Best Practices

1. **Regular Backups**: Schedule weekly or monthly exports
2. **Selective Exports**: Export only active projects to reduce file size
3. **Conflict Strategy**: Use "skip" for regular backups, "rename" for shared imports
4. **Media Management**: Exclude media for text-only backups to save space
5. **Job Monitoring**: Always check job status for async operations
6. **Error Recovery**: Keep original files until import is confirmed successful

## Quota Limits

| User Tier | Daily Exports | Daily Imports | Max File Size | Retention |
|-----------|--------------|---------------|---------------|-----------|
| Free | 5 | 5 | 100 MB | 7 days |
| Basic | 20 | 20 | 500 MB | 30 days |
| Premium | 100 | 100 | 2 GB | 90 days |
| Enterprise | Unlimited | Unlimited | 10 GB | 365 days |

## Security Considerations

1. **File Validation**: All uploads are validated for format and content
2. **Path Traversal**: File paths are sanitized to prevent directory traversal
3. **Size Limits**: Enforced to prevent resource exhaustion
4. **User Isolation**: Content is strictly isolated between users
5. **Temporary Files**: Cleaned up automatically after processing
6. **Encryption**: Consider encrypting chatbooks before sharing

---

*Last updated: 2025-12-29*
*API Version: 1.0.0*
## Configuration

Environment variables controlling Chatbooks job backends and downloads:

- `CHATBOOKS_JOBS_BACKEND`: Core-only; overrides are ignored (kept for compatibility).
- `CHATBOOKS_CORE_WORKER_ENABLED`: `true|false` to start the Chatbooks worker.
- Deprecated: `TLDW_USE_PROMPT_STUDIO_QUEUE` (ignored; legacy backend removed).

Signed downloads:
- `CHATBOOKS_SIGNED_URLS=true|false` - enable HMAC-signed download URLs.
- `CHATBOOKS_SIGNING_SECRET` - shared secret for signing.
- `CHATBOOKS_ENFORCE_EXPIRY=true|false` - enforce job `expires_at`.
- `CHATBOOKS_URL_TTL_SECONDS` - default expiry TTL for generated links.

Retention and cleanup:
- `CHATBOOKS_EXPORT_RETENTION_DEFAULT_HOURS` - retention window for completed exports before expiry (default `24`).
- `CHATBOOKS_CLEANUP_INTERVAL_SEC` - scheduled cleanup cadence in seconds (set to `0` to disable scheduling).

Export limits:
- `CHATBOOKS_EVAL_EXPORT_MAX_ROWS` - max rows exported per evaluation run (default `200`).
- `CHATBOOKS_BINARY_LIMITS_MB` - JSON map of content type to max bundled size in MB (for example, `{"media": 0, "conversations": 10, "generated_docs": 25}`).

Template/import controls:
- `CHATBOOKS_TEMPLATE_MODE` - default manifest template mode for exports (`pass_through|render_on_export`, default `pass_through`).
- `CHATBOOKS_TEMPLATE_DEFAULTS_JSON` - optional JSON object merged into template defaults.
- `CHATBOOKS_TEMPLATE_TIMEZONE` - default timezone used for template rendering (default `UTC`).
- `CHATBOOKS_TEMPLATE_LOCALE` - optional default locale for template rendering.
- `CHATBOOKS_IMPORT_DICT_STRICT` - when `true`, Chatbooks import skips embedded dictionaries that contain fatal validation errors and continues best-effort for the rest.
