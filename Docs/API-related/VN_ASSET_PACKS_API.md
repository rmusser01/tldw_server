# VN Asset Packs API

VN Asset Packs provide offline visual-novel asset planning, image generation, upload, review, and approved-only manifest export. V1 is a workbench for creating and reviewing assets; it does not include VN playback.

Canonical base path: `/api/v1/vn/vn-assets`

The previous top-level `/api/v1/vn-assets` path is not part of the VN platform API contract.

## Authentication And Ownership

All endpoints require the normal tldw_server API authentication:

- Single-user mode: `X-API-KEY: <key>`
- Multi-user mode: `Authorization: Bearer <jwt>`

Pack metadata is owned by the authenticated user and stored in that user's `ChaChaNotes.db`, beside character cards and world-book data. Image bytes are stored through AuthNZ generated-file storage with `source_feature=vn_assets`; API responses expose `generated_file_id` and VN item metadata, not raw storage paths.

## Limits

- A pack may contain up to 300 item records.
- A slot may request up to 6 variants.
- V1 supports one `primary_character_id` per pack.
- Generated variants start as `draft` and must be approved before they appear in the runtime manifest.

## Idempotency

Side-effecting VN asset commands that create work or delete/upload files support durable idempotency. A repeated request with the same key and same payload returns the original response. Reusing the same key with a different payload returns `409` with a stable VN error detail object whose `code` is `idempotency_key_conflict`.

Use these keys:

- JSON `idempotency_key`: `/packs/{pack_id}/generate`, `/packs/{pack_id}/slots/{slot_id}/retry`, `/packs/{pack_id}/items/{item_id}/regenerate`, `/packs/{pack_id}/cleanup` when `dry_run=false`.
- Multipart form `idempotency_key`: `/packs/{pack_id}/items/upload`.
- JSON `request_id`: `/packs/{pack_id}/export` and `/import/commit`.
- Multipart form `request_id`: `/import/previews`.

Dry-run cleanup requests are intentionally not persisted as idempotent mutations.

## Endpoint Summary

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/starter-matrices` | List built-in matrix templates. |
| `POST` | `/packs` | Create a pack. |
| `GET` | `/packs` | List the current user's packs. |
| `GET` | `/packs/{pack_id}` | Get one pack. |
| `PATCH` | `/packs/{pack_id}` | Update pack metadata. |
| `DELETE` | `/packs/{pack_id}` | Soft-delete pack metadata only. |
| `POST` | `/packs/{pack_id}/cleanup` | Preview or execute generated-file cleanup. |
| `POST` | `/packs/{pack_id}/export` | Start a pack export job. |
| `GET` | `/portability/exports/{job_id}` | Read export status. |
| `GET` | `/portability/exports/{job_id}/download` | Download a completed export archive. |
| `POST` | `/portability/exports/{job_id}/cancel` | Cancel an export job. |
| `POST` | `/import/previews` | Upload an import archive and start preview validation. |
| `GET` | `/import/previews/{preview_id}` | Read import preview status and proposed plan. |
| `POST` | `/import/previews/{preview_id}/cancel` | Cancel import preview validation. |
| `DELETE` | `/import/previews/{preview_id}` | Delete a cancellable import preview and staged archive. |
| `POST` | `/import/commit` | Start an import commit job from a completed preview. |
| `GET` | `/portability/imports/{job_id}` | Read import commit status. |
| `POST` | `/portability/imports/{job_id}/cancel` | Cancel an import commit job. |
| `POST` | `/packs/{pack_id}/matrix/apply` | Expand a starter matrix into slots. |
| `GET` | `/packs/{pack_id}/slots` | List slots. |
| `POST` | `/packs/{pack_id}/slots` | Create a custom slot. |
| `PATCH` | `/packs/{pack_id}/slots/{slot_id}` | Update a slot. |
| `DELETE` | `/packs/{pack_id}/slots/{slot_id}` | Delete a slot, if it has no dependents. |
| `GET` | `/packs/{pack_id}/items` | List generated or uploaded item variants. |
| `PATCH` | `/packs/{pack_id}/items/{item_id}/review` | Set item review status and optional preferred flag. |
| `GET` | `/packs/{pack_id}/items/{item_id}/content` | Stream item image content. |
| `GET` | `/packs/{pack_id}/items/{item_id}/preview` | Stream item image preview content. |
| `POST` | `/packs/{pack_id}/items/bulk-review` | Apply one review status to many items. |
| `POST` | `/packs/{pack_id}/items/upload` | Upload an image for a slot. |
| `POST` | `/packs/{pack_id}/items/{item_id}/preferred` | Mark one item preferred for its slot. |
| `GET` | `/packs/{pack_id}/manifest` | Export approved-only runtime manifest. |
| `GET` | `/packs/{pack_id}/readiness` | Check runtime readiness. |
| `POST` | `/packs/{pack_id}/prompt-preview` | Preview assembled prompt text and truncation diagnostics. |
| `POST` | `/packs/{pack_id}/generate` | Enqueue a parent generation batch job. |
| `GET` | `/packs/{pack_id}/generation` | Read latest generation batch status. |
| `POST` | `/packs/{pack_id}/generation/cancel` | Request cancellation for active generation. |
| `POST` | `/packs/{pack_id}/slots/{slot_id}/retry` | Retry generation for one slot. |
| `POST` | `/packs/{pack_id}/items/{item_id}/regenerate` | Regenerate one item variant. |

## Pack Setup

Create a pack:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn/vn-assets/packs" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Orbital Library",
    "primary_character_id": 42,
    "content_rating": "general",
    "scenario_notes": "Quiet after-hours archive scene.",
    "style_prompt": "cinematic VN sprites, clean linework",
    "negative_prompt": "low quality, distorted hands",
    "default_backend": "stable_diffusion_cpp",
    "default_dimensions": { "width": 768, "height": 1024, "format": "png" },
    "apply_starter_matrix": false
  }'
```

Minimal response:

```json
{
  "id": 1,
  "owner_user_id": 1,
  "title": "Orbital Library",
  "primary_character_id": 42,
  "status": "draft",
  "content_rating": "general",
  "source_world_book_ids": [],
  "planned_output_count": 0,
  "version": 1,
  "deleted": false
}
```

Apply the built-in starter matrix:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn/vn-assets/packs/1/matrix/apply" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "matrix_key": "starter",
    "overrides": { "variant_count": 3 }
  }'
```

Each returned slot contains its `asset_type`, `slot_key`, labels, prompt templates, dimensions, backend/model overrides, review requirement, runtime requirement, and derived status.

## Prompt Preview

Use prompt preview before generation to inspect the assembled prompt without starting a job:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn/vn-assets/packs/1/prompt-preview" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "slot_id": 10,
    "variant_index": 0,
    "budgets": { "character": 1200, "world_book": 800 }
  }'
```

Response:

```json
{
  "prompt": "character and scene prompt text",
  "negative_prompt": "low quality",
  "omitted_source_counts": { "world_book": 2 },
  "token_estimates": { "total": 512 },
  "warnings": ["world_book_truncated"]
}
```

Do not log full prompt previews in production logs; world-book and scenario content may contain private user data.

## Generation Lifecycle

Start generation:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn/vn-assets/packs/1/generate" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "slot_ids": [10, 11],
    "variant_count": 2,
    "idempotency_key": "generate-pack-1",
    "options": { "priority": "normal" }
  }'
```

Generation creates one parent fanout job (`vn_asset_enqueue_batch`) in the `vn_assets` domain. The parent job creates idempotent variant jobs (`vn_asset_generate_variant`) gradually, so API callers do not enqueue hundreds of child jobs directly.

Status response:

```json
{
  "batch_id": 7,
  "job_batch_id": "vn_assets:user:1:pack:1:batch:7",
  "status": "queued",
  "total_slots": 2,
  "total_variants": 4,
  "planned_count": 4,
  "enqueued_count": 0,
  "completed_count": 0,
  "failed_count": 0,
  "cancelled_count": 0,
  "enqueue_error": null
}
```

Poll `GET /packs/{pack_id}/generation` for status. Use `POST /packs/{pack_id}/generation/cancel` to request cancellation. Use the slot retry or item regenerate endpoints for targeted retries after failures.

## Review And Manifest

Generated and uploaded items start as `draft`. Valid review statuses are `draft`, `approved`, `rejected`, and `hidden`.

Approve one item and make it preferred:

```bash
curl -X PATCH "http://127.0.0.1:8000/api/v1/vn/vn-assets/packs/1/items/101/review" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "review_status": "approved", "preferred": true }'
```

Bulk review:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn/vn-assets/packs/1/items/bulk-review" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "item_ids": [101, 102], "review_status": "approved" }'
```

Runtime readiness:

```json
{
  "ready": true,
  "status": "ready_with_warnings",
  "warnings": ["depth_unavailable"],
  "errors": []
}
```

Manifest export includes approved items only:

```json
{
  "schema_version": "vn_asset_manifest.v1",
  "pack_id": 1,
  "title": "Orbital Library",
  "primary_character_id": 42,
  "content_rating": "general",
  "assets": {
    "sprite": [
      {
        "slot_key": "sprite_neutral",
        "item_id": 101,
        "generated_file_id": 9001,
        "preferred": true
      }
    ]
  }
}
```

## Uploads And Content

Upload a manually prepared image into an existing slot:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn/vn-assets/packs/1/items/upload" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -F "slot_id=10" \
  -F "variant_index=0" \
  -F "idempotency_key=upload-slot-10-v0" \
  -F "file=@sprite_neutral.png;type=image/png"
```

The upload endpoint stores the bytes through generated-file storage and returns a draft item. Use `GET /packs/{pack_id}/items/{item_id}/content` or `GET /packs/{pack_id}/items/{item_id}/preview` to stream image bytes.

Both streaming endpoints validate before serving bytes:

- the pack and item belong to the authenticated user;
- the generated-file row belongs to the same user;
- `source_feature` is `vn_assets`;
- `source_ref` is `vn_asset_item:{item_id}`;
- the generated file is not deleted;
- the stored media type is `image/png`, `image/jpeg`, or `image/webp`;
- optional `file_category`, when present, is `image`;
- policy metadata does not block the item.

Provenance, ownership, missing-file, path, and media-type failures return 404 to avoid leaking asset existence. Policy-blocked access returns 403 with stable VN error `code: "policy_blocked"`.

## Cleanup Safety

`DELETE /packs/{pack_id}` only soft-deletes metadata. It does not physically delete generated image files.

Use `POST /packs/{pack_id}/cleanup` for generated-file cleanup. The default request is a dry-run against rejected and hidden items:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/vn/vn-assets/packs/1/cleanup" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "dry_run": true }'
```

Response:

```json
{
  "dry_run": true,
  "removed_item_ids": [],
  "removed_file_count": 0,
  "files_would_delete": 3,
  "files_deleted": 0,
  "skipped_file_ids": [],
  "blocked_count": 0,
  "cleanup_blocked": [],
  "reclaimed_bytes": 1536000
}
```

To execute cleanup for rejected and hidden items:

```json
{
  "dry_run": false,
  "statuses": ["rejected", "hidden"],
  "idempotency_key": "cleanup-pack-1"
}
```

Approved-item cleanup requires both `include_approved: true` and explicit confirmation. The confirmation text/token is:

```text
DELETE APPROVED VN ASSETS
```

Example approved cleanup request:

```json
{
  "dry_run": false,
  "statuses": ["approved"],
  "include_approved": true,
  "confirmation_text": "DELETE APPROVED VN ASSETS"
}
```

Cleanup skips any generated file still referenced by another item. It also consults a pluggable cleanup blocker provider before deleting files; blocked files are added to `skipped_file_ids`, counted in `blocked_count`, and reported in `cleanup_blocked` with `item_id`, `file_id`, and blocker details.

## Worker And Backend Configuration

Enable workers with environment flags:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VN_ASSET_JOBS_WORKER_ENABLED` | disabled | Starts the parent fanout worker. |
| `VN_ASSET_GENERATION_JOBS_WORKER_ENABLED` | disabled | Starts the variant generation worker. |
| `VN_ASSET_JOBS_QUEUE` | `default` | Queue for parent fanout jobs. |
| `VN_ASSET_GENERATION_JOBS_QUEUE` | `generation` | Queue for variant generation jobs. |
| `VN_ASSET_JOBS_WORKER_ID` | process-derived | Parent worker ID override. |
| `VN_ASSET_GENERATION_JOBS_WORKER_ID` | process-derived | Generation worker ID override. |
| `VN_ASSET_JOBS_LEASE_SECONDS` | `JOBS_LEASE_SECONDS` or `120` | Job lease duration. |
| `VN_ASSET_JOBS_RENEW_THRESHOLD_SECONDS` | `10` | Lease renewal threshold. |
| `VN_ASSET_JOBS_RENEW_JITTER_SECONDS` | `0` | Lease renewal jitter. |

Image generation uses the shared image-generation adapter registry. `default_backend` and `default_model` on the pack can be overridden per slot.

Backend concurrency is process-local and defaults to:

- Local/GPU image backends: `VN_ASSETS_LOCAL_BACKEND_CONCURRENCY=1`
- Remote image backends: `VN_ASSETS_REMOTE_BACKEND_CONCURRENCY=4`
- Per-backend override: `VN_ASSETS_BACKEND_CONCURRENCY_<BACKEND_NAME>`

For example, `VN_ASSETS_BACKEND_CONCURRENCY_STABLE_DIFFUSION_CPP=1` keeps local `stable_diffusion_cpp` generation serialized to reduce VRAM exhaustion risk.

## Known V1 Limitations

- No VN playback UI or choose-your-own-adventure runtime is included.
- Each pack has one primary character.
- Depth companions are experimental and not generated for every background by default.
- Sprite cutout/background removal is deferred.
- Realtime image generation is out of scope for V1; generation is Jobs-backed and offline-first.
