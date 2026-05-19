# VN Asset Packs Design

Status: Draft
Date: 2026-04-24
Owner: Core/WebUI maintainers
Scope: Offline image asset generation for future choose-your-own-adventure / visual-novel style play

## Summary

Add a standalone VN Asset Packs module that generates, stores, reviews, and serves reusable visual assets for character/persona-driven visual-novel workflows. The first version focuses on offline asset generation, not VN playback. Packs are anchored to one primary existing character card and optional world books, use the existing image-generation adapter layer, run through Jobs-backed batch processing, and expose only approved assets through a future-runtime-friendly manifest.

This design intentionally keeps asset packs independent from the draft Story Engine. Story/VN runtime work can consume approved packs later without becoming a prerequisite for generation.

## Goals

- Generate full VN-style asset packs of roughly 100-300 outputs per character/location set.
- Cover sprites, backgrounds, background depth companions, and CG/event images.
- Reuse existing image-generation backends where practical.
- Keep generated assets separate from character records and character `extensions`.
- Require human review before generated assets are considered runtime-ready.
- Store enough metadata for reproducibility, regeneration, filtering, and future playback.
- Provide a thin WebUI workbench for setup, generation monitoring, and review.

## Non-Goals

- No VN/CYOA playback UI in this first project.
- No dependency on Story Engine internals.
- No first-class ComfyUI workflow integration in the initial implementation.
- No guarantee that all image backends can produce consistent sprites or usable cutouts.
- No hosted-platform content policy enforcement by default for self-hosted deployments.
- No multi-primary-character pack generation in the initial implementation.

## Existing Project Context

tldw_server already has useful foundations:

- Character cards, character chat sessions, world books, import/export, and prompt presets.
- File artifact image generation through `/api/v1/files/create` and `FileArtifactsService`.
- Image-generation adapters for stable-diffusion.cpp and remote providers.
- Core Jobs infrastructure with leases, retries, cancellation, and per-domain queues.
- Generated-file storage and quota tracking through AuthNZ storage services.
- Next.js/WebUI routes and existing Characters/World Books management flows.

The design should reuse these pieces, but not assume their current public endpoints are sufficient for VN playback.

## Key Design Corrections

### File Artifacts Are Not Durable Playback URLs

Existing file-artifact exports are temporary and may be one-shot. `/api/v1/files/{file_id}/export` marks an export consumed and deletes the temp file after download. A VN runtime cannot rely on those URLs.

VN asset items may reference `file_artifact_id` for structured generation metadata, but runtime serving must use durable generated-file references or VN-specific content endpoints.

### Workers Should Not Call Internal HTTP

Generation workers should call core services/adapters directly rather than posting to `/api/v1/files/create`. This avoids unnecessary auth, routing, HTTP timeout, and retry complexity inside the server process.

### Depth Maps Are Depth Companions

Prompting an image backend for a grayscale depth map does not guarantee geometrically aligned depth. The first version should call these `depth_companions` and record a `depth_kind`:

- `prompted`: image-backend-generated companion, experimental
- `estimated`: real depth adapter output, future/local adapter path
- `uploaded`: user-provided depth companion
If depth is requested but unsupported or failed, no `vn_asset_items` row is created because there is no image candidate to serve. The depth companion slot records `status=failed` and `last_error=depth_unavailable` or a backend-specific failure code. Readiness and manifest responses may report `depth_companion_status=unavailable`, but `depth_kind=unavailable` is not stored on item rows.

### Consistency And Cutout Quality Are First-Class Risks

Most image backends will not automatically preserve character identity, transparency, sprite bounds, or crop anchors across hundreds of outputs. The schema and workbench must represent these facts explicitly instead of hiding them.

## Architecture

New backend module:

- `tldw_Server_API/app/core/VN_Assets/`
  - pack service
  - matrix expansion
  - prompt assembly
  - review/promotion state machine
  - generation orchestration
  - depth companion strategy
  - runtime manifest builder
- `tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py`
  - pack, slot, item, and batch metadata stored in the per-user ChaChaNotes database
- `tldw_Server_API/app/api/v1/endpoints/vn_assets.py`
  - pack CRUD, matrix editing, generation control, review, and manifest endpoints
- `tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py`
  - Pydantic request/response models
- Jobs domain: `vn_assets`
  - slot/variant generation jobs
  - progress aggregation
  - retries and cancellation

Existing services reused:

- Image-generation adapter registry for actual model calls.
- FileArtifactsService or lower-level file artifact adapters for structured generation metadata.
- Generated-file storage for durable image bytes and quota tracking.
- Character and world-book services for source context.
- AuthNZ user scoping and role checks.

Metadata storage boundary:

- VN pack metadata lives in the user's existing `ChaChaNotes.db`, alongside character cards and world books.
- `VNAssetPacks_DB.py` owns the VN tables and migration helpers, but uses the same per-user database boundary as Characters/World Books.
- Image bytes are never stored in ChaChaNotes. `vn_asset_items.generated_file_id` points to AuthNZ generated-file tracking, and VN services validate that the generated-file record belongs to the same user before serving content.
- Cross-database references to generated files are application-validated rather than foreign-key enforced.
- This keeps character/world-book backup scope coherent while letting generated-file cleanup and quota accounting stay in AuthNZ storage services.

## Data Model

### `vn_asset_packs`

Stores pack-level ownership and generation defaults.

Important fields:

- `id`
- `owner_user_id`
- `title`
- `description`
- `status`: `draft`, `generating`, `reviewing`, `ready`, `archived`, `failed`
- `content_rating`: `general`, `suggestive`, `mature`, `violent`, or custom label
- `primary_character_id`
- `source_world_book_ids_json`
- `scenario_notes`
- `style_prompt`
- `negative_prompt`
- `default_backend`
- `default_model`
- `default_dimensions_json`
- `style_lock_json`
- `generation_budget_json`
- `created_at`
- `updated_at`
- `version`
- `deleted`

`style_lock_json` should support canonical reference image IDs, fixed seed policy, model/backend preferences, and notes warning when selected backends lack reference-image support.

V1 pack generation supports one primary character. CG items may carry optional participating-character metadata for future runtimes, but the workbench and prompt assembly generate from the primary character only.

Pack status transitions:

- `draft`: pack exists, matrix may still be edited, no active generation batch.
- `generating`: at least one slot/variant job is queued or processing.
- `reviewing`: generation is complete or cancelled and at least one draft item needs review, or one required slot lacks an approved item.
- `ready`: no active jobs, every `required_for_runtime` slot has at least one approved item, and every approved sprite/background/CG item has the minimum manifest metadata required for playback. Optional failed or skipped slots may remain as readiness warnings.
- `failed`: generation ended without any usable draft or approved items, or all required slots failed.
- `archived`: user-hidden from normal pack lists, but not deleted.

Status is derived from batch, slot, and item state where possible; direct status writes should be limited to user-driven archive/delete actions.

### `vn_asset_slots`

Each slot represents one desired output, before variants are generated.

Important fields:

- `id`
- `pack_id`
- `asset_type`: `sprite`, `background`, `depth_companion`, `cg`
- `slot_key`: stable key such as `sprite.happy.front.default`
- `labels_json`: expression, pose, outfit, location, time, scenario, camera, etc.
- `prompt_template`
- `negative_prompt_template`
- `variant_count`
- `width`
- `height`
- `backend_override`
- `model_override`
- `seed_policy_json`
- `requires_review`
- `required_for_runtime`
- `depends_on_slot_id`
- `status`: `planned`, `queued`, `generating`, `reviewing`, `approved`, `skipped`, `failed`, `cancelled`
- `last_error`
- `created_at`
- `updated_at`

Depth companion slots depend on background slots where applicable.

Slot status transitions:

- `planned`: slot exists but has no queued generation work.
- `queued`: at least one variant job is waiting.
- `generating`: at least one variant job is actively processing.
- `reviewing`: at least one draft candidate needs review, or a required slot has no approved item after generation.
- `approved`: slot has at least one approved item and no draft candidate still requires review.
- `skipped`: optional slot was intentionally excluded or accepted without an asset.
- `failed`: all requested variants failed and no approved item exists. This blocks readiness only when `required_for_runtime=true`.
- `cancelled`: all queued work for the slot was cancelled before producing a candidate.

Slot status is derived with explicit precedence so review counters and runtime readiness do not disagree:

1. `generating`: any variant job is actively processing.
2. `queued`: no active job exists, but at least one variant job is queued or scheduled.
3. `skipped`: the optional slot was intentionally skipped by the user.
4. `failed`: all requested variants failed, no approved candidate exists, and the slot is not skipped.
5. `reviewing`: at least one draft candidate exists, or a required slot has no approved item after generation.
6. `approved`: at least one approved item exists and no draft candidate still requires review.
7. `cancelled`: all queued work was cancelled before producing a candidate.
8. `planned`: none of the above applies.

For readiness, `approved` is a slot-level derived state. It does not mean every variant is approved. Optional failed or skipped slots are non-blocking warnings in readiness and manifest diagnostics.

### `vn_asset_items`

Each generated, uploaded, or replaced image candidate.

Important fields:

- `id`
- `pack_id`
- `slot_id`
- `variant_index`
- `file_artifact_id`: optional structured generation metadata
- `generated_file_id`: durable generated-file record for playback serving
- `storage_ref`: opaque storage reference if needed by the storage service
- `mime_type`
- `width`
- `height`
- `bytes`
- `review_status`: `draft`, `approved`, `rejected`, `hidden`
- `preferred`
- `source`: `generated`, `uploaded`, `imported`
- `generation_job_id`
- `source_prompt_snapshot_json`
- `source_context_snapshot_json`
- `backend_metadata_json`
- `depth_kind`: nullable except for depth companion items; valid item values are `prompted`, `estimated`, or `uploaded`
- `parent_item_id`: for depth companions tied to a background item
- `has_alpha`
- `crop_box_json`
- `anchor_json`
- `scale_hint`
- `trim_status`: `unknown`, `clean`, `needs_trim`, `processed`, `failed`
- `quality_flags_json`
- `created_at`
- `updated_at`

Only `approved` items are visible in the runtime manifest. A slot may have multiple approved variants, with one optional preferred item.

Depth companion failures do not create item rows. They remain represented by the depth slot/job state and by background manifest diagnostics.

Item review transitions:

- New generated/imported/uploaded items start as `draft`.
- `draft` can become `approved`, `rejected`, or `hidden`.
- `approved` can become `hidden` or `rejected`; doing so may make a required slot no longer runtime-ready.
- `rejected` and `hidden` can return to `draft` or become `approved` through explicit user action.
- A slot may have only one `preferred` approved item at a time.

### `vn_asset_batches`

Tracks pack-level generation runs and maps to Jobs records.

Important fields:

- `id`
- `pack_id`
- `job_batch_id`
- `requested_by_user_id`
- `status`
- `total_slots`
- `total_variants`
- `completed_count`
- `failed_count`
- `cancelled_count`
- `started_at`
- `completed_at`
- `options_json`

## Generation Matrix

The first workbench ships with an editable starter matrix:

- Sprites: expressions, poses, outfits, optional camera distance, variant count.
- Backgrounds: locations, time of day, mood/lighting, variant count.
- Depth companions: enabled per background slot, strategy configured per pack. Default policy generates depth only for approved/preferred background variants; generating depth for every background variant is an explicit opt-in.
- CGs: common event/scenario slots, camera framing, primary-character state, optional participating-character metadata, variant count.

Users can edit labels and prompts, add/remove rows, and change variant counts before generation.

## Prompt Assembly

Prompt assembly uses a bounded source context:

- Primary character fields: name, description, personality, scenario, first message, creator notes, avatar/style notes.
- Selected world-book entries or summaries.
- Pack scenario notes.
- Pack style prompt and negative prompt.
- Slot labels and slot prompt template.
- Style lock hints: canonical reference, model, seed, and consistency notes.

The workbench should expose a prompt/source preview before generation. The backend should enforce source budgets so a large world book cannot silently dominate every prompt.

Prompt snapshots are stored per item for auditability and reproducibility. Sensitive secrets must never be included in snapshots or logs.

Default prompt budgets:

- Character source budget: `1,500` tokens.
- World-book source budget: `1,000` tokens.
- Pack scenario/style budget: `750` tokens.
- Slot template and labels budget: `750` tokens.
- Total assembled prompt budget before backend-specific truncation: `4,000` tokens.

Budget units should use the project token-estimation helper where available; fallback estimate is 4 characters per token. Truncation order is deterministic: preserve slot template, labels, pack style, and negative prompt first; then character core fields; then world-book snippets by priority/relevance. The prompt preview endpoint returns the assembled prompt, omitted-source counts, token estimates by source bucket, and warnings when truncation occurred.

## Generation Flow

1. User creates a pack from a primary character and optional world books.
2. User selects the editable VN starter matrix.
3. Backend expands the matrix into deterministic slots and estimates output count.
4. User starts generation.
5. Backend creates a `vn_assets` batch and enqueues generation work through the fanout strategy below.
6. Workers process jobs directly through core services/adapters.
7. Each result is saved to durable generated-file storage and registered for quota tracking.
8. Optional file artifact metadata is stored for structured generation provenance.
9. Items start as `draft`.
10. User approves, rejects, hides, uploads replacements, or regenerates variants.
11. Runtime manifest exposes approved items only.

Failures are per slot or per variant. One failed CG should not fail an entire pack.

## Jobs And Progress

Use slot-level or variant-level Jobs for generation work, but do not enqueue hundreds of child jobs directly from the API request.

Requirements:

- Pack progress aggregates child job states.
- Cancellation can stop queued jobs and request cancellation for active jobs.
- Retry can target failed slot jobs or selected variants.
- Depth companion jobs can wait for successful and approved/preferred background items by default.
- Lease renewal and stale-worker handling follow core Jobs rules.
- Job payloads contain IDs and options, not large image bytes or prompt blobs.

Fanout strategy:

- `POST /packs/{pack_id}/generate` validates pack limits, storage estimate, active batch limits, and relevant Jobs quota signals before creating a `vn_asset_batches` row.
- The API enqueues one lightweight parent job, `vn_asset_enqueue_batch`, with idempotency key `vn_assets:batch:{batch_id}:enqueue`.
- The parent fanout job creates child generation jobs gradually, records `planned_count`, `enqueued_count`, and `enqueue_error` on the batch, and can back off or resume if Jobs quotas reject more child jobs.
- Child jobs use deterministic idempotency keys such as `vn_assets:batch:{batch_id}:slot:{slot_id}:variant:{variant_index}:attempt:{attempt}` so retries do not duplicate successful work.
- If the Jobs layer later exposes safe transactional multi-create, the fanout implementation can switch to transactional child enqueue without changing the public API contract.

Backend concurrency:

- VN workers must acquire a per-backend generation gate before invoking the image adapter.
- Local or GPU-bound backends default to one active generation job globally unless configured otherwise.
- Remote/API backends may use a higher configurable default, but the gate is still keyed by backend and optionally model.
- If the gate is unavailable, the worker should reschedule or back off without holding a long idle lease.

Suggested domain:

- `domain="vn_assets"`
- queues: `default`, optional `low`, optional `high`
- job types:
  - `vn_asset_enqueue_batch`
  - `vn_asset_generate_variant`
  - `vn_asset_generate_depth_companion`
  - `vn_asset_import_item`
  - future: `vn_asset_trim_sprite`

## Storage And Serving

Generated images should be registered through durable generated-file storage, not only temp file artifact exports. The VN asset item stores a durable generated-file reference. The API exposes pack-specific authenticated content endpoints, for example:

- `GET /api/v1/vn-assets/packs/{pack_id}/items/{item_id}/content`
- `GET /api/v1/vn-assets/packs/{pack_id}/manifest`

The content endpoint enforces pack ownership and item access. It should stream durable storage content without consuming or deleting it.

Generated-file registration must use a dedicated source feature: `vn_assets`. Implementation should add this source feature to generated-file storage validation/migrations rather than overloading `image_gen`. VN asset records should set `source_ref` to a stable reference such as `vn_asset_item:{item_id}`, `folder_tag` to `vn-assets/{pack_id}`, and tags for pack ID, asset type, content label, backend, and model. This keeps quota reports, cleanup, and ownership attribution distinguishable from ad hoc image generation.

Deletion behavior:

- `DELETE /packs/{pack_id}` soft-deletes pack metadata by default and retains durable generated files.
- Physical file deletion is handled only by `POST /packs/{pack_id}/cleanup`, not by a query flag on `DELETE`.
- Cleanup requests should support dry-run mode, item-status filters, explicit `include_approved` opt-in, and a confirmation token or confirmation text before deleting bytes.
- Cleanup can remove bytes for selected draft, rejected, or hidden items. Approved runtime assets are preserved unless the request explicitly includes approved assets and passes confirmation checks.
- Deleting/rejecting an item should not immediately hard-delete bytes unless the user requests cleanup.
- Cleanup policies should distinguish drafts, rejected items, hidden items, and approved runtime assets.

Quotas:

- Preflight estimates warn about likely storage usage.
- Each generated image counts against user storage quotas.
- Jobs fail gracefully with quota errors and preserve partial results.

Initial hard limits:

- Default maximum planned generated items per pack: `300`.
- Default maximum variants per slot: `6`.
- Default maximum active generation batches per user: `1`.
- Default maximum active local/GPU image generations across all VN workers: `1`.
- Generation requests that exceed limits return validation errors before enqueueing jobs.
- Limits should be configurable through server settings/env vars, but tests must cover the defaults.

Preflight storage estimate should use configured width, height, format, variant count, and a conservative bytes-per-pixel fallback when exact backend estimates are unavailable.

## API Surface

Base path: `/api/v1/vn-assets`

Pack lifecycle:

- `POST /packs`
- `GET /packs`
- `GET /packs/{pack_id}`
- `PATCH /packs/{pack_id}`
- `DELETE /packs/{pack_id}`
- `POST /packs/{pack_id}/cleanup`

Matrix and slots:

- `GET /starter-matrices`
- `POST /packs/{pack_id}/matrix/apply`
- `GET /packs/{pack_id}/slots`
- `POST /packs/{pack_id}/slots`
- `PATCH /packs/{pack_id}/slots/{slot_id}`
- `DELETE /packs/{pack_id}/slots/{slot_id}`

Generation:

- `POST /packs/{pack_id}/generate`
- `GET /packs/{pack_id}/generation`
- `POST /packs/{pack_id}/generation/cancel`
- `POST /packs/{pack_id}/slots/{slot_id}/retry`
- `POST /packs/{pack_id}/items/{item_id}/regenerate`

Review:

- `GET /packs/{pack_id}/items`
- `PATCH /packs/{pack_id}/items/{item_id}/review`
- `POST /packs/{pack_id}/items/bulk-review`
- `POST /packs/{pack_id}/items/upload`
- `POST /packs/{pack_id}/items/{item_id}/preferred`

Runtime/read-only:

- `GET /packs/{pack_id}/manifest`
- `GET /packs/{pack_id}/items/{item_id}/content`

Diagnostics:

- `POST /packs/{pack_id}/prompt-preview`
- `GET /packs/{pack_id}/readiness`

## Runtime Manifest Contract

The runtime manifest should be stable enough for future Story Engine/VN playback:

- schema version, initially `vn_asset_manifest.v1`
- pack metadata
- primary character ID
- optional participating character IDs on CG items
- content rating labels
- asset groups by type
- approved items only
- slot labels
- item IDs
- dimensions
- sprite anchor/crop/scale metadata
- background/depth companion pairings
- CG scenario labels
- preferred variant markers

The manifest must omit drafts, rejected items, hidden items, and hidden world-book content.

Minimum manifest metadata for runtime readiness:

- Every approved item: `item_id`, `slot_key`, `asset_type`, `content_url`, `mime_type`, `width`, `height`, `labels`.
- Sprite items: `anchor` (default `{ "x": 0.5, "y": 1.0 }`), `scale_hint` (default `1.0`), `has_alpha` (boolean, default `false`), `crop_box` (nullable), and `trim_status` not equal to `failed`.
- Background items: dimensions, content URL, and optional `depth_companion_item_id`; if a depth companion was requested but unavailable, include `depth_companion_status: "unavailable"` and keep the background runtime-ready unless the pack explicitly marks depth as required.
- Depth companion items: `parent_item_id`, `depth_kind`, dimensions, and content URL. Dimensions should match the parent background when possible; mismatches produce manifest warnings.
- CG items: dimensions, content URL, labels, and optional participating character IDs. In V1, participating IDs are consumer metadata only; generation and prompt assembly remain primary-character based.

Readiness validation applies these defaults server-side so deferred sprite trim/cutout processing does not block a pack unless an item is explicitly marked `trim_status=failed` or a required slot has no approved item.

Compact example:

```json
{
  "schema_version": "vn_asset_manifest.v1",
  "pack_id": 12,
  "title": "Gura VN starter pack",
  "primary_character_id": 4,
  "content_rating": "general",
  "assets": {
    "sprites": [
      {
        "slot_key": "sprite.happy.front.default",
        "item_id": 101,
        "preferred": true,
        "labels": {"expression": "happy", "pose": "front", "outfit": "default"},
        "content_url": "/api/v1/vn-assets/packs/12/items/101/content",
        "width": 768,
        "height": 1024,
        "anchor": {"x": 0.5, "y": 1.0},
        "crop_box": null,
        "has_alpha": true
      }
    ],
    "backgrounds": [
      {
        "slot_key": "background.bedroom.evening",
        "item_id": 201,
        "content_url": "/api/v1/vn-assets/packs/12/items/201/content",
        "depth_companion_item_id": 202,
        "depth_companion_status": "available"
      }
    ],
    "cgs": []
  }
}
```

## Workbench UX

Route: `/vn-assets` for the first implementation.

Views:

- Pack list: primary character, status, approved/draft/failed counts, runtime readiness, last generated.
- Pack setup: primary character picker, world book picker, starter matrix selector, style prompt, negative prompt, content label.
- Matrix editor: editable sections for sprites, backgrounds/depth companions, and CG scenarios.
- Prompt preview: source snippets and final prompt preview for representative slots.
- Generation monitor: progress, queue status, failed slots, retry/cancel controls.
- Review board: grouped grid by asset type and slot.

Review board requirements for 100-300 assets:

- Keyboard approve/reject shortcuts.
- Approve one preferred variant per slot.
- Bulk reject failed or low-quality variants.
- Bulk hide/unhide.
- Retry failed slots.
- Upload replacement images or depth companions.
- Runtime readiness counters.

The workbench should not attempt full VN playback in this phase.

## Safety And Governance

Self-hosted default:

- Content labels and audit metadata are required.
- Generation is not blocked by default.
- Admin policy gates can be added later.

Pack and item metadata should record:

- creator user ID
- backend
- model
- prompt snapshot
- primary character ID
- optional participating character IDs for CG metadata
- source world-book IDs
- seed/settings
- content rating labels

Logs should avoid full prompt/body dumps in production. Use IDs, counts, provider/model, and failure codes.

## Security

- AuthNZ enforced on every endpoint.
- Pack ownership checked for all pack, slot, item, content, and job operations.
- Generated-file access is mediated through VN endpoints, not raw paths.
- Uploads validate MIME type, extension, dimensions, size, and image decoding.
- Prompt snapshots redact secrets and avoid hidden lore unless explicitly permitted.
- Jobs payloads do not contain large image data.
- World-book context respects the same user/character ownership rules as existing world-book APIs.

## Rollout Plan

### Phase 1: Backend Pack Core

- DB schema and migrations.
- Pack CRUD.
- Matrix definitions and expansion.
- Slot CRUD.
- Item review state machine.
- Runtime manifest for approved assets.

### Phase 2: Generation Jobs

- Jobs domain and worker.
- Parent fanout job with idempotent child job creation.
- Per-backend concurrency gate for synchronous/local image adapters.
- Direct adapter/service invocation.
- Durable generated-file registration.
- Per-slot/variant status.
- Retry, cancellation, and quota handling.
- Sprite trim/background removal remains out of scope for this phase. The worker records `has_alpha`, `crop_box`, `anchor`, `scale_hint`, and `trim_status`, but does not attempt automatic matting or cutout correction.

### Phase 3: Depth Companions

- Prompted depth companion mode for approved/preferred background variants by default.
- Uploaded depth companion attachment.
- Metadata for future estimated/local adapter support.

### Phase 4: Workbench UI

- Pack setup.
- Matrix editor.
- Prompt preview.
- Generation monitor.
- Review board.
- Runtime readiness view.

### Phase 5: Consumer Contract

- Stabilize manifest shape.
- Add docs for future Story Engine/VN runtime consumption.
- Optional export/import format for packs remains future scope and should not be included in the first implementation plan unless the user explicitly re-scopes the work.

### Future: Sprite Processing Adapter

- Optional background removal, alpha matting, crop detection, and anchor normalization.
- New job type: `vn_asset_trim_sprite`.
- Workbench review affordances for before/after comparisons.

This is explicitly deferred so the first implementation can ship generation, storage, and review without adding another image-processing model/toolchain.

## Testing Strategy

Backend:

- Matrix expansion is deterministic.
- Slot counts and variant counts respect configured limits.
- Prompt assembly respects source budgets and produces previews.
- AuthNZ prevents cross-user pack, item, content, and job access.
- Jobs handle partial failures, retries, cancellation, and dependencies.
- Jobs fanout is idempotent and does not duplicate child work after retries.
- Backend concurrency gates prevent overlapping local/GPU generations by default.
- Durable generated-file references are served without one-shot deletion.
- Review-state transitions enforce approved-only manifests.
- Slot derived-status precedence is deterministic.
- Quota failures preserve partial completed results and mark remaining jobs failed.
- Depth companion states distinguish prompted, uploaded, estimated, and unavailable.

Frontend:

- Pack setup flow.
- Matrix editing and count preview.
- Prompt preview display.
- Generation monitor states.
- Review board keyboard and bulk actions.
- Runtime readiness counters.

Security:

- Upload validation.
- Bandit on touched backend paths during implementation.
- No raw storage paths exposed.
- Prompt snapshots and logs exclude secrets.

## Risks And Mitigations

- Character consistency may be poor on unsupported backends. Mitigate with style-lock metadata, reference-image capability checks, fixed seed options, and explicit warnings.
- Sprites may lack transparency or clean crop bounds. Mitigate with sprite metadata, replacement uploads, and future trim/background-removal adapter slots.
- Prompted depth companions may not align with backgrounds. Mitigate by labeling them experimental and supporting uploaded/estimated alternatives.
- Full packs can consume large storage. Mitigate with preflight estimates, quotas, cleanup policies, and review bulk actions.
- Long generation batches can be brittle. Mitigate with parent fanout, slot/variant Jobs, partial success, retries, cancellation, idempotent enqueue, and resumable progress.
- Local image generation can exhaust VRAM if multiple workers invoke it concurrently. Mitigate with per-backend concurrency gates that default local/GPU backends to one active generation.
- World-book context can overwhelm prompts. Mitigate with source budgets and user-visible prompt previews.

## Open Questions

Blocking decisions resolved for initial planning:

- Generated-file source feature: use dedicated `vn_assets`.
- First hard default limit: 300 planned generated items per pack, 6 variants per slot.
- Sprite trim/background removal: defer to a future sprite processing adapter.
- Metadata storage: VN tables live in per-user `ChaChaNotes.db`; generated-file bytes and quota records stay in AuthNZ generated-file storage.
- File cleanup: metadata deletion and physical file deletion are separate; physical cleanup uses an explicit POST flow with dry-run and confirmation.

Future questions:

- Which image backends should be considered reference-image capable for style locking?
- Should pack export/import include bytes, metadata only, or both?
- How should future Story Engine map story locations and scene beats onto this independent pack manifest?
- What is the right shape for future multi-character pack generation beyond V1's single-primary-character model?
