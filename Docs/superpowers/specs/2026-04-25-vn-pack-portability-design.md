# VN Pack Portability Design

Status: Draft
Date: 2026-04-25
Owner: Core/WebUI maintainers
Scope: Backup-grade import/export for VN asset packs

## Summary

Add VN-native import/export for asset packs through a zip-compatible `.tldw-vnpack`
archive. The first version is a backup/restore format, not a playback-only sharing
format. It preserves full pack workbench state, image bytes, review state, slot
metadata, item metadata, and enough provenance to restore or inspect a pack on
another tldw_server install.

The feature builds on the VN Asset Packs module. It reuses the existing Jobs,
AuthNZ, generated-file storage, audit, and Chatbooks archive-safety patterns, but
does not reuse the Chatbooks content model. VN packs have different semantics:
slots, variants, review state, generated-file remapping, manifest readiness, and
future VN runtime consumption.

## Goals

- Export one VN asset pack as a portable `.tldw-vnpack` archive.
- Include image bytes by default for backup-grade restore.
- Preserve full pack state, including draft, rejected, hidden, and approved items.
- Redact prompt/source snapshots by default while preserving reproducibility
  metadata and prompt hashes.
- Optionally include source character and world-book data at export time.
- Import through an async preview and commit flow.
- Detect conflicts and allow create, link, skip, fail, and update-existing policies.
- Support trusted restore and untrusted import review behavior.
- Remap all database IDs and generated-file records safely on import.
- Store checksums and canonical fingerprints for integrity, conflict detection, and
  future signatures.

## Non-Goals

- No playable VN runtime in this project.
- No Story Engine integration in this project.
- No runtime/share bundle profile in v1. The default bundle is a backup bundle.
- No password-encrypted archives in v1. The format reserves encryption metadata so
  later encrypted bundles fail clearly on older servers.
- No automatic destructive update of existing packs, characters, world books, or
  generated files.
- No direct import from ComfyUI, Ren'Py, or other VN tool formats.

## Existing Project Context

Useful foundations already exist:

- VN asset pack metadata lives in the per-user `ChaChaNotes.db`.
- VN image bytes are tracked through AuthNZ generated-file storage with
  `source_feature="vn_assets"` and source refs such as `vn_asset_item:{item_id}`.
- VN runtime manifests already expose approved items through stable content URLs.
- Jobs support user-visible progress, cancellation, retries, and status inspection.
- Chatbooks already has secure archive patterns: zip validation, per-user temp
  directories, archive limits, download retention, and audit logging.

The portability design should copy proven Chatbooks patterns where they apply, but
it should remain a VN-specific module under `tldw_Server_API/app/core/VN_Assets/`.

## Architecture

New backend package:

- `tldw_Server_API/app/core/VN_Assets/portability/`
  - archive format constants
  - export assembler
  - archive validator
  - import preview planner
  - import commit executor
  - conflict detector
  - canonical fingerprint builder
  - import journal helpers

Existing modules extended:

- `VNAssetPacks_DB.py`
  - portability job records
  - import preview records
  - import journal rows
  - old-to-new ID map storage
- `vn_assets.py`
  - export, preview, commit, job status, download endpoints
- `vn_asset_schemas.py`
  - portability requests and responses
- `VN_Assets/jobs.py`
  - job type constants and idempotency keys for portability work

Jobs domain:

- Reuse `domain="vn_assets"`.
- Add job types:
  - `vn_pack_export`
  - `vn_pack_import_preview`
  - `vn_pack_import_commit`
  - optional future: `vn_pack_cleanup_import`

Job status source of truth:

- The core Jobs manager owns lifecycle state: queued, processing, completed, failed,
  cancelled, retry count, leases, and worker ownership.
- `vn_pack_portability_jobs` owns VN-specific progress: current stage, stage counts,
  archive path, download URL, warnings, validation summaries, preview IDs, import
  IDs, and cleanup status.
- API status responses compose both records. When lifecycle fields disagree, the
  Jobs manager wins. When stage/progress fields disagree, the VN portability row wins
  unless the Jobs manager is terminal and the VN row is still non-terminal.
- Reads reconcile drift: a terminal Jobs record moves stale VN portability rows into
  the matching terminal state and preserves the last recorded stage/warnings.
- Cancellation and retry requests go through the Jobs manager first, then append
  VN-specific cancellation/retry notes to the portability row.
- Workers update VN stage/progress rows idempotently after acquiring a job lease and
  before releasing or completing the job.

Storage boundary:

- Export reads existing generated-file records and streams bytes from durable
  generated-file storage.
- Import always creates new generated-file records for imported image bytes.
- Imported metadata never reuses generated-file IDs from the source install.
- Cross-store consistency is managed by an import journal because ChaChaNotes and
  AuthNZ generated-file storage cannot participate in one database transaction.

## Archive Format

The archive extension is `.tldw-vnpack`. The content is a standard ZIP file with a
VN-specific schema marker.

Schema version:

- `tldw.vnpack.v1`

Layout:

```text
manifest.json
README.md
assets/items/<source_item_key>.<ext>
metadata/pack.json
metadata/slots.json
metadata/items.json
metadata/batches.json
metadata/character.json
metadata/world_books.json
metadata/provenance.json
metadata/runtime_manifest.json
checksums/sha256.json
signatures/README.md
```

Required files:

- `manifest.json`
- `metadata/pack.json`
- `metadata/slots.json`
- `metadata/items.json`
- `checksums/sha256.json`

Optional files:

- `README.md`
- `metadata/batches.json`
- `metadata/character.json`
- `metadata/world_books.json`
- `metadata/provenance.json`
- `metadata/runtime_manifest.json`
- `assets/items/*`
- `signatures/README.md`

Unexpected top-level files are rejected. Future-compatible sections must live under
known top-level directories and be ignored only when the manifest explicitly marks
them optional.

## Manifest

`manifest.json` contains:

- `schema_version`
- `exported_by`: app name and version if available
- `exported_at`
- `archive_profile`: `backup`
- `pack_title`
- `content_rating`
- `source_pack_fingerprint`
- `canonical_payload_fingerprint`
- `counts`: slots, items, approved, draft, rejected, hidden, assets with bytes
- `include_images`: true for v1 default full bundles
- `include_character`: boolean
- `include_world_books`: boolean
- `provenance_mode`: `redacted` or `full`
- `trust_hints`: local source hints and non-authoritative source identifiers
- `encryption`: `{ "encrypted": false, "scheme": null }`
- `sections`: file paths and expected checksums for metadata sections
- `warnings`: export-time warnings

The manifest must not be the only source of detailed pack state. Detailed records
live in the `metadata/*.json` files so validators can check section-level schema and
checksums independently.

The final archive SHA-256 is not stored inside `manifest.json`. Writing the exact
ZIP hash into a file inside the ZIP would change the ZIP bytes and invalidate the
hash. Export jobs record `archive_sha256` in job/download metadata after the archive
is closed. Import preview computes the uploaded archive hash from the received file
and stores it in preview/job metadata outside the archive payload.

## Export Scope

V1 exports a backup-grade bundle:

- Pack metadata.
- Slot metadata.
- Every item record, regardless of review state.
- Preferred markers.
- Draft, approved, rejected, and hidden review states.
- Batch summary metadata.
- Runtime-readiness diagnostics.
- Approved-only runtime manifest snapshot.
- Image bytes for every item with readable durable generated-file content.
- Redacted provenance by default.

Export options:

- `include_character_payload`: default false.
- `include_world_book_payloads`: default false.
- `include_full_provenance`: default false.
- `strict`: default false. Missing image bytes become warnings in non-strict mode.
- `warn_for_sharing`: default true.

The UI must label this as a backup bundle. It should warn before export when:

- Character or world-book data is included.
- Full provenance/prompts are included.
- Draft, rejected, or hidden assets are included.
- Content rating is mature or custom.
- Encryption is not available in v1.

## Provenance Redaction

Default `metadata/provenance.json` keeps:

- backend
- model
- dimensions
- seed and deterministic generation settings when available
- item hashes
- prompt-present flags
- prompt hashes
- source context hashes
- generation job IDs and batch IDs where useful for local audit

Default redaction removes:

- full prompt text
- full negative prompt text
- full source context snapshots
- hidden world-book text
- creator notes unless explicitly included through character export
- local raw storage paths

Full provenance export is explicit. Even in full mode, secrets and API keys must be
redacted before writing any archive file.

## Export Flow

Export is always Jobs-backed:

1. API validates pack ownership and export options.
2. API creates a `vn_pack_export` job with an idempotency key derived from user ID,
   pack ID, options hash, and request nonce.
3. Worker loads pack, slots, items, batches, readiness, and optional character/world
   book data.
4. Worker checks generated-file ownership for every item with bytes.
5. Worker estimates archive size and validates export limits.
6. Worker writes a per-user staging directory.
7. Worker streams image bytes into `assets/items/`.
8. Worker writes metadata JSON and section checksums.
9. Worker computes canonical payload fingerprint.
10. Worker creates the zip archive.
11. Worker records file size, archive checksum, expiry, warnings, and download URL.

Failed reads of individual item bytes are warnings unless `strict=true`. Strict
exports fail if any expected item byte cannot be read or validated.

Non-strict exports may include item metadata for assets whose bytes were missing or
unreadable. Those item records must carry `asset_bytes_status="missing"` and no
`asset_path`. The source review state remains in metadata, but import must not make
missing-byte items runtime-ready.

## Import Overview

Import is split into async preview and async commit:

1. Upload archive and start preview job.
2. Inspect preview result and choose conflict policies.
3. Start commit job from an immutable preview.

The preview stage validates and plans. The commit stage writes data.

This separation is required because full bundles may contain hundreds of images,
large checksums, and conflict-detection work that can exceed normal request
timeouts.

## Import Preview Flow

Endpoint:

- `POST /api/v1/vn-assets/import/previews`

Behavior:

- Uploads the archive to a per-user temp directory.
- Creates a `vn_pack_import_preview` job.
- Returns `preview_job_id` and `preview_id`.

Preview job stages:

1. Validate archive member paths.
2. Validate schema version.
3. Enforce per-file and total uncompressed size limits.
4. Reject symlinks, duplicate normalized paths, absolute paths, drive letters, null
   bytes, and `..` components.
5. Validate required files.
6. Validate metadata JSON schemas.
7. Validate section checksums.
8. Decode/sniff every image that will be importable.
9. Compare image MIME, extension, dimensions, and metadata.
10. Compute archive checksum and canonical payload fingerprint.
11. Estimate storage quota impact.
12. Detect pack, character, world-book, slot, and item conflicts.
13. Produce an immutable preview plan.

Preview status endpoint:

- `GET /api/v1/vn-assets/import/previews/{preview_id}`

Preview result includes:

- validation status
- bundle summary
- warnings
- required choices
- conflict list
- proposed actions
- quota estimate
- trust-mode options
- update-existing diff if a target pack is selected

Preview records expire. Commit is rejected when the preview expired, the archive
was removed, or the archive checksum no longer matches.

## Required Character Resolution

Imports must never create a VN pack with a dangling `primary_character_id`.

For each imported pack, preview returns one required character action:

- `import_included_character`
- `link_existing_character`
- `create_placeholder_character`
- `fail_import`

Rules:

- If the archive includes character data, preview may propose import or link.
- If the archive does not include character data, preview may propose link or fail.
- Placeholder creation is explicit. It creates a minimal character record marked as
  an imported placeholder and marks the pack as needing source-character repair.
- The commit endpoint rejects plans that do not resolve the primary character.

World books are optional. Missing, skipped, or unresolved world books generate
warnings and preserve source fingerprints in import metadata.

## Conflict Detection

Conflict detection uses deterministic fingerprints and human-readable candidates.

Pack conflict signals:

- canonical payload fingerprint
- source pack fingerprint
- pack title
- primary character fingerprint
- matching slot keys and item checksums

Character conflict signals:

- included character card hash
- normalized character name
- creator/source identifiers when present
- avatar/image hash when included

World-book conflict signals:

- title
- canonical content hash
- source identifiers when present

Item conflict signals:

- slot key
- variant index
- asset type
- item checksum
- source item fingerprint

Preview must not auto-merge based on weak signals. It may recommend actions, but
the accepted plan records the user's or API client's explicit policy.

## Import Policies

API-supported policies:

- `create_new`
- `link_existing`
- `skip`
- `fail_on_conflict`
- `update_existing`

The WebUI uses the same preview/commit API as headless clients.

Default import behavior:

- Create a new pack.
- Import image bytes as new generated-file records.
- Link existing character only when selected.
- Reset review state when trust mode is `untrusted_import`.

Missing asset bytes:

- If an imported item has `asset_bytes_status="missing"` or lacks an asset file, the
  item metadata can be imported but `generated_file_id`, `storage_ref`, and content
  URL remain null/unavailable.
- Missing-byte items are forced to `review_status="hidden"` in both trust modes.
- The original source review state is preserved in import metadata for audit and
  possible later repair.
- Required slots that only contain missing-byte items are not runtime-ready.
- Import preview reports missing-byte item counts and required-slot impact before
  commit.
- Update-existing mode never replaces a local item that has bytes with an imported
  missing-byte item.

## Trust Modes

Trust mode controls whether imported review state is active after import.

`trusted_restore`:

- Preserves review status.
- Preserves preferred markers.
- Preserves pack readiness where possible.
- Intended for personal backups or known-safe bundles.

`untrusted_import`:

- Imports assets as draft/needs-review.
- Preserves original source review state in import metadata for audit.
- Does not expose imported items through runtime manifest until reviewed.
- Intended for bundles from other people or unknown sources.

V1 trust is explicit user/API choice. Bundles store fingerprints and checksums so
future versions can add signatures or local-export trust without changing the
archive shape.

## Update Existing Mode

V1 update mode is non-destructive by default.

Allowed without extra per-diff confirmation:

- Add missing slots.
- Add missing items.
- Add new generated-file records.
- Add import metadata and source fingerprints.
- Link imported assets to selected existing character/world-book records.

Identity rules:

- Existing slots match imported slots by `asset_type + slot_key`.
- If a matched slot has different labels, prompt templates, dependency metadata,
  dimensions, or `required_for_runtime`, the difference is reported as a slot
  metadata diff and requires explicit confirmation before update.
- Existing items first match by `source_item_fingerprint` when present.
- If no source item fingerprint exists, items match by exact asset checksum under a
  matched slot.
- If no checksum exists, items may only be suggested as possible matches by matched
  slot plus `variant_index`; this is ambiguous and requires explicit user/API
  selection.
- Duplicate or conflicting candidate matches block automatic update and must be
  resolved in the accepted preview plan.

Requires explicit per-diff confirmation:

- Change review status of an existing local item.
- Replace a preferred item.
- Modify pack style, scenario, prompt, or content-rating fields.
- Hide or reject an existing local item.
- Relink the pack to a different primary character.

Out of scope for v1 unless explicitly re-scoped later:

- Hard-delete local items.
- Hard-delete local generated files.
- Rewrite existing generated-file records.
- Auto-overwrite character or world-book records.

The update preview must show a diff summary and exact destructive-risk actions.
Commit rejects any update action not present in the accepted preview.

## Two-Phase Import Commit

Metadata and image bytes span different stores. Import must use an import journal,
not a naive loop.

Commit endpoint:

- `POST /api/v1/vn-assets/import/commit`

Commit request contains:

- `preview_id`
- `trust_mode`
- conflict decisions
- optional `target_pack_id` for update mode
- explicit confirmation token for update-existing risky diffs

Commit job stages:

1. Revalidate preview status, archive checksum, and canonical payload fingerprint.
2. Create import journal row with idempotency key.
3. Preflight storage quota using decoded byte totals.
4. Stage image bytes in per-user temp storage. V1 does not create pending
   generated-file records before item IDs exist.
5. Insert or update VN pack metadata in `ChaChaNotes.db`.
6. Insert slots and item rows using new local IDs. New imported item rows start in a
   temporary non-runtime-visible state with `generated_file_id=null`; source review
   state is stored in import metadata.
7. Register staged bytes as generated-file records with
   `source_ref="vn_asset_item:{new_item_id}"`.
8. Update item rows with generated-file IDs and apply final review state according
   to trust mode. Missing-byte items remain hidden.
9. Persist old-to-new ID maps.
10. Mark journal complete and expose the imported pack.

If failure happens after generated-file registration, the journal records created
generated-file IDs and runs best-effort cleanup. Retry can resume from the last safe
stage when possible. Cleanup can be retried through a maintenance endpoint or a
future `vn_pack_cleanup_import` job.

## Import Journal

The import journal stores:

- import ID
- owner user ID
- preview ID
- job ID
- archive checksum
- canonical payload fingerprint
- status
- current stage
- trust mode
- target mode: create or update
- created pack IDs
- created slot IDs
- created item IDs
- created generated-file IDs
- linked character/world-book IDs
- old-to-new ID maps
- warnings
- cleanup status
- started/completed timestamps

The journal is application-level recovery metadata. It is not a replacement for
database transactions inside one database.

## Generated-File Import Rules

Every imported image byte is registered as a new generated-file record:

- `source_feature="vn_assets"`
- `source_ref="vn_asset_item:{new_item_id}"`
- `folder_tag="vn-assets/{new_pack_id}"`
- tags include pack ID, item ID, asset type, slot key, and imported bundle
  fingerprint

The import job must avoid permanent records with source refs pointing to old source
item IDs. V1 avoids this by staging bytes in temp storage until new item IDs exist.
Generated-file registration happens only after item rows are created, and item rows
are finalized only after the matching generated-file record is active. If
registration succeeds but item finalization fails, the import journal records the
generated-file ID for cleanup or retry and the imported pack remains hidden from
normal runtime selection until recovery completes.

Quota checks:

- Preview estimates storage impact.
- Commit performs authoritative quota preflight.
- Each stored imported image counts against user storage quota.
- Quota failure aborts before durable commit where possible.

## Fingerprints And Checksums

Use two hash concepts:

- `archive_sha256`: exact hash of the final `.tldw-vnpack` bytes.
- `canonical_payload_fingerprint`: deterministic hash over canonical metadata and
  per-asset checksums.

Canonical payload fingerprint excludes:

- export timestamp
- export job ID
- download URL
- archive comments
- local database IDs where remappable
- local generated-file IDs
- non-semantic ordering

Canonicalization rules:

- JSON encoded as UTF-8.
- Object keys sorted.
- Lists sorted by stable semantic keys such as `slot_key`, `asset_type`,
  `variant_index`, `source_item_fingerprint`, and section path.
- Null and missing optional fields normalized by schema rules.
- Checksums lowercase hex SHA-256.

The archive checksum proves exact file integrity. The canonical fingerprint supports
conflict detection across re-exports.

## API Surface

Base path: `/api/v1/vn-assets`

Export:

- `POST /packs/{pack_id}/export`
- `GET /portability/exports/{job_id}`
- `GET /portability/exports/{job_id}/download`
- `POST /portability/exports/{job_id}/cancel`

Import preview:

- `POST /import/previews`
- `GET /import/previews/{preview_id}`
- `POST /import/previews/{preview_id}/cancel`
- `DELETE /import/previews/{preview_id}`

Import commit:

- `POST /import/commit`
- `GET /portability/imports/{job_id}`
- `POST /portability/imports/{job_id}/cancel`

Maintenance:

- `POST /portability/imports/{import_id}/cleanup`
- `GET /portability/jobs`

The exact naming can be adjusted during implementation to match existing route
patterns, but preview and commit must remain separate public operations.

## Data Model Additions

Tables in per-user `ChaChaNotes.db`:

- `vn_pack_portability_jobs`
- `vn_pack_import_previews`
- `vn_pack_import_journal`

`vn_pack_portability_jobs`:

- `id`
- `job_id`
- `owner_user_id`
- `operation`: `export`, `import_preview`, `import_commit`, `cleanup`
- `status`
- `stage`
- `pack_id`
- `preview_id`
- `import_id`
- `archive_path`
- `archive_sha256`
- `canonical_payload_fingerprint`
- `progress_json`
- `warnings_json`
- `error_code`
- `error_message`
- `download_url`
- `expires_at`
- `created_at`
- `updated_at`

`vn_pack_import_previews`:

- `id`
- `owner_user_id`
- `job_id`
- `status`
- `archive_path`
- `archive_sha256`
- `canonical_payload_fingerprint`
- `schema_version`
- `bundle_summary_json`
- `validation_warnings_json`
- `conflicts_json`
- `proposed_plan_json`
- `quota_estimate_json`
- `required_choices_json`
- `expires_at`
- `created_at`
- `updated_at`

`vn_pack_import_journal`:

- `id`
- `owner_user_id`
- `preview_id`
- `job_id`
- `status`
- `stage`
- `trust_mode`
- `target_mode`
- `target_pack_id`
- `archive_sha256`
- `canonical_payload_fingerprint`
- `id_maps_json`
- `created_records_json`
- `cleanup_status_json`
- `warnings_json`
- `error_code`
- `error_message`
- `created_at`
- `updated_at`
- `completed_at`

## WebUI

Export UX:

- Export action from pack detail/review page.
- Backup-bundle labeling.
- Toggles for character data, world-book data, and full provenance.
- Warnings for sensitive choices and no encryption.
- Job progress and download link.

Import UX:

- Upload `.tldw-vnpack`.
- Preview progress.
- Validation summary.
- Bundle summary.
- Conflict list with recommended actions.
- Required primary-character resolution.
- Trust mode selector.
- Update-existing diff view when selected.
- Commit progress and final imported pack link.

The UI should avoid presenting backup bundles as safe public sharing artifacts.

## Security

- AuthNZ enforced on every endpoint.
- All archive operations are per-user scoped.
- Archive paths are normalized and validated before any extraction or read.
- No raw archive path or raw generated-file path is persisted in public responses.
- Imported files are MIME-sniffed and image-decoded before storage.
- Per-file and total uncompressed limits are enforced during preview.
- Duplicate normalized archive members are rejected.
- Full prompt/source export requires explicit opt-in.
- Logs and audit events store IDs, counts, hashes, and warning codes rather than
  prompts, lore text, or raw image paths.
- Download URLs expire.
- Import preview archives expire and are cleaned up.
- Generated-file ownership is checked on export before reading bytes.

## Rollout Plan

### Phase 1: Format And Export

- Define archive schemas.
- Export pack metadata and all item states.
- Export image bytes and checksums.
- Redacted provenance.
- Jobs-backed export status and download.

### Phase 2: Async Preview

- Upload and validate `.tldw-vnpack`.
- Compute checksums and fingerprints.
- Decode images and estimate quota.
- Generate conflict plan and required choices.

### Phase 3: Import Commit

- Import journal.
- Create-new import mode.
- Generated-file remapping.
- Trusted and untrusted review-state behavior.
- Character resolution.

### Phase 4: Update Existing

- Non-destructive update-existing mode.
- Diff preview.
- Explicit confirmation for risky diffs.

### Phase 5: WebUI

- Export wizard.
- Import preview wizard.
- Conflict resolution.
- Job status and final pack navigation.

## Testing Strategy

Backend:

- Export contains required files and valid checksums.
- Export includes draft, rejected, hidden, and approved items.
- Export omits full prompts in redacted mode.
- Full provenance opt-in includes prompt snapshots after redaction checks.
- Character/world-book payloads are included only when requested.
- Archive validation rejects traversal, absolute paths, drive letters, null bytes,
  duplicate normalized paths, symlinks, unexpected top-level files, missing
  manifests, malformed JSON, checksum mismatches, and zip bombs.
- Preview is Jobs-backed and exposes progress.
- Preview detects required character choices.
- Preview conflict output is deterministic.
- Commit rejects expired or mutated previews.
- Trusted restore preserves review state and preferred markers.
- Untrusted import resets imported items to draft while preserving source review
  state in metadata.
- Import remaps pack, slot, item, character, world-book, and generated-file IDs.
- Imported generated files belong to the importing user and use
  `source_feature="vn_assets"`.
- Quota failures abort without orphaned active items.
- Journal cleanup handles failures after generated-file registration.
- Update-existing mode is non-destructive unless explicit diff confirmations are
  present.

Frontend:

- Export wizard warnings.
- Export job progress and download.
- Import upload and preview states.
- Conflict resolution controls.
- Required character resolution.
- Trust mode selector.
- Update-existing diff confirmation.
- Import job progress and final pack link.

Security:

- Bandit on touched backend paths.
- Tests for path traversal and unsafe archive members.
- Tests that logs and API responses do not expose raw prompt snapshots in redacted
  mode.

## Risks And Mitigations

- Full backup bundles can leak private images, hidden variants, lore, or prompt
  context. Mitigate with backup labeling, warnings, redacted provenance default,
  and explicit opt-ins for character/world-book/full provenance data.
- Import spans two storage systems. Mitigate with preview, quota preflight, import
  journal, idempotency keys, old-to-new maps, and cleanup retry.
- Update-existing can destroy local work. Mitigate with non-destructive defaults and
  explicit per-diff confirmation.
- Archive validation can be expensive for large packs. Mitigate with Jobs-backed
  preview, size limits, streaming validation, and progress reporting.
- Weak conflict detection can cause bad links. Mitigate with canonical
  fingerprints, strong checksum matches, and user-visible decisions for ambiguous
  conflicts.
- Future encryption/signatures could be hard to add if the format is too rigid.
  Mitigate by reserving encryption and signatures metadata in v1.

## Open Questions

- Should backup bundles eventually support password-based encryption, platform key
  encryption, or both?
- Should a future runtime/share profile export approved assets only from the same
  `.tldw-vnpack` format or use a distinct extension?
- Should import previews support partial item selection in v1, or only whole-pack
  import?
- What is the maximum default `.tldw-vnpack` uncompressed size for local desktop
  installs versus multi-user server installs?
