# Persona Visual Pack Duplicate-To-Persona Design

Date: 2026-05-09
Status: Approved by spec review; pending user review
Owner: Codex brainstorming pass
Backlog: TASK-192
GitHub: #1449, #1450

## Summary

Add a focused Persona/Buddy visual-pack workflow that lets a user duplicate an
existing Persona Visual pack from one of their personas to another of their
personas as a draft.

The duplicate is a real pack attached to the target persona. It owns its own
asset rows and copied asset files, has a remapped manifest that references the
new asset IDs, and is not activated automatically. The source pack, source
persona, target persona active pack, and existing import/export behavior remain
unchanged.

This is Phase 3 library/reuse groundwork from the Persona Live Visual Packs
PRD. It should reuse the existing Persona Visual portability model where useful,
especially asset-byte copying and manifest asset-reference remapping, without
turning the first duplicate action into a background import/export archive flow.

## Goals

- Let users reuse an existing Persona Visual pack on another same-user persona.
- Preserve the one-persona-default attachment model by creating a distinct
  target pack with distinct target asset records.
- Copy asset bytes into the target persona/pack storage path so the duplicate
  does not depend on source-pack storage cleanup behavior.
- Remap all manifest asset references to the duplicated asset IDs.
- Create the duplicate as `draft` and require normal review/activation.
- Preserve same-user lineage from the duplicate back to the source pack.
- Keep the workflow in Persona Garden/Visuals and Buddy/persona language.
- Provide enough backend, frontend, error, and test coverage for a small
  implementation PR.

## Non-Goals

- Do not support cross-user sharing.
- Do not add a public marketplace or shared library browser.
- Do not auto-activate the duplicated pack.
- Do not add Live2D renderer support or change renderer contracts.
- Do not add VN/CYOA asset-pack behavior or UI.
- Do not expose external MCP-compatible visual providers.
- Do not build a general archive import/export replacement.
- Do not copy generated candidates unless they are already accepted pack assets.
- Do not support same-persona "make a draft copy" duplication in V1.
- Do not add idempotency keys in V1; duplicate is a synchronous command and a
  repeated submit creates another draft.

## Existing Context

Current Persona Visual support already includes:

- Persona-scoped visual packs and assets in the ChaChaNotes persona data layer.
- `PersonaVisualService` for upload validation, file placement, candidate
  review, activation, and active-pack validation.
- API routes under
  `/api/v1/persona/profiles/{persona_id}/visual-packs`.
- Frontend services/types in `apps/packages/ui/src/services/persona-visuals.ts`
  and `apps/packages/ui/src/types/persona-visuals.ts`.
- A Persona Garden Visual Pack editor in
  `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`.
- Buddy shell rendering and diagnostics in
  `apps/packages/ui/src/components/Common/PersonaBuddy/`.
- Import/export Jobs for `.tldw-persona-vpack` archives, including an import
  commit path that creates draft packs and remaps asset references.

Important current details:

- `persona_visual_packs.parent_pack_id` already exists.
- Pack creation currently validates `parent_pack_id` as same persona and same
  user.
- Allowed visual provenance values are `uploaded`, `generated`, `imported`,
  and `mixed`.
- Existing import commit code copies asset bytes through
  `PersonaVisualService.create_asset_from_upload()` and remaps manifest fields:
  `frames[].asset_id`, `asset_ids[]`, and `preview_asset_id`.
- Existing exporter resolves `persona_visuals/...` storage keys under
  `DatabasePaths.get_user_persona_visuals_dir(user_id)` and guards path escape.

## User-Confirmed Product Rules

1. Duplicates should use the physical-copy model, not metadata-only shared
   storage keys.
2. Duplicates should preserve same-user cross-persona lineage where possible.
3. The first implementation target is duplicate-to-persona as draft.
4. This remains Persona/Buddy visual-pack work, not VN/CYOA runtime work.

## Recommended Approach

Implement a synchronous same-user duplicate operation in the Persona Visual
service and expose it through one API route from the source pack.

The service should:

1. Validate source pack ownership by `source_persona_id`, `pack_id`, and
   `user_id`.
2. Validate target persona ownership by `target_persona_id` and `user_id`.
3. Create a target draft pack with a copied or user-provided title.
4. Copy each source asset referenced by the source manifest into the target
   storage path.
5. Create target asset rows with new IDs and copied metadata.
6. Build an old-asset-ID to new-asset-ID map.
7. Remap the source visual manifest through that map.
8. Validate the remapped manifest with `require_activatable=False`.
9. Save the remapped manifest on the target draft pack.
10. Return the target draft pack with assets.

This approach is recommended because it keeps the user model simple: a duplicated
pack is now the target persona's pack and is independent from the source pack.
It costs more disk space than shared storage keys, but avoids cleanup coupling,
cross-persona storage paths, and surprising breakage if the source pack is later
deleted or repaired.

## Alternatives Considered

### Shared Storage-Key Copy

The service could create new target pack and asset rows while keeping the source
asset `storage_key` values.

This is smaller on disk, but it breaks the "attached to one persona by default"
mental model. It also means the target persona's pack depends on files stored in
the source persona/pack directory, which makes future deletion, repair, and
export diagnostics harder.

### Export/Import Loopback

The UI could export a pack archive and immediately import it into another
persona through existing Jobs.

This reuses the portability stack most completely, but it is too heavy for an
in-app duplicate action. It would introduce staging archives, asynchronous job
state, and import-preview language for a workflow that should feel like copying
a draft.

## API Design

Add a route under the source pack:

```text
POST /api/v1/persona/profiles/{source_persona_id}/visual-packs/{pack_id}/duplicate
```

Request body:

```json
{
  "target_persona_id": "persona_target",
  "title": "Optional copied title"
}
```

Response:

```json
{
  "id": "new_pack_id",
  "persona_id": "persona_target",
  "status": "draft",
  "parent_pack_id": "source_pack_id",
  "assets": []
}
```

Use the existing `PersonaVisualPackResponse` shape. Do not expose
`asset_id_map` in the public response. The service may return the map internally
for tests and diagnostics, but client code should rely on the returned pack and
its remapped manifest/assets.

The endpoint should require normal persona auth and use the current authenticated
user. It must not accept a source or target user ID from the client.

## Persistence and Lineage

The duplicate should create a new `persona_visual_packs` row:

- `persona_id`: target persona
- `user_id`: current user
- `status`: `draft`
- `renderer_type`: copied from source
- `manifest_version`: copied from validated remapped manifest
- `manifest_json`: remapped source manifest
- `parent_pack_id`: source pack ID when same-user cross-persona parent lineage
  is allowed
- `revision_number`: `1`
- `provenance`: `mixed`

Current `create_persona_visual_pack()` validates `parent_pack_id` against the
same persona. The implementation should either:

1. add a narrow DB helper/path for duplicate creation that validates the parent
   pack as same user while allowing a different persona, or
2. extend parent validation with an explicit same-user cross-persona mode that is
   only used by the duplicate service.

The implementation should not weaken the default pack-creation path. Ordinary
pack creation should still avoid arbitrary parent references.

## Asset Selection and Copying

The duplicate should copy only source assets referenced by the source manifest.
This prevents unaccepted generated candidates, stale uploads, and rejected review
artifacts from leaking into the target pack. The duplicate service should collect
source asset IDs from:

- `animations.*.frames[].asset_id`
- `animations.*.asset_ids[]`
- `animations.*.preview_asset_id`

If the source manifest references an asset ID that does not belong to the source
pack, the duplicate should fail with `invalid_manifest` before creating target
records.

Referenced source assets should be physically copied into the target pack's
storage path. The implementation can do this by reading the source asset file,
then passing the bytes back through `PersonaVisualService.create_asset_from_upload()`
with:

- `persona_id`: target persona
- `pack_id`: target pack
- `asset_role`: copied source role
- `mime_type`: copied source MIME type
- `original_filename`: copied source original filename
- `provenance`: `mixed`

This reuses upload image validation, checksum calculation, safe storage-target
construction, and DB asset creation. The implementation should preflight source
manifest validity, referenced asset membership, source asset file existence, and
source asset checksums before creating target records. If using
`create_asset_from_upload()` makes all-or-nothing cleanup difficult, add a
service-private copy helper that preserves the same image validation and storage
safety properties while keeping target metadata creation atomic.

Source asset path resolution should follow the exporter pattern:

- Strip the `persona_visuals/` prefix if present.
- Resolve under `DatabasePaths.get_user_persona_visuals_dir(user_id)`.
- Reject paths that escape that base.
- Treat missing files as a failed duplicate, not as a partial success.

If any asset copy or manifest update fails after target files are written, the
service should remove created target asset files where practical and leave no
active-pack changes. The preferred outcome is no visible target draft. If the
existing DB layer makes full rollback impractical, the service must mark the
target pack `failed` rather than leaving a partial `draft`.

## Manifest Remapping

The duplicate must remap every source asset ID referenced by the manifest to its
new target asset ID. At minimum this includes the existing import-remap fields:

- `animations.*.frames[].asset_id`
- `animations.*.asset_ids[]`
- `animations.*.preview_asset_id`

The implementation should avoid duplicating this traversal logic. The current
import helper `_remap_visual_manifest_assets()` can be promoted to a shared
module under `tldw_Server_API/app/core/Persona/` or
`visual_portability/` if that keeps dependencies clean.

After remapping, validate the manifest with:

- all target asset IDs as `available_asset_ids`
- target asset dimensions as `available_asset_dimensions`
- `require_activatable=False`

The duplicate is a draft, so it may be incomplete, but it must not reference
missing source asset IDs after remap.

## Frontend Workflow

Add a small duplicate affordance in the Visual Pack editor. The preferred shape
is a command near existing export/import pack actions:

- "Duplicate to persona"
- target persona selector that excludes the source persona
- optional title field defaulting to `Copy of {source title}`
- confirmation copy that says the result is a draft and will not replace the
  target persona's active pack

On success:

- show the target draft pack title/status
- refresh pack data
- if the editor can switch personas cleanly, move the user to the target persona
  and select the new draft pack
- otherwise provide a clear link/action to open the target persona's Visuals
  editor

The UI copy should avoid library/marketplace language in this slice. It should
say this copies a pack to another of the user's personas for review.

## Error Handling

Expected failures:

- source pack not found or not owned by user: `404`
- target persona not found or not owned by user: `404`
- target persona equals source persona: `400` with a stable code such as
  `same_persona_target_unsupported`
- source pack has no assets: allowed only if the manifest validates as a draft
  without assets; otherwise return a validation error
- source manifest references unowned or missing asset rows: `400` with
  `invalid_manifest`
- source asset file missing: conflict or validation error with a stable code
  such as `source_asset_missing`
- source asset checksum mismatch: conflict or validation error with a stable
  code such as `source_asset_checksum_mismatch`
- remapped manifest invalid: `400` with `invalid_manifest`

The operation must never archive or activate packs as part of failure handling.

## Testing

Backend service tests should cover:

- duplicate creates a target draft pack for another same-user persona
- target asset IDs differ from source asset IDs
- target storage keys point at the target persona/pack path
- source and target asset checksums match
- manifest references are remapped for `frames`, `asset_ids`, and
  `preview_asset_id`
- unreferenced source assets and unaccepted generated candidates are not copied
- source active pack and target active pack remain unchanged
- same-persona duplicate is rejected with the stable V1 error code
- missing source asset rows or files fail before a usable target draft appears
- missing source file fails without activating anything
- cross-user target/source access fails

API tests should cover:

- successful duplicate response shape
- `404` for missing source pack
- `404` for missing or unauthorized target persona
- error mapping for invalid source assets or manifests

Frontend tests should cover:

- duplicate action opens target selection
- submit calls the duplicate service with source persona, source pack, and
  target persona
- source persona is not offered as a target
- success state communicates draft/review status
- active-pack copy does not say it replaces the target active pack

No E2E browser flow is required for the first implementation unless the UI
change materially alters navigation.

## Documentation

Update the Persona Live Visual Packs PRD or the relevant WebUI help copy to note
that Phase 3 duplicate-to-persona is implemented once the code lands.

Implementation docs should explicitly preserve the distinction between:

- duplicate-to-persona: same-user local copy into a target draft
- import/export: archive-based portability
- shared library: future work
- external providers: future MCP-compatible work

## Open Questions for Implementation Planning

No product-level open questions remain for the first implementation plan.
Implementation planning may still choose the exact private helper boundaries for
asset copy atomicity.

## Success Criteria

- The spec can be converted into a small implementation plan without revisiting
  the product model.
- The implementation can be reviewed as one focused PR.
- The duplicate workflow advances Persona/Buddy visual-pack reuse without
  introducing shared-library or marketplace semantics early.
