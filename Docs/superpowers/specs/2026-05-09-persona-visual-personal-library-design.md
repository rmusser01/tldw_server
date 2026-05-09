# Persona Visual Personal Library Design

## Status

Design for GitHub issue #1468 under epic #1449.

This design covers the first personal library layer for Persona/Buddy visual
packs. It builds on PR #1467, which added same-user duplicate-to-persona as a
reviewed draft workflow.

## Problem

Persona Visual Packs are currently attached to one persona by default. Users can
export a pack, import an archive, or duplicate one pack directly to another
persona, but there is no durable personal catalog of reusable packs. That makes
reuse depend on remembering which persona has the pack and navigating into that
persona first.

The next slice should make reuse discoverable without turning Persona Garden
into a public marketplace or introducing a second asset ownership model.

## Goals

1. Let a user save an existing Persona Visual pack into a personal library.
2. Let the user browse saved reusable packs across their own personas.
3. Let the user apply a saved library entry to another same-user persona as a
   draft through the existing duplicate-to-persona semantics.
4. Preserve explicit review and activation. Saving or using a library entry must
   never silently change an active Buddy renderer.
5. Keep the library user-scoped and persona-aware so future import/export,
   duplication, or shared-library work can extend the format without rewriting
   the core pack model.

## Non-Goals

1. No cross-user publishing.
2. No shared community marketplace.
3. No organization-wide library.
4. No archive-backed snapshots in the first slice.
5. No asset deduplication or detached asset ownership model.
6. No automatic activation after applying a library entry.
7. No Live2D, external visual provider, VN Play, or CYOA runtime work.

## Recommended Approach

Use a reference-backed personal library.

A library entry is user-owned metadata pointing at an existing user-owned
`persona_visual_packs` row. The source pack remains attached to its original
persona. The library entry does not copy files, create a new manifest, or claim a
separate ownership boundary.

When the user chooses "Use for persona," the backend loads the referenced source
pack and calls the same duplication workflow used by PR #1467. The result is a
draft pack on the target persona. The target persona's active pack is unchanged.

This approach is intentionally smaller than a full reusable asset store:

1. It gives users the product affordance they need now: "I saved this pack for
   reuse."
2. It keeps all asset validation, checksum checks, manifest remapping, and
   draft creation in one existing path.
3. It leaves room for a later archive-backed snapshot or shared library without
   changing `persona_visual_packs` or asset storage in this first slice.

## Data Model

Add a new table owned by the ChaChaNotes persona store:

```sql
CREATE TABLE persona_visual_library_items (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  source_persona_id TEXT NOT NULL,
  source_pack_id TEXT NOT NULL,
  title TEXT NOT NULL,
  notes TEXT,
  tags_json TEXT NOT NULL DEFAULT '[]',
  source_pack_version INTEGER,
  created_at DATETIME NOT NULL,
  last_modified DATETIME NOT NULL,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1,
  FOREIGN KEY(source_persona_id) REFERENCES persona_profiles(id),
  FOREIGN KEY(source_pack_id) REFERENCES persona_visual_packs(id)
);
```

Indexes:

1. `(user_id, deleted, last_modified)` for library listing.
2. A unique active-entry index for `(user_id, source_persona_id,
   source_pack_id)` where `deleted = false`, or the closest existing backend
   equivalent, to prevent duplicate saved entries for the same live pack.
3. Optional `(user_id, title)` for simple filtering once the UI adds search.

`source_pack_version` records the pack version when the item was saved. Because
the first slice is reference-backed, applying the item uses the current source
pack state, not a snapshot. If the source pack version has changed since save,
the API should surface `source_changed: true` so the UI can show a small status
label.

## Library Item Shape

API responses should include:

```json
{
  "id": "library-item-id",
  "source_persona_id": "persona-a",
  "source_pack_id": "pack-a",
  "source_pack_version": 3,
  "source_current_version": 5,
  "source_changed": true,
  "source_available": true,
  "source_persona_name": "Research Buddy",
  "source_pack_title": "Warm desk assistant",
  "title": "Warm desk assistant",
  "notes": "Good default for focused research sessions.",
  "tags": ["research", "calm"],
  "created_at": "...",
  "last_modified": "...",
  "version": 1
}
```

`user_id` should remain an ownership field in persistence and service logic. The
API does not need to echo it back to the WebUI because all returned entries are
already scoped to the authenticated/current user.

If the source persona or pack was deleted, the item can still be listed with
`source_available: false`, but "Use for persona" must be disabled and the
backend must reject the apply request. This keeps users able to remove stale
library entries.

## Backend API

Add endpoints under the existing persona API namespace:

1. `GET /api/v1/persona/visual-library`
   - Lists current user's non-deleted library entries.
   - Supports later query parameters such as `q`, `tag`, `limit`, and `offset`,
     but V1 can ship with pagination only if existing endpoint patterns require
     it.

2. `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/library`
   - Saves a pack to the user's personal library.
   - Requires the source persona and pack to belong to the current user.
   - If the pack is already saved, return the existing entry or update the
     metadata depending on request body. The recommended V1 behavior is an
     idempotent upsert keyed by user/source persona/source pack.

3. `PATCH /api/v1/persona/visual-library/{item_id}`
   - Updates library metadata only: title, notes, tags.
   - Does not mutate the source pack or assets.

4. `DELETE /api/v1/persona/visual-library/{item_id}`
   - Soft-deletes the library entry.
   - Does not delete the source pack or assets.

5. `POST /api/v1/persona/visual-library/{item_id}/use`
   - Request body: `target_persona_id`, optional `title`.
   - Validates that the library item belongs to the current user.
   - Validates that the referenced source persona and pack still exist and
     belong to the current user.
   - Calls `PersonaVisualService.duplicate_pack_to_persona(...)`.
   - Returns the created `PersonaVisualPackResponse`.

The `use` endpoint should preserve the existing V1 rule that duplicate-to-persona
targets a different persona. If same-persona reuse becomes important later, that
should be a separate product decision because it changes the semantics from
"reuse across personas" to "clone a draft into the same persona."

## Backend Service Boundary

Add a small service layer beside `PersonaVisualService`, for example
`PersonaVisualLibraryService`.

Responsibilities:

1. Normalize library metadata.
2. Own idempotent save/upsert behavior.
3. Compose library rows with source persona/pack display metadata.
4. Enforce item ownership before update/delete/use.
5. Delegate actual pack copy to `PersonaVisualService.duplicate_pack_to_persona`.

Do not duplicate manifest remapping, asset copying, checksum validation, or draft
status logic in the library service.

## WebUI Flow

Add library affordances in the existing Visuals editor rather than introducing a
new top-level marketplace page.

V1 UI:

1. In the selected pack header or portability area, add "Save to library."
2. Show saved state for the selected pack:
   - unsaved
   - saved
   - source changed since save
3. Add a "Personal library" panel in the Visuals tab. It lists saved entries with
   title, source persona, source pack, tags, and source availability.
4. Each entry supports:
   - "Use for persona" with a target persona select.
   - "Edit details" for title, notes, tags.
   - "Remove from library."
5. "Use for persona" creates a draft and shows the same handoff affordance as PR
   #1467: open the target persona's Visuals tab.

The copy should stay explicit:

1. "Save to library" means save this pack as reusable.
2. "Use for persona" means create a draft copy for another persona.
3. "Activate" remains a separate action on the target persona.

## Error Handling

Stable service/API error codes should include:

1. `library_item_not_found`
2. `source_pack_not_found`
3. `source_pack_unavailable`
4. `target_persona_not_found`
5. `same_persona_target_unsupported`
6. `invalid_library_metadata`
7. `library_item_conflict`

HTTP mapping:

1. 404 for missing item/source/target not owned by user.
2. 409 for stale or unavailable source pack states.
3. 422 for malformed title, notes, tags, or target payload.

Source-unavailable entries should still be removable from the library.

## Migration Compatibility

The migration must be additive:

1. Existing `persona_visual_packs` rows remain unchanged.
2. Existing duplicate, import, export, activation, and MCP paths do not need to
   know about library entries.
3. Deleting a source pack should not cascade-delete library entries in V1. It
   should make entries unavailable so users can see and remove stale references.
4. Future archive-backed snapshots can add a nullable snapshot/archive pointer
   to library entries or introduce a parallel item type without changing the
   source-pack reference behavior.

## Testing

Backend unit/integration tests:

1. Saving a pack creates a user-owned library entry.
2. Saving the same pack twice is idempotent or updates metadata according to the
   chosen upsert behavior.
3. Listing entries includes source persona and pack display metadata.
4. Deleted source pack/persona entries list as unavailable and cannot be used.
5. Updating/removing a library entry affects only the entry, not the source pack.
6. Using a library entry creates a draft on the target persona via duplicate
   semantics and preserves active pack state on both personas.
7. Cross-user access to entries, source packs, and targets is rejected.

Frontend tests:

1. VisualPackEditor renders save-to-library state for selected pack.
2. Personal library panel lists entries and unavailable states.
3. Use-for-persona sends the expected target payload and shows draft handoff.
4. Remove entry updates the panel without deleting the pack.
5. Source-changed and source-unavailable copy is visible when returned by API.

Security checks:

1. Run Bandit on touched backend modules.
2. Confirm no library endpoint exposes entries across users.
3. Confirm delete is soft-delete of library metadata only.

## Documentation Updates

Update:

1. `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
   - Mark personal library as the next Phase 3 slice.
   - Clarify reference-backed V1 behavior and non-goals.
2. `Docs/Code_Documentation/Persona_Visual_Packs.md`
   - Add a "Personal Library" section.
   - Explain that save-to-library is metadata only in V1.
   - Explain that use-for-persona creates a draft via duplicate semantics.

## Rollout

Recommended implementation order:

1. DB/schema helpers and tests for library item persistence.
2. Library service tests and implementation.
3. API schemas/endpoints and integration tests.
4. Frontend service/types and VisualPackEditor library panel tests.
5. Documentation updates and final focused verification.

This should be one focused feature PR if kept to reference-backed V1 behavior.
If archive snapshots, shared libraries, or same-persona cloning are added, split
them into separate issues and PRs.
