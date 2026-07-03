# VN Visual Identity Bridge Design

Date: 2026-07-02
Status: Approved for spec review
Owner: Codex brainstorming pass
Backlog: TASK-12090.2

## Summary

Add the next two Visual Identity Expression Packs stages:

1. Stage 11A: a VN generated-file import bridge that lets reviewed/generated VN image files become Visual Identity expression assets without duplicating upload logic.
2. Stage 11B: a stateless VN role/casting resolver that can resolve a character or persona as themselves, or as a role override, when a VN-style surface needs an expression asset.

The work builds on the Stage 10 chat runtime and the existing Visual Identity generated-file endpoint. It does not add a new VN generation workflow, full VN scene compositor, persisted cast tables, or route-level VN workbench UI.

Source reference: [TavernSprite SillyTavern expression guide](https://tavernsprite.com/blog/sillytavern-add-use-character-expressions-guide/)

## Goals

- Let VN-generated image files become Visual Identity expression assets through a VN-aware bridge.
- Keep the existing generic generated-file import endpoint as the backend API surface.
- Add asset-level provenance so the system can trace which VN asset workflow produced a Visual Identity asset.
- Preserve Visual Identity support for animated raster expression assets, including GIF, WebP, and capability-gated AVIF, when a valid generated-file source already provides those formats.
- Provide a reusable frontend import action/hook for future VN asset surfaces without creating route-level VN UI in this stage.
- Support partial success when asset import succeeds but draft slot assignment fails.
- Add a stateless resolver for future VN role/casting flows with strict ownership and override validation.
- Keep selected character/persona default Visual Identity bindings as the normal fallback when no role override is provided.

## Non-Goals

- Do not implement VN asset generation or prompt orchestration.
- Do not implement a full VN workbench UI.
- Do not implement full VN scene composition, backgrounds, blocking, CG/event layers, script timing, or multi-actor staging.
- Do not add persisted VN cast assignment tables in this stage.
- Do not replace the existing Visual Identity generated-file endpoint.
- Do not make personas fall back to character legacy mood images.
- Do not silently substitute unrelated packs when a strict role override is invalid or incomplete.

## Existing Context

The current repository already has the important building blocks:

- `Visual_Identities` owns packs, drafts, versions, assets, bindings, storage validation, generated-file import, asset serving, chat runtime resolver hooks, and animated raster image support.
- The generic endpoint `POST /api/v1/visual-identities/packs/{pack_id}/assets/from-generated-file` already imports from generated-file records into a Visual Identity pack.
- VN Assets own generated asset records under the `vn_assets` source feature and already model durable generated files for visual-novel workflows.
- Frontend Visual Identity client code already exposes generated-file import and draft slot update calls.
- The frontend route registry has VN routes, but no concrete VN asset workbench component was found during design review, so this stage must expose reusable bridge UI primitives rather than a new route-level screen.

The missing pieces are asset-level provenance, a VN-specific bridge wrapper around the generic import path, reusable UI behavior for import-then-assign, and a resolver contract for future role casting.

## Stage 11A: VN Generated-File Import Bridge

### Backend API Boundary

Keep this endpoint as the public import API:

```text
POST /api/v1/visual-identities/packs/{pack_id}/assets/from-generated-file
```

Extend its request/response schema and service path to support optional asset provenance:

```json
{
  "generated_file_id": 42,
  "expression_key": "happy",
  "draft_id": 7,
  "source_feature": "vn_assets",
  "idempotency_key": "vn-assets:42:pack:5:draft:7:happy",
  "source_context": {
    "source_feature": "vn_assets",
    "generated_file_id": 42,
    "filename": "maya_happy.webp",
    "vn_pack_id": 3,
    "vn_slot_id": 11,
    "vn_item_id": 29,
    "vn_asset_type": "sprite",
    "vn_slot_key": "happy",
    "vn_slot_label": "Happy",
    "source_ref": "vn_asset_item:29"
  }
}
```

The lower-level endpoint remains flexible enough for other generated-file sources. The VN bridge service defaults `source_feature` to `vn_assets` and rejects non-VN source features unless the caller explicitly opts into the generic path outside the bridge.

The VN bridge must treat client-provided provenance as a request hint, not as trusted authority. It always derives `source_feature`, `generated_file_id`, filename, MIME metadata, and generated-file source references from the generated-file record. If `vn_item_id` is present, the generated-file record must match the VN asset item source reference, such as `vn_asset_item:{vn_item_id}`. A mismatch fails before asset creation. VN pack, slot, and item ids are stored only after they are verified against VN data. Client-provided labels, such as `vn_slot_label`, are stored as display labels only and are not used as ownership or source-of-truth checks.

Use this backend module:

```text
tldw_Server_API/app/core/Visual_Identities/vn_bridge.py
```

The bridge stays small. It validates VN defaults, builds the provenance object, and delegates to the existing Visual Identity generated-file import service.

### Asset Provenance

Add asset-level provenance to `visual_identity_assets`:

```sql
source_context_json TEXT NOT NULL DEFAULT '{}'
```

Migration requirements:

- The migration is idempotent and checks whether the column exists before adding it.
- Existing rows receive `{}` and remain valid.
- Draft assets and activated version assets both carry `source_context_json`.
- Activation copies `source_context_json` from draft assets into version assets in both the SQL insert and the returned activation asset list.
- Version manifests preserve asset `source_context` where they serialize asset metadata.
- `VisualIdentityAssetResponse` includes `source_context`, including generated-file import responses and draft asset lists that the frontend needs to verify or display.

`source_context` is bounded metadata, not a hidden payload channel:

- root value must be a JSON object
- serialized UTF-8 size must be at most 8 KiB
- maximum nesting depth is 4
- maximum total keys is 50
- keys are strings of 1 to 64 characters
- scalar strings are at most 512 characters
- strings beginning with `data:` are rejected
- known prompt text keys are rejected: `prompt`, `negative_prompt`, `system_prompt`, `user_prompt`, and `prompt_text`
- short prompt references are allowed only under reference keys such as `prompt_id`, `prompt_ref`, or `prompt_label`
- binary data, base64-like payload strings, and full prompt text are rejected

The minimum useful provenance fields are `source_feature`, `generated_file_id`, and filename. VN-specific fields are optional because different generation paths may know different amounts of VN context.

### Idempotency

The generated-file import idempotency payload hash must include the validated, canonicalized `source_context` after server-derived fields are merged and rejected fields are removed.

If the same owner, scope, resource, and idempotency key are reused with a different `source_context`, the request conflicts rather than returning a stale response. This prevents two VN generation records from being accidentally collapsed into one Visual Identity asset assignment.

Canonicalization uses deterministic JSON key ordering and normalized scalar values so semantically equivalent metadata does not conflict only because object keys arrived in a different order.

### Frontend Reusable Import Action

Add a reusable hook rather than a route-level VN UI:

```text
useGeneratedFileImportAction
```

Inputs:

- `packId`
- `draftId`
- `slotKey`
- `generatedFileId`
- optional VN provenance fields
- optional client override for tests or host surfaces

Default behavior:

1. Call `createVisualIdentityAssetFromGeneratedFile` with `source_feature: "vn_assets"` and bounded `source_context`.
2. Call `updateVisualIdentityDraftSlot` with the created asset id.
3. Return a discriminated result to the host surface.

Result contract:

```ts
type GeneratedFileImportActionResult =
  | { status: "assigned"; assetId: number; slotKey: string }
  | { status: "imported_unassigned"; assetId: number; slotKey: string; error: unknown }
  | { status: "failed"; error: unknown };
```

If asset import succeeds and slot update fails, the action returns `imported_unassigned`. It does not auto-delete the created asset. The host can show a retry affordance or let the user manually assign the imported asset.

Stage 11A does not expand the current VN Asset upload/generation format matrix. VN Assets currently produce or accept only the formats supported by their own module. Visual Identity can still preserve GIF/WebP/AVIF animation when the generated-file record is already a valid image source accepted by the Visual Identity generated-file import path.

## Stage 11B: VN Role/Casting Resolver

### Resolver Shape

Add a stateless resolver for VN-like surfaces. It takes actor identity, optional role metadata, requested expression, and optional strict pack/version override. It does not persist cast assignments.

Stage 11B extends the existing resolver API instead of adding a separate VN-only route:

```text
GET /api/v1/visual-identities/bindings/resolve
```

The endpoint keeps the current actor/expression/manual/mood query parameters and adds optional query parameters for `role_id`, `role_label`, `override_pack_id`, `override_pack_version_id`, and `allow_override_fallback`. The response model extends `VisualIdentityResolveResponse` with `role_id`, `role_label`, and `resolution_source`.

Resolver input fields:

- `actor_kind`: `character` or `persona`
- `actor_id`
- `expression_key`
- optional `role_id`
- optional `role_label`
- optional `override_pack_id`
- optional `override_pack_version_id`
- optional `allow_override_fallback`

Resolver output fields:

- `actor_kind`
- `actor_id`
- `role_id`
- `role_label`
- `pack_id`
- `pack_version_id`
- `expression_key`
- `requested_expression_key`
- `asset_id`
- `asset_url`
- `content_type`
- `is_animated`
- `fallback_reason`
- `resolution_source`

### Resolution Order

Resolution order is deterministic:

1. Explicit role override pack/version, when provided.
2. Actor default Visual Identity binding for the selected character/persona.
3. Legacy fallback that is valid for that actor kind.
4. Neutral placeholder.

Character actors may use legacy character mood images only through the existing character fallback path. Persona actors do not fall back to character legacy mood images.

### Strict Overrides

Explicit pack/version override is strict by default:

- invalid actor returns a typed error
- cross-user pack/version returns a typed error
- mismatched pack/version returns a typed error
- valid override with missing requested/default expression returns `override_expression_missing`

When `allow_override_fallback=true`, the resolver may fall back within the override pack and then to the normal legacy/placeholder fallback path. The response must include `fallback_reason` and `resolution_source` so the caller can distinguish a true override match from a fallback.

The resolver must not silently substitute the actor default binding when an explicit override is invalid. Actor default binding is used only when no explicit override is supplied, or when override fallback is explicitly allowed after a valid override pack/version fails expression matching.

### Future Persisted Casting

This stage documents but does not implement persisted cast assignment storage.

A later persisted shape can bind a VN role to:

- `role_id`
- `role_label`
- `actor_kind`
- `actor_id`
- `pack_id`
- `pack_version_id`
- default expression mapping overrides
- creation/update metadata

The Stage 11B resolver stays stateless so VN surfaces can use the contract before committing to table ownership and lifecycle rules for saved cast assignments.

## Error Handling

Stage 11A import errors:

- missing generated file: no asset row is created
- wrong owner: no asset row is created
- non-image generated file: no asset row is created
- unsupported MIME or failed image validation: no asset row is created
- invalid `source_context`: no asset row is created
- VN source-ref mismatch: no asset row is created
- idempotency conflict: no new asset row is created
- slot update failure after successful import: asset remains and frontend returns `imported_unassigned`

Stage 11B resolver errors:

- `actor_not_found`
- `pack_not_found`
- `pack_not_owned`
- `pack_version_not_found`
- `pack_version_mismatch`
- `override_expression_missing`
- `expression_missing`
- `asset_missing`
- `asset_unrenderable`

Resolver responses must expose fallback reasons. Silent fallback to unrelated packs is not allowed.

## Testing Strategy

### Backend Stage 11A

- Idempotent migration adds `source_context_json` to existing `visual_identity_assets` rows and running it twice is safe.
- Existing rows receive `{}` and draft activation continues to work.
- Generated-file import records VN provenance on draft assets.
- Draft activation copies provenance to version assets.
- Same idempotency key with different `source_context` returns a conflict.
- Missing, wrong-owner, non-image, and invalid generated files create no asset rows.
- `source_context` validation rejects invalid root type, excessive size, excessive depth, excessive key count, long keys, long scalar strings, binary/base64 payloads, data URIs, and long prompt text.
- `source_context` validation rejects known prompt text keys, while allowing short prompt reference keys.
- VN bridge defaults `source_feature` to `vn_assets`.
- VN bridge rejects non-VN generated-file sources unless the generic import path is explicitly used.
- VN bridge rejects a generated-file record whose `source_ref` does not match the provided `vn_item_id`.
- Idempotency uses validated/canonicalized `source_context`; reordered equivalent objects replay, while materially different context conflicts.
- API response includes asset-level `source_context` on `VisualIdentityAssetResponse`, generated-file import responses, draft asset lists, and version/manifest asset metadata where exposed.

### Frontend Stage 11A

- Reusable import action/hook uses a fake client in tests.
- Test exact generated-file import payload, including `source_feature` and `source_context`.
- Test exact draft slot update payload.
- Test `assigned` result on full success.
- Test `imported_unassigned` result when import succeeds and slot update fails.
- Test `failed` result when import fails.
- Do not add route-level VN workbench tests in this stage.

### Backend Stage 11B

- Actor default binding resolves without role override.
- Existing `/bindings/resolve` query parameters keep working when no Stage 11B override fields are supplied.
- Explicit override resolves the requested expression.
- Valid override missing the requested/default expression returns `override_expression_missing` by default.
- `allow_override_fallback=true` permits fallback and records `fallback_reason`.
- Invalid actor fails with a typed error.
- Cross-user pack/version fails with a typed error.
- Mismatched pack/version fails with a typed error.
- Persona actor resolution does not use character legacy mood image fallback.

### Verification

- Run focused pytest coverage for Visual Identity DB migration, generated-file import, VN bridge behavior, and resolver behavior.
- Run focused Vitest coverage for the new frontend reusable import action/hook.
- Run Bandit on touched backend Visual Identity paths, endpoint schemas, and endpoint files.
- Run TypeScript checks for the frontend package. If the full repository baseline is noisy, record that no diagnostics match new or touched Stage 11 frontend files.

## Rollout

Implement in two stages:

1. Stage 11A: add asset provenance, VN bridge helper, request/response/client type extensions, reusable frontend import action, and tests.
2. Stage 11B: add stateless role/casting resolver contracts, strict override behavior, fallback reasons, and tests.

Stage 11A must land before Stage 11B because role/casting resolution needs provenance-rich assets and a stable path for VN-generated files to enter Visual Identity packs.

## Risks And Mitigations

- Risk: provenance becomes an unbounded storage channel. Mitigation: strict JSON object limits and rejection of binary/base64/prompt payloads.
- Risk: import succeeds but assignment fails, leaving orphan-looking assets. Mitigation: explicit `imported_unassigned` result and retry/manual assignment path.
- Risk: idempotency hides provenance changes. Mitigation: include `source_context` in the idempotency payload hash.
- Risk: role override silently falls back to the wrong character/persona pack. Mitigation: strict override errors by default and typed fallback only when opted in.
- Risk: future VN UI needs a different surface. Mitigation: ship reusable client/action primitives instead of hard-coding a route-level workbench component now.
