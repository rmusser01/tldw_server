# VN Play Setup Options Design

Status: Draft
Date: 2026-05-09
Owner: Core/WebUI maintainers
Scope: Usable VN Play session creation with backend-computed setup options
Tracking: https://github.com/rmusser01/tldw_server/issues/1407

## Summary

Add a backend setup-options endpoint for VN Play session creation. The endpoint
aggregates minimal character selector data, bounded VN asset pack selector data,
pack readiness, compatibility, trust provenance, content-rating notes, and warning
severity into one contract the WebUI can render directly.

The first implementation keeps `POST /api/v1/vn-play/sessions` compatible with the
existing `VNPlaySessionCreate` contract. Setup options are advisory and
user-facing. The WebUI should warn but allow problematic asset pack choices, with
an explicit acknowledgement step only for high-risk warnings.

## Goals

- Let users create VN Play sessions without knowing raw character or asset pack
  database IDs.
- Centralize setup compatibility and readiness logic in the backend instead of
  duplicating it in the frontend.
- Preserve the existing session creation payload shape.
- Keep unready or incompatible packs visible and selectable, with clear reasons.
- Require WebUI acknowledgement for high-risk setup warnings.
- Keep setup queries bounded so users with many characters or packs do not trigger
  unbounded readiness evaluation.
- Provide empty and failure states that direct users to character creation, asset
  pack review, import, or manual ID fallback.
- Preserve the selected character label even when pagination or search would
  otherwise omit it from the current character page.

## Non-Goals

- No realtime image generation.
- No new VN turn/runtime mechanics.
- No story branch graph authoring.
- No large VN asset pack editor redesign.
- No backend hard block for warned setup choices in V1.
- No full character response or image payload embedded in the setup-options
  response.

## Existing Project Context

The current `/vn-play` setup dialog uses numeric `primary_character_id` and
`vn_asset_pack_id` fields. This is functional for early development, but not
usable as the normal entry point.

Relevant existing surfaces:

- `GET /api/v1/characters/` and `GET /api/v1/characters/query` return character
  records.
- `GET /api/v1/vn-assets/packs` lists VN asset packs.
- `GET /api/v1/vn-assets/packs/{pack_id}/readiness` evaluates runtime readiness.
- `POST /api/v1/vn-play/sessions` already creates a session with
  `primary_character_id`, `vn_asset_pack_id`, mode, title, rating, linked chat,
  trust level, and settings.

The new setup endpoint should reuse the same per-user `ChaChaNotes.db` boundary
as characters, VN assets, and VN Play. It should not read or serve image bytes.

Current VN asset pack list helpers return all non-deleted packs. This design
requires adding a paginated/searchable repository or service method for setup
options before computing readiness. A compliant implementation must not fetch all
packs into Python and then slice the result.

## Design Choices

### Backend Aggregated Setup Contract

Use a new endpoint:

```text
GET /api/v1/vn-play/setup-options
```

This endpoint returns selector data and compatibility/readiness summaries in one
response. The frontend should not independently decide whether a pack is
compatible or high risk.

### Warn But Allow

VN Play setup should not hide or permanently block problematic packs. Users may
choose an incompatible or unready pack after seeing the issue. This matters for
self-hosted experimentation, imported packs, and partially prepared assets.

The WebUI may still make the recommended path easier by sorting compatible ready
packs first and visually de-emphasizing high-risk choices.

### High-Risk Acknowledgement

High-risk warnings require an explicit WebUI acknowledgement before submit. This
is a UX guard, not a new backend policy gate in V1.

The frontend can store acknowledgement metadata in the existing session
`settings` object, for example:

```json
{
  "setup_acknowledgements": {
    "warning_codes": ["pack_character_mismatch"],
    "acknowledged_at": "2026-05-09T00:00:00Z"
  }
}
```

The backend should accept the normal session creation payload without requiring
this metadata.

The WebUI must persist acknowledgement metadata whenever a session is created
after accepting high-risk setup warnings. The metadata should include the warning
codes and enough snapshot context to debug the decision later, such as selected
character ID, selected pack ID, readiness status, and timestamp. This is still not
a backend gate in V1; API clients that bypass setup-options remain compatible.

### Minimal Character Metadata

The setup response should expose only selector-safe character fields:

- `id`
- `name`
- `description_preview`
- `tags`
- `favorite`, defaulting to false if not cheaply available
- `deleted`, defaulting to false because normal setup should exclude deleted
  characters
- `has_image`

Do not embed full prompts, greetings, full image base64, or private character
payload fields in the setup response.

### Bounded Summary Mode

The endpoint returns bounded pages for characters and asset packs. It computes
readiness and warnings only for the returned pack page.

Default limits should be conservative, for example:

- `character_limit=25`
- `pack_limit=25`
- maximum `character_limit=100`
- maximum `pack_limit=100`

The response includes separate pagination metadata for characters and asset
packs. This avoids one large selector call becoming a slow readiness sweep.

Pack filtering, sorting, limit, and offset must happen at the repository/service
query boundary. The implementation can use a `LIMIT pack_limit + 1` query to
derive `has_more`, or return an exact total if the repository can do that cheaply.
Readiness fanout happens only for the final returned pack rows.

## API Contract

### Query Parameters

```text
GET /api/v1/vn-play/setup-options
  ?mode=story
  &character_query=mira
  &pack_query=library
  &character_limit=25
  &character_offset=0
  &pack_limit=25
  &pack_offset=0
  &selected_character_id=42
  &content_rating=general
```

Parameters:

- `mode`: optional `freeform` or `story`; used for defaults and future mode
  specific warnings.
- `character_query`: optional search text for character selector pages.
- `pack_query`: optional search text for asset pack selector pages.
- `character_limit` / `character_offset`: bounded character pagination.
- `pack_limit` / `pack_offset`: bounded pack pagination.
- `selected_character_id`: optional current character selection. When present,
  pack compatibility and sort hints should be computed against this character.
- `content_rating`: optional intended session content rating. When present,
  content-rating mismatch warnings should be computed against this rating.

### Response Shape

```json
{
  "characters": [
    {
      "id": 42,
      "name": "Mira",
      "description_preview": "Archivist with a careful eye...",
      "tags": ["sci-fi", "guide"],
      "favorite": true,
      "deleted": false,
      "has_image": true
    }
  ],
  "selected_character": {
    "id": 42,
    "name": "Mira",
    "description_preview": "Archivist with a careful eye...",
    "tags": ["sci-fi", "guide"],
    "favorite": true,
    "deleted": false,
    "has_image": true
  },
  "asset_packs": [
    {
      "id": 7,
      "title": "Mira - Archive Pack",
      "primary_character_id": 42,
      "content_rating": "general",
      "status": "ready",
      "trust_level": "local",
      "trust_source": "local_pack",
      "ready": true,
      "readiness_status": "ready",
      "readiness_warnings": [],
      "readiness_errors": [],
      "compatibility": {
        "status": "compatible",
        "reason_codes": []
      },
      "warning_summary": {
        "highest_severity": "info",
        "requires_acknowledgement": false,
        "warnings": []
      },
      "recommended": true
    }
  ],
  "defaults": {
    "mode": "story",
    "character_id": 42,
    "asset_pack_id": 7,
    "content_rating": "general"
  },
  "pagination": {
    "characters": {
      "limit": 25,
      "offset": 0,
      "has_more": false,
      "total": null
    },
    "asset_packs": {
      "limit": 25,
      "offset": 0,
      "has_more": true,
      "total": null
    }
  },
  "empty_states": [
    {
      "code": "no_compatible_packs",
      "scope": "page",
      "message": "No compatible packs were found in this page of results."
    }
  ],
  "generated_at": "2026-05-09T00:00:00Z"
}
```

### Pagination Object

Character and pack pagination metadata should use the same object shape:

- `limit`: effective bounded page size after applying server maximums.
- `offset`: effective zero-based offset.
- `has_more`: true when another page is known to exist.
- `total`: exact total for the current selector query when cheap to compute,
  otherwise null.

When `total` is null, the implementation should still derive `has_more` from a
bounded overfetch such as `LIMIT limit + 1`. It must not mark `has_more=false`
just because the total count was skipped.

### Warning Object

Warnings should be typed and stable enough for frontend tests:

```json
{
  "code": "pack_character_mismatch",
  "severity": "high_risk",
  "message": "This pack was generated for a different primary character.",
  "requires_acknowledgement": true
}
```

Allowed severities:

- `info`
- `warning`
- `high_risk`

Initial warning codes:

- `pack_character_mismatch`: pack primary character differs from selected
  character. High risk.
- `pack_not_ready`: pack readiness is not ready. High risk.
- `pack_has_readiness_errors`: readiness response contains errors. High risk.
- `pack_missing_required_assets`: required runtime assets are missing or lack
  approved variants. High risk.
- `content_rating_mismatch`: pack rating differs from intended session rating.
  High risk when the pack rating is more permissive or unknown; warning otherwise.
- `pack_untrusted_import`: pack was last committed from an untrusted import
  flow. Warning.
- `pack_deleted_or_archived`: pack is hidden from normal use. High risk.
- `readiness_unavailable`: readiness could not be computed for this pack. Warning.

Implementation may add warning codes later, but existing codes must remain stable
once shipped.

### Content Rating Comparison

Use a small ordered baseline for known labels:

```text
general < suggestive < mature < violent
```

Custom, missing, or unknown labels cannot be reliably ordered. If the pack rating
differs from the requested session rating and either side is unknown, emit
`content_rating_mismatch` as high risk. If both labels are known and the pack
rating is more permissive than the requested session rating, emit high risk. If
the pack rating is less permissive than the requested session rating, emit a
warning.

### Compatibility Status

Compatibility is scoped to the selected character when one is provided:

- `compatible`: pack primary character matches selected character.
- `different_character`: pack primary character differs.
- `unknown`: no selected character or missing metadata.

Compatibility does not decide whether the pack can be submitted. It only drives
warning severity, sort order, and acknowledgement behavior.

### Trust Provenance

V1 asset pack rows do not store a dedicated trust field. Setup options should
derive `trust_level` as follows:

- `local`: no completed import journal is associated with the pack for this user.
- `trusted_restore`: the latest completed import journal for the pack used
  `trust_mode=trusted_restore`.
- `untrusted_import`: the latest completed import journal for the pack used
  `trust_mode=untrusted_import`.
- `unknown`: provenance lookup failed or the pack predates provenance data in a
  way the service cannot classify.

The source should be reported in `trust_source`:

- `local_pack`
- `latest_import_journal`
- `unknown`

The latest completed import journal means the newest committed import journal row
for the same owner boundary and target pack ID. Failed, canceled, preview-only, or
missing-target import rows must not affect trust classification.

If this derivation is too expensive to perform per returned pack page, the
implementation should add a small repository helper that returns latest completed
import provenance for the returned pack IDs in one query. It should not do one
unbounded journal scan per pack.

### Defaults

Defaults are optional. The backend should suggest a default only when the choice is
unambiguous:

1. Prefer selected character when `selected_character_id` is valid.
2. Otherwise prefer one non-deleted favorite character when exactly one exists in
   the returned page.
3. Prefer a ready compatible pack for the selected/default character.
4. If multiple equivalent safe packs exist, omit `asset_pack_id`.

The frontend must tolerate absent defaults.

When `selected_character_id` is supplied and resolves to an active character, the
response must include `selected_character` even if that character is outside the
current `characters` page or does not match `character_query`. If it does not
resolve, `selected_character` is null and setup options should include a scoped
empty/error hint such as `selected_character_not_found`.

### Sorting

Within the returned pack page, the backend should order rows for setup usefulness:

1. compatible and ready packs
2. compatible but warned packs
3. unknown-compatibility packs
4. different-character or high-risk packs

This sort order is only for presentation. It must not hide warned packs.

## Backend Implementation Notes

Add setup-option schemas to `vn_play_schemas.py`, keeping them separate from
session creation models:

- `VNPlaySetupOptionsResponse`
- `VNPlaySetupCharacterOption`
- `VNPlaySetupAssetPackOption`
- `VNPlaySetupWarning`
- `VNPlaySetupCompatibility`
- `VNPlaySetupPagination`
- `VNPlaySetupDefaults`
- `VNPlaySetupEmptyState`

Add endpoint logic to `vn_play.py` near session creation routes:

```text
GET /setup-options
```

Recommended service boundary:

- Add a small setup helper under `core/VN_Play/` if the endpoint logic grows past
  straightforward aggregation.
- Reuse existing `VNAssetPackService` or repository methods where available for
  pack readiness. Do not call internal HTTP endpoints.
- Add paginated/searchable VN asset pack listing at the repository/service layer
  before readiness fanout. Do not use the existing all-packs list as the setup
  source of truth.
- Add a bulk provenance lookup for latest completed import journal rows by
  returned pack IDs if trust warnings are implemented in V1.
- Reuse character DB query/list helpers rather than duplicating SQL in the
  endpoint module. Character setup queries should never request `image_base64`.

Readiness fanout must stay bounded to the returned pack page. If readiness fails
for one pack, the endpoint should include that pack with a
`readiness_unavailable` warning rather than failing the entire setup-options
request.

## Frontend Implementation Notes

Update `NewSessionDialog` to load setup options when opened.

Recommended component split:

- Keep `NewSessionDialog` as the orchestration component.
- Add small internal selector helpers only if needed:
  - character selector
  - asset pack selector
  - warning acknowledgement panel

Behavior:

1. Load setup options on open.
2. Render named character options.
3. Render named asset pack options with readiness and warning badges.
4. Re-fetch setup options when search text, pagination, selected character, mode,
   or content rating changes.
5. If selected pack warning summary requires acknowledgement, disable create until
   the acknowledgement checkbox is checked.
6. Submit the existing `VNPlaySessionCreate` payload shape.
7. Include acknowledgement metadata under `settings` whenever high-risk warnings
   were accepted.
8. If setup-options fails, show an error and reveal manual ID fields as a fallback.

When `selected_character` is present, the frontend should use it as the selected
label even if that row is not present in the current `characters` page.

The fallback preserves the current raw-ID capability but should not be the default
path.

## Empty And Error States

The endpoint can return empty state hints:

- `no_characters`: user has no available characters.
- `no_asset_packs`: user has no VN asset packs.
- `no_ready_packs`: packs exist but none are runtime-ready in the relevant result
  scope.
- `no_compatible_packs`: packs exist but none match the selected character in the
  relevant result scope.
- `selected_character_not_found`: supplied `selected_character_id` does not
  resolve to an active character for this user.

Each empty state must include a `scope`:

- `global`: true for the user's complete dataset.
- `filter`: true for the current query/filter but not necessarily global.
- `page`: true only for the returned page.

If the backend cannot cheaply prove a global condition, it must return `filter` or
`page` rather than implying there are no matching records anywhere.

The frontend should link users toward existing character creation/import and VN
asset pack creation/review/import surfaces when possible.

## Testing

Backend tests:

- setup options returns bounded character and pack pages.
- pack list filtering and pagination happen before readiness fanout.
- ready compatible pack returns no high-risk warnings.
- different-character pack returns `pack_character_mismatch`.
- selected character outside the current page still appears in
  `selected_character`.
- unready pack returns high-risk readiness warnings but remains in `asset_packs`.
- untrusted import provenance emits `pack_untrusted_import` when provenance is
  available.
- per-pack readiness failure degrades to `readiness_unavailable` instead of
  failing the whole endpoint.
- pagination metadata is correct and readiness is evaluated only for returned
  packs.
- empty-state scope is page/filter/global accurate.

Frontend tests:

- dialog renders selector labels instead of raw ID fields when setup options load.
- selected character changes asset pack warning state.
- high-risk pack selection requires acknowledgement before create.
- warning-but-allow flow still submits after acknowledgement.
- accepted high-risk warnings are persisted into the session creation `settings`
  payload.
- setup-options failure exposes manual ID fallback.
- create payload remains compatible with `VNPlaySessionCreate`.

Verification should run focused VN Play backend tests, focused VN Play frontend
tests, `git diff --check`, and Bandit only if Python code is implemented in the
later implementation task. This design-only task does not require Bandit.

## Rollout And Compatibility

This is additive. Existing API consumers can continue posting raw IDs to
`POST /api/v1/vn-play/sessions`.

The WebUI should prefer setup-options but keep manual fallback. If the endpoint is
unavailable because a user is running an older backend, the current manual flow
still works.

## Risks

- Readiness evaluation can become expensive if pack limits are too high. Keep
  bounded defaults and maximums, and apply them before readiness fanout.
- Character query and asset pack query may have different performance profiles.
  Keep pagination independent.
- Warning severity can drift between frontend and backend if the frontend embeds
  its own rules. Treat backend warning objects as authoritative.
- High-risk acknowledgement is a WebUI guard only in V1. API clients can still
  create warned sessions directly.
- Trust provenance is derived from import journal state in V1 rather than a
  pack-level trust column. If that proves too costly or ambiguous, defer trust
  warnings until pack-level provenance is persisted.
