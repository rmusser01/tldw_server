# Persona Ambient Companion Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship deterministic, idle-only Off, Expressive, and grounded Roaming behavior for the existing raster Persona Buddy while hardening visual-pack review, activation, asset loading, preferences, and reduced-motion behavior.

**Architecture:** The backend stores validated pack-level behavior metadata, versioned Buddy preferences, and fingerprint-bound reviews while enforcing immutable active pack payloads. A renderer-neutral browser engine resolves user mode, semantic-state leases, seeded scheduling, generation fences, interaction, and transient x-axis movement; the existing sprite renderer consumes its intent through one authenticated Blob asset loader.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL migrations, existing ChaChaNotes DB abstractions, React 18, TypeScript, Zustand, Vitest/Testing Library, Playwright, pytest, Bandit.

**Spec:** `Docs/superpowers/specs/2026-08-23-persona-ambient-companion-transparent-video-design.md`

## Global Constraints

- Stage 1 is web-app-only and uses the existing Buddy mount surfaces; it does not add native, desktop, or narrow-mobile Buddy positioning.
- Ambient behavior runs only while the winning semantic state is idle.
- The wire values are exactly `off`, `expressive`, and `roaming`; a successful missing preference resolves to `expressive`, while a failed or unauthorized preference read fails ambient scheduling closed to `off`.
- `roaming` is allowed only on the full-web surface and is coerced to `expressive` on the side panel.
- The engine uses deterministic declared metadata and local browser state only; it must not import LLM/model clients, call model endpoints, or perform arbitrary network access.
- One focused Buddy remains visible; switching focus advances the generation and changes the visible companion.
- Pack behavior stays in nullable pack-level `companion_behavior_json`; the strict `sprite_frames` version 1 manifest is unchanged.
- Active pack payloads and assets are immutable; edits fork an inactive revision, and activation requires an expected version plus a current reviewed fingerprint.
- Reduced motion presents a genuinely non-animated PNG still, with no sprite animation, nudge, crossfade, or roaming.
- User drag persists only the anchor; ambient roaming writes only an in-memory x offset and never calls the persisted position setter per frame.
- Existing archive envelope `tldw.persona_visual_pack.v1` and one-active-pack invariant remain unchanged.
- Licensing remains outside the Persona Visual workflow.

---

### Task 1: Persist Behavior, Preferences, Reviews, and Immutable Revisions

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`
- Modify: `tldw_Server_API/app/core/Persona/buddy.py`
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_migration_v62_persona_companion.py`
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v62_persona_companion.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py`

**Interfaces:**
- Consumes: Existing SQLite/PostgreSQL migration dispatch, `persona_buddies`, `persona_visual_packs`, `persona_visual_assets`, and the one-active-pack transaction.
- Produces: schema version 62; `persona_visual_packs.companion_behavior_json`; `persona_buddy_preferences`; `persona_visual_pack_reviews`; `get_persona_buddy_preferences(user_id) -> dict[str, Any] | None`; `upsert_persona_buddy_preferences(user_id, ambient_mode, expected_version) -> dict[str, Any]`; `patch_persona_buddy_overlay_preferences(persona_id, user_id, patch, expected_version) -> dict[str, Any]`; `create_persona_visual_pack_review(pack_id, user_id, reviewer_user_id, fingerprint, expected_pack_version) -> dict[str, Any]`; inactive-only `update_persona_visual_pack_payload(pack_id, user_id, manifest, companion_behavior, expected_version) -> dict[str, Any]`; and version/fingerprint-aware `activate_persona_visual_pack(pack_id, persona_id, user_id, expected_version, reviewed_fingerprint) -> dict[str, Any]`.

- [ ] **Step 1: Write the migration tests**

```python
def test_v62_adds_companion_tables_and_pack_column(migrated_db):
    assert "companion_behavior_json" in migrated_db.table_columns("persona_visual_packs")
    assert migrated_db.table_exists("persona_buddy_preferences")
    assert migrated_db.table_exists("persona_visual_pack_reviews")


def test_v62_preference_mode_constraint_rejects_unknown_mode(migrated_db):
    with pytest.raises(Exception):
        migrated_db.execute(
            "INSERT INTO persona_buddy_preferences "
            "(user_id, ambient_mode, version, created_at, updated_at) VALUES (?, ?, 1, ?, ?)",
            ("user-1", "chaotic", "2026-08-23T00:00:00Z", "2026-08-23T00:00:00Z"),
        )
```

Create equivalent PostgreSQL assertions against the existing v61 migration fixture, including rejected `ambient_mode='chaotic'`, unique `user_id`, and unique `(pack_id, fingerprint)` review constraints.

- [ ] **Step 2: Run the migration tests and confirm the red state**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_migration_v62_persona_companion.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v62_persona_companion.py -v`

Expected: FAIL because schema version 62 and its tables/column are absent.

- [ ] **Step 3: Add migration 61→62 to both database engines**

Add nullable `companion_behavior_json TEXT` to `persona_visual_packs`. Add a user-owned preference table with `ambient_mode IN ('off','expressive','roaming')`, integer `version >= 1`, and timestamps. Add a review table with pack/user ownership, reviewer identity, 64-character fingerprint, pack version, and timestamps. Register both migrations and set `_CURRENT_SCHEMA_VERSION = 62`.

```sql
CREATE TABLE persona_buddy_preferences (
    user_id TEXT PRIMARY KEY,
    ambient_mode TEXT NOT NULL CHECK (ambient_mode IN ('off', 'expressive', 'roaming')),
    version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE persona_visual_pack_reviews (
    id TEXT PRIMARY KEY,
    pack_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    reviewer_user_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    pack_version INTEGER NOT NULL CHECK (pack_version >= 1),
    reviewed_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(pack_id, fingerprint),
    FOREIGN KEY(pack_id) REFERENCES persona_visual_packs(id)
);
```

Use the repository's PostgreSQL timestamp and foreign-key conventions in the PostgreSQL migration.

- [ ] **Step 4: Run both migration tests to green**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_migration_v62_persona_companion.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v62_persona_companion.py -v`

Expected: PASS for SQLite; PostgreSQL PASS when its fixture is available or its established environment skip otherwise.

- [ ] **Step 5: Write store tests for optimistic updates and immutability**

```python
def test_overlay_patch_preserves_unrelated_preferences(persona_db, buddy):
    updated = persona_db.patch_persona_buddy_overlay_preferences(
        persona_id=buddy["persona_id"],
        user_id=buddy["user_id"],
        patch={"ambient_mode": "roaming"},
        expected_version=buddy["version"],
    )
    assert updated["overlay_preferences"] == {
        "accessory_id": "scarf",
        "eye_style": "round",
        "ambient_mode": "roaming",
    }


def test_active_pack_payload_update_is_rejected(persona_db, active_pack):
    with pytest.raises(InputError, match="active visual pack payload is immutable"):
        persona_db.update_persona_visual_pack_payload(
            pack_id=active_pack["id"],
            user_id=active_pack["user_id"],
            manifest=active_pack["manifest"],
            companion_behavior=None,
            expected_version=active_pack["version"],
        )
```

Add named tests for stale global preference version, stale per-Persona patch, preservation of unknown overlay keys, active asset update/delete rejection, active pack delete rejection, inactive deletion preserving the active pack, and activation rejecting a missing or mismatched review fingerprint.

- [ ] **Step 6: Implement row adapters and version-checked methods**

Decode/encode `companion_behavior_json`; preserve every existing overlay key while validating only `ambient_mode`; use `UPDATE ... WHERE version = ?` and affected-row count for conflicts. Restrict payload and asset membership/byte mutations to inactive statuses. In one activation transaction, verify ownership, expected version, current review row, and reviewed fingerprint before switching the active pack.

```python
def normalize_persona_buddy_overlay_preferences(value: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(value or {})
    mode = normalized.get("ambient_mode")
    if mode is not None and mode not in {"off", "expressive", "roaming"}:
        raise ValueError("ambient_mode must be off, expressive, or roaming")
    return normalized
```

- [ ] **Step 7: Run the focused database suite**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v62_persona_companion.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v62_persona_companion.py -v`

Expected: PASS.

- [ ] **Step 8: Commit the persistence boundary**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py tldw_Server_API/app/core/Persona/buddy.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v62_persona_companion.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v62_persona_companion.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py
git commit -m "feat(persona): persist companion preferences and immutable reviews"
```

### Task 2: Validate Behavior Metadata and Complete Pack Fingerprints

**Files:**
- Create: `tldw_Server_API/app/core/Persona/companion_behavior.py`
- Modify: `tldw_Server_API/app/core/Persona/visuals.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_asset_constraints.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_service.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_portability/fingerprints.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_portability/exporter.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_portability/importer.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_starter_fixtures.py`
- Create: `tldw_Server_API/tests/Persona/test_persona_companion_behavior.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_service.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_portability.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py`

**Interfaces:**
- Consumes: Task 1 stores, `validate_persona_visual_manifest`, reachable-asset traversal, and canonical JSON fingerprints.
- Produces: `CompanionBehaviorValidationError`; `normalize_companion_behavior(value, resolvable_state_ids) -> dict[str, Any] | None`; `build_persona_visual_pack_fingerprint(pack, assets) -> str`; `PersonaVisualService.review_pack(pack_id, user_id, reviewer_user_id, expected_version) -> dict[str, Any]`; pure `PersonaVisualService.activate_pack(pack_id, persona_id, user_id, expected_version, reviewed_fingerprint) -> dict[str, Any]`; and explicit starter-pack behavior declarations.

- [ ] **Step 1: Write behavior validation tests**

```python
def test_normalize_behavior_preserves_relative_weights():
    normalized = normalize_companion_behavior(
        {
            "schema_version": 1,
            "entries": [{
                "state": "ambient.look",
                "trigger": "ambient",
                "category": "idle_variant",
                "suggested_weight": 3,
                "suggested_cooldown_ms": 45_000,
            }],
        },
        resolvable_state_ids={"idle", "ambient.look"},
    )
    assert normalized["entries"][0]["suggested_weight"] == 3.0


@pytest.mark.parametrize("weight", [-1, float("inf"), float("nan")])
def test_behavior_rejects_invalid_weights(weight):
    with pytest.raises(CompanionBehaviorValidationError):
        normalize_companion_behavior(
            {"schema_version": 1, "entries": [{
                "state": "ambient.look",
                "trigger": "ambient",
                "category": "idle_variant",
                "suggested_weight": weight,
            }]},
            resolvable_state_ids={"ambient.look"},
        )
```

Cover version 1, maximum 128 entries, 128-character state IDs, unique `(trigger,state)`, triggers `ambient|click|drag`, categories `idle_variant|move|reaction`, finite weights `0..10000`, cooldowns `0..86400000`, resolvable states, and horizontal movement ratios in `0..1` with start no later than end. Add raster activation cases proving all nine built-in states resolve to a static selection, selected static bytes are PNG with exactly one decoded frame, and animated GIF/WebP cannot satisfy reduced-motion coverage.

- [ ] **Step 2: Run the validator test and confirm failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_companion_behavior.py -v`

Expected: FAIL because `companion_behavior.py` does not exist.

- [ ] **Step 3: Implement strict canonical normalization**

```python
ALLOWED_TRIGGERS = frozenset({"ambient", "click", "drag"})
ALLOWED_CATEGORIES = frozenset({"idle_variant", "move", "reaction"})
MAX_BEHAVIOR_ENTRIES = 128


def normalize_companion_behavior(
    value: Mapping[str, Any] | None,
    *,
    resolvable_state_ids: set[str],
) -> dict[str, Any] | None:
    if value is None:
        return None
    if value.get("schema_version") != 1:
        raise CompanionBehaviorValidationError("unsupported companion behavior schema_version")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) > MAX_BEHAVIOR_ENTRIES:
        raise CompanionBehaviorValidationError("companion behavior entries are invalid")
    return {"schema_version": 1, "entries": _normalize_entries(entries, resolvable_state_ids)}
```

Reject invalid pack data; do not clamp it and do not infer actions when metadata is absent.

- [ ] **Step 4: Write fingerprint, review, fork, and portability tests**

```python
def test_behavior_changes_invalidate_review(visual_service, reviewed_pack):
    old_fingerprint = reviewed_pack["review"]["fingerprint"]
    fork = visual_service.fork_pack_revision(
        pack_id=reviewed_pack["pack"]["id"],
        user_id="user-1",
        manifest=reviewed_pack["pack"]["manifest"],
        companion_behavior={"schema_version": 1, "entries": []},
    )
    new_review = visual_service.review_pack(
        pack_id=fork["id"],
        user_id="user-1",
        reviewer_user_id="user-1",
        expected_version=fork["version"],
    )
    assert new_review["fingerprint"] != old_fingerprint
```

Add assertions that asset checksum/metadata, normalized manifest, behavior, and converter/provenance version affect the fingerprint; timestamps/status do not; activation writes no normalized payload; stale review fails; duplication/native export/native import preserve behavior.

- [ ] **Step 5: Implement complete fingerprints and pure lifecycle methods**

```python
def build_persona_visual_pack_fingerprint(
    pack: Mapping[str, Any],
    assets: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "renderer_type": pack["renderer_type"],
        "manifest_version": pack["manifest_version"],
        "manifest": pack["manifest"],
        "companion_behavior": pack.get("companion_behavior"),
        "provenance_version": pack.get("provenance_version"),
        "assets": [
            _fingerprint_asset(asset)
            for asset in sorted(assets, key=lambda row: row["id"])
        ],
    }
    return canonical_payload_fingerprint(payload)
```

`review_pack` validates without mutating payload fields, records reviewer/time/fingerprint/version, and returns the review. `fork_pack_revision` copies assets into a new inactive pack before applying edits. Native duplication/import/export includes behavior beside the strict manifest.

Add `validate_sprite_static_coverage(manifest, assets) -> PersonaVisualValidationResult` in `visuals.py`; resolve each built-in state's animation and select `preview_frame`, then `preview_asset_id`, then first frame. Inspect candidate bytes with the existing bounded raster decoder, require `image/png` and exactly one decoded frame, and call this pure validator from review/activation. Existing active packs are not rewritten; a future fork/review must satisfy the gate.

- [ ] **Step 6: Add explicit behavior to bundled raster fixtures**

Declare only states resolvable by each pack. If a fixture lacks a distinct authored idle variant, use an empty entry list.

```python
"companion_behavior": {
    "schema_version": 1,
    "entries": [{
        "state": "ambient.look",
        "trigger": "ambient",
        "category": "idle_variant",
        "suggested_weight": 1,
        "suggested_cooldown_ms": 45_000,
    }],
}
```

- [ ] **Step 7: Run behavior, service, portability, and starter tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_companion_behavior.py tldw_Server_API/tests/Persona/test_persona_visuals_core.py tldw_Server_API/tests/Persona/test_persona_visual_service.py tldw_Server_API/tests/Persona/test_persona_visual_portability.py tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -v`

Expected: PASS.

- [ ] **Step 8: Commit the reviewed payload contract**

```bash
git add tldw_Server_API/app/core/Persona/companion_behavior.py tldw_Server_API/app/core/Persona/visuals.py tldw_Server_API/app/core/Persona/visual_asset_constraints.py tldw_Server_API/app/core/Persona/visual_service.py tldw_Server_API/app/core/Persona/visual_portability/fingerprints.py tldw_Server_API/app/core/Persona/visual_portability/exporter.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/core/Persona/visual_starter_fixtures.py tldw_Server_API/tests/Persona/test_persona_companion_behavior.py tldw_Server_API/tests/Persona/test_persona_visuals_core.py tldw_Server_API/tests/Persona/test_persona_visual_service.py tldw_Server_API/tests/Persona/test_persona_visual_portability.py tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py
git commit -m "feat(persona): validate and review companion pack behavior"
```

### Task 3: Expose Versioned Preferences and Review/Activation APIs

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_buddy_api.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`

**Interfaces:**
- Consumes: Tasks 1–2 stores/services and existing AuthNZ ownership/rate-limit dependencies.
- Produces: `GET /api/v1/persona/buddy/preferences`; `PATCH /api/v1/persona/buddy/preferences`; `PATCH /api/v1/persona/profiles/{persona_id}/buddy/preferences`; `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/reviews`; expected-version/fingerprint activation body; pack `companion_behavior`, `version`, and `review` response fields.

- [ ] **Step 1: Write API contract tests**

```python
def test_missing_global_preference_returns_expressive_default(client, auth_headers):
    response = client.get("/api/v1/persona/buddy/preferences", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {
        "ambient_mode": "expressive",
        "version": None,
        "stored": False,
    }


def test_stale_persona_override_patch_returns_conflict(client, auth_headers, persona):
    response = client.patch(
        f"/api/v1/persona/profiles/{persona['id']}/buddy/preferences",
        headers=auth_headers,
        json={"ambient_mode": "roaming", "expected_version": 1},
    )
    assert response.status_code == 409
```

Add API-key and bearer ownership cases, invalid mode 422, missing preference versus backend read failure, review creation, activation with current review, stale activation conflict, and behavior round-trip.

- [ ] **Step 2: Run the API tests and confirm failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_buddy_api.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -v`

Expected: FAIL on missing routes and fields.

- [ ] **Step 3: Add request and response schemas**

```python
PersonaAmbientMode = Literal["off", "expressive", "roaming"]


class PersonaBuddyPreferencesUpdate(BaseModel):
    ambient_mode: PersonaAmbientMode
    expected_version: int | None = Field(default=None, ge=1)


class PersonaBuddyPreferencesResponse(BaseModel):
    ambient_mode: PersonaAmbientMode
    version: int | None
    stored: bool


class PersonaVisualPackReviewRequest(BaseModel):
    expected_version: int = Field(ge=1)


class PersonaVisualPackActivateRequest(BaseModel):
    expected_version: int = Field(ge=1)
    reviewed_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
```

Add `companion_behavior: dict[str, Any] | None` to pack create, inactive payload update, and responses.

- [ ] **Step 4: Implement routes with ownership and explicit conflict handling**

A GET returns `stored=false` only after a successful no-row read. AuthNZ/store failures propagate. PATCH uses version-checked stores. Review calls the pure service. Activation maps stale review/version to HTTP 409.

```python
@router.get("/buddy/preferences", response_model=PersonaBuddyPreferencesResponse)
async def get_buddy_preferences(current_user=Depends(get_current_user)):
    row = _persona_db().get_persona_buddy_preferences(user_id=str(current_user.id))
    if row is None:
        return PersonaBuddyPreferencesResponse(
            ambient_mode="expressive",
            version=None,
            stored=False,
        )
    return PersonaBuddyPreferencesResponse(
        ambient_mode=row["ambient_mode"],
        version=row["version"],
        stored=True,
    )
```

Use the endpoint module's existing dependency and exception helpers.

- [ ] **Step 5: Run the focused API tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_buddy_api.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the API contract**

```bash
git add tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/tests/Persona/test_persona_buddy_api.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py
git commit -m "feat(api): expose persona companion preferences and reviews"
```

### Task 4: Add Frontend Types, Preference Client, and Authenticated Asset Loader

**Files:**
- Modify: `apps/packages/ui/src/types/persona-visuals.ts`
- Create: `apps/packages/ui/src/types/persona-buddy.ts`
- Create: `apps/packages/ui/src/services/persona-buddy.ts`
- Create: `apps/packages/ui/src/services/persona-visual-assets.ts`
- Modify: `apps/packages/ui/src/services/persona-visuals.ts`
- Create: `apps/packages/ui/src/services/__tests__/persona-buddy.test.ts`
- Create: `apps/packages/ui/src/services/__tests__/persona-visual-assets.test.ts`

**Interfaces:**
- Consumes: Task 3 JSON contracts and `TldwApiClient.fetchWithAuth` with `responseType: "arrayBuffer"` and `AbortSignal`.
- Produces: `PersonaAmbientMode`; `PersonaCompanionBehavior`; `PersonaBuddyPreferences`; `getBuddyPreferences()`, `updateBuddyPreferences(input)`, `updatePersonaBuddyPreferences(personaId,input)`; and reference-counted `acquirePersonaVisualAsset(asset, options) -> Promise<PersonaVisualAssetHandle>` with immutable `url`, `mimeType`, and `release()`.

- [ ] **Step 1: Write preference and protected-byte tests**

```typescript
it("distinguishes a failed preference read from a missing stored row", async () => {
  mockFetchWithAuth.mockRejectedValueOnce(new Error("unauthorized"))
  await expect(getBuddyPreferences()).rejects.toThrow("unauthorized")
})

it("loads protected bytes through the authenticated client", async () => {
  const handle = await acquirePersonaVisualAsset(asset, { maxBytes: 1024 })
  expect(mockFetchWithAuth).toHaveBeenCalledWith(
    asset.url,
    expect.objectContaining({ responseType: "arrayBuffer" }),
  )
  handle.release()
  expect(URL.revokeObjectURL).toHaveBeenCalledTimes(1)
})
```

Add cache-key `${asset.id}:${asset.checksum_sha256}`, shared reference count, declared/received size, checksum mismatch, MIME mismatch, abort, idempotent release, eviction, and clear-cache cases. Exercise both API-key and bearer-authenticated client setups.

- [ ] **Step 2: Run service tests and confirm failure**

Run: `cd apps/packages/ui && bunx vitest run src/services/__tests__/persona-buddy.test.ts src/services/__tests__/persona-visual-assets.test.ts`

Expected: FAIL because the services/types are absent.

- [ ] **Step 3: Add exact frontend types**

```typescript
export type PersonaAmbientMode = "off" | "expressive" | "roaming"

export type PersonaCompanionBehaviorEntry = {
  state: PersonaVisualStateId
  trigger: "ambient" | "click" | "drag"
  category: "idle_variant" | "move" | "reaction"
  suggested_weight?: number
  suggested_cooldown_ms?: number
  movement?: {
    direction: "horizontal"
    motion_start_ratio: number
    motion_end_ratio: number
  }
}

export type PersonaCompanionBehavior = {
  schema_version: 1
  entries: PersonaCompanionBehaviorEntry[]
}
```

Add `companion_behavior`, `version`, preference, review, and activation types matching Task 3 exactly.

- [ ] **Step 4: Implement preference calls and the bounded Blob cache**

```typescript
export type PersonaVisualAssetHandle = {
  readonly url: string
  readonly mimeType: string
  release(): void
}

export async function acquirePersonaVisualAsset(
  asset: Pick<PersonaVisualAsset, "id" | "url" | "checksum_sha256" | "byte_size" | "mime_type">,
  options: { signal?: AbortSignal; maxBytes?: number } = {},
): Promise<PersonaVisualAssetHandle> {
  const maxBytes = options.maxBytes ?? 16 * 1024 * 1024
  if (asset.byte_size > maxBytes) throw new PersonaVisualAssetError("asset_too_large")
  const bytes = await tldwClient.fetchWithAuth<ArrayBuffer>(asset.url, {
    responseType: "arrayBuffer",
    signal: options.signal,
  })
  if (bytes.byteLength > maxBytes) throw new PersonaVisualAssetError("asset_too_large")
  await verifySha256(bytes, asset.checksum_sha256)
  return retainCachedObjectUrl(asset, bytes)
}
```

Revoke only when the cache reference count reaches zero or the cache is cleared.

- [ ] **Step 5: Run service tests to green**

Run: `cd apps/packages/ui && bunx vitest run src/services/__tests__/persona-buddy.test.ts src/services/__tests__/persona-visual-assets.test.ts`

Expected: PASS.

- [ ] **Step 6: Commit the browser data boundary**

```bash
git add apps/packages/ui/src/types/persona-visuals.ts apps/packages/ui/src/types/persona-buddy.ts apps/packages/ui/src/services/persona-buddy.ts apps/packages/ui/src/services/persona-visual-assets.ts apps/packages/ui/src/services/persona-visuals.ts apps/packages/ui/src/services/__tests__/persona-buddy.test.ts apps/packages/ui/src/services/__tests__/persona-visual-assets.test.ts
git commit -m "feat(ui): add companion preferences and protected asset loader"
```

### Task 5: Build the Renderer-Neutral Companion Engine

**Files:**
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/personaCompanionPolicy.ts`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/personaCompanionEngine.ts`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/usePersonaCompanion.ts`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualState.ts`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualDiagnostics.ts`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaCompanionPolicy.test.ts`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaCompanionEngine.test.ts`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts`

**Interfaces:**
- Consumes: Task 4 behavior/mode types, existing semantic state resolution, visibility/focus signals, and injected monotonic clock/timer/PRNG.
- Produces: `resolveEffectiveAmbientMode(input) -> PersonaAmbientMode`; `resolveWinningPersonaVisualIntent(input) -> PersonaVisualStateId`; source-scoped `PersonaVisualStateLease`; `createPersonaCompanionEngine(runtime) -> PersonaCompanionEngine`; and `usePersonaCompanion(input) -> PersonaCompanionSnapshot` with requested state, phase, facing, transient x, generation, and suspension. The injected runtime emits local-only `ambient_selected|ambient_skipped|ambient_preempted|stale_generation` diagnostics without external telemetry.

- [ ] **Step 1: Write pure policy tests**

```typescript
it("uses persona, then global, then Expressive", () => {
  expect(resolveEffectiveAmbientMode({ persona: "roaming", global: "off", readFailed: false, surface: "web" })).toBe("roaming")
  expect(resolveEffectiveAmbientMode({ persona: null, global: "off", readFailed: false, surface: "web" })).toBe("off")
  expect(resolveEffectiveAmbientMode({ persona: null, global: null, readFailed: false, surface: "web" })).toBe("expressive")
})

it("fails closed and coerces sidepanel roaming", () => {
  expect(resolveEffectiveAmbientMode({ persona: null, global: null, readFailed: true, surface: "web" })).toBe("off")
  expect(resolveEffectiveAmbientMode({ persona: "roaming", global: null, readFailed: false, surface: "sidepanel" })).toBe("expressive")
})
```

Add precedence assertions in this exact order: error, approval, offline, wake/listening/thinking/speaking/tool, interaction, ambient, idle.

- [ ] **Step 2: Write fake-clock engine tests**

```typescript
it("starts a fresh interval after hidden-tab resume", () => {
  const runtime = createFakeCompanionRuntime({ random: [0, 0.5] })
  const engine = createPersonaCompanionEngine(runtime)
  engine.update(idleInput({ visibility: "visible", mode: "expressive" }))
  runtime.advanceBy(29_999)
  expect(engine.getSnapshot().phase).toBe("idle")
  engine.update(idleInput({ visibility: "hidden", mode: "expressive" }))
  runtime.advanceBy(120_000)
  engine.update(idleInput({ visibility: "visible", mode: "expressive" }))
  runtime.advanceBy(29_999)
  expect(engine.getSnapshot().phase).toBe("idle")
})
```

Cover seeded repeat avoidance, relative weights, empty sets, cadence/cooldown/duration/distance clamps, lease release/expiry, higher-priority preemption, controls/focus/drag suspension, reduced motion, viewport reclamping, generation invalidation, and no persisted-position calls during motion. Prove Off still permits direct click/Space reactions, and prove a direction change selects a declared `ambient.turn.*` entry before changing facing; without a turn state, facing changes only when the target state is declared mirror-safe.

- [ ] **Step 3: Run engine tests and confirm failure**

Run: `cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaCompanionPolicy.test.ts src/components/Common/PersonaBuddy/__tests__/personaCompanionEngine.test.ts src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts`

Expected: FAIL because the engine modules are absent and precedence differs.

- [ ] **Step 4: Implement policy, leases, scheduler, and generation fence**

```typescript
export type PersonaCompanionSnapshot = {
  generation: number
  phase: "idle" | "action"
  requestedState: PersonaVisualStateId
  facing: "left" | "right"
  transientOffsetX: number
  suspension: "none" | "semantic" | "hidden" | "controls" | "focus" | "drag" | "reduced_motion" | "surface"
}

export interface PersonaCompanionEngine {
  update(input: PersonaCompanionInput): void
  react(trigger: "click" | "drag"): boolean
  acquireLease(source: PersonaVisualLeaseSource, state: PersonaVisualStateId, ttlMs: number): PersonaVisualStateLease
  getSnapshot(): PersonaCompanionSnapshot
  subscribe(listener: () => void): () => void
  dispose(): void
}
```

Clamp ambient interval to 30–90 seconds, action duration to 150–8000 ms, cooldown to 0–86400000 ms, and movement to current horizontal bounds. Packs supply relative weights only. Avoid the immediately previous larger action when another eligible action exists. A successful declared turn completion commits the new facing; a failed turn preserves facing unless the subsequent animation is mirror-safe. Emit only user-safe local diagnostics with Persona/pack/state IDs and failure class.

- [ ] **Step 5: Add React lifecycle and dependency-boundary tests**

`usePersonaCompanion` owns one engine instance per mounted Buddy, updates it from React input, subscribes with `useSyncExternalStore`, and disposes timers/leases on unmount. Install a throwing `globalThis.fetch`, advance fake timers through ambient/reaction paths, and assert no network call. Scan engine imports for service/model coupling.

```typescript
expect(sourceText).not.toMatch(/services\/tldw|LLM|model-client|\bfetch\s*\(/)
expect(globalThis.fetch).not.toHaveBeenCalled()
```

- [ ] **Step 6: Run the engine suite to green**

Run: `cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaCompanionPolicy.test.ts src/components/Common/PersonaBuddy/__tests__/personaCompanionEngine.test.ts src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts`

Expected: PASS.

- [ ] **Step 7: Commit the shared engine**

```bash
git add apps/packages/ui/src/components/Common/PersonaBuddy/personaCompanionPolicy.ts apps/packages/ui/src/components/Common/PersonaBuddy/personaCompanionEngine.ts apps/packages/ui/src/components/Common/PersonaBuddy/usePersonaCompanion.ts apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualState.ts apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualDiagnostics.ts apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaCompanionPolicy.test.ts apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaCompanionEngine.test.ts apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts
git commit -m "feat(ui): add deterministic idle companion engine"
```

### Task 6: Integrate Raster Rendering, Gestures, Settings, and Reduced Motion

**Files:**
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellPopover.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/SpriteFrameRenderer.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualRenderers.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualAssets.ts`
- Modify: `apps/packages/ui/src/store/persona-buddy-shell.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellDock.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellPopover.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

**Interfaces:**
- Consumes: Tasks 3–5 API, loader, and engine snapshot plus the persisted anchor store.
- Produces: renderer props `{ requestedState, generation, reducedMotion, onReady, onFailure, onComplete }`; authenticated raster Blob rendering; deterministic static selection; deferred click/double-click/drag arbitration; keyboard/touch controls; first-use hint; layered mode controls; and review-then-activate Visual Garden flow.

- [ ] **Step 1: Write renderer and reduced-motion tests**

```typescript
it("selects a static PNG and never starts a sprite timer under reduced motion", async () => {
  render(<SpriteFrameRenderer {...props} reducedMotion />)
  await screen.findByRole("img", { name: /persona buddy/i })
  expect(acquirePersonaVisualAsset).toHaveBeenCalledWith(
    expect.objectContaining({ mime_type: "image/png" }),
    expect.any(Object),
  )
  expect(vi.getTimerCount()).toBe(0)
})
```

Add static selection order `preview_frame`, `preview_asset_id`, first frame; animated GIF/WebP rejection; previous visual until next Blob is ready; stale generation cleanup; auth-mode coverage; and object URL release.

- [ ] **Step 2: Write interaction and settings tests**

```typescript
it("cancels a pending click when double click opens controls", async () => {
  await user.pointer({ keys: "[MouseLeft]", target: buddy })
  await user.pointer({ keys: "[MouseLeft]", target: buddy })
  vi.advanceTimersByTime(500)
  expect(onReact).not.toHaveBeenCalled()
  expect(onOpenControls).toHaveBeenCalledTimes(1)
})

it("does not persist ambient x movement", () => {
  rerender(<BuddyShellHost engineSnapshot={{ ...snapshot, transientOffsetX: 48 }} />)
  expect(setPersistedPosition).not.toHaveBeenCalled()
})
```

Cover 8-pixel drag threshold, pointer capture, final anchor persistence, click, Enter, Space, touch tap, visible focus, touch/focus controls button, idle-only reactions, renderer-neutral inner nudge, reduced-motion nudge suppression, first-use hint, global/per-Persona conflicts, sidepanel coercion, focused Persona switch, preservation of other-session badges/status, controls reachable in every semantic state, and resting chrome without text except approval/error/offline.

- [ ] **Step 3: Run component tests and confirm failure**

Run: `cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellDock.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellPopover.test.tsx src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

Expected: FAIL on the new behavior.

- [ ] **Step 4: Integrate the engine without broadening mount scope**

Load preferences and active pack in the host, feed semantic/suspension input into `usePersonaCompanion`, and apply anchor plus transient x only to the outer Buddy container. Keep the current root-web desktop gate and sidepanel mount.

```typescript
const companion = usePersonaCompanion({
  personaId: focusedPersonaId,
  packId: activePack?.id ?? null,
  packVersion: activePack?.version ?? 0,
  behavior: activePack?.companion_behavior ?? null,
  globalMode: preference.data?.ambient_mode ?? null,
  personaMode: buddy.overlay_preferences?.ambient_mode ?? null,
  preferenceReadFailed: preference.isError,
  surface,
  semanticState,
  controlsOpen,
  focusWithin,
  dragging,
  reducedMotion,
  viewport,
})
```

- [ ] **Step 5: Implement adaptive gesture arbitration and controls**

Defer click 300 ms; cancel on double-click or movement beyond 8 CSS pixels; capture the pointer during drag. Enter opens controls. Space prevents page scroll and reacts. Touch tap reacts. The small button has `aria-label="Open Buddy controls"` and remains visible on touch/focus surfaces.

```typescript
const scheduleClickReaction = () => {
  clearPendingClick()
  pendingClick.current = window.setTimeout(() => {
    pendingClick.current = null
    requestIdleReaction("click")
  }, 300)
}
```

When no declared click action exists, animate only the inner visual wrapper for at most 160 ms; the shared reduced-motion rule disables it.

- [ ] **Step 6: Use authenticated handles and exact static selection**

Replace direct protected `asset.url` sources with `acquirePersonaVisualAsset`. Under reduced motion resolve the state's animation, select `preview_frame`, then `preview_asset_id`, then the first frame, require `image/png`, render once, and allocate no timer. Fence every completion by generation and release stale handles.

- [ ] **Step 7: Add layered settings and review-before-activation UI**

The popover shows a global radio group and a per-Persona override with `Use global`, `Off`, `Expressive`, `Roaming`, effective-mode text, and sidepanel coercion copy. A 409 refetches and reports a stale-update message. In `VisualPackEditor`, `Review revision` precedes `Activate`; activation submits review fingerprint/current version, active edits call `fork_pack_revision`, and review warns when multiple built-in states share one still without rejecting complete coverage.

- [ ] **Step 8: Run focused component tests**

Run: `cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__ src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/services/__tests__/persona-buddy.test.ts src/services/__tests__/persona-visual-assets.test.ts`

Expected: PASS.

- [ ] **Step 9: Commit the integrated experience**

```bash
git add apps/packages/ui/src/components/Common/PersonaBuddy apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx apps/packages/ui/src/store/persona-buddy-shell.ts
git commit -m "feat(ui): integrate ambient Buddy behavior and accessible controls"
```

### Task 7: Prove Stage 1 End to End and Document It

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/persona-buddy-interaction.spec.ts`
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
- Create: `Docs/Code_Documentation/Persona_Ambient_Companion.md`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaCompanionEngine.test.ts`

**Interfaces:**
- Consumes: Complete Stage 1 backend/API/browser feature.
- Produces: release proof for idle-only scheduling, focused-Persona invalidation, authenticated raster fallback, grounded transient roaming, accessibility, and no model/network dependency; operator and author documentation.

- [ ] **Step 1: Add focused Playwright scenarios**

```typescript
test("ambient Buddy stays idle while speaking and resumes with a fresh interval", async ({ page }) => {
  await seedBuddyMode(page, "roaming")
  await openPersonaWithBuddy(page)
  await setPersonaRuntimeState(page, "speaking")
  await advanceBuddyClock(page, 90_000)
  await expect(page.getByTestId("persona-buddy")).toHaveAttribute("data-companion-phase", "idle")
  await setPersonaRuntimeState(page, "idle")
  await advanceBuddyClock(page, 29_999)
  await expect(page.getByTestId("persona-buddy")).toHaveAttribute("data-companion-phase", "idle")
})
```

Add Off/Expressive/Roaming, horizontal bounds, no persisted write per ambient frame, click/double-click/drag, keyboard/touch, reduced-motion static PNG, hidden-tab fresh interval through the existing E2E harness, preference read failure, focused Persona switch, and protected raster failure retaining the previous visual.

- [ ] **Step 2: Run focused Playwright**

Run: `cd apps/tldw-frontend && bunx playwright test e2e/workflows/persona-buddy-interaction.spec.ts --project=chromium --reporter=line`

Expected: PASS after adding deterministic test-only clock/state seams to the existing E2E harness; production behavior is unchanged when the harness is absent.

- [ ] **Step 3: Document runtime and authoring contracts**

Document modes, preference order, missing versus failed reads, idle eligibility, precedence, behavior limits, review fingerprints, immutability, authenticated Blob loading, gestures, reduced motion, surfaces, and this minimal example:

```json
{
  "schema_version": 1,
  "entries": [{
    "state": "ambient.look",
    "trigger": "ambient",
    "category": "idle_variant",
    "suggested_weight": 3,
    "suggested_cooldown_ms": 45000
  }]
}
```

State that missing behavior means base idle only and technical validation does not establish licensing rights.

- [ ] **Step 4: Run the focused verification matrix**

Backend:

`source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_companion_behavior.py tldw_Server_API/tests/Persona/test_persona_buddy_api.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py tldw_Server_API/tests/Persona/test_persona_visuals_core.py tldw_Server_API/tests/Persona/test_persona_visual_service.py tldw_Server_API/tests/Persona/test_persona_visual_portability.py tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v62_persona_companion.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v62_persona_companion.py -v`

Frontend:

`cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__ src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/services/__tests__/persona-buddy.test.ts src/services/__tests__/persona-visual-assets.test.ts`

Lint and typecheck:

`cd apps/tldw-frontend && bunx eslint ../packages/ui/src/components/Common/PersonaBuddy ../packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx ../packages/ui/src/services/persona-buddy.ts ../packages/ui/src/services/persona-visual-assets.ts ../packages/ui/src/types/persona-buddy.ts ../packages/ui/src/types/persona-visuals.ts`

`cd apps/tldw-frontend && bunx tsc --noEmit`

E2E:

`cd apps/tldw-frontend && bunx playwright test e2e/workflows/persona-buddy-interaction.spec.ts --project=chromium --reporter=line`

Expected: every command PASS, with only the established PostgreSQL environment skip when unavailable.

- [ ] **Step 5: Run security and diff gates**

`source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Persona tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py -f json -o /tmp/bandit_persona_ambient_stage1.json`

`git diff --check`

Expected: Bandit exits 0 with no new findings; diff check exits 0.

- [ ] **Step 6: Commit Stage 1 release proof**

```bash
git add apps/tldw-frontend/e2e/workflows/persona-buddy-interaction.spec.ts Docs/Code_Documentation/Persona_Visual_Packs.md Docs/Code_Documentation/Persona_Ambient_Companion.md tldw_Server_API/tests/Persona/test_persona_visuals_api.py apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaCompanionEngine.test.ts
git commit -m "test(persona): verify ambient Buddy stage one"
```

## Stage 1 Completion Gate

- [ ] Off, Expressive, and Roaming pass over current raster packs; Roaming is full-web-only.
- [ ] Error, approval, offline, and active Persona states preempt interaction and ambient intent in the approved order.
- [ ] Hidden time, controls, focus, drag, and reduced motion suspend the relevant ambient behavior.
- [ ] Preferences are versioned, preserve unrelated overlay keys, and distinguish absent data from read failure.
- [ ] Active revisions/assets are immutable; reviews bind complete fingerprints; activation is atomic and pure.
- [ ] Raster bytes load only through authenticated bounded Blob handles and are revoked safely.
- [ ] Reduced motion uses deterministic non-animated PNG stills and allocates no animation/video work.
- [ ] Ambient roaming remains transient and never churns persisted anchor storage.
- [ ] The engine boundary test proves no LLM/model client or arbitrary network coupling.
- [ ] Focused backend, frontend, E2E, lint, typecheck, Bandit, and diff checks pass.
