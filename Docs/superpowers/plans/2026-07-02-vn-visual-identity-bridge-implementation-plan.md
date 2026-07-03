# VN Visual Identity Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved VN Visual Identity Bridge in two stages: Stage 11A imports VN generated files into Visual Identity expression assets with provenance, and Stage 11B resolves VN role/casting overrides through the existing Visual Identity resolver.

**Architecture:** Stage 11A keeps the existing generic generated-file import endpoint and adds bounded asset-level `source_context`, a small VN bridge helper, and a reusable frontend import hook. Stage 11B extends the existing `/api/v1/visual-identities/bindings/resolve` endpoint and service resolver with strict optional pack/version overrides, role metadata, typed fallback reasons, and backward-compatible default-binding behavior.

**Tech Stack:** FastAPI, Pydantic, SQLite ChaChaNotes, existing AuthNZ generated-file repository, existing VN Asset repository helpers, React, TypeScript, Vitest, pytest, Bandit.

**Boundary Spec:** `Docs/superpowers/specs/2026-07-02-vn-visual-identity-bridge-design.md`

**Backlog Plan Task:** `TASK-12090.3`

---

## Scope Check

- [ ] Implement Stage 11A before Stage 11B.
- [ ] Keep `POST /api/v1/visual-identities/packs/{pack_id}/assets/from-generated-file` as the import endpoint.
- [ ] Do not create route-level VN workbench UI.
- [ ] Do not add persisted VN cast tables.
- [ ] Do not expand VN Asset generation/upload format support beyond existing VN Asset behavior.
- [ ] Do preserve Visual Identity animated raster support when generated-file records are valid for it.
- [ ] Do not use persona actors as a path to character legacy mood images.

## File Structure

### Backend Files

- Create: `tldw_Server_API/app/core/Visual_Identities/source_context.py`
  - Validates, canonicalizes, and serializes Visual Identity asset `source_context`.
  - Owns size/depth/key/string/prompt/data URI checks.
- Create: `tldw_Server_API/app/core/Visual_Identities/vn_bridge.py`
  - Builds trusted VN provenance from generated-file records and request hints.
  - Defaults `source_feature` to VN Assets and validates `vn_item_id` source refs.
- Modify: `tldw_Server_API/app/core/Visual_Identities/__init__.py`
  - Export new helpers only if local patterns call for it.
- Modify: `tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py`
  - Add idempotent `source_context_json` migration for `visual_identity_assets`.
  - Persist source context on draft assets and copy it to version assets.
  - Include source context in activation manifest asset metadata.
- Modify: `tldw_Server_API/app/api/v1/schemas/visual_identity_schemas.py`
  - Add `source_context` to `VisualIdentityAssetResponse`.
  - Add `source_context` to `VisualIdentityGeneratedFileAssetRequest`.
  - Extend `VisualIdentityResolveResponse` with role/casting fields.
- Modify: `tldw_Server_API/app/api/v1/endpoints/visual_identities.py`
  - Include source context in asset responses.
  - Hash validated/canonical source context in generated-file idempotency.
  - Use VN bridge helper for VN generated-file provenance.
  - Extend `/bindings/resolve` query parameters and response fields.
- Modify: `tldw_Server_API/app/core/Visual_Identities/service.py`
  - Add stateless role/casting resolver behavior.
  - Preserve existing default-binding resolver behavior.

### Backend Test Files

- Create: `tldw_Server_API/tests/Visual_Identities/test_visual_identity_source_context.py`
- Create: `tldw_Server_API/tests/Visual_Identities/test_visual_identity_vn_bridge.py`
- Modify: `tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py`
- Modify: `tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py`
- Modify: `tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py`

### Frontend Files

- Modify: `apps/packages/ui/src/types/visual-identities.ts`
  - Add source context fields and resolver override fields.
- Modify: `apps/packages/ui/src/services/tldw/domains/visual-identities.ts`
  - Send source context on generated-file import.
  - Include Stage 11B resolver query parameters.
- Create: `apps/packages/ui/src/components/Common/VisualIdentity/useGeneratedFileImportAction.ts`
  - Reusable hook/action for import-then-slot-assignment.
- Create: `apps/packages/ui/src/components/Common/VisualIdentity/__tests__/useGeneratedFileImportAction.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.visual-identities.test.ts`
- Modify: `apps/packages/ui/src/hooks/useVisualIdentityResolver.ts`
  - Add optional role/casting resolver inputs and cache-key fields.
- Modify: `apps/packages/ui/src/hooks/__tests__/useVisualIdentityResolver.test.tsx`

---

## Stage 11A: VN Generated-File Import Bridge

### Task 1: Source Context Validation Helper

**Files:**
- Create: `tldw_Server_API/app/core/Visual_Identities/source_context.py`
- Create: `tldw_Server_API/tests/Visual_Identities/test_visual_identity_source_context.py`

- [ ] **Step 1: Write failing tests for accepted and canonical source context**

Add tests:

```python
def test_canonical_source_context_sorts_keys_and_preserves_short_metadata() -> None:
    context = canonicalize_source_context({
        "vn_slot_label": "Happy",
        "generated_file_id": 42,
        "source_feature": "vn_assets",
    })

    assert context == {
        "generated_file_id": 42,
        "source_feature": "vn_assets",
        "vn_slot_label": "Happy",
    }
    assert source_context_payload_hash(context) == source_context_payload_hash({
        "source_feature": "vn_assets",
        "vn_slot_label": "Happy",
        "generated_file_id": 42,
    })
```

Also cover prompt references as allowed short metadata:

```python
def test_source_context_allows_short_prompt_references() -> None:
    context = canonicalize_source_context({
        "prompt_id": "prompt-123",
        "prompt_ref": "vn-pack/maya/happy",
        "prompt_label": "Maya happy sprite",
    })

    assert context == {
        "prompt_id": "prompt-123",
        "prompt_label": "Maya happy sprite",
        "prompt_ref": "vn-pack/maya/happy",
    }
```

- [ ] **Step 2: Write failing tests for rejected source context**

Cover forbidden roots, prompt text, data URIs, base64-like payloads, binary values, serialized size, depth, key count, key length, and scalar string length:

```python
@pytest.mark.parametrize("value", [[], "text", 7, None])
def test_source_context_rejects_non_object_roots(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_source_context"):
        canonicalize_source_context(value)


@pytest.mark.parametrize("context", [
    {"prompt": "draw a full character sprite"},
    {"user_prompt": "draw a full character sprite"},
    {"image": "data:image/png;base64,AAAA"},
    {"blob": "QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFB"},
])
def test_source_context_rejects_prompt_text_and_payloads(context: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid_source_context"):
        canonicalize_source_context(context)


@pytest.mark.parametrize("context", [
    {"binary": b"\x00\x01"},
    {"too_long": "x" * 513},
    {"x" * 65: "value"},
    {f"k{i}": i for i in range(51)},
    {"a": {"b": {"c": {"d": {"e": "too deep"}}}}},
    {f"k{i}": "x" * 512 for i in range(17)},
])
def test_source_context_rejects_bounds_violations(context: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid_source_context"):
        canonicalize_source_context(context)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Visual_Identities/test_visual_identity_source_context.py -q
```

Expected: FAIL because `source_context.py` does not exist.

- [ ] **Step 4: Implement the helper**

Implement constants and helpers:

```python
MAX_SOURCE_CONTEXT_BYTES = 8 * 1024
MAX_SOURCE_CONTEXT_DEPTH = 4
MAX_SOURCE_CONTEXT_KEYS = 50
MAX_SOURCE_CONTEXT_KEY_LENGTH = 64
MAX_SOURCE_CONTEXT_STRING_LENGTH = 512
PROMPT_TEXT_KEYS = {"prompt", "negative_prompt", "system_prompt", "user_prompt", "prompt_text"}
PROMPT_REFERENCE_KEYS = {"prompt_id", "prompt_ref", "prompt_label"}
```

Functions:

```python
def canonicalize_source_context(value: object) -> dict[str, Any]:
    """Return a bounded, deterministic source context object."""


def source_context_payload_hash(value: Mapping[str, Any]) -> str:
    """Hash canonical source context JSON using sorted keys."""
```

Implementation requirements:

- Root must be `Mapping`.
- Nested mappings and arrays are allowed only within depth and total key-count limits.
- Total keys across the root and all nested mappings must not exceed 50.
- Keys must be strings from 1 to 64 characters.
- Scalar strings must be at most 512 characters.
- Scalar values are limited to JSON-compatible strings, numbers, booleans, and null; reject bytes and unsupported object values.
- Reject strings beginning with `data:`.
- Reject prompt text keys case-insensitively unless they are one of the explicit reference keys.
- Reject long base64-like strings with a conservative helper.
- Serialize with `json.dumps(..., sort_keys=True, separators=(",", ":"))` and reject if the UTF-8 payload exceeds 8 KiB.

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Visual_Identities/test_visual_identity_source_context.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Visual_Identities/source_context.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_source_context.py
git commit -m "TASK-12090.3 add visual identity source context validation"
```

### Task 2: Persist Asset Source Context Through DB And Activation

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py`
- Modify: `tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py`
- Modify: `tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py`

- [ ] **Step 1: Write failing DB migration and create-asset tests**

Add to `test_visual_identity_db.py`:

```python
def test_asset_source_context_column_is_added_idempotently(chacha_db: CharactersRAGDB) -> None:
    ensure_visual_identity_tables(chacha_db)
    ensure_visual_identity_tables(chacha_db)

    columns = {
        row[1]
        for row in chacha_db.execute_query(
            "PRAGMA table_info(visual_identity_assets)"
        ).fetchall()
    }
    assert "source_context_json" in columns


def test_create_asset_persists_source_context(chacha_db: CharactersRAGDB) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    draft = repo.create_draft(
        owner_user_id=1,
        title="Context Draft",
        source_kind="generated",
        status="ready_for_review",
    )
    asset = repo.create_asset(
        owner_user_id=1,
        draft_id=draft["id"],
        expression_key="happy",
        source_filename="happy.webp",
        storage_relpath="packs/draft-1/happy.webp",
        content_type="image/webp",
        bytes=12,
        sha256="sha256-happy",
        width=64,
        height=64,
        source_context={"source_feature": "vn_assets", "generated_file_id": 42},
    )

    assert json.loads(asset["source_context_json"]) == {
        "generated_file_id": 42,
        "source_feature": "vn_assets",
    }
```

- [ ] **Step 2: Write failing activation-copy test**

Add to `test_visual_identity_service.py`:

```python
def test_activation_copies_asset_source_context_to_version(
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    draft = repo.create_draft(
        owner_user_id=OWNER_USER_ID,
        title="Context Draft",
        source_kind="generated",
        status="ready_for_review",
        default_expression_key="neutral",
    )
    repo.create_asset(
        owner_user_id=OWNER_USER_ID,
        draft_id=draft["id"],
        expression_key="neutral",
        source_filename="neutral.webp",
        storage_relpath="visual_identities/neutral.webp",
        content_type="image/webp",
        bytes=12,
        sha256="sha256-neutral-context",
        width=64,
        height=64,
        source_context={"source_feature": "vn_assets", "generated_file_id": 42},
    )

    activation = service.activate_draft(draft_id=draft["id"])
    version_assets = repo.list_assets_for_version(
        activation.pack_version_id,
        owner_user_id=OWNER_USER_ID,
    )

    assert json.loads(version_assets[0]["source_context_json"]) == {
        "generated_file_id": 42,
        "source_feature": "vn_assets",
    }
    manifest = json.loads(
        repo.get_pack_version(
            activation.pack_version_id,
            owner_user_id=OWNER_USER_ID,
        )["manifest_json"]
    )
    assert manifest["assets"][0]["source_context"] == {
        "generated_file_id": 42,
        "source_feature": "vn_assets",
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py \
  -q
```

Expected: FAIL because `visual_identity_assets.source_context_json` and `create_asset(source_context=...)` are not implemented.

- [ ] **Step 4: Implement DB schema, migration, and create_asset support**

In `VisualIdentity_DB.py`:

- Add `source_context_json TEXT NOT NULL DEFAULT '{}'` to `visual_identity_assets`.
- Add `_ensure_visual_identity_asset_columns(conn)` and call it from `ensure_visual_identity_tables`.
- Add `source_context: Mapping[str, Any] | None = None` to `create_asset`.
- Canonicalize source context before storing.
- Include `source_context_json` in the asset insert.

Use this shape:

```python
def _ensure_visual_identity_asset_columns(conn: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(visual_identity_assets)").fetchall()
    }
    if "source_context_json" not in columns:
        conn.execute(
            "ALTER TABLE visual_identity_assets ADD COLUMN source_context_json TEXT NOT NULL DEFAULT '{}'"
        )
```

- [ ] **Step 5: Implement activation copy and manifest source context**

In `activate_draft_as_version`:

- Select draft asset `source_context_json`.
- Insert version asset `source_context_json`.
- Add `source_context_json` to `copied_assets`.
- Update `_build_activation_manifest` so asset entries include:

```python
"source_context": _json_loads_dict(asset.get("source_context_json"))
```

Add a small private JSON dict loader if one does not already exist.

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py \
  -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py
git commit -m "TASK-12090.3 persist visual identity asset provenance"
```

### Task 3: API Source Context Schema And Idempotency

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/visual_identity_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/visual_identities.py`
- Modify: `tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py`

- [ ] **Step 1: Write failing API response and source-context tests**

Add tests to `test_visual_identities_api.py`:

```python
def test_generated_file_asset_import_returns_source_context(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    storage_root: Path,
    outputs_root: Path,
) -> None:
    pack = repo.create_pack(owner_user_id=1, title="Generated Expressions")
    source_path = outputs_root / "1" / "image_gen" / "happy.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(_png_bytes(color="blue"))
    files_repo = FakeGeneratedFilesRepo({
        77: {
            "id": 77,
            "user_id": 1,
            "is_deleted": False,
            "file_category": "image",
            "source_feature": "image_gen",
            "storage_path": "image_gen/happy.png",
            "mime_type": "image/png",
            "original_filename": "happy.png",
        }
    })

    response = _client(chacha_db, files_repo=files_repo).post(
        f"/api/v1/visual-identities/packs/{pack['id']}/assets/from-generated-file",
        json={
            "generated_file_id": 77,
            "expression_key": "happy",
            "source_context": {"source_feature": "image_gen", "generated_file_id": 77},
            "idempotency_key": "generated-context-1",
        },
    )

    assert response.status_code == 201
    assert response.json()["source_context"] == {
        "generated_file_id": 77,
        "source_feature": "image_gen",
    }
```

- [ ] **Step 2: Write failing idempotency-context tests**

Add:

```python
def test_generated_file_asset_import_idempotency_uses_canonical_source_context(...):
    # Same idempotency key with reordered equivalent source_context replays.
    # Same key with materially different source_context returns 409.
```

Use the existing `test_generated_file_asset_import_replays_idempotency` fixture pattern and assert only one generated file access for replay.

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py -q
```

Expected: FAIL because request/response schemas and payload hash do not include asset source context.

- [ ] **Step 4: Implement schema fields**

In `visual_identity_schemas.py`:

```python
class VisualIdentityAssetResponse(BaseModel):
    ...
    source_context: dict[str, Any] = Field(default_factory=dict)


class VisualIdentityGeneratedFileAssetRequest(BaseModel):
    ...
    source_context: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 5: Implement endpoint response and hash behavior**

In `visual_identities.py`:

- Import `canonicalize_source_context`.
- Include `source_context=_json_mapping(row, "source_context_json")` in `_asset_response`.
- Canonicalize request source context before idempotency claim.
- Include canonical source context in `_canonical_payload_hash`.
- Pass canonical source context into `_create_asset_from_stored_metadata`.

Expected payload hash shape:

```python
canonical_context = canonicalize_source_context(request.source_context)
payload_hash = _canonical_payload_hash({
    "pack_id": pack_id,
    "generated_file_id": request.generated_file_id,
    "expression_key": normalized_expression,
    "draft_id": request.draft_id,
    "source_feature": request.source_feature,
    "source_context": canonical_context,
})
```

- [ ] **Step 6: Extend `_create_asset_from_stored_metadata`**

Add a `source_context` parameter and pass it to `repo.create_asset`.

Manual uploads use `{}` by default.

- [ ] **Step 7: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py -q
```

Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/visual_identity_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/visual_identities.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py
git commit -m "TASK-12090.3 expose asset provenance in visual identity API"
```

### Task 4: VN Bridge Provenance Helper

**Files:**
- Create: `tldw_Server_API/app/core/Visual_Identities/vn_bridge.py`
- Create: `tldw_Server_API/tests/Visual_Identities/test_visual_identity_vn_bridge.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/visual_identities.py`
- Modify: `tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py`

- [ ] **Step 1: Write failing pure helper tests**

Create `test_visual_identity_vn_bridge.py`:

```python
def test_vn_bridge_derives_trusted_context_from_generated_file(...) -> None:
    context = build_vn_visual_identity_source_context(
        user_id=OWNER_USER_ID,
        vn_repository=vn_repository,
        generated_file_record={
            "id": 42,
            "source_feature": "vn_assets",
            "source_ref": "vn_asset_item:29",
            "mime_type": "image/webp",
            "original_filename": "maya_happy.webp",
        },
        requested_context={
            "source_feature": "client-lie",
            "generated_file_id": 999,
            "vn_item_id": 29,
            "vn_slot_label": "Happy",
        },
    )

    assert context["source_feature"] == "vn_assets"
    assert context["generated_file_id"] == 42
    assert context["filename"] == "maya_happy.webp"
    assert context["source_ref"] == "vn_asset_item:29"
    assert context["vn_item_id"] == 29
    assert context["vn_slot_label"] == "Happy"
```

Add a repository-backed structural ID test. Use existing VN Asset DB fixtures if available; otherwise use a small fake repository with `get_item`, `get_slot`, and `get_pack` methods:

```python
def test_vn_bridge_verifies_structural_vn_ids_before_persisting(...):
    # VN item 29 belongs to slot 11; slot 11 belongs to pack 7; pack 7 belongs to OWNER_USER_ID.
    context = build_vn_visual_identity_source_context(
        user_id=OWNER_USER_ID,
        vn_repository=vn_repository,
        generated_file_record={
            "id": 42,
            "source_feature": "vn_assets",
            "source_ref": "vn_asset_item:29",
            "mime_type": "image/webp",
            "original_filename": "maya_happy.webp",
        },
        requested_context={
            "vn_item_id": 29,
            "vn_pack_id": 7,
            "vn_slot_id": 11,
            "vn_slot_key": "happy",
            "vn_asset_type": "character_sprite",
        },
    )

    assert context["vn_item_id"] == 29
    assert context["vn_pack_id"] == 7
    assert context["vn_slot_id"] == 11
    assert context["vn_slot_key"] == "happy"
    assert context["vn_asset_type"] == "character_sprite"
```

Add mismatch test:

```python
def test_vn_bridge_rejects_item_source_ref_mismatch() -> None:
    with pytest.raises(ValueError, match="vn_generated_file_context_mismatch"):
        build_vn_visual_identity_source_context(
            user_id=OWNER_USER_ID,
            vn_repository=vn_repository,
            generated_file_record={"id": 42, "source_feature": "vn_assets", "source_ref": "vn_asset_item:29"},
            requested_context={"vn_item_id": 30},
        )


@pytest.mark.parametrize("requested_context", [
    {"vn_item_id": 29, "vn_pack_id": 8},
    {"vn_item_id": 29, "vn_slot_id": 12},
    {"vn_item_id": 29, "vn_slot_key": "sad"},
    {"vn_item_id": 29, "vn_asset_type": "background"},
])
def test_vn_bridge_rejects_unverified_structural_hints(requested_context: dict[str, object], ...):
    with pytest.raises(ValueError, match="vn_generated_file_context_mismatch"):
        build_vn_visual_identity_source_context(
            user_id=OWNER_USER_ID,
            vn_repository=vn_repository,
            generated_file_record={"id": 42, "source_feature": "vn_assets", "source_ref": "vn_asset_item:29"},
            requested_context=requested_context,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Visual_Identities/test_visual_identity_vn_bridge.py -q
```

Expected: FAIL because `vn_bridge.py` does not exist.

- [ ] **Step 3: Implement `vn_bridge.py`**

Implement:

```python
VN_SOURCE_FEATURE = SOURCE_FEATURE_VN_ASSETS


def build_vn_visual_identity_source_context(
    *,
    user_id: str,
    vn_repository: VNAssetPacksRepositoryProtocol,
    generated_file_record: Mapping[str, Any],
    requested_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ...
```

Rules:

- Always derive `source_feature`, `generated_file_id`, `filename`, `mime_type`, and `source_ref` from generated-file record.
- Determine `vn_item_id` from request context when present; otherwise parse it from `source_ref` when it uses `vn_asset_item:{item_id}`.
- If `vn_item_id` is present or derived, require `source_ref == vn_asset_source_ref(vn_item_id)`.
- Verify `vn_item_id` by loading the VN item for the current user before storing it.
- Verify `vn_slot_id`, `vn_slot_key`, `vn_pack_id`, and `vn_asset_type` against the loaded item, slot, and pack before storing them.
- If any structural VN keys are supplied but no VN item can be verified, reject the request with `ValueError("vn_generated_file_context_mismatch")`.
- Reject mismatched structural hints with `ValueError("vn_generated_file_context_mismatch")`.
- Do not persist client-provided structural VN IDs that were not verified against VN data.
- Keep short client display labels such as `vn_slot_label` as labels only.
- Canonicalize using `canonicalize_source_context`.
- Raise `ValueError("vn_generated_file_context_mismatch")` for source-ref mismatch.

- [ ] **Step 4: Run helper tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Visual_Identities/test_visual_identity_vn_bridge.py -q
```

Expected: PASS.

- [ ] **Step 5: Write failing API VN-source tests**

In `test_visual_identities_api.py`, add:

```python
def test_generated_file_asset_import_records_vn_context_and_rejects_item_mismatch(...):
    # Build generated-file record with source_feature "vn_assets" and source_ref "vn_asset_item:29".
    # Seed VN pack/slot/item rows that verify item 29 belongs to the requesting user.
    # POST with source_context {"vn_item_id": 29, "vn_pack_id": 7, "vn_slot_id": 11, "vn_slot_key": "happy", "vn_slot_label": "Happy"} returns 201 and derived context.
    # POST with source_context {"vn_item_id": 30} returns 422 or 404 and creates no asset rows.
    # POST with mismatched vn_pack_id/vn_slot_id/vn_slot_key returns 422 or 404 and creates no asset rows.
```

- [ ] **Step 6: Integrate VN bridge in endpoint**

In `create_visual_identity_asset_from_generated_file`:

- If normalized `source_feature` equals `SOURCE_FEATURE_VN_ASSETS`, call `build_vn_visual_identity_source_context`.
- Pass the current `user_id` and a VN Asset repository instance into the bridge helper.
- Otherwise canonicalize generic `request.source_context`.
- Use the resolved context for idempotency hash and `repo.create_asset`.
- Map `vn_generated_file_context_mismatch` through `_handle_value_error`.

- [ ] **Step 7: Run API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_vn_bridge.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Visual_Identities/vn_bridge.py \
  tldw_Server_API/app/api/v1/endpoints/visual_identities.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_vn_bridge.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py
git commit -m "TASK-12090.3 add VN generated-file provenance bridge"
```

### Task 5: Frontend Generated-File Import Action

**Files:**
- Modify: `apps/packages/ui/src/types/visual-identities.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/visual-identities.ts`
- Create: `apps/packages/ui/src/components/Common/VisualIdentity/useGeneratedFileImportAction.ts`
- Create: `apps/packages/ui/src/components/Common/VisualIdentity/__tests__/useGeneratedFileImportAction.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.visual-identities.test.ts`

- [ ] **Step 1: Write failing client contract test**

In `tldw-api-client.visual-identities.test.ts`, add:

```ts
it("imports a generated file asset with source context", async () => {
  mocks.bgRequest.mockResolvedValue({ id: 12 })

  await visualIdentityMethods.createVisualIdentityAssetFromGeneratedFile.call({}, 5, {
    generated_file_id: 42,
    expression_key: "happy",
    draft_id: 7,
    source_feature: "vn_assets",
    source_context: { vn_item_id: 29, vn_slot_label: "Happy" },
    idempotency_key: "vn-assets:42:happy"
  })

  expect(mocks.bgRequest).toHaveBeenCalledWith({
    path: "/api/v1/visual-identities/packs/5/assets/from-generated-file",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: {
      generated_file_id: 42,
      expression_key: "happy",
      draft_id: 7,
      source_feature: "vn_assets",
      source_context: { vn_item_id: 29, vn_slot_label: "Happy" },
      idempotency_key: "vn-assets:42:happy"
    }
  })
})
```

- [ ] **Step 2: Write failing hook tests**

Create `useGeneratedFileImportAction.test.ts` with a fake client:

```ts
it("returns assigned after import and slot update succeed", async () => {
  const client = {
    createVisualIdentityAssetFromGeneratedFile: vi.fn(async () => ({ id: 44 })),
    updateVisualIdentityDraftSlot: vi.fn(async () => ({ id: 7 }))
  }
  const result = await importGeneratedFileAndAssignSlot({
    client,
    packId: 5,
    draftId: 7,
    slotKey: "happy",
    generatedFileId: 42,
    sourceContext: { vn_item_id: 29 },
  })

  expect(result).toEqual({ status: "assigned", assetId: 44, slotKey: "happy" })
})
```

Also test:

- `imported_unassigned` when slot update throws after import.
- `failed` when import throws.
- Exact import payload defaults `source_feature` to `vn_assets`.

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/services/__tests__/tldw-api-client.visual-identities.test.ts \
  src/components/Common/VisualIdentity/__tests__/useGeneratedFileImportAction.test.ts
```

Expected: FAIL because type/hook support does not exist.

- [ ] **Step 4: Extend frontend types**

In `visual-identities.ts`:

```ts
export interface VisualIdentityAssetResponse {
  ...
  source_context: Record<string, unknown>
}

export interface VisualIdentityGeneratedFileAssetRequest {
  ...
  source_context?: Record<string, unknown>
}
```

- [ ] **Step 5: Implement reusable action helper and hook**

Create `useGeneratedFileImportAction.ts`:

```ts
export type GeneratedFileImportActionResult =
  | { status: "assigned"; assetId: number; slotKey: string }
  | { status: "imported_unassigned"; assetId: number; slotKey: string; error: unknown }
  | { status: "failed"; error: unknown }

export async function importGeneratedFileAndAssignSlot(args: ImportGeneratedFileAndAssignSlotArgs): Promise<GeneratedFileImportActionResult> {
  try {
    const asset = await args.client.createVisualIdentityAssetFromGeneratedFile(args.packId, {
      generated_file_id: args.generatedFileId,
      expression_key: args.slotKey,
      draft_id: args.draftId,
      source_feature: args.sourceFeature ?? "vn_assets",
      source_context: args.sourceContext ?? {},
      idempotency_key: args.idempotencyKey ?? `vn-assets:${args.generatedFileId}:pack:${args.packId}:draft:${args.draftId}:${args.slotKey}`,
    })
    try {
      await args.client.updateVisualIdentityDraftSlot(args.draftId, args.slotKey, {
        asset_id: asset.id,
        expression_key: args.slotKey,
      })
      return { status: "assigned", assetId: asset.id, slotKey: args.slotKey }
    } catch (error) {
      return { status: "imported_unassigned", assetId: asset.id, slotKey: args.slotKey, error }
    }
  } catch (error) {
    return { status: "failed", error }
  }
}
```

Also export a `useGeneratedFileImportAction` hook that memoizes a callback around the helper and defaults to `tldwClient`.

- [ ] **Step 6: Run frontend tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/services/__tests__/tldw-api-client.visual-identities.test.ts \
  src/components/Common/VisualIdentity/__tests__/useGeneratedFileImportAction.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/types/visual-identities.ts \
  apps/packages/ui/src/services/tldw/domains/visual-identities.ts \
  apps/packages/ui/src/components/Common/VisualIdentity/useGeneratedFileImportAction.ts \
  apps/packages/ui/src/components/Common/VisualIdentity/__tests__/useGeneratedFileImportAction.test.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.visual-identities.test.ts
git commit -m "TASK-12090.3 add VN generated-file import action"
```

### Task 6: Stage 11A Verification Checkpoint

**Files:**
- Modify: `backlog/tasks/task-12090.3 - Plan-VN-visual-identity-bridge-and-resolver-implementation.md` if recording results works through MCP, otherwise record results in final response.

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_source_context.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_vn_bridge.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/services/__tests__/tldw-api-client.visual-identities.test.ts \
  src/components/Common/VisualIdentity/__tests__/useGeneratedFileImportAction.test.ts
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on Stage 11A backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Visual_Identities \
  tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py \
  tldw_Server_API/app/api/v1/endpoints/visual_identities.py \
  tldw_Server_API/app/api/v1/schemas/visual_identity_schemas.py \
  -f json -o /tmp/bandit_vn_visual_identity_stage11a.json
```

Expected: exit 0 or only pre-existing non-touched findings documented with rationale.

- [ ] **Step 4: Commit verification notes if any repository file changes were made**

Only commit if Backlog metadata was successfully updated through MCP:

```bash
git add "backlog/tasks/task-12090.3 - Plan-VN-visual-identity-bridge-and-resolver-implementation.md"
git commit -m "TASK-12090.3 record Stage 11A verification"
```

---

## Stage 11B: VN Role/Casting Resolver

### Task 7: Service Resolver Override Semantics

**Files:**
- Modify: `tldw_Server_API/app/core/Visual_Identities/service.py`
- Modify: `tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py`

Resolution source values are part of the contract:

- `binding`: actor binding resolved the requested expression exactly.
- `binding_fallback`: actor binding pack used a configured/default/neutral fallback.
- `override`: explicit override pack/version resolved the requested expression exactly.
- `override_fallback`: explicit override pack/version used a configured/default/neutral fallback.
- `override_binding_fallback`: override pack/version had no usable asset and `allow_override_fallback=true` fell through to normal binding resolution.
- `override_legacy_fallback`: override pack/version had no usable asset and `allow_override_fallback=true` fell through to character legacy mood resolution.
- `override_placeholder_fallback`: override pack/version had no usable asset and `allow_override_fallback=true` fell through to placeholder resolution.
- `legacy_character_mood`: character actor used legacy mood image fallback without an override path.
- `placeholder`: no pack or legacy asset was available.

- [x] **Step 1: Write failing service tests for strict override**

Add to `test_visual_identity_service.py`:

```python
def test_resolver_explicit_override_resolves_requested_expression(...):
    # Create actor binding pack A and separate pack B/version with happy asset.
    # Resolve actor with override_pack_id=B, override_pack_version_id=Bv, expression_key="happy".
    # Assert returned pack/version is B/Bv and resolution_source == "override".


def test_resolver_override_missing_expression_is_strict_by_default(...):
    # Valid override pack/version exists but has no "sad" or default asset.
    # Resolve expression_key="sad".
    # Assert ValueError("override_expression_missing").


def test_resolver_rejects_override_pack_version_mismatch(...):
    # Version belongs to a different pack than override_pack_id.
    # Assert ValueError("pack_version_mismatch") and no actor binding fallback.
```

- [x] **Step 2: Write failing fallback and actor-kind tests**

Add:

```python
def test_resolver_override_fallback_opt_in_records_reason(...):
    # allow_override_fallback=True falls back to neutral inside override pack.
    # Assert fallback_reason is not None and resolution_source indicates override fallback.


def test_resolver_override_fallback_opt_in_can_fall_through_to_normal_binding(...):
    # Valid override pack/version exists but has no requested/default/neutral asset.
    # Actor has a normal binding with the requested expression.
    # Resolve with allow_override_fallback=True.
    # Assert returned asset comes from the normal binding, fallback_reason includes override_expression_missing,
    # and resolution_source == "override_binding_fallback".


def test_persona_without_pack_does_not_use_character_legacy_mood(...):
    # Create character with legacy mood image and persona with no binding.
    # Resolve persona actor.
    # Assert placeholder, not legacy_character_mood.


def test_resolver_rejects_invalid_actor(...):
    # Resolve an actor_id that does not exist for the requested actor_kind.
    # Assert ValueError("visual_identity_character_not_found") or ValueError("visual_identity_persona_not_found") and no placeholder masking.


def test_resolver_rejects_cross_user_override_pack(...):
    # Create an override pack owned by a different user.
    # Assert ValueError("pack_not_found") or ValueError("pack_not_owned") and no default binding fallback.
```

- [x] **Step 3: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py -q
```

Expected: FAIL because override arguments are not implemented.

- [x] **Step 4: Extend dataclass and service method signature**

In `VisualIdentityResolvedAsset`, add:

```python
role_id: str | None = None
role_label: str | None = None
resolution_source: str = "binding"
```

Extend `resolve_expression_asset` with keyword-only args:

```python
role_id: str | None = None,
role_label: str | None = None,
override_pack_id: int | None = None,
override_pack_version_id: int | None = None,
allow_override_fallback: bool = False,
```

- [x] **Step 5: Implement override validation**

Add private helpers:

```python
def _require_owned_override_version(
    self,
    *,
    pack_id: int,
    pack_version_id: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ...
```

Rules:

- Actor kind/id must be validated before resolving bindings or overrides. Reuse existing typed errors: `visual_identity_actor_kind_invalid`, `visual_identity_character_not_found`, and `visual_identity_persona_not_found`.
- Pack missing or not owned: `pack_not_found` or `pack_not_owned`.
- Pack version missing: `pack_version_not_found`.
- Version exists but not for pack: `pack_version_mismatch`.
- Do not fall back to actor default binding when override validation fails.

- [x] **Step 6: Implement override expression lookup**

Use `repository.list_assets_for_version`. Candidate order for override:

1. requested expression
2. pack default expression, only when `allow_override_fallback=True`
3. `neutral`, `default`, `normal`, only when `allow_override_fallback=True`

If no candidate matches and fallback is not allowed, raise `ValueError("override_expression_missing")`.

If no candidate matches and fallback is allowed:

1. Re-run the normal resolver with override arguments cleared.
2. Preserve `role_id` and `role_label`.
3. Set `fallback_reason` to include `override_expression_missing` even when the normal resolver also has its own fallback reason.
4. Set `resolution_source` to `override_binding_fallback`, `override_legacy_fallback`, or `override_placeholder_fallback` based on the normal resolver result.

Do not use character legacy mood images for persona actors in either normal or override fallback paths.

- [x] **Step 7: Run service tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Visual_Identities/service.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py
git commit -m "TASK-12090.3 add visual identity casting resolver overrides"
```

### Task 8: Resolver API Extension

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/visual_identity_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/visual_identities.py`
- Modify: `tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py`

- [x] **Step 1: Write failing endpoint compatibility and override tests**

In `test_visual_identities_api.py`, add:

```python
def test_resolve_endpoint_preserves_existing_query_contract(...):
    # Use existing actor binding setup.
    # GET /bindings/resolve with existing params only.
    # Assert status 200 and no role fields required from caller.


def test_resolve_endpoint_accepts_role_override_fields(...):
    # Create override pack/version with happy asset.
    # GET /bindings/resolve with role_id, role_label, override_pack_id,
    # override_pack_version_id, allow_override_fallback=false.
    # Assert response includes role_id, role_label, resolution_source.
```

- [x] **Step 2: Write failing typed error tests**

Add:

```python
def test_resolve_endpoint_reports_override_expression_missing(...):
    response = client.get("/api/v1/visual-identities/bindings/resolve", params={...})
    assert response.status_code in {409, 422}
    assert response.json()["detail"] == "override_expression_missing"


def test_resolve_endpoint_reports_invalid_actor_without_placeholder_masking(...):
    response = client.get("/api/v1/visual-identities/bindings/resolve", params={...})
    assert response.status_code in {404, 422}
    assert response.json()["detail"] in {
        "visual_identity_actor_kind_invalid",
        "visual_identity_character_not_found",
        "visual_identity_persona_not_found",
    }


def test_resolve_endpoint_reports_cross_user_override_pack(...):
    response = client.get("/api/v1/visual-identities/bindings/resolve", params={...})
    assert response.status_code == 404
    assert response.json()["detail"] in {"pack_not_found", "pack_not_owned"}


def test_resolve_endpoint_reports_pack_version_mismatch(...):
    response = client.get("/api/v1/visual-identities/bindings/resolve", params={...})
    assert response.status_code in {409, 422}
    assert response.json()["detail"] == "pack_version_mismatch"
```

Use existing `_handle_value_error` conventions. If needed, map override errors to 409 for valid resource but missing expression and 404 for not-found ownership failures.

- [x] **Step 3: Run endpoint tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py -q
```

Expected: FAIL because endpoint query params and response fields are missing.

- [x] **Step 4: Extend Pydantic response schema**

In `VisualIdentityResolveResponse`:

```python
role_id: str | None = None
role_label: str | None = None
resolution_source: str | None = None
```

- [x] **Step 5: Extend endpoint query parameters**

In `resolve_visual_identity_binding` add:

```python
role_id: str | None = Query(default=None),
role_label: str | None = Query(default=None),
override_pack_id: int | None = Query(default=None, ge=1),
override_pack_version_id: int | None = Query(default=None, ge=1),
allow_override_fallback: bool = Query(default=False),
```

Pass these through to `service.resolve_expression_asset`.

- [x] **Step 6: Include response fields**

Return:

```python
role_id=resolved.role_id,
role_label=resolved.role_label,
resolution_source=resolved.resolution_source,
```

- [x] **Step 7: Run endpoint tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py -q
```

Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/visual_identity_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/visual_identities.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py
git commit -m "TASK-12090.3 expose visual identity casting resolver API"
```

### Task 9: Frontend Resolver Types And Cache Keys

**Files:**
- Modify: `apps/packages/ui/src/types/visual-identities.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/visual-identities.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.visual-identities.test.ts`
- Modify: `apps/packages/ui/src/hooks/useVisualIdentityResolver.ts`
- Modify: `apps/packages/ui/src/hooks/__tests__/useVisualIdentityResolver.test.tsx`

- [ ] **Step 1: Write failing client query test**

Add to `tldw-api-client.visual-identities.test.ts`:

```ts
it("resolves actor bindings with role override query parameters", async () => {
  mocks.bgRequest.mockResolvedValue({ asset_id: 9 })

  await visualIdentityMethods.resolveVisualIdentityBinding.call({}, {
    actor_kind: "character",
    actor_id: 123,
    expression_key: "happy",
    role_id: "hero",
    role_label: "Hero",
    override_pack_id: 5,
    override_pack_version_id: 6,
    allow_override_fallback: true
  })

  expect(mocks.bgRequest).toHaveBeenCalledWith({
    path:
      "/api/v1/visual-identities/bindings/resolve?actor_kind=character&actor_id=123&expression_key=happy&role_id=hero&role_label=Hero&override_pack_id=5&override_pack_version_id=6&allow_override_fallback=true",
    method: "GET"
  })
})
```

- [ ] **Step 2: Write failing hook cache-key test**

In `useVisualIdentityResolver.test.tsx`, add a test that renders twice with same actor/expression but different `override_pack_id`, and assert the fake client is called for the second override instead of returning stale cached default-binding data.

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/services/__tests__/tldw-api-client.visual-identities.test.ts \
  src/hooks/__tests__/useVisualIdentityResolver.test.tsx
```

Expected: FAIL because query and cache-key fields are missing.

- [ ] **Step 4: Extend frontend types**

In `VisualIdentityResolveRequest`:

```ts
role_id?: string | null
role_label?: string | null
override_pack_id?: number | null
override_pack_version_id?: number | null
allow_override_fallback?: boolean | null
```

In `VisualIdentityResolveResponse`:

```ts
role_id?: string | null
role_label?: string | null
resolution_source?: string | null
```

- [ ] **Step 5: Extend client query builder**

In `resolveVisualIdentityBinding`, include the new optional fields in `buildQuery`.

- [ ] **Step 6: Extend `useVisualIdentityResolver` options and cache key**

Add options:

```ts
roleId?: string | null
roleLabel?: string | null
overridePackId?: number | null
overridePackVersionId?: number | null
allowOverrideFallback?: boolean | null
```

Include these in `buildResolverCacheKey` and pass them to `client.resolveVisualIdentityBinding`.

- [ ] **Step 7: Run frontend tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/services/__tests__/tldw-api-client.visual-identities.test.ts \
  src/hooks/__tests__/useVisualIdentityResolver.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/types/visual-identities.ts \
  apps/packages/ui/src/services/tldw/domains/visual-identities.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.visual-identities.test.ts \
  apps/packages/ui/src/hooks/useVisualIdentityResolver.ts \
  apps/packages/ui/src/hooks/__tests__/useVisualIdentityResolver.test.tsx
git commit -m "TASK-12090.3 add frontend casting resolver parameters"
```

### Task 10: Stage 11B And Final Verification

**Files:**
- Modify: `Docs/superpowers/plans/2026-07-02-vn-visual-identity-bridge-implementation-plan.md` only if updating plan task checkboxes during execution.
- Modify: Backlog task metadata through MCP if available.

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_source_context.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_vn_bridge.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/services/__tests__/tldw-api-client.visual-identities.test.ts \
  src/components/Common/VisualIdentity/__tests__/useGeneratedFileImportAction.test.ts \
  src/hooks/__tests__/useVisualIdentityResolver.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run TypeScript diagnostics for frontend package**

Run:

```bash
cd apps/packages/ui && bunx tsc --noEmit --pretty false
```

Expected: PASS. If the full package has known baseline diagnostics, record every diagnostic that references new/touched Stage 11 frontend files and fix those before finishing.

- [ ] **Step 4: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Visual_Identities \
  tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py \
  tldw_Server_API/app/api/v1/endpoints/visual_identities.py \
  tldw_Server_API/app/api/v1/schemas/visual_identity_schemas.py \
  -f json -o /tmp/bandit_vn_visual_identity_stage11.json
```

Expected: exit 0 or only pre-existing non-touched findings documented with rationale.

- [ ] **Step 5: Run final status and diff review**

Run:

```bash
git status --short
git diff --stat
```

Expected: only intended implementation/plan/Backlog files are modified.

- [ ] **Step 6: Commit final verification metadata if needed**

If any plan or Backlog metadata was updated:

```bash
git add Docs/superpowers/plans/2026-07-02-vn-visual-identity-bridge-implementation-plan.md \
  "backlog/tasks/task-12090.3 - Plan-VN-visual-identity-bridge-and-resolver-implementation.md"
git commit -m "TASK-12090.3 record VN bridge verification"
```

---

## Implementation Notes

- Use TDD for each task: write failing tests first, run them, implement the smallest code path, run focused tests, then commit.
- Keep Stage 11A reviewable independently. Do not start Stage 11B if Stage 11A tests or Bandit fail.
- Keep provenance validation server-side. The frontend may send hints, but backend must derive trusted fields from generated-file and VN data.
- Preserve existing resolver behavior when no new Stage 11B query parameters are supplied.
- Do not alter the unrelated untracked watchlist template files currently present in the worktree.
