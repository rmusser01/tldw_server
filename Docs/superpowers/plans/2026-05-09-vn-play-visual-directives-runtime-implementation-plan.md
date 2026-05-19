# VN Play Visual Directives Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve VN Play turn `visual_directives` against approved VN asset packs and return scene-ready visual payloads from `/api/v1/vn-play`.

**Architecture:** Keep VN Play backend-authoritative for visual state. The durable scene state remains compact and replayable using approved asset IDs plus sprite payload JSON; API responses enrich that state from the approved manifest so custom frontends do not need to call VN asset-pack internals. Directive failures append audit events and warnings without failing the narrative turn.

**Tech Stack:** FastAPI, Pydantic, SQLite-backed ChaChaNotes repositories, VNAssetPackService manifest builder, pytest.

---

## File Structure

- Modify `tldw_Server_API/app/core/VN_Play/assets.py`
  - Normalize manifest collection names and directive asset type aliases (`background`/`backgrounds`, `sprite`/`sprites`, `depth`/`depth_companion`/`depth_companions`, `cg`/`cgs`).
  - Keep deterministic approved-only resolution in the existing resolver.
- Modify `tldw_Server_API/app/core/VN_Play/state.py`
  - Replay `visual_directive_applied` events into `current_background_item_id`, `current_depth_item_id`, and `active_sprite_items`.
  - Replay rejected directives as warnings with stable reason codes.
- Modify `tldw_Server_API/app/core/VN_Play/service.py`
  - Build the approved VN asset manifest from the session pack during turn completion.
  - Append requested/applied/rejected visual directive events.
  - Merge applied directives into the `scene_state_changed` payload before persisting state.
  - Add a response enrichment helper for `background`, `depth`, and `active_sprites` in API-facing scene state payloads.
  - Treat missing pack/manifest resolver failures as non-fatal warnings once the narrative turn has been accepted.
- Modify `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
  - Return enriched scene state from session and turn responses.
- Modify `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
  - Add permissive optional `background`, `depth`, and `active_sprites` fields to `VNPlaySceneStateResponse`.
- Modify `Docs/API-related/VN_PLAY_API.md`
  - Document directive application/rejection events and scene asset payload shape.
- Add/modify tests:
  - `tldw_Server_API/tests/VN_Play/test_vn_play_assets.py`
  - `tldw_Server_API/tests/VN_Play/test_vn_play_state.py`
  - `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`
  - `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`

## Baseline

- Existing focused suite passed before changes:
  - ` /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q`
  - Result: `39 passed, 5 warnings`

---

### Task 1: Normalize Manifest Asset Resolution

**Files:**
- Modify: `tldw_Server_API/app/core/VN_Play/assets.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_assets.py`

- [x] **Step 1: Write failing resolver alias tests**

Add tests showing a runtime manifest with plural collections resolves singular directives:

```python
def test_resolver_supports_manifest_collection_aliases() -> None:
    manifest = {
        "assets": {
            "backgrounds": [
                {
                    "item_id": 10,
                    "slot_key": "background.library",
                    "asset_type": "background",
                    "labels": {"location": "library"},
                    "review_status": "approved",
                    "content_url": "/api/v1/vn-assets/packs/1/items/10/content",
                }
            ],
            "sprites": [
                {
                    "item_id": 20,
                    "slot_key": "sprite.happy",
                    "asset_type": "sprite",
                    "labels": {"emotion": "happy"},
                    "review_status": "approved",
                    "content_url": "/api/v1/vn-assets/packs/1/items/20/content",
                }
            ],
        }
    }

    background = resolve_visual_directive(
        manifest,
        {"asset_type": "background", "labels": {"location": "library"}},
        seed="seed",
    )
    sprite = resolve_visual_directive(
        manifest,
        {"asset_type": "sprite", "labels": {"emotion": "happy"}},
        seed="seed",
    )

    assert background.applied is True
    assert background.item["item_id"] == 10
    assert sprite.applied is True
    assert sprite.item["item_id"] == 20
```

- [x] **Step 2: Verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_assets.py::test_resolver_supports_manifest_collection_aliases -q
```

Expected: FAIL because `assets.py` looks up the directive `asset_type` literally.

- [x] **Step 3: Implement alias normalization**

Add a private alias helper and route `_iter_manifest_items()` through it:

```python
_ASSET_TYPE_COLLECTION_ALIASES = {
    "background": ("backgrounds", "background"),
    "backgrounds": ("backgrounds", "background"),
    "sprite": ("sprites", "sprite"),
    "sprites": ("sprites", "sprite"),
    "depth": ("depth_companions", "depth_companion"),
    "depth_companion": ("depth_companions", "depth_companion"),
    "depth_companions": ("depth_companions", "depth_companion"),
    "cg": ("cgs", "cg"),
    "cgs": ("cgs", "cg"),
}
```

Keep legacy tests with singular `assets["sprite"]` passing by checking both collection aliases and direct keys.

- [x] **Step 4: Verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_assets.py -q
```

Expected: PASS.

---

### Task 2: Replay Applied Visual Directive Events

**Files:**
- Modify: `tldw_Server_API/app/core/VN_Play/state.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_state.py`

- [x] **Step 1: Write failing replay tests**

Add tests showing applied visual directives update scene state:

```python
def test_replay_applies_visual_directive_assets() -> None:
    state = derive_scene_state(
        [
            {
                "event_type": "visual_directive_applied",
                "event_payload": {
                    "asset_type": "background",
                    "item": {"item_id": 101, "content_url": "/content/bg"},
                    "scene_version": 1,
                },
            },
            {
                "event_type": "visual_directive_applied",
                "event_payload": {
                    "asset_type": "sprite",
                    "item": {"item_id": 201, "content_url": "/content/sprite"},
                    "scene_version": 1,
                },
            },
        ]
    )

    assert state.current_background_item_id == 101
    assert state.active_sprite_items == [{"item_id": 201, "content_url": "/content/sprite"}]
    assert state.scene_version == 1
```

- [x] **Step 2: Verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_state.py::test_replay_applies_visual_directive_assets -q
```

Expected: FAIL because `visual_directive_applied` is ignored.

- [x] **Step 3: Implement replay support**

Import `EVENT_VISUAL_DIRECTIVE_APPLIED` and add `_apply_visual_directive_applied()`. For V1:
- background sets `current_background_item_id`.
- depth/depth_companion sets `current_depth_item_id`.
- sprite appends the resolved item payload to `active_sprite_items`.
- rejected directives keep existing warning behavior.

- [x] **Step 4: Verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_state.py -q
```

Expected: PASS.

---

### Task 3: Apply Directives During Turn Completion

**Files:**
- Modify: `tldw_Server_API/app/core/VN_Play/service.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`

- [x] **Step 1: Write failing service tests**

Add a custom adapter returning visual directives and fixture helpers that create an approved VN asset pack in the same ChaCha DB:

```python
class VisualDirectiveAdapter:
    async def generate_turn(self, context):
        return TurnResult(
            narrative_text="The library appears.",
            dialogue=[{"speaker": "Narrator", "text": "The library appears."}],
            visual_directives=[
                {"asset_type": "background", "labels": {"location": "library"}},
                {"asset_type": "sprite", "labels": {"emotion": "happy"}},
            ],
            scene_updates={"location_key": "library"},
        )
```

Assertions:
- response events include `visual_directive_requested` and `visual_directive_applied`.
- persisted scene state has `current_background_item_id` and `active_sprite_items`.
- repeated submit with the same idempotency key replays the same response.

- [x] **Step 2: Verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py::test_turn_applies_visual_directives_to_scene_state -q
```

Expected: FAIL because service currently stores model `visual_directives` but does not resolve or apply them.

- [x] **Step 3: Implement directive application**

In `VNPlayService._complete_turn()`:
- Pass the accepted `session` into `_complete_turn()` from `submit_turn()`.
- Build the approved manifest with `VNAssetPackService(self.repo.db, owner_user_id=self.owner_user_id).build_manifest(session.vn_asset_pack_id)`.
- Resolve `result.visual_directives` with `resolve_scene_directives()`.
- Append `visual_directive_requested` for each directive.
- Append `visual_directive_applied` with resolved item payloads.
- Append `visual_directive_rejected` with stable `reason` for misses.
- Merge applied background/depth/sprite results into `scene_payload` before `scene_state_changed`.
- If manifest loading or directive resolution raises unexpectedly, append rejected events or warnings with `reason=manifest_unavailable` or `reason=resolver_error`, keep the narrative/model turn, and include those warnings in the returned `VNPlayTurnResponse`.

- [x] **Step 4: Verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q
```

Expected: PASS.

---

### Task 4: Enrich API Scene State Responses

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`

- [x] **Step 1: Write failing API response test**

Add a test that creates an approved pack, submits a turn with a visual directive adapter, and verifies `GET /sessions/{id}` or the turn response includes:

```python
assert body["scene_state"]["background"]["content_url"].endswith("/items/<id>/content")
assert body["scene_state"]["active_sprites"][0]["content_url"].endswith("/items/<id>/content")
```

- [x] **Step 2: Verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py::test_turn_response_includes_resolved_scene_assets -q
```

Expected: FAIL because `VNPlaySceneStateResponse` has no `background`, `depth`, or `active_sprites` fields.

- [x] **Step 3: Implement API enrichment**

- Add optional `background`, `depth`, and `active_sprites` fields to `VNPlaySceneStateResponse`.
- Add a service helper such as `get_enriched_scene_state(session_id)` that loads the session, reads repository scene state, builds the approved manifest for the session pack, and enriches current item IDs into payload objects. If the manifest cannot load, return the durable scene state plus warnings rather than failing `GET /sessions/{id}`.
- Use enriched state in `_session_response()` and `_turn_response()`.

- [x] **Step 4: Verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q
```

Expected: PASS.

---

### Task 5: Documentation And Full Verification

**Files:**
- Modify: `Docs/API-related/VN_PLAY_API.md`
- Modify: `backlog/tasks/task-172 - Resolve-VN-Play-visual-directives-into-scene-assets.md`

- [x] **Step 1: Update API docs**

Document:
- `visual_directive_requested`
- `visual_directive_applied`
- `visual_directive_rejected`
- `scene_state.background`
- `scene_state.depth`
- `scene_state.active_sprites`
- Warning-only behavior for unresolved directives.

- [x] **Step 2: Run focused backend tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q
```

Expected: PASS.

- [x] **Step 3: Run API docs/schema-adjacent checks**

Run:

```bash
git diff --check
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/VN_Play tldw_Server_API/app/api/v1/endpoints/vn_play.py tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py -f json -o /tmp/bandit_vn_play_visual_directives.json
```

Expected: PASS or only pre-existing/non-touched findings with notes.

- [x] **Step 4: Update Backlog task**

Check acceptance criteria and DoD, record verification commands and known skips, and add a final summary.
