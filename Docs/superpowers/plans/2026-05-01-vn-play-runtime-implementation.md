# VN Play Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first VN Play runtime: durable Freeform and Story/CYOA sessions that consume approved VN asset pack manifests, process server-authoritative turns, persist event-sourced state, and expose a `/vn-play` workspace.

**Architecture:** Add a `VN_Play` backend module with a per-user `ChaChaNotes.db` repository, event replay/state derivation, runtime gates, approved-manifest asset resolution, turn idempotency/concurrency, and mocked-provider-friendly turn adapters. Add `/api/v1/vn-play` endpoints and a new frontend workspace that calls those endpoints and renders the current scene, dialogue, choices, warnings, and checkpoint controls. Keep V1 interactive and request/response; do not add Jobs or realtime image generation.

**Tech Stack:** FastAPI, Pydantic, SQLite via `CharactersRAGDB`, existing VN asset pack services/manifests, pytest, Next.js/React, existing frontend API helpers, Vitest, Playwright smoke tests.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-05-01-vn-play-runtime-design.md`
- Existing VN asset API docs: `Docs/API-related/VN_ASSET_PACKS_API.md`
- Existing VN asset DB/API patterns:
  - `tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py`
  - `tldw_Server_API/app/core/VN_Assets/service.py`
  - `tldw_Server_API/app/api/v1/endpoints/vn_assets.py`
  - `tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py`
  - `tldw_Server_API/tests/VN_Assets/`
  - `apps/tldw-frontend/components/vn-assets/`

## File Map

Backend files to create:

- `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
  - Owns VN Play schema migration helpers and repository methods against per-user `ChaChaNotes.db`.
- `tldw_Server_API/app/core/VN_Play/__init__.py`
  - Module exports.
- `tldw_Server_API/app/core/VN_Play/constants.py`
  - Modes, statuses, event types, turn statuses, error codes.
- `tldw_Server_API/app/core/VN_Play/models.py`
  - Internal dataclasses or typed dicts for turn result, scene state, gates, and resolver output.
- `tldw_Server_API/app/core/VN_Play/state.py`
  - Event replay and scene-state derivation.
- `tldw_Server_API/app/core/VN_Play/gates.py`
  - Session, manifest, content-rating, character safety, and provider capability gates.
- `tldw_Server_API/app/core/VN_Play/assets.py`
  - Approved-manifest asset resolver and deterministic seeded selection.
- `tldw_Server_API/app/core/VN_Play/parser.py`
  - Structured model-output parser and normalized turn result validation.
- `tldw_Server_API/app/core/VN_Play/adapters.py`
  - Freeform and Story turn adapter interfaces plus deterministic test adapter.
- `tldw_Server_API/app/core/VN_Play/service.py`
  - Orchestrates sessions, events, turn requests, retries, checkpoints, and branches.
- `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
  - API request/response schemas.
- `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
  - `/api/v1/vn-play` router.

Backend files to modify:

- `tldw_Server_API/app/main.py`
  - Register `vn_play` router behind route flag, mirroring `vn_assets`.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Only if needed for schema initialization wiring; prefer `VNPlay_DB.py` repository initialization if possible.

Backend tests to create:

- `tldw_Server_API/tests/VN_Play/test_vn_play_db.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_state.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_assets.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_gates.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`
- `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`

Frontend files to create:

- `apps/tldw-frontend/types/vn-play.ts`
- `apps/tldw-frontend/lib/api/vnPlay.ts`
- `apps/tldw-frontend/pages/vn-play.tsx`
- `apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx`
- `apps/tldw-frontend/components/vn-play/SessionList.tsx`
- `apps/tldw-frontend/components/vn-play/SceneStage.tsx`
- `apps/tldw-frontend/components/vn-play/DialoguePanel.tsx`
- `apps/tldw-frontend/components/vn-play/ChoicePanel.tsx`
- `apps/tldw-frontend/components/vn-play/SceneInspector.tsx`
- `apps/tldw-frontend/components/vn-play/NewSessionDialog.tsx`

Frontend tests to create:

- `apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts`
- `apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx`
- `apps/tldw-frontend/__tests__/vn-play/SceneStage.test.tsx`
- `apps/tldw-frontend/e2e/smoke/vn-play.spec.ts`

## Task 1: VN Play Database Schema And Repository

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_db.py`

- [ ] **Step 1: Write failing DB schema tests**

Create `tldw_Server_API/tests/VN_Play/test_vn_play_db.py` with fixtures matching `tldw_Server_API/tests/VN_Assets/test_vn_asset_packs_db.py`.

```python
def test_initialized_creates_session_event_turn_and_state_tables(chacha_db):
    repo = VNPlayRepository.initialized(chacha_db)

    session = repo.create_session(
        owner_user_id=42,
        mode="freeform",
        title="Library night",
        primary_character_id=1,
        vn_asset_pack_id=10,
        content_rating="general",
        seed="seed-1",
        settings={},
    )
    event = repo.append_event(
        session_id=session["id"],
        owner_user_id=42,
        event_type="session_started",
        event_payload={"schema_version": 1},
        source="system",
    )
    turn = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="turn-1",
        request_payload_hash="hash-1",
        base_scene_version=0,
    )

    assert session["scene_version"] == 0
    assert event["sequence_number"] == 1
    assert turn["status"] == "pending"
```

Add a second test for idempotency uniqueness:

```python
def test_turn_request_idempotency_key_is_unique_per_session(chacha_db):
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="freeform",
        title="Library night",
        primary_character_id=1,
        vn_asset_pack_id=10,
        content_rating="general",
        seed="seed-1",
        settings={},
    )
    repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="same",
        request_payload_hash="hash-a",
        base_scene_version=0,
    )

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.create_turn_request(
            session_id=session["id"],
            owner_user_id=42,
            idempotency_key="same",
            request_payload_hash="hash-b",
            base_scene_version=0,
        )
```

- [ ] **Step 2: Run failing DB tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py -q
```

Expected: import failure for `VNPlayRepository`.

- [ ] **Step 3: Implement `VNPlay_DB.py`**

Implement:

- `VN_PLAY_SCHEMA_SQL`
- `VNPlayRepository.initialized(db: CharactersRAGDB)`
- `create_session`
- `get_session`
- `list_sessions`
- `update_session`
- `append_event`
- `list_events`
- `create_turn_request`
- `get_turn_request_by_key`
- `update_turn_request`
- `set_scene_state`
- `get_scene_state`
- `create_branch`
- `list_branches`
- `create_checkpoint`
- `list_checkpoints`

Schema must include:

- `vn_play_sessions`
- `vn_play_events`
- `vn_play_turn_requests`
- `vn_play_scene_state`
- `vn_play_branches`
- `vn_play_checkpoints`

Use JSON helper patterns from `VNAssetPacks_DB.py`: store JSON as text, return decoded dict/list fields where practical, and keep ownership filters explicit.

- [ ] **Step 4: Run DB tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py -q
```

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/tests/VN_Play/test_vn_play_db.py
git commit -m "feat(vn-play): add session repository"
```

## Task 2: API Schemas And Constants

**Files:**
- Create: `tldw_Server_API/app/core/VN_Play/__init__.py`
- Create: `tldw_Server_API/app/core/VN_Play/constants.py`
- Create: `tldw_Server_API/app/core/VN_Play/models.py`
- Create: `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`

- [ ] **Step 1: Write failing schema validation tests**

Add tests:

```python
def test_turn_request_requires_exactly_one_input_field():
    VNPlayTurnRequest(input_text="hello", client_scene_version=0, idempotency_key="k")

    with pytest.raises(ValidationError):
        VNPlayTurnRequest(
            input_text="hello",
            choice_id="choice-1",
            client_scene_version=0,
            idempotency_key="k",
        )
```

```python
def test_create_session_defaults_linked_chat_to_read_only():
    request = VNPlaySessionCreate(
        mode="freeform",
        title="Test",
        primary_character_id=1,
        vn_asset_pack_id=2,
    )

    assert request.linked_chat_mode == "read_only_context"
```

- [ ] **Step 2: Run failing schema tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py::test_turn_request_requires_exactly_one_input_field tldw_Server_API/tests/VN_Play/test_vn_play_api.py::test_create_session_defaults_linked_chat_to_read_only -q
```

Expected: import failure for `vn_play_schemas`.

- [ ] **Step 3: Implement constants, internal models, and schemas**

Add constants:

- Modes: `freeform`, `story`
- Session statuses: `active`, `paused`, `completed`, `archived`, `failed`
- Turn statuses from the spec.
- Event types from the spec.
- Error codes: `stale_scene_version`, `turn_in_progress`, `idempotency_key_conflict`, `runtime_gate_failed`, `model_turn_parse_failed`.

Add Pydantic schemas:

- `VNPlaySessionCreate`
- `VNPlaySessionUpdate`
- `VNPlaySessionResponse`
- `VNPlaySceneStateResponse`
- `VNPlayEventResponse`
- `VNPlayTurnRequest`
- `VNPlayTurnResponse`
- `VNPlayCheckpointCreate`
- `VNPlayCheckpointResponse`
- `VNPlayRestoreRequest`
- `VNPlayBranchResponse`

Use strict validation where existing schemas do: reject coercion for mode/status, require `idempotency_key`, require `client_scene_version`, and enforce exactly one of `input_text`, `choice_id`, `custom_action`.

- [ ] **Step 4: Run schema tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q
```

Expected: schema tests pass; endpoint tests may still be skipped or absent at this task.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/VN_Play tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py tldw_Server_API/tests/VN_Play/test_vn_play_api.py
git commit -m "feat(vn-play): add api schemas"
```

## Task 3: Event Replay And Scene State

**Files:**
- Create: `tldw_Server_API/app/core/VN_Play/state.py`
- Modify: `tldw_Server_API/app/core/VN_Play/models.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_state.py`

- [ ] **Step 1: Write failing replay tests**

Create tests:

```python
def test_replay_applies_scene_state_changed_event():
    events = [
        {"event_type": "session_started", "event_payload": {"schema_version": 1}},
        {
            "event_type": "scene_state_changed",
            "event_payload": {
                "background_item_id": 101,
                "active_sprite_items": [{"character_id": 1, "item_id": 201}],
                "location_key": "library",
                "scene_version": 1,
            },
        },
    ]

    state = derive_scene_state(events)

    assert state.current_background_item_id == 101
    assert state.location_key == "library"
    assert state.scene_version == 1
```

```python
def test_replay_keeps_warning_for_rejected_visual_directive():
    state = derive_scene_state([
        {
            "event_type": "visual_directive_rejected",
            "event_payload": {"reason": "asset_not_found", "slot_key": "sprite.happy"},
        }
    ])

    assert state.warnings[0]["reason"] == "asset_not_found"
```

- [ ] **Step 2: Run failing replay tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_state.py -q
```

Expected: import failure for `VN_Play.state`.

- [ ] **Step 3: Implement scene-state derivation**

Implement:

- `SceneState` dataclass or Pydantic model.
- `derive_scene_state(events: Iterable[Mapping[str, Any]]) -> SceneState`.
- Event handlers for:
  - `session_started`
  - `scene_state_changed`
  - `choice_presented`
  - `choice_selected`
  - `session_restored`
  - `visual_directive_rejected`
  - `turn_failed`
  - `model_turn_parse_failed`

Keep this module pure: no DB calls, no HTTP, no model calls.

- [ ] **Step 4: Run replay tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_state.py -q
```

Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/VN_Play/state.py tldw_Server_API/app/core/VN_Play/models.py tldw_Server_API/tests/VN_Play/test_vn_play_state.py
git commit -m "feat(vn-play): derive scene state from events"
```

## Task 4: Runtime Gates And Approved-Manifest Asset Resolver

**Files:**
- Create: `tldw_Server_API/app/core/VN_Play/gates.py`
- Create: `tldw_Server_API/app/core/VN_Play/assets.py`
- Modify: `tldw_Server_API/app/core/VN_Play/models.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_gates.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_assets.py`

- [ ] **Step 1: Write failing gate tests**

```python
def test_unknown_character_metadata_warns_for_general_rating():
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira"},
        content_rating="general",
        settings={},
        trust_level="local",
    )

    assert result.allowed is True
    assert result.status == "unknown"
    assert result.warning_code == "character_safety_unknown"
```

```python
def test_unknown_character_metadata_requires_override_for_mature_rating():
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira"},
        content_rating="mature",
        settings={},
        trust_level="local",
    )

    assert result.allowed is False
    assert result.error_code == "character_safety_unknown_requires_override"
```

- [ ] **Step 2: Write failing asset resolver tests**

```python
def test_resolver_prefers_preferred_approved_item():
    manifest = {
        "assets": {
            "sprite": [
                {"item_id": 1, "slot_key": "sprite.happy", "labels": {"emotion": "happy"}, "preferred": False},
                {"item_id": 2, "slot_key": "sprite.happy.alt", "labels": {"emotion": "happy"}, "preferred": True},
            ]
        }
    }

    resolved = resolve_visual_directive(manifest, {"asset_type": "sprite", "labels": {"emotion": "happy"}}, seed="s")

    assert resolved.applied is True
    assert resolved.item["item_id"] == 2
```

```python
def test_resolver_rejects_unmatched_directive():
    resolved = resolve_visual_directive({"assets": {"sprite": []}}, {"slot_key": "sprite.missing"}, seed="s")

    assert resolved.applied is False
    assert resolved.reason == "asset_not_found"
```

- [ ] **Step 3: Run failing tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_gates.py tldw_Server_API/tests/VN_Play/test_vn_play_assets.py -q
```

Expected: import failures.

- [ ] **Step 4: Implement gates and resolver**

Implement:

- `evaluate_character_safety`
- `evaluate_runtime_gates`
- `resolve_visual_directive`
- `resolve_scene_directives`

Rules:

- Approved manifest entries only.
- Preferred item wins.
- Ambiguity resolves with stable seeded ordering.
- Unknown character safety metadata is explicit and mode-consistent.
- Imported/untrusted safety metadata requires opt-in for missing/conflicting cases.

- [ ] **Step 5: Run tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_gates.py tldw_Server_API/tests/VN_Play/test_vn_play_assets.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/VN_Play/gates.py tldw_Server_API/app/core/VN_Play/assets.py tldw_Server_API/app/core/VN_Play/models.py tldw_Server_API/tests/VN_Play/test_vn_play_gates.py tldw_Server_API/tests/VN_Play/test_vn_play_assets.py
git commit -m "feat(vn-play): add runtime gates and asset resolver"
```

## Task 5: Turn Request Lifecycle And VN Play Service

**Files:**
- Create: `tldw_Server_API/app/core/VN_Play/service.py`
- Modify: `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`

- [x] **Step 1: Write failing idempotency tests**

```python
async def test_duplicate_completed_turn_returns_stored_response(service, ready_session):
    first = await service.submit_turn(
        ready_session.id,
        input_text="Hello",
        client_scene_version=0,
        idempotency_key="turn-1",
    )
    second = await service.submit_turn(
        ready_session.id,
        input_text="Hello",
        client_scene_version=0,
        idempotency_key="turn-1",
    )

    assert second.turn_request_id == first.turn_request_id
    assert second.events == first.events
```

```python
async def test_same_idempotency_key_different_payload_conflicts(service, ready_session):
    await service.submit_turn(ready_session.id, input_text="Hello", client_scene_version=0, idempotency_key="turn-1")

    with pytest.raises(VNPlayConflictError, match="idempotency_key_conflict"):
        await service.submit_turn(ready_session.id, input_text="Different", client_scene_version=0, idempotency_key="turn-1")
```

- [x] **Step 2: Write failing concurrency and failure tests**

```python
async def test_stale_scene_version_conflicts(service, ready_session):
    await service.submit_turn(ready_session.id, input_text="First", client_scene_version=0, idempotency_key="first")

    with pytest.raises(VNPlayConflictError, match="stale_scene_version"):
        await service.submit_turn(ready_session.id, input_text="Second", client_scene_version=0, idempotency_key="second")
```

```python
async def test_model_failure_marks_turn_failed_and_clears_lock(service_with_failing_adapter, ready_session):
    with pytest.raises(VNPlayTurnError):
        await service_with_failing_adapter.submit_turn(
            ready_session.id,
            input_text="Break",
            client_scene_version=0,
            idempotency_key="fail-1",
        )

    session = service_with_failing_adapter.get_session(ready_session.id)
    assert session.active_turn_request_id is None
```

- [x] **Step 3: Run failing turn tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q
```

Expected: service import failures.

- [x] **Step 4: Implement `VNPlayService`**

Implement:

- `create_session`
- `list_sessions`
- `get_session`
- `submit_turn`
- `retry_last_turn`
- `list_events`
- `create_checkpoint`
- `restore_checkpoint`
- `list_branches`

Turn behavior:

- Require idempotency key and client scene version.
- Hash normalized input payload.
- Check idempotency before stale scene validation so duplicate completed requests can replay their stored response even after the session scene version has advanced.
- Enforce one active turn per session.
- Reject stale scene version with current state.
- Open pre-model transaction for turn request, `turn_started`, and input event.
- Call adapter outside transaction.
- Open post-model transaction for model events, visual events, scene-state update, stored response, and lock clear.
- Mark timeout/provider failure as `model_failed`.
- Mark parser failure as `parse_failed`.
- `retry_last_turn` links to failed/abandoned request without duplicating input unless changed.

Use a deterministic test adapter for service tests and a monkeypatched `ChatVNPlayTurnAdapter` test for the real chat-service boundary. The shipped V1 adapter must be capable of calling the configured chat provider through `perform_chat_api_call_async`; tests must never call an external provider.

- [x] **Step 5: Run turn tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q
```

Expected: all pass.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/VN_Play/service.py tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py
git commit -m "feat(vn-play): orchestrate turn lifecycle"
```

## Task 6: Structured Parser And Freeform/Story Adapters

**Files:**
- Create: `tldw_Server_API/app/core/VN_Play/parser.py`
- Create: `tldw_Server_API/app/core/VN_Play/adapters.py`
- Modify: `tldw_Server_API/app/core/VN_Play/service.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`

- [x] **Step 1: Write failing parser tests**

```python
def test_parse_structured_turn_result():
    result = parse_model_turn(
        {
            "narration": "The library lights flicker.",
            "dialogue": [{"speaker": "Mira", "text": "Stay close."}],
            "scene_directives": {"background": {"labels": {"location": "library"}}},
            "choices": [{"id": "choice-1", "text": "Inspect the shelves"}],
            "summary": "Mira enters the library.",
        },
        mode="story",
    )

    assert result.narration.startswith("The library")
    assert result.choices[0].text == "Inspect the shelves"
```

```python
def test_story_parser_requires_two_to_five_choices():
    with pytest.raises(VNPlayParseError):
        parse_model_turn({"narration": "No choice", "choices": []}, mode="story")
```

Add an adapter test that monkeypatches the existing chat service call:

```python
async def test_chat_adapter_calls_existing_chat_service(monkeypatch):
    captured = {}

    async def fake_chat_call(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"narration":"Hi","dialogue":[{"speaker":"Mira","text":"Hello."}],"summary":"Greeting"}'
                    }
                }
            ]
        }

    monkeypatch.setattr(vn_play_adapters, "perform_chat_api_call_async", fake_chat_call)
    adapter = ChatVNPlayTurnAdapter(provider="openai", model="gpt-test")

    result = await adapter.generate_turn(context=make_turn_context(mode="freeform"))

    assert result.dialogue[0].text == "Hello."
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-test"
```

- [x] **Step 2: Run failing parser tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py::test_parse_structured_turn_result tldw_Server_API/tests/VN_Play/test_vn_play_turns.py::test_story_parser_requires_two_to_five_choices -q
```

Expected: import failure for parser/adapter.

- [x] **Step 3: Implement parser and adapter interfaces**

Implement:

- `parse_model_turn(raw: Any, mode: str) -> NormalizedTurnResult`
- `VNPlayTurnAdapter` protocol/interface.
- `DeterministicVNPlayAdapter` for tests and mocked UI.
- `ChatVNPlayTurnAdapter` that calls `tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async`.
- `FreeformVNPlayAdapter` and `StoryVNPlayAdapter` wrappers that assemble mode-specific messages and delegate to `ChatVNPlayTurnAdapter`.

Adapter boundaries:

- Keep provider-native structured output integration behind `ChatVNPlayTurnAdapter`.
- Tests must monkeypatch `perform_chat_api_call_async`; no external provider call is allowed in tests.
- The adapter should request non-streaming output in V1.
- The adapter must add VN structured-output instructions, recent VN event summaries, character context, optional read-only linked chat snapshot, and the current scene summary to the chat messages.
- Provider/model selection comes from session settings first, then configured defaults.
- On provider errors, raise a typed VN Play model error so Task 5 failure handling marks the turn `model_failed`.

- [x] **Step 4: Run parser/turn tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q
```

Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/VN_Play/parser.py tldw_Server_API/app/core/VN_Play/adapters.py tldw_Server_API/app/core/VN_Play/service.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py
git commit -m "feat(vn-play): parse structured turns"
```

## Task 7: VN Play API Endpoints

**Files:**
- Create: `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
- Modify: `tldw_Server_API/app/main.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`

- [x] **Step 1: Write failing endpoint tests**

```python
def test_create_session_endpoint_returns_scene_state(client, ready_pack_id, character_id):
    response = client.post(
        "/api/v1/vn-play/sessions",
        json={
            "mode": "freeform",
            "title": "Library night",
            "primary_character_id": character_id,
            "vn_asset_pack_id": ready_pack_id,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["mode"] == "freeform"
    assert body["scene_state"]["scene_version"] == 0
```

```python
def test_turn_endpoint_rejects_stale_scene_version(client, session_id):
    first = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={"input_text": "Hello", "client_scene_version": 0, "idempotency_key": "a"},
    )
    assert first.status_code == 200

    stale = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={"input_text": "Again", "client_scene_version": 0, "idempotency_key": "b"},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"] == "stale_scene_version"
```

- [x] **Step 2: Run failing endpoint tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q
```

Expected: router import failure or 404.

- [x] **Step 3: Implement router and register in `main.py`**

Implement endpoints from the spec:

- `POST /api/v1/vn-play/sessions`
- `GET /api/v1/vn-play/sessions`
- `GET /api/v1/vn-play/sessions/{session_id}`
- `PATCH /api/v1/vn-play/sessions/{session_id}`
- `DELETE /api/v1/vn-play/sessions/{session_id}`
- `POST /api/v1/vn-play/sessions/{session_id}/turn`
- `POST /api/v1/vn-play/sessions/{session_id}/retry-last-turn`
- `GET /api/v1/vn-play/sessions/{session_id}/events`
- `POST /api/v1/vn-play/sessions/{session_id}/checkpoint`
- `GET /api/v1/vn-play/sessions/{session_id}/checkpoints`
- `POST /api/v1/vn-play/sessions/{session_id}/restore`
- `GET /api/v1/vn-play/sessions/{session_id}/branches`

Map service exceptions:

- `not_found` -> 404
- `stale_scene_version`, `turn_in_progress`, `idempotency_key_conflict` -> 409
- validation/gate failures -> 400
- unexpected parse/model errors -> 502 or 500 only after a structured event/turn status is recorded.

Register route in `main.py` similarly to `vn_assets`, behind route key `vn-play`.

- [x] **Step 4: Run endpoint tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/vn_play.py tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py tldw_Server_API/app/main.py tldw_Server_API/tests/VN_Play/test_vn_play_api.py
git commit -m "feat(vn-play): expose runtime api"
```

## Task 8: Frontend API Client And Types

**Files:**
- Create: `apps/tldw-frontend/types/vn-play.ts`
- Create: `apps/tldw-frontend/lib/api/vnPlay.ts`
- Test: `apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts`

- [x] **Step 1: Write failing API client tests**

```ts
it('creates a VN play session', async () => {
  mockFetchOnce({ id: 1, mode: 'freeform', title: 'Library', scene_state: { scene_version: 0 } });

  const session = await createVNPlaySession({
    mode: 'freeform',
    title: 'Library',
    primary_character_id: 1,
    vn_asset_pack_id: 2,
  });

  expect(session.id).toBe(1);
  expect(fetch).toHaveBeenCalledWith(
    expect.stringContaining('/api/v1/vn-play/sessions'),
    expect.objectContaining({ method: 'POST' })
  );
});
```

```ts
it('submits a VN play turn with idempotency key and scene version', async () => {
  mockFetchOnce({ events: [], scene_state: { scene_version: 1 } });

  await submitVNPlayTurn(1, {
    input_text: 'Hello',
    client_scene_version: 0,
    idempotency_key: 'turn-1',
  });

  expect(JSON.parse((fetch as Mock).mock.calls[0][1].body)).toMatchObject({
    input_text: 'Hello',
    client_scene_version: 0,
    idempotency_key: 'turn-1',
  });
});
```

- [x] **Step 2: Run failing frontend API tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play/vnPlayApi.test.ts
```

Expected: module import failure.

- [x] **Step 3: Implement frontend types and API functions**

Implement:

- `VNPlaySession`
- `VNPlaySceneState`
- `VNPlayEvent`
- `VNPlayTurnRequest`
- `VNPlayTurnResponse`
- `VNPlayChoice`
- `VNPlayCheckpoint`
- `VNPlayBranch`

API functions:

- `createVNPlaySession`
- `listVNPlaySessions`
- `getVNPlaySession`
- `updateVNPlaySession`
- `deleteVNPlaySession`
- `submitVNPlayTurn`
- `retryLastVNPlayTurn`
- `listVNPlayEvents`
- `createVNPlayCheckpoint`
- `listVNPlayCheckpoints`
- `restoreVNPlaySession`
- `listVNPlayBranches`

Follow `apps/tldw-frontend/lib/api/vnAssets.ts` style for URL building and JSON errors.

- [x] **Step 4: Run frontend API tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play/vnPlayApi.test.ts
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add apps/tldw-frontend/types/vn-play.ts apps/tldw-frontend/lib/api/vnPlay.ts apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts
git commit -m "feat(vn-play): add frontend api client"
```

## Task 9: `/vn-play` Workspace Shell

**Files:**
- Create: `apps/tldw-frontend/pages/vn-play.tsx`
- Create: `apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx`
- Create: `apps/tldw-frontend/components/vn-play/SessionList.tsx`
- Create: `apps/tldw-frontend/components/vn-play/NewSessionDialog.tsx`
- Test: `apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx`

- [x] **Step 1: Write failing workspace tests**

```tsx
it('renders freeform and story session actions', async () => {
  mockVNPlayApi({ sessions: [] });

  render(<VNPlayWorkspace />);

  expect(await screen.findByRole('button', { name: /new freeform/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /new story/i })).toBeInTheDocument();
});
```

```tsx
it('loads and selects the first session', async () => {
  mockVNPlayApi({ sessions: [{ id: 1, title: 'Library', mode: 'freeform', scene_state: { scene_version: 0 } }] });

  render(<VNPlayWorkspace />);

  expect(await screen.findByText('Library')).toBeInTheDocument();
});
```

- [x] **Step 2: Run failing workspace tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx
```

Expected: component import failure.

- [x] **Step 3: Implement workspace shell**

Build:

- Route with `dynamic(() => import(...), { ssr: false })`, matching `/vn-assets`.
- Left rail session list.
- Mode filter.
- New Freeform and New Story buttons.
- Minimal new-session dialog fields:
  - title
  - mode
  - primary character ID
  - VN asset pack ID
  - optional linked chat ID
  - content rating
- Initial selected-session load and error state.

Keep center/right panels as placeholders until Task 10.

- [x] **Step 4: Run workspace tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play/VNPlayWorkspace.test.tsx
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add apps/tldw-frontend/pages/vn-play.tsx apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx apps/tldw-frontend/components/vn-play/SessionList.tsx apps/tldw-frontend/components/vn-play/NewSessionDialog.tsx apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx
git commit -m "feat(vn-play): add workspace shell"
```

## Task 10: Scene Stage, Turns, Choices, And Inspector

**Files:**
- Create: `apps/tldw-frontend/components/vn-play/SceneStage.tsx`
- Create: `apps/tldw-frontend/components/vn-play/DialoguePanel.tsx`
- Create: `apps/tldw-frontend/components/vn-play/ChoicePanel.tsx`
- Create: `apps/tldw-frontend/components/vn-play/SceneInspector.tsx`
- Modify: `apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx`
- Test: `apps/tldw-frontend/__tests__/vn-play/SceneStage.test.tsx`
- Test: `apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx`
- Test: `apps/tldw-frontend/e2e/smoke/vn-play.spec.ts`

- [x] **Step 1: Write failing scene component tests**

```tsx
it('renders background, sprite, dialogue, and warnings', () => {
  render(
    <SceneStage
      sceneState={{
        scene_version: 1,
        background: { content_url: '/bg.png', labels: { location: 'library' } },
        active_sprites: [{ item_id: 2, content_url: '/sprite.png', labels: { emotion: 'happy' } }],
        warnings: [{ reason: 'asset_not_found', slot_key: 'sprite.angry' }],
      }}
      events={[{ id: 1, event_type: 'model_turn', event_payload: { dialogue: [{ speaker: 'Mira', text: 'Hello.' }] } }]}
    />
  );

  expect(screen.getByAltText(/background/i)).toHaveAttribute('src', '/bg.png');
  expect(screen.getByText('Hello.')).toBeInTheDocument();
  expect(screen.getByText(/asset_not_found/i)).toBeInTheDocument();
});
```

```tsx
it('submits a story choice with current scene version', async () => {
  mockVNPlayApi({ turnResponse: { scene_state: { scene_version: 2 }, events: [] } });
  render(<ChoicePanel sessionId={1} sceneVersion={1} choices={[{ id: 'c1', text: 'Open the door' }]} onTurn={vi.fn()} />);

  await userEvent.click(screen.getByRole('button', { name: /open the door/i }));

  expect(submitVNPlayTurn).toHaveBeenCalledWith(1, expect.objectContaining({
    choice_id: 'c1',
    client_scene_version: 1,
  }));
});
```

- [x] **Step 2: Write failing Playwright smoke**

Create `apps/tldw-frontend/e2e/smoke/vn-play.spec.ts` that:

- Mocks `GET /api/v1/vn-play/sessions`.
- Mocks `POST /api/v1/vn-play/sessions`.
- Mocks `POST /api/v1/vn-play/sessions/1/turn`.
- Opens `/vn-play`.
- Creates a Story session.
- Clicks one mocked choice.
- Verifies dialogue and updated scene text.

- [x] **Step 3: Run failing UI tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play
```

Expected: missing component/test failures.

- [x] **Step 4: Implement stage and interactions**

Build:

- `SceneStage`
  - stable stage dimensions
  - background image
  - optional depth/parallax layer when URLs exist
  - sprite image layer
  - no decorative cards inside cards
- `DialoguePanel`
  - latest narration/dialogue from events
  - freeform input for Freeform mode
- `ChoicePanel`
  - story choices
  - optional custom action input
  - idempotency key generated per submit
- `SceneInspector`
  - pack, character, scene version, warnings, branch/checkpoint summary
- Workspace wiring:
  - refresh selected session after turn
  - handle `409 stale_scene_version` by reloading session
  - handle `409 turn_in_progress` by showing current status

- [x] **Step 5: Run UI tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play
```

Expected: all pass.

- [x] **Step 6: Run Playwright smoke**

Run:

```bash
cd apps/tldw-frontend
bunx playwright test e2e/smoke/vn-play.spec.ts --reporter=line
```

Expected: 1 passed.

- [x] **Step 7: Commit**

```bash
git add apps/tldw-frontend/components/vn-play apps/tldw-frontend/__tests__/vn-play apps/tldw-frontend/e2e/smoke/vn-play.spec.ts
git commit -m "feat(vn-play): render playable scenes"
```

## Task 11: Documentation And Final Verification

**Files:**
- Create: `Docs/API-related/VN_PLAY_API.md`
- Modify: `Docs/superpowers/plans/2026-05-01-vn-play-runtime-implementation.md`

- [x] **Step 1: Add API documentation**

Document:

- Session creation/list/get/update/delete.
- Turn request idempotency requirements.
- `409 stale_scene_version`.
- `409 turn_in_progress`.
- Read-only linked chat behavior.
- Character safety metadata behavior.
- Freeform and Story example requests.

- [x] **Step 2: Run backend VN Play tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q
```

Expected: all pass.

- [x] **Step 3: Run existing VN Asset tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Assets -q
```

Expected: all pass.

- [x] **Step 4: Run frontend VN Play tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-play
```

Expected: all pass.

- [x] **Step 5: Run existing VN Asset frontend tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-assets
```

Expected: all pass.

- [x] **Step 6: Run Playwright VN Play smoke**

Run:

```bash
cd apps/tldw-frontend
bunx playwright test e2e/smoke/vn-play.spec.ts --reporter=line
```

Expected: 1 passed.

- [x] **Step 7: Run Bandit on touched backend scope**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/VN_Play tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/app/api/v1/endpoints/vn_play.py tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py -f json -o /tmp/bandit_vn_play_runtime.json
```

Expected: command exits 0 and `/tmp/bandit_vn_play_runtime.json` has `"results": []`.

- [x] **Step 8: Run diff check**

Run:

```bash
git diff --check
```

Expected: no output.

- [x] **Step 9: Commit docs and final plan status**

```bash
git add Docs/API-related/VN_PLAY_API.md Docs/superpowers/plans/2026-05-01-vn-play-runtime-implementation.md
git commit -m "docs(vn-play): document runtime api"
```

## Final Handoff

After all tasks pass:

- Confirm `git status --short` is clean.
- Summarize all commits.
- If the work remains on the existing PR branch, push and update the PR description with the VN Play runtime scope and test plan.
- Do not merge until the human-owned `Change summary` requirement is satisfied.
