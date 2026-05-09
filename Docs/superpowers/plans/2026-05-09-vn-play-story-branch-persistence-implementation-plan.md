# VN Play Story Branch Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist selected Story/CYOA choices as durable VN Play branch metadata while keeping the API server authoritative for validation, idempotency, retry, and scene state.

**Architecture:** Add one atomic repository helper for accepted Story choices, then call it from `VNPlayService.submit_turn()` after the turn lock is acquired and before model work starts. Keep `branch_path_json` top-level list-shaped for current API compatibility, persist replay-derived scene state before model calls, and make `retry-last-turn` failure-retry only with the failed turn request's `input_event_id` as the source of truth.

**Tech Stack:** FastAPI, Pydantic v2, SQLite-backed `CharactersRAGDB`, VN Play service/repository modules, pytest, Bandit.

---

## File Map

- Modify `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
  - Add atomic repository helper for Story choice acceptance.
  - Add small transaction-local helpers for inserting events and setting scene state.
  - Add turn-request lookup helpers needed by retry.
  - Preserve `branch_path` as `list[Any]`.
- Modify `tldw_Server_API/app/core/VN_Play/constants.py`
  - Add `choice_not_allowed`, `invalid_choice_id`, and `retry_last_turn_not_failed`.
- Modify `tldw_Server_API/app/core/VN_Play/service.py`
  - Validate Story/Freeform turn inputs.
  - Validate selected choices against persisted scene state.
  - Create Story branches and `choice_selected` events before model work.
  - Build retry context from failed turn request `input_event_id`.
- Modify `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
  - Import new stable error constants only if conflict/error mapping needs them.
- Modify `Docs/API-related/VN_PLAY_API.md`
  - Document Story choice validation, `choice_selected`, branch path shape, and retry errors.
- Test `tldw_Server_API/tests/VN_Play/test_vn_play_db.py`
  - Repository helper, branch path compatibility, and pre-model scene persistence.
- Test `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`
  - Service-level Story choice, validation, idempotency, failure, and retry behavior.
- Test `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`
  - Endpoint status/error behavior and branch response validation.
- Test `tldw_Server_API/tests/VN_Play/test_vn_play_state.py`
  - Add or tighten replay assertion for `active_branch_node_id`.

## Preflight

- [ ] **Step 1: Confirm branch and task context**

Run:

```bash
git status --short --branch
```

Expected: branch is `codex/vn-play-story-branch-persistence`, clean except intentional task/plan edits when executing this plan.

- [ ] **Step 2: Create or reuse an implementation Backlog task**

Use Backlog.md MCP before runtime code edits. Suggested task title:

```text
Implement VN Play Story/CYOA branch persistence
```

Reference:

```text
https://github.com/rmusser01/tldw_server/issues/1434
```

Expected: new implementation task exists and is marked `In Progress`.

### Task 1: Add Atomic Story Choice Repository Helper

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_db.py`

- [x] **Step 1: Write failing repository test for atomic accepted-choice persistence**

Add a test like:

```python
def test_record_story_choice_selection_creates_branch_event_turn_and_state(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Door",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    choice_presented = repo.append_event(
        session_id=session["id"],
        owner_user_id=42,
        event_type="choice_presented",
        event_payload={
            "choices": [{"id": "open", "text": "Open the door"}],
            "scene_version": 1,
        },
    )
    repo.set_scene_state(
        session_id=session["id"],
        owner_user_id=42,
        last_event_id=choice_presented["id"],
        visible_choices=[{"id": "open", "text": "Open the door"}],
        current_background_item_id=100,
        location_key="hall",
        scene_version=1,
    )
    repo.update_session(session["id"], {"scene_version": 1}, owner_user_id=42)
    turn = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="choice-1",
        request_payload_hash="hash-choice-1",
        base_scene_version=1,
    )

    result = repo.record_story_choice_selection(
        session_id=session["id"],
        owner_user_id=42,
        turn_request_id=turn["id"],
        client_scene_version=1,
        selected_choice={"id": "open", "text": "Open the door"},
        parent_event_id=choice_presented["id"],
        branch_label="Open the door",
        branch_path=[
            {
                "schema_version": 1,
                "type": "choice",
                "choice_id": "open",
                "choice_text": "Open the door",
                "choice_presented_event_id": choice_presented["id"],
                "scene_version": 1,
            }
        ],
    )

    assert result["branch"]["branch_path"][0]["choice_id"] == "open"
    assert result["turn_started"]["event_type"] == "turn_started"
    assert result["choice_selected"]["event_type"] == "choice_selected"
    assert result["choice_selected"]["branch_node_id"] == result["branch"]["id"]

    state = repo.get_scene_state(session["id"], owner_user_id=42)
    assert state["last_event_id"] == result["choice_selected"]["id"]
    assert state["active_branch_node_id"] == result["branch"]["id"]
    assert state["visible_choices"] == []
    assert state["current_background_item_id"] == 100
    assert state["location_key"] == "hall"

    updated_turn = repo.get_turn_request(turn["id"])
    assert updated_turn["status"] == "model_calling"
    assert updated_turn["turn_started_event_id"] == result["turn_started"]["id"]
    assert updated_turn["input_event_id"] == result["choice_selected"]["id"]
```

- [x] **Step 2: Run the new repository test and verify it fails**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py::test_record_story_choice_selection_creates_branch_event_turn_and_state -q
```

Expected: fails because `VNPlayRepository.record_story_choice_selection` does not exist.

- [x] **Step 3: Implement transaction-local helpers**

In `VNPlay_DB.py`, add private helpers that accept a transaction connection and do not open their own transaction:

```python
def _insert_event(
    conn: Any,
    *,
    session_id: int,
    owner_user_id: int,
    event_type: str,
    event_payload: Mapping[str, Any] | None = None,
    source: str = "runtime",
    model_provider: str | None = None,
    model_name: str | None = None,
    branch_node_id: int | None = None,
) -> int:
    sequence_cursor = conn.execute(
        """
        SELECT COALESCE(MAX(sequence_number), 0) + 1 AS next_sequence
        FROM vn_play_events
        WHERE session_id = ?
        """,
        (session_id,),
    )
    sequence_number = int(sequence_cursor.fetchone()["next_sequence"])
    cursor = conn.execute(
        """
        INSERT INTO vn_play_events (
            session_id,
            owner_user_id,
            sequence_number,
            event_type,
            event_payload_json,
            source,
            model_provider,
            model_name,
            branch_node_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            owner_user_id,
            sequence_number,
            event_type,
            _json_dump(dict(event_payload or {})),
            source,
            model_provider,
            model_name,
            branch_node_id,
        ),
    )
    return int(cursor.lastrowid)
```

Refactor `append_event()` to call `_insert_event()` inside its existing transaction.

- [x] **Step 4: Implement `record_story_choice_selection()`**

Add a public repository method. It must:

- Insert one branch with list-shaped `branch_path_json`.
- Insert `turn_started`.
- Insert `choice_selected` with `branch_node_id`.
- Update `vn_play_turn_requests.status` to `model_calling`.
- Set `turn_started_event_id` and `input_event_id`.
- Upsert `vn_play_scene_state` with `last_event_id=choice_selected.id`, `active_branch_node_id=branch.id`, `visible_choices=[]`, and unchanged base `scene_version`.
- Preserve all unrelated scene fields during the upsert, including background,
  depth, sprites, location, mood, time of day, weather, transcript cursor, and
  safety warnings.
- Return decoded `branch`, `turn_started`, `choice_selected`, and `scene_state`.

Keep this whole operation inside one `with self.db.transaction() as conn:` block.

- [x] **Step 5: Run repository tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py -q
```

Expected: all VN Play DB tests pass.

- [x] **Step 6: Commit Task 1**

Run:

```bash
git add tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/tests/VN_Play/test_vn_play_db.py
git commit -m "Add VN Play story choice persistence helper"
```

### Task 2: Wire Story Choice Validation And Branch Persistence Into Submit Turn

**Files:**
- Modify: `tldw_Server_API/app/core/VN_Play/constants.py`
- Modify: `tldw_Server_API/app/core/VN_Play/service.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_state.py`

- [ ] **Step 1: Write failing service tests for valid Story choice**

Add a Story helper in `test_vn_play_turns.py`:

```python
class InspectingStoryAdapter:
    def __init__(self, repo: VNPlayRepository, owner_user_id: int) -> None:
        self.repo = repo
        self.owner_user_id = owner_user_id
        self.seen_contexts: list[VNPlayTurnContext] = []

    async def generate_turn(self, context: VNPlayTurnContext) -> TurnResult:
        self.seen_contexts.append(context)
        persisted = self.repo.get_scene_state(
            context.session.id,
            owner_user_id=self.owner_user_id,
        )
        assert persisted is not None
        assert persisted["active_branch_node_id"] is not None
        assert persisted["visible_choices"] == []
        return TurnResult(
            narrative_text="The door opens.",
            dialogue=[{"speaker": "Narrator", "text": "The door opens."}],
            choices=[
                {"id": "inside", "text": "Step inside"},
                {"id": "wait", "text": "Wait outside"},
            ],
        )
```

Add a helper that creates a Story session with one visible choice by appending
`choice_presented`, persisting scene state with `visible_choices`, and updating
the session `scene_version`.

Test:

```python
@pytest.mark.asyncio
async def test_story_choice_creates_branch_and_choice_selected_before_model(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    adapter = InspectingStoryAdapter(repo, owner_user_id=42)
    service = VNPlayService(repo=repo, owner_user_id=42, adapter=adapter)
    session = create_story_session_with_visible_choice(service, repo)

    response = await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="story-choice-1",
    )

    event_types = [event["event_type"] for event in response.events]
    assert event_types[:2] == ["turn_started", "choice_selected"]
    assert adapter.seen_contexts[0].scene_state.active_branch_node_id is not None

    branches = service.list_branches(session.id)
    assert len(branches) == 1
    assert branches[0]["branch_path"][0]["choice_id"] == "open"

    state = repo.get_scene_state(session.id, owner_user_id=42)
    assert state["active_branch_node_id"] == branches[0]["id"]
    assert state["visible_choices"] == [{"id": "inside", "text": "Step inside"}, {"id": "wait", "text": "Wait outside"}]
```

- [ ] **Step 2: Write failing tests for invalid mode/input rules**

Add tests:

- Story unknown `choice_id` raises `VNPlayTurnError("invalid_choice_id")`.
- Freeform `choice_id` raises `VNPlayTurnError("choice_not_allowed")`.
- Story `input_text` raises `VNPlayTurnError("choice_not_allowed")`.
- Story `custom_action` appends `user_turn` and creates no branch.

- [ ] **Step 3: Run the new service tests and verify they fail**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VN_Play/test_vn_play_turns.py::test_story_choice_creates_branch_and_choice_selected_before_model \
  tldw_Server_API/tests/VN_Play/test_vn_play_turns.py::test_story_unknown_choice_id_fails_before_model \
  tldw_Server_API/tests/VN_Play/test_vn_play_turns.py::test_freeform_choice_id_is_not_allowed \
  tldw_Server_API/tests/VN_Play/test_vn_play_turns.py::test_story_input_text_is_not_allowed \
  tldw_Server_API/tests/VN_Play/test_vn_play_turns.py::test_story_custom_action_remains_non_branching \
  -q
```

Expected: fail because Story validation/branch persistence is not implemented.

- [ ] **Step 4: Add constants**

In `constants.py`, add:

```python
ERROR_CHOICE_NOT_ALLOWED = "choice_not_allowed"
ERROR_INVALID_CHOICE_ID = "invalid_choice_id"
ERROR_RETRY_LAST_TURN_NOT_FAILED = "retry_last_turn_not_failed"
```

Add them to `VN_PLAY_ERROR_CODES`.

- [ ] **Step 5: Add service helpers**

In `service.py`, add focused helpers:

- `_validate_turn_input_for_mode(session, input_payload) -> None`
- `_selected_visible_choice(state, choice_id) -> dict[str, Any]`
- `_latest_restore_sequence(events) -> int`
- `_parent_choice_event_id(events, scene_last_event_id, choice_id) -> int | None`
- `_choice_text(choice) -> str`
- `_branch_path_for_choice(choice, *, scene_version, choice_presented_event_id) -> list[dict[str, Any]]`

Use string extraction that accepts both `text` and `label` because existing tests and state use both shapes.

- [ ] **Step 6: Wire Story choice submit path**

In `submit_turn()`:

- Keep idempotency lookup first.
- Load session and persisted scene state before validation.
- Validate `client_scene_version` against session.
- Validate mode/input rules.
- For Story `choice_id`, validate against persisted `visible_choices`.
- Create turn request.
- Acquire lock.
- For Story `choice_id`, call `repo.record_story_choice_selection(...)`.
- For other input, keep `_append_accepted_turn_events(...)`.
- Derive scene state after accepted input and before adapter call.
- Pass selected choice metadata in `input_payload`, for example:

```python
{"choice_id": "open", "choice": {"id": "open", "text": "Open the door"}}
```

- [ ] **Step 7: Tighten replay test for branch ID**

In `test_vn_play_state.py`, update `test_replay_replaces_visible_choices_after_selection()` so the `choice_selected` event payload includes `branch_node_id`, and assert:

```python
assert state.active_branch_node_id == 12
```

- [ ] **Step 8: Run focused service/state tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py tldw_Server_API/tests/VN_Play/test_vn_play_state.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 2**

Run:

```bash
git add \
  tldw_Server_API/app/core/VN_Play/constants.py \
  tldw_Server_API/app/core/VN_Play/service.py \
  tldw_Server_API/tests/VN_Play/test_vn_play_turns.py \
  tldw_Server_API/tests/VN_Play/test_vn_play_state.py
git commit -m "Persist VN Play story choice branches before model calls"
```

### Task 3: Implement Failure-Only Retry For Story Choices

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
- Modify: `tldw_Server_API/app/core/VN_Play/service.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`

- [ ] **Step 1: Write failing retry tests**

Add tests:

```python
@pytest.mark.asyncio
async def test_retry_failed_story_choice_reuses_original_branch(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    failing = VNPlayService(repo=repo, owner_user_id=42, adapter=FailingStoryAdapter())
    session = create_story_session_with_visible_choice(failing, repo)

    with pytest.raises(VNPlayTurnError, match="model_failed"):
        await failing.submit_turn(
            session.id,
            choice_id="open",
            client_scene_version=1,
            idempotency_key="story-fail-1",
        )

    branches_before = failing.list_branches(session.id)
    retrying = VNPlayService(repo=repo, owner_user_id=42, adapter=InspectingStoryAdapter(repo, 42))

    response = await retrying.retry_last_turn(
        session.id,
        client_scene_version=1,
        idempotency_key="story-retry-1",
    )

    assert response.status == "completed"
    assert retrying.list_branches(session.id) == branches_before
    events = retrying.list_events(session.id)
    assert [event["event_type"] for event in events].count("choice_selected") == 1
```

Also add:

```python
@pytest.mark.asyncio
async def test_retry_completed_story_choice_is_not_failed(
    chacha_db: CharactersRAGDB,
) -> None:
    ...
    with pytest.raises(VNPlayTurnError, match="retry_last_turn_not_failed"):
        await service.retry_last_turn(session.id, client_scene_version=2, idempotency_key="retry-completed")
```

- [ ] **Step 2: Run retry tests and verify they fail**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VN_Play/test_vn_play_turns.py::test_retry_failed_story_choice_reuses_original_branch \
  tldw_Server_API/tests/VN_Play/test_vn_play_turns.py::test_retry_completed_story_choice_is_not_failed \
  -q
```

Expected: fail because retry currently searches only `user_turn` events and resubmits through `submit_turn()`.

- [ ] **Step 3: Add repository helper for latest failed turn**

In `VNPlay_DB.py`, add a method like:

```python
def latest_retryable_turn_request(
    self,
    *,
    session_id: int,
    owner_user_id: int,
) -> dict[str, Any] | None:
    ...
```

It should return the newest turn request for the session/owner where:

- `status IN ('model_failed', 'parse_failed', 'abandoned')`
- `input_event_id IS NOT NULL`

Order by `updated_at DESC, id DESC`.

- [ ] **Step 4: Rewrite `retry_last_turn()`**

In `service.py`:

- Load session and check `client_scene_version`.
- Reject active turn with `turn_in_progress`.
- Fetch latest retryable turn request.
- If none exists, raise `VNPlayTurnError(ERROR_RETRY_LAST_TURN_NOT_FAILED)`.
- Fetch the original input event by `input_event_id`.
- Normalize retry input from the original event type:
  - `choice_selected`: use `choice_id`, `choice`, and `branch_node_id`.
  - `user_turn`: use `event_payload.input`.
- Create a new turn request with a normalized retry payload hash that includes:
  - `session_id`
  - `retry_of_turn_request_id`
  - original input event ID
  - original input payload
- Acquire turn lock.
- Append a `turn_started` event with retry metadata.
- Set new turn request `input_event_id` to original input event ID and `turn_started_event_id` to the new event ID.
- Do not append another `choice_selected` or `user_turn`.
- Build adapter context from original input payload plus current replayed state.
- Complete/fail using existing completion/failure paths.

- [ ] **Step 5: Run retry and full service tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q
```

Expected: all service turn tests pass.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/app/core/VN_Play/service.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py
git commit -m "Make VN Play story choice retry branch-safe"
```

### Task 4: Update API Coverage And Documentation

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
- Modify: `Docs/API-related/VN_PLAY_API.md`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`

- [ ] **Step 1: Write failing API tests**

In `test_vn_play_api.py`, add tests for:

- Story choice endpoint creates branch and returns `active_branch_node_id`.
- Unknown Story choice returns HTTP 400 with `invalid_choice_id`.
- Completed Story retry returns HTTP 400 with `retry_last_turn_not_failed`.
- `GET /branches` returns `branch_path` as a list.

Use the existing TestClient fixture and direct `VNPlayService` setup where necessary to seed a Story session with visible choices.

- [ ] **Step 2: Run API tests and verify failures**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VN_Play/test_vn_play_api.py::test_story_choice_turn_returns_branch_state \
  tldw_Server_API/tests/VN_Play/test_vn_play_api.py::test_story_unknown_choice_returns_invalid_choice_id \
  tldw_Server_API/tests/VN_Play/test_vn_play_api.py::test_story_retry_completed_turn_returns_not_failed \
  tldw_Server_API/tests/VN_Play/test_vn_play_api.py::test_branch_list_keeps_branch_path_list_shape \
  -q
```

Expected: fail until endpoint-visible behavior and docs are updated.

- [ ] **Step 3: Adjust endpoint error imports if needed**

`_http_error_for_service_exception()` already returns HTTP 400 for generic `VNPlayTurnError`, which is correct for `choice_not_allowed`, `invalid_choice_id`, and `retry_last_turn_not_failed`. Only import constants into `vn_play.py` if tests or schema documentation require explicit mapping.

- [ ] **Step 4: Update API docs**

In `Docs/API-related/VN_PLAY_API.md`, document:

- Story `choice_id` validation against current `visible_choices`.
- Story `custom_action` as non-branching `user_turn`.
- `choice_selected` event payload.
- `branch_path` list-shaped response.
- `invalid_choice_id`, `choice_not_allowed`, and `retry_last_turn_not_failed`.
- Failure-only retry semantics.

- [ ] **Step 5: Run API tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q
```

Expected: all VN Play API tests pass.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add tldw_Server_API/app/api/v1/endpoints/vn_play.py Docs/API-related/VN_PLAY_API.md tldw_Server_API/tests/VN_Play/test_vn_play_api.py
git commit -m "Document VN Play story branch API behavior"
```

### Task 5: Full Verification And Closeout

**Files:**
- Modify: Backlog task for the implementation work.
- No runtime files unless verification finds a defect.

- [ ] **Step 1: Run full focused VN Play tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VN_Play/test_vn_play_db.py \
  tldw_Server_API/tests/VN_Play/test_vn_play_state.py \
  tldw_Server_API/tests/VN_Play/test_vn_play_turns.py \
  tldw_Server_API/tests/VN_Play/test_vn_play_api.py \
  -q
```

Expected: all focused VN Play tests pass.

- [ ] **Step 2: Run Bandit on touched backend scope**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit \
  tldw_Server_API/app/core/DB_Management/VNPlay_DB.py \
  tldw_Server_API/app/core/VN_Play/service.py \
  tldw_Server_API/app/core/VN_Play/constants.py \
  tldw_Server_API/app/api/v1/endpoints/vn_play.py \
  -f json \
  -o /tmp/bandit_vn_play_story_branch.json
```

Expected: no new findings in touched code. If existing baseline findings appear, inspect and document whether they are pre-existing or introduced.

- [ ] **Step 3: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no whitespace or conflict-marker issues.

- [ ] **Step 4: Update implementation Backlog task**

Record:

- Test commands and outcomes.
- Bandit output path.
- Any known skips or blockers.
- Final summary with what changed and why.

- [ ] **Step 5: Final commit**

Run:

```bash
git add <implementation-backlog-task-file>
git commit -m "Close VN Play story branch persistence implementation"
```

If the Backlog task was already included in the previous task commits and there are no changes, skip this commit and note that in the final summary.

## Final Verification Before PR

Before opening or updating a PR, run:

```bash
git status --short --branch
git log --oneline origin/dev..HEAD
```

Expected:

- Working tree clean.
- Branch contains the design/plan commits plus implementation commits.
- No unrelated files from the main dirty checkout.
