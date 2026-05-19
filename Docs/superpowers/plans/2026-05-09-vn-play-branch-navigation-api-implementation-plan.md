# VN Play Branch Navigation API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backend-owned VN Play branch navigation API with active path/range data, branch-aware event filtering, and guarded branch/checkpoint restore idempotency.

**Architecture:** Keep branch navigation as a derived read model built from existing branch rows, ordered events, and persisted scene state. Add session-action persistence and a shared session mutation gate so branch/checkpoint restore cannot race with turns. Expose the read model and restore flow through the existing VN Play service, schemas, endpoints, and API docs.

**Tech Stack:** FastAPI, Pydantic v2, SQLite-backed `CharactersRAGDB`, VN Play repository/service modules, pytest, Bandit.

---

## File Map

- Create `tldw_Server_API/app/core/VN_Play/branch_navigation.py`
  - Pure branch navigation read-model builder.
  - Warning payload helpers.
  - Direct `event_range` vs `subtree_event_range` derivation.
  - Branch-aware event filtering helpers.
  - No database calls.
- Modify `tldw_Server_API/app/core/VN_Play/constants.py`
  - Add branch navigation/restore error codes.
  - Add branch restore target constants.
  - Add branch navigation replay cap constant.
- Modify `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
  - Add `active_session_action_id` to `vn_play_sessions`.
  - Add `vn_play_session_actions`.
  - Add migration helpers for existing SQLite DBs.
  - Add session-action create/get/update helpers.
  - Add shared turn/restore mutation gate helpers.
  - Extend `list_events()` with bounded pagination fields if useful.
- Modify `tldw_Server_API/app/core/VN_Play/service.py`
  - Add `get_branch_navigation()`.
  - Add branch-aware `list_events()` parameters.
  - Add a branch-event listing helper that returns events plus warning metadata while preserving the existing bare-list service/API compatibility path.
  - Add branch restore orchestration.
  - Enforce checkpoint restore idempotency through session actions.
  - Attach current active `branch_node_id` to completed Story branch events.
- Modify `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
  - Add branch navigation response models.
  - Add branch restore request/response models.
  - Add optional event filter query response compatibility only where needed.
- Modify `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
  - Add `GET /sessions/{session_id}/branch-navigation`.
  - Add `POST /sessions/{session_id}/branches/{branch_id}/restore`.
  - Extend `GET /sessions/{session_id}/events` query params.
  - Emit branch-filter warning metadata through `X-VN-Play-Warnings` when the list response cannot carry warning payloads without breaking compatibility.
  - Map new stable errors to `400`, `404`, and `409`.
- Modify `Docs/API-related/VN_PLAY_API.md`
  - Document branch navigation payload.
  - Document branch-aware event filtering.
  - Document branch restore and checkpoint restore idempotency semantics.
- Test `tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py`
  - Pure navigation builder and warning payload coverage.
- Test `tldw_Server_API/tests/VN_Play/test_vn_play_db.py`
  - Schema, session-action, mutation-gate, and idempotency helpers.
- Test `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`
  - Service branch navigation, branch restore, checkpoint idempotency, and turn/restore lock interactions.
- Test `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`
  - Endpoint response shapes and HTTP error mappings.
  - Branch-event warning header behavior.
- Keep existing `tldw_Server_API/tests/VN_Play/test_vn_play_state.py`
  - Add assertions only if replay semantics need tightening.

## Preflight

- [ ] **Step 1: Confirm branch, spec, and clean state**

Run:

```bash
git status --short --branch
test -f Docs/superpowers/specs/2026-05-09-vn-play-branch-navigation-api-design.md
```

Expected: branch is `codex/vn-play-branch-navigation-api`; worktree is clean before implementation edits.

- [ ] **Step 2: Create or reuse an implementation Backlog task**

Use Backlog.md MCP before runtime code edits. Suggested task:

```text
Implement VN Play branch navigation API
```

References:

```text
https://github.com/rmusser01/tldw_server/issues/1463
Docs/superpowers/specs/2026-05-09-vn-play-branch-navigation-api-design.md
Docs/superpowers/plans/2026-05-09-vn-play-branch-navigation-api-implementation-plan.md
```

Expected: implementation task exists and is marked `In Progress`.

- [ ] **Step 3: Run focused baseline**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q
```

Expected: current focused VN Play baseline passes. Previously observed baseline on this branch: `71 passed, 5 warnings`.

## Task 1: Pure Branch Navigation Read Model

**Files:**
- Create: `tldw_Server_API/app/core/VN_Play/branch_navigation.py`
- Modify: `tldw_Server_API/app/core/VN_Play/constants.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py`

- [ ] **Step 1: Write failing tests for active path and parent derivation**

Create `test_vn_play_branch_navigation.py` with focused pure-function tests. Start with:

```python
from tldw_Server_API.app.core.VN_Play.branch_navigation import build_branch_navigation


def _event(
    event_id: int,
    sequence: int,
    event_type: str,
    payload: dict,
    branch_node_id: int | None = None,
) -> dict:
    return {
        "id": event_id,
        "session_id": 1,
        "owner_user_id": 42,
        "sequence_number": sequence,
        "event_type": event_type,
        "event_payload": payload,
        "source": "runtime",
        "branch_node_id": branch_node_id,
    }


def test_navigation_derives_active_path_and_parent_branch_ids() -> None:
    branches = [
        {
            "id": 10,
            "session_id": 1,
            "owner_user_id": 42,
            "parent_event_id": 2,
            "branch_label": "Open the door",
            "branch_path": [
                {
                    "type": "choice",
                    "choice_id": "open",
                    "choice_presented_event_id": 2,
                    "scene_version": 1,
                }
            ],
            "status": "active",
        },
        {
            "id": 11,
            "session_id": 1,
            "owner_user_id": 42,
            "parent_event_id": 6,
            "branch_label": "Step inside",
            "branch_path": [
                {
                    "type": "choice",
                    "choice_id": "open",
                    "choice_presented_event_id": 2,
                    "scene_version": 1,
                },
                {
                    "type": "choice",
                    "choice_id": "inside",
                    "choice_presented_event_id": 6,
                    "scene_version": 2,
                },
            ],
            "status": "active",
        },
    ]
    events = [
        _event(2, 2, "choice_presented", {"choices": [{"id": "open"}], "scene_version": 1}),
        _event(3, 3, "choice_selected", {"choice_id": "open", "branch_node_id": 10, "scene_version": 1}, 10),
        _event(6, 6, "choice_presented", {"choices": [{"id": "inside"}], "scene_version": 2}, 10),
        _event(7, 7, "choice_selected", {"choice_id": "inside", "branch_node_id": 11, "scene_version": 2}, 11),
    ]

    navigation = build_branch_navigation(
        session={"id": 1, "mode": "story", "scene_version": 3},
        branches=branches,
        events=events,
        scene_state={"active_branch_node_id": 11, "last_event_id": 7, "scene_version": 3},
    )

    assert [step["branch_id"] for step in navigation["active_path"]] == [10, 11]
    node = next(item for item in navigation["branches"] if item["branch_id"] == 11)
    assert node["parent_branch_id"] == 10
    assert node["depth"] == 2
    assert node["is_active"] is True
    assert node["is_on_active_path"] is True
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py::test_navigation_derives_active_path_and_parent_branch_ids -q
```

Expected: fails because `branch_navigation.py` does not exist.

- [ ] **Step 3: Add branch navigation constants**

In `constants.py`, add:

```python
BRANCH_RESTORE_TARGET_LATEST = "branch_latest"
BRANCH_RESTORE_TARGET_CHOICE_POINT = "choice_point"
VN_PLAY_BRANCH_RESTORE_TARGETS = (
    BRANCH_RESTORE_TARGET_LATEST,
    BRANCH_RESTORE_TARGET_CHOICE_POINT,
)
VN_PLAY_BRANCH_NAV_MAX_REPLAY_EVENTS = 5000

ERROR_BRANCH_NOT_FOUND = "branch_not_found"
ERROR_BRANCH_RESTORE_NOT_ALLOWED = "branch_restore_not_allowed"
ERROR_BRANCH_RESTORE_TARGET_UNAVAILABLE = "branch_restore_target_unavailable"
ERROR_BRANCH_RESTORE_AMBIGUOUS = "branch_restore_ambiguous"
ERROR_RESTORE_ACTION_IN_PROGRESS = "restore_action_in_progress"
ERROR_BRANCH_NAVIGATION_UNAVAILABLE = "branch_navigation_unavailable"
ERROR_BRANCH_INTERVAL_REPLAY_LIMIT_EXCEEDED = "branch_interval_replay_limit_exceeded"
```

Also add those error codes to `VN_PLAY_ERROR_CODES`.

- [ ] **Step 4: Implement minimal navigation builder**

Create `branch_navigation.py` with a pure API shaped like:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_branch_navigation(
    *,
    session: Mapping[str, Any],
    branches: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    scene_state: Mapping[str, Any] | None,
    replay_limit: int = 5000,
) -> dict[str, Any]:
    ...
```

Minimum responsibilities:

- normalize branch path steps by `choice_id`, `choice_presented_event_id`, `scene_version`
- derive `parent_branch_id`
- derive active path from `scene_state.active_branch_node_id`
- expose branch fields without mutating inputs
- return top-level `warnings: []`

- [ ] **Step 5: Run the test and verify it passes**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py::test_navigation_derives_active_path_and_parent_branch_ids -q
```

Expected: pass.

- [ ] **Step 6: Add tests for direct vs subtree event ranges**

Add tests that prove:

- child branch events do not extend parent `event_range`
- child branch events do extend parent `subtree_event_range`
- `branch_latest` target data uses direct `event_range`

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py -q
```

Expected: new range tests fail before implementation.

- [ ] **Step 7: Implement event range derivation**

In `branch_navigation.py`, add helpers:

```python
def filter_branch_events(
    *,
    branch_id: int,
    branches: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    include_descendants: bool = False,
    after_sequence: int | None = None,
    limit: int = 100,
    replay_limit: int = 5000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ...
```

Use explicit `event["branch_node_id"]` first. For missing tags, derive active intervals by replaying `choice_selected` and `session_restored` payloads until `replay_limit`.

- [ ] **Step 8: Add warning payload tests**

Add tests for stable warning shape:

```python
def test_navigation_warning_payloads_are_frontend_safe() -> None:
    ...
    warning = navigation["warnings"][0]
    assert set(warning) >= {"code", "severity", "recoverable"}
    assert "Traceback" not in str(warning)
```

- [ ] **Step 9: Run pure navigation tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py -q
```

Expected: pass.

- [ ] **Step 10: Commit Task 1**

Run:

```bash
git add tldw_Server_API/app/core/VN_Play/branch_navigation.py tldw_Server_API/app/core/VN_Play/constants.py tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py
git commit -m "Add VN Play branch navigation read model"
```

## Task 2: Repository Session Actions And Shared Mutation Gate

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_db.py`

- [ ] **Step 1: Write failing schema test for session actions and active action column**

Add to `test_vn_play_db.py`:

```python
def test_initialized_creates_session_actions_and_active_session_action_column(
    chacha_db: CharactersRAGDB,
) -> None:
    VNPlayRepository.initialized(chacha_db)

    tables = chacha_db.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vn_play_session_actions'"
    ).fetchall()
    assert tables

    columns = chacha_db.execute_query("PRAGMA table_info(vn_play_sessions)").fetchall()
    assert "active_session_action_id" in {row["name"] for row in columns}
```

- [ ] **Step 2: Run schema test and verify it fails**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py::test_initialized_creates_session_actions_and_active_session_action_column -q
```

Expected: fail because table/column do not exist.

- [ ] **Step 3: Add schema and migration helpers**

In `VNPlay_DB.py`:

- add `active_session_action_id INTEGER` to `vn_play_sessions`
- add `vn_play_session_actions`
- add indexes:
  - `idx_vn_play_session_actions_session`
  - `idx_vn_play_session_actions_owner_status`
- add a migration helper after table creation:

```python
def _ensure_column(conn: Any, table_name: str, column_name: str, definition: str) -> None:
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    if column_name not in {row["name"] for row in columns}:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
```

Use this only with hard-coded table/column names, not user input.

- [ ] **Step 4: Run schema test and existing DB tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py -q
```

Expected: pass.

- [ ] **Step 5: Write failing session action idempotency tests**

Add tests for:

- create action row
- same idempotency key and same hash returns existing action
- same idempotency key and different hash raises `ValueError("idempotency_key_conflict")`
- same idempotency key used for another action type conflicts because hash includes action type

Suggested helper API:

```python
action = repo.create_session_action(
    session_id=session["id"],
    owner_user_id=42,
    action_type="branch_restore",
    idempotency_key="restore-1",
    request_payload_hash="hash-branch-restore-1",
)
```

- [ ] **Step 6: Implement session action helpers**

Add repository methods:

```python
def create_session_action(...)
def get_session_action(...)
def get_session_action_by_key(...)
def update_session_action(...)
def latest_active_session_action(...)
```

Use `_mapped_update_values()` pattern already used for sessions and turn requests. Add `_SESSION_ACTION_UPDATE_COLUMNS`, `_SESSION_ACTION_UPDATE_STATEMENTS`, and `_decode_session_action()` so `response_payload_json` and `error_json` are decoded like turn requests.

- [ ] **Step 7: Write failing shared mutation gate tests**

Add repository tests:

- `try_acquire_turn_lock()` returns false when `active_session_action_id` is set
- `try_acquire_session_action_lock()` returns false when `active_turn_request_id` is set
- stale `scene_version` returns false
- successful action lock sets `active_session_action_id`
- clear helper removes `active_session_action_id`

Suggested repository API:

```python
locked = repo.try_acquire_session_action_lock(
    session_id=session["id"],
    owner_user_id=42,
    action_id=action["id"],
    expected_scene_version=0,
)
```

- [ ] **Step 8: Implement shared mutation gate**

Update `try_acquire_turn_lock()` SQL to require:

```sql
AND active_session_action_id IS NULL
```

Add:

```python
def try_acquire_session_action_lock(...)
def clear_session_action_lock(...)
```

Use scene-version compare-and-swap and owner checks.

- [ ] **Step 9: Run repository tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py -q
```

Expected: pass.

- [ ] **Step 10: Commit Task 2**

Run:

```bash
git add tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/tests/VN_Play/test_vn_play_db.py
git commit -m "Add VN Play session action locking"
```

## Task 3: Service Branch Navigation And Branch-Aware Event Filtering

**Files:**
- Modify: `tldw_Server_API/app/core/VN_Play/service.py`
- Modify: `tldw_Server_API/app/core/VN_Play/branch_navigation.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py`

- [ ] **Step 1: Write failing service navigation test**

Add to `test_vn_play_turns.py`:

```python
@pytest.mark.asyncio
async def test_branch_navigation_service_returns_active_path(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(repo=repo, owner_user_id=42, adapter=DeterministicVNPlayTurnAdapter())
    session = create_story_session_with_visible_choice(service, repo)

    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="choice-open",
    )

    navigation = service.get_branch_navigation(session.id)
    assert navigation["active_branch_node_id"] == navigation["active_path"][-1]["branch_id"]
    assert navigation["branches"][0]["restore"]["supported"] is True
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py::test_branch_navigation_service_returns_active_path -q
```

Expected: fail because `VNPlayService.get_branch_navigation` does not exist.

- [ ] **Step 3: Implement `VNPlayService.get_branch_navigation()`**

In `service.py`:

```python
def get_branch_navigation(self, session_id: int) -> dict[str, Any]:
    session = self.get_session(session_id)
    branches = self.repo.list_branches(session_id, owner_user_id=self.owner_user_id)
    events = self.repo.list_events(session_id)
    scene_state = self.repo.get_scene_state(session_id, owner_user_id=self.owner_user_id)
    return build_branch_navigation(
        session=asdict(session),
        branches=branches,
        events=events,
        scene_state=scene_state,
    )
```

Use the existing `dataclasses.asdict` import if available; otherwise add it.

- [ ] **Step 4: Write failing event filter service tests**

Add tests for:

- direct branch events only
- `include_descendants=True` includes child branch events
- invalid branch id raises `VNPlayNotFoundError("branch_not_found")`
- `limit` is bounded by service or endpoint layer

Keep the existing `list_events(session_id)` compatibility path returning only a list. Add a metadata-aware helper for the endpoint and tests:

```python
def list_events_with_metadata(
    self,
    session_id: int,
    *,
    branch_id: int | None = None,
    after_sequence: int | None = None,
    limit: int | None = None,
    include_descendants: bool = False,
) -> dict[str, Any]:
```

Return shape:

```python
{"events": [...], "warnings": [...]}
```

- [ ] **Step 5: Implement branch-aware service event filtering**

Keep default behavior unchanged when `branch_id is None`.

When `branch_id` is provided:

- verify branch belongs to the session/user
- call `filter_branch_events()`
- apply `after_sequence` and `limit`
- return warning payloads from `list_events_with_metadata()`
- keep `list_events()` as a compatibility wrapper that returns only `events`

- [ ] **Step 6: Attach active branch id to new Story branch events**

In `_complete_turn()`:

- derive current active branch id from persisted scene state or `derive_scene_state(...)`
- pass `branch_node_id=active_branch_id` to:
  - `model_turn`
  - visual directive requested/applied/rejected events
  - `choice_presented`
  - `scene_state_changed`
  - `turn_completed`

Avoid tagging events when no branch is active or session mode is Freeform.

- [ ] **Step 7: Run service and pure navigation tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q
```

Expected: pass.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add tldw_Server_API/app/core/VN_Play/branch_navigation.py tldw_Server_API/app/core/VN_Play/service.py tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py
git commit -m "Expose VN Play branch navigation service"
```

## Task 4: Guarded Branch And Checkpoint Restore

**Files:**
- Modify: `tldw_Server_API/app/core/VN_Play/service.py`
- Modify: `tldw_Server_API/app/core/VN_Play/branch_navigation.py`
- Modify: `tldw_Server_API/app/core/VN_Play/constants.py`
- Modify: `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_db.py`

- [ ] **Step 1: Write failing branch restore service test**

Add to `test_vn_play_turns.py`:

```python
@pytest.mark.asyncio
async def test_branch_restore_latest_advances_scene_version_and_replays_response(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(repo=repo, owner_user_id=42, adapter=DeterministicVNPlayTurnAdapter())
    session = create_story_session_with_visible_choice(service, repo)
    turn = await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="choice-open-restore",
    )
    branch_id = service.list_branches(session.id)[0]["id"]

    response = service.restore_branch(
        session.id,
        branch_id=branch_id,
        client_scene_version=turn.scene_version,
        idempotency_key="restore-branch-open",
        target="branch_latest",
    )

    assert response["scene_version"] == turn.scene_version + 1
    assert response["branch_id"] == branch_id
    assert response["branch_navigation"]["active_branch_node_id"] == branch_id

    replayed = service.restore_branch(
        session.id,
        branch_id=branch_id,
        client_scene_version=turn.scene_version,
        idempotency_key="restore-branch-open",
        target="branch_latest",
    )
    assert replayed["replayed"] is True
    events = repo.list_events(session.id)
    assert [event["event_type"] for event in events].count("session_restored") == 1
```

- [ ] **Step 2: Run restore test and verify it fails**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py::test_branch_restore_latest_advances_scene_version_and_replays_response -q
```

Expected: fail because `restore_branch()` does not exist.

- [ ] **Step 3: Add restore response/request helper logic**

In `service.py`, implement:

```python
def restore_branch(
    self,
    session_id: int,
    *,
    branch_id: int,
    client_scene_version: int,
    idempotency_key: str,
    target: str = BRANCH_RESTORE_TARGET_LATEST,
) -> dict[str, Any]:
    ...
```

Flow:

1. Validate session exists and mode is Story.
2. Validate branch belongs to the session/user.
3. Build request payload hash including action type, branch id, target, and client scene version.
4. Look up/create the session action by idempotency key and compare the request payload hash.
5. If the action is already completed with the same hash, replay the stored response before stale scene-version or active-lock checks.
6. Acquire the session action lock for new or resumable pending actions.
7. Derive target event id from branch navigation.
8. Derive target scene snapshot by replaying events through target event.
9. Commit restore completion through one repository transaction helper that appends `session_restored`, derives/persists scene state, updates session scene version, clears `active_session_action_id`, stores the action response payload, and returns the committed event/response data.

Do not stitch the restore completion phase together from independent service-level repository calls. A partial restore must not leave a committed `session_restored` event without matching scene/session/action state.

- [ ] **Step 4: Write failing `choice_point` restore test**

Test that:

- restore target is parent `choice_presented`
- restored scene exposes visible choices
- restored `active_branch_node_id` is parent branch or `None`, not selected branch

- [ ] **Step 5: Implement `choice_point` target**

Use branch `parent_event_id`; verify it points to `choice_presented`. Reject with `VNPlayTurnError(ERROR_BRANCH_RESTORE_TARGET_UNAVAILABLE)` or equivalent stable service error if unavailable.

- [ ] **Step 6: Write failing lock/idempotency service tests**

Add tests:

- branch restore rejects stale scene version
- branch restore rejects active turn
- branch restore rejects active restore action
- turn submission rejects active restore action
- idempotency conflict on same key with different branch target
- branch restore is not allowed for Freeform sessions

- [ ] **Step 7: Implement lock/idempotency behavior**

Coordinate service and repository helpers:

- session action statuses: `pending`, `completed`, `failed`, `abandoned`
- replay completed response if hash matches
- perform completed-response replay before rejecting the duplicate request as stale against the now-advanced session scene version
- conflict if hash differs
- clear `active_session_action_id` on success/failure/abandonment
- ensure `try_acquire_turn_lock` sees active restore action

- [ ] **Step 8: Harden checkpoint restore idempotency**

Change `restore_checkpoint()` signature to:

```python
def restore_checkpoint(
    self,
    session_id: int,
    checkpoint_id: int,
    *,
    idempotency_key: str,
) -> dict[str, Any]:
    ...
```

Preserve endpoint request shape and wire `request.idempotency_key`.

Add tests:

- duplicate checkpoint restore with same key replays response
- same key with different checkpoint id conflicts
- checkpoint restore advances scene version by one

- [ ] **Step 9: Run restore-focused tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q
```

Expected: pass.

- [ ] **Step 10: Commit Task 4**

Run:

```bash
git add tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/app/core/VN_Play/branch_navigation.py tldw_Server_API/app/core/VN_Play/constants.py tldw_Server_API/app/core/VN_Play/service.py tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py
git commit -m "Add guarded VN Play branch restore"
```

## Task 5: API Schemas And Endpoints

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
- Test: `tldw_Server_API/tests/VN_Play/test_vn_play_api.py`

- [ ] **Step 1: Write failing API tests**

Add tests for:

- `GET /api/v1/vn-play/sessions/{session_id}/branch-navigation`
- `GET /api/v1/vn-play/sessions/{session_id}/events?branch_id=...&limit=...&include_descendants=...`
- `GET /api/v1/vn-play/sessions/{session_id}/events?...` emits `X-VN-Play-Warnings` with stable JSON warning payloads when branch replay is capped or ambiguous
- `POST /api/v1/vn-play/sessions/{session_id}/branches/{branch_id}/restore`
- stale restore returns `409 stale_scene_version`
- missing branch returns `404 not_found` or `404 branch_not_found` consistently with existing endpoint style

- [ ] **Step 2: Run one new API test and verify it fails**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py::test_branch_navigation_endpoint_returns_active_path -q
```

Expected: fail because endpoint/schema does not exist.

- [ ] **Step 3: Add Pydantic schemas**

In `vn_play_schemas.py`, add:

```python
VNPlayBranchRestoreTarget = Literal["branch_latest", "choice_point"]
VNPlayBranchWarningSeverity = Literal["info", "warning", "high_risk"]
```

Add models:

- `VNPlayBranchWarning`
- `VNPlayBranchEventRange`
- `VNPlayBranchRestoreCapability`
- `VNPlayBranchPathStep`
- `VNPlayBranchNavigationNode`
- `VNPlayBranchNavigationResponse`
- `VNPlayBranchRestoreRequest`
- `VNPlayBranchRestoreResponse`

Add all new exported names to `__all__`.

- [ ] **Step 4: Add endpoints and query params**

In `vn_play.py`:

- import new schemas
- extend `list_events()` parameters:
  - `branch_id: int | None = Query(default=None, ge=1)`
  - `after_sequence: int | None = Query(default=None, ge=0)`
  - `limit: int = Query(default=100, ge=1, le=250)`
  - `include_descendants: bool = Query(default=False)`
- accept a `Response` parameter and set `X-VN-Play-Warnings` to a compact JSON list when `list_events_with_metadata()` returns warnings
- add `branch_navigation()`
- add `restore_branch()`
- pass checkpoint restore idempotency key to service

- [ ] **Step 5: Extend HTTP error mapping**

Map:

- `ERROR_BRANCH_NOT_FOUND` -> `404`
- `ERROR_STALE_SCENE_VERSION`, `ERROR_TURN_IN_PROGRESS`, `ERROR_RESTORE_ACTION_IN_PROGRESS`, `ERROR_IDEMPOTENCY_KEY_CONFLICT` -> `409`
- branch restore target/allowed/ambiguous errors -> `400`

Keep existing `not_found` behavior for missing sessions.

- [ ] **Step 6: Run API tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q
```

Expected: pass.

- [ ] **Step 7: Commit Task 5**

Run:

```bash
git add tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py tldw_Server_API/app/api/v1/endpoints/vn_play.py tldw_Server_API/tests/VN_Play/test_vn_play_api.py
git commit -m "Expose VN Play branch navigation API"
```

## Task 6: API Docs And Final Verification

**Files:**
- Modify: `Docs/API-related/VN_PLAY_API.md`
- Modify: Backlog implementation task file created during Preflight

- [ ] **Step 1: Update VN Play API docs**

Add sections:

- Branch Navigation
- Branch-Aware Event Listing
- Branch Restore
- Restore Idempotency
- Error Codes

Document that `GET /events` keeps its existing list body for compatibility and uses `X-VN-Play-Warnings` for branch-filter warning payloads.

Include examples for:

```bash
curl "http://127.0.0.1:8000/api/v1/vn-play/sessions/1/branch-navigation" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY"

curl "http://127.0.0.1:8000/api/v1/vn-play/sessions/1/events?branch_id=12&include_descendants=true&limit=100" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY"

curl -X POST "http://127.0.0.1:8000/api/v1/vn-play/sessions/1/branches/12/restore" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "client_scene_version": 6,
    "idempotency_key": "session-1-restore-branch-12",
    "target": "choice_point"
  }'
```

- [ ] **Step 2: Run focused test suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q
```

Expected: pass.

- [ ] **Step 3: Run Bandit on touched backend scope**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/VN_Play tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/app/api/v1/endpoints/vn_play.py tldw_Server_API/app/api/v1/schemas/vn_play_schemas.py -f json -o /tmp/bandit_vn_play_branch_navigation.json
```

Expected: exit code `0`; `/tmp/bandit_vn_play_branch_navigation.json` contains `"results": []` or only pre-existing findings outside touched code. Fix any new findings before continuing.

- [ ] **Step 4: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output and exit code `0`.

- [ ] **Step 5: Update Backlog implementation task**

Record:

- tests run
- Bandit result path
- known skips/blockers
- final summary

- [ ] **Step 6: Commit docs and task finalization**

Run:

```bash
git add Docs/API-related/VN_PLAY_API.md backlog/tasks/<implementation-task-file>.md
git commit -m "Document VN Play branch navigation API"
```

- [ ] **Step 7: Prepare PR**

Run:

```bash
git status --short
git log --oneline origin/dev..HEAD
```

Expected: clean worktree with reviewable commits for Tasks 1-6.

Open a PR against `dev` after final verification and include a human-editable change summary. The summary must explain why the implementation uses a derived navigation model plus session action locking instead of frontend reconstruction or a persisted navigation table.
