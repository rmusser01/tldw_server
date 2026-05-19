# Sandbox & ACP Competitive Improvements — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 17 improvements to tldw_server's Sandbox and ACP modules inspired by ClaudeCodeUI, Codexia, and 1code.dev — covering sandbox runtimes, streaming, automation, and permissions with a unified governance model.

**Architecture:** All permission features build on a unified GovernanceFilter → MCPHub integration (item 4.0) that makes MCPHub the single authority. Streaming features share a replay abstraction used by both SSE and WebSocket. Automation features compose the existing Scheduler and WorkflowsScheduler infrastructure. Sandbox features add new RuntimeType variants following existing runner patterns.

**Tech Stack:** FastAPI, SQLite/PostgreSQL, asyncio, APScheduler, Docker CLI, git CLI, fnmatch

**Design doc:** `docs/plans/2026-04-05-sandbox-acp-competitive-improvements-design.md`

---

## Phase 0: Endpoint File Split (Prep)

Split `agent_client_protocol.py` (2,552 lines, 25 routes) into sub-modules before adding ~12 new endpoints.

### Task 0.1: Create ACP endpoint sub-module structure

**Files:**
- Create: `tldw_Server_API/app/api/v1/endpoints/acp_schedules.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/acp_triggers.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/acp_permissions.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- Modify: `tldw_Server_API/app/main.py` (router registration)

**Step 1: Create stub routers for each new endpoint file**

Each file gets an `APIRouter` with appropriate prefix and tags:

```python
# acp_schedules.py
from fastapi import APIRouter
router = APIRouter(prefix="/acp/schedules", tags=["acp-schedules"])

# acp_triggers.py
from fastapi import APIRouter
router = APIRouter(prefix="/acp/triggers", tags=["acp-triggers"])

# acp_permissions.py
from fastapi import APIRouter
router = APIRouter(prefix="/acp/permissions", tags=["acp-permissions"])
```

**Step 2: Register new routers in `main.py`**

Find the existing `_include_if_enabled("acp", acp_router, ...)` pattern around line 6804 and add:

```python
from tldw_Server_API.app.api.v1.endpoints.acp_schedules import router as acp_schedules_router
from tldw_Server_API.app.api.v1.endpoints.acp_triggers import router as acp_triggers_router
from tldw_Server_API.app.api.v1.endpoints.acp_permissions import router as acp_permissions_router

# After existing ACP router registration:
_include_if_enabled("acp", acp_schedules_router, prefix=API_V1_PREFIX)
_include_if_enabled("acp", acp_triggers_router, prefix=API_V1_PREFIX)
_include_if_enabled("acp", acp_permissions_router, prefix=API_V1_PREFIX)
```

**Step 3: Verify server starts and existing ACP tests pass**

Run: `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/ -x -q`

**Step 4: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/acp_*.py tldw_Server_API/app/main.py
git commit -m "refactor: split ACP endpoints into sub-modules for schedule, trigger, and permission CRUD"
```

---

## Phase 1: Foundations

### Task 1.1: Permission Unification — GovernanceFilter accepts MCPHub snapshot (Item 4.0)

**Files:**
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/governance_filter.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_governance_filter_unified.py`

**Step 1: Write failing test — GovernanceFilter denies tool in snapshot.denied_tools**

```python
# test_governance_filter_unified.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from tldw_Server_API.app.core.Agent_Client_Protocol.governance_filter import GovernanceFilter
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind


@pytest.fixture
def bus():
    b = MagicMock()
    b.publish = AsyncMock()
    return b


@pytest.fixture
def snapshot_with_denied():
    """Snapshot that denies 'dangerous_tool'."""
    snap = MagicMock()
    snap.resolved_policy_document = {
        "denied_tools": ["dangerous_tool"],
        "allowed_tools": [],
        "tool_tier_overrides": {},
    }
    snap.approval_summary = {}
    return snap


def _make_tool_call(tool_name: str) -> AgentEvent:
    return AgentEvent(
        session_id="test-session",
        kind=AgentEventKind.TOOL_CALL,
        payload={"tool_name": tool_name},
        metadata={},
    )


@pytest.mark.asyncio
async def test_denied_tool_blocked_immediately(bus, snapshot_with_denied):
    gf = GovernanceFilter(bus=bus, policy_snapshot=snapshot_with_denied)
    event = _make_tool_call("dangerous_tool")
    await gf.process(event)

    # Should NOT publish the tool call; should publish a TOOL_RESULT error
    calls = bus.publish.call_args_list
    assert len(calls) == 1
    published = calls[0][0][0]
    assert published.kind == AgentEventKind.TOOL_RESULT
    assert "denied" in published.payload.get("error", "").lower()


@pytest.mark.asyncio
async def test_allowed_tool_auto_approved(bus, snapshot_with_denied):
    snap = MagicMock()
    snap.resolved_policy_document = {
        "denied_tools": [],
        "allowed_tools": ["safe_read"],
        "tool_tier_overrides": {},
    }
    snap.approval_summary = {}
    gf = GovernanceFilter(bus=bus, policy_snapshot=snap)
    event = _make_tool_call("safe_read")
    await gf.process(event)

    calls = bus.publish.call_args_list
    assert len(calls) == 1
    assert calls[0][0][0].kind == AgentEventKind.TOOL_CALL


@pytest.mark.asyncio
async def test_no_snapshot_falls_through_to_tier(bus):
    """Without snapshot, existing tier logic applies."""
    gf = GovernanceFilter(bus=bus, policy_snapshot=None)
    event = _make_tool_call("read_file")  # heuristic → "auto"
    await gf.process(event)
    assert bus.publish.call_count == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_governance_filter_unified.py -v`
Expected: FAIL — `GovernanceFilter.__init__()` doesn't accept `policy_snapshot`

**Step 3: Implement unified GovernanceFilter**

Modify `governance_filter.py` `__init__` (line 56) to accept optional snapshot and session metadata:

```python
def __init__(
    self,
    bus: SessionEventBus,
    default_timeout_sec: int = 300,
    policy_snapshot: ACPRuntimePolicySnapshot | None = None,
    session_metadata: dict[str, Any] | None = None,
    permission_decision_service: Any | None = None,  # Added in 4.2
) -> None:
    self._bus = bus
    self._default_timeout_sec = default_timeout_sec
    self._pending: dict[str, _PendingEntry] = {}
    self._snapshot = policy_snapshot
    self._session_metadata = session_metadata or {}
    self._perm_service = permission_decision_service
```

Add a `_check_snapshot_policy` method before `process()`:

```python
def _check_snapshot_policy(self, tool_name: str) -> str | None:
    """Check MCPHub snapshot for tool permission. Returns tier or None."""
    if self._snapshot is None:
        return None
    doc = self._snapshot.resolved_policy_document
    denied = doc.get("denied_tools", [])
    allowed = doc.get("allowed_tools", [])
    overrides = doc.get("tool_tier_overrides", {})

    # Step 1: denied_tools → deny
    if any(fnmatch.fnmatch(tool_name, p) for p in denied):
        return "_deny"
    # Step 2: allowed_tools → auto
    if any(fnmatch.fnmatch(tool_name, p) for p in allowed):
        return "auto"
    # Step 3: tool_tier_overrides → use tier
    for pattern, tier in overrides.items():
        if fnmatch.fnmatch(tool_name, pattern):
            return tier
    return None
```

Modify `process()` (after line 95, the `tool_name` extraction) to check snapshot first:

```python
    # --- MCPHub snapshot check (unified hierarchy steps 1-4) ---
    snapshot_tier = self._check_snapshot_policy(tool_name)
    if snapshot_tier == "_deny":
        deny_event = AgentEvent(
            session_id=event.session_id,
            kind=AgentEventKind.TOOL_RESULT,
            payload={"tool_call_id": event.payload.get("tool_call_id", ""),
                     "error": f"Tool '{tool_name}' denied by policy"},
            metadata={"governance_action": "denied_by_snapshot"},
        )
        await self._bus.publish(deny_event)
        return
    if snapshot_tier is not None:
        tier = snapshot_tier
    else:
        # Fall through to existing determine_permission_tier()
        tier = event.payload.get("permission_tier") or determine_permission_tier(tool_name)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_governance_filter_unified.py -v`
Expected: PASS

**Step 5: Run full ACP test suite for regression**

Run: `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/ -x -q`
Expected: PASS (existing tests use `GovernanceFilter(bus=bus)` which still works since new params are optional)

**Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Agent_Client_Protocol/governance_filter.py \
       tldw_Server_API/tests/Agent_Client_Protocol/test_governance_filter_unified.py
git commit -m "feat(acp): unify GovernanceFilter with MCPHub policy snapshot

GovernanceFilter now checks MCPHub snapshot (denied_tools, allowed_tools,
tool_tier_overrides) before falling back to ACP tier heuristics.
This establishes MCPHub as single authority for tool permissions."
```

---

### Task 1.2: Migrate ACP permission policies to database (Item 4.0 continued)

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py` — add `permission_policies` table
- Modify: `tldw_Server_API/app/services/admin_acp_sessions_service.py` — replace in-memory dict with DB calls
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_permission_policies_db.py`

**Step 1: Write test — permission policies persist across service restart**

```python
@pytest.mark.asyncio
async def test_permission_policy_survives_restart(tmp_path):
    db_path = tmp_path / "test_acp.db"
    db = ACPSessionsDB(str(db_path))
    db.create_permission_policy(
        name="test-policy",
        rules=[{"tool_pattern": "Bash(*)", "tier": "individual"}],
        priority=10,
    )
    # Simulate restart: create new DB instance
    db2 = ACPSessionsDB(str(db_path))
    policies = db2.list_permission_policies()
    assert len(policies) == 1
    assert policies[0]["name"] == "test-policy"
```

**Step 2: Add `permission_policies` table to `ACP_Sessions_DB.py`**

In the schema creation section, add:

```sql
CREATE TABLE IF NOT EXISTS permission_policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    rules_json TEXT NOT NULL,
    org_id TEXT,
    team_id TEXT,
    priority INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

Bump `_SCHEMA_VERSION` from 7 to 8. Add CRUD methods (`create_permission_policy`, `list_permission_policies`, `update_permission_policy`, `delete_permission_policy`, `resolve_permission_tier`).

**Step 3: Update `admin_acp_sessions_service.py` to use DB instead of in-memory dict**

Replace `_permission_policies: dict[int, PermissionPolicy]` with delegation to `ACPSessionsDB` methods. The `resolve_permission_tier()` method now queries the DB.

**Step 4: Run tests, commit**

```bash
git commit -m "feat(acp): migrate permission policies from in-memory to ACP_Sessions_DB

Policies now survive server restarts. Schema version bumped to 8."
```

---

### Task 1.3: Shared Replay Abstraction (Item 2.2 foundation)

**Files:**
- Create: `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/replay_utils.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_replay_utils.py`

**Step 1: Write test**

```python
import pytest
import asyncio
from unittest.mock import MagicMock
from tldw_Server_API.app.core.Agent_Client_Protocol.consumers.replay_utils import replay_events
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind


@pytest.mark.asyncio
async def test_replay_sends_buffered_events_in_order():
    bus = MagicMock()
    events = [
        AgentEvent(session_id="s1", kind=AgentEventKind.TEXT, payload={"text": str(i)}, metadata={})
        for i in range(5)
    ]
    for i, ev in enumerate(events):
        ev.sequence = i + 1
    bus.snapshot.return_value = events[2:]  # from_sequence=3

    sent = []
    async def emit(ev):
        sent.append(ev)

    await replay_events(bus, from_sequence=3, emit_fn=emit)
    assert len(sent) == 3
    assert [e.sequence for e in sent] == [3, 4, 5]
```

**Step 2: Implement**

```python
# replay_utils.py
from __future__ import annotations
from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
    from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent


async def replay_events(
    bus: SessionEventBus,
    from_sequence: int,
    emit_fn: Callable[[AgentEvent], Awaitable[None]],
) -> int:
    """Replay buffered events from *from_sequence* via *emit_fn*.

    Returns the number of events replayed.
    """
    if from_sequence <= 0:
        return 0
    buffered = bus.snapshot(from_sequence=from_sequence)
    for event in buffered:
        await emit_fn(event)
    return len(buffered)
```

**Step 3: Run test, commit**

```bash
git commit -m "feat(acp): add shared replay_events helper for SSE and WS catch-up"
```

---

### Task 1.4: WebSocket Reconnect with Catch-Up (Item 2.2)

**Files:**
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/ws_broadcaster.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` — accept `last_sequence` query param
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_ws_reconnect.py`

**Step 1: Write test — replay on add_connection with from_sequence**

```python
@pytest.mark.asyncio
async def test_add_connection_with_replay(bus_with_events, ws_broadcaster):
    sent = []
    async def mock_send(data):
        sent.append(data)

    await ws_broadcaster.add_connection("conn1", mock_send, from_sequence=3)
    # Should have replayed events 3, 4, 5 before going live
    assert len(sent) == 3
```

**Step 2: Modify `add_connection()` to accept `from_sequence` and replay**

```python
async def add_connection(
    self,
    conn_id: str,
    send_callback: SendCallback,
    verbosity: str = "full",
    from_sequence: int = 0,
) -> None:
    """Register a WebSocket connection. Replays buffered events if from_sequence > 0."""
    self._connections[conn_id] = _ConnectionInfo(conn_id, send_callback, verbosity)
    if from_sequence > 0 and self._bus is not None:
        async def _emit(ev: AgentEvent) -> None:
            serialized = self._serialize_event(ev, verbosity)
            if serialized is not None:
                await send_callback(serialized)
        await replay_events(self._bus, from_sequence, _emit)
```

**Step 3: Modify WS endpoint to accept `last_sequence` query param**

In the WebSocket route handler, extract `last_sequence` from query params and pass to `add_connection()`.

**Step 4: Run tests, commit**

```bash
git commit -m "feat(acp): add WebSocket reconnect with catch-up replay from last_sequence"
```

---

### Task 1.5: Orchestrator Run Queue Durability (Item 1.2)

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/store.py` — add `enqueue_run`/`dequeue_run` to abstract + 3 backends
- Modify: `tldw_Server_API/app/core/Sandbox/orchestrator.py` — migrate `_queue` to store
- Test: `tldw_Server_API/tests/sandbox/test_orchestrator_durability.py`

**Step 1: Write test — enqueued run survives store recreation**

```python
def test_enqueue_dequeue_persists(sqlite_store):
    sqlite_store.enqueue_run("run-1", "user-1", priority=0)
    sqlite_store.enqueue_run("run-2", "user-1", priority=1)

    # Simulate restart
    store2 = SQLiteStore(sqlite_store._db_path)
    run = store2.dequeue_run("worker-1")
    assert run is not None
    assert run["run_id"] == "run-2"  # higher priority first
```

**Step 2: Add abstract methods to `SandboxStore`**

```python
def enqueue_run(self, run_id: str, user_id: str, priority: int = 0) -> None:
    raise NotImplementedError

def dequeue_run(self, worker_id: str) -> dict | None:
    raise NotImplementedError
```

**Step 3: Implement in all 3 backends** (InMemoryStore uses a list; SQLiteStore/PostgresStore use a `run_queue` table).

**Step 4: Migrate orchestrator `_queue` to use store methods**

**Step 5: Audit all 9 `_sessions` write paths for write-through consistency**

**Step 6: Run sandbox tests, commit**

```bash
git commit -m "feat(sandbox): migrate orchestrator run queue to durable store backends"
```

---

### Task 1.6: Bash Prefix Matching (Item 4.1)

**Files:**
- Modify: `tldw_Server_API/app/services/mcp_hub_policy_resolver.py` — add `tool_tier_overrides` to resolved document
- Modify: `tldw_Server_API/app/services/acp_runtime_policy_service.py` — propagate to snapshot
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/runner_client.py` — extend `_resolve_runtime_permission_outcome()`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_bash_prefix_matching.py`

**Step 1: Write test — fnmatch pattern matching for tool tiers**

```python
@pytest.mark.asyncio
async def test_bash_git_auto_approved(bus):
    snap = MagicMock()
    snap.resolved_policy_document = {
        "denied_tools": [],
        "allowed_tools": [],
        "tool_tier_overrides": {"Bash(git:*)": "auto", "Bash(rm:*)": "individual"},
    }
    snap.approval_summary = {}
    gf = GovernanceFilter(bus=bus, policy_snapshot=snap)
    event = _make_tool_call("Bash(git:status)")
    await gf.process(event)
    assert bus.publish.call_args_list[0][0][0].kind == AgentEventKind.TOOL_CALL  # auto-forwarded
```

**Step 2: Implement** — `_check_snapshot_policy` already handles `tool_tier_overrides` (added in Task 1.1). This task adds the overrides to the MCPHub policy resolver output and snapshot schema.

**Step 3: Run tests, commit**

```bash
git commit -m "feat(acp): add bash prefix matching via tool_tier_overrides in MCPHub policy"
```

---

### Task 1.7: Permission Decision Persistence (Item 4.2)

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py` — add `permission_decisions` table
- Create: `tldw_Server_API/app/core/Agent_Client_Protocol/permission_decision_service.py`
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/governance_filter.py` — check/persist decisions
- Create: `tldw_Server_API/app/api/v1/endpoints/acp_permissions.py` — CRUD endpoints
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_permission_persistence.py`

**Step 1: Write test — remembered decision auto-applies**

```python
@pytest.mark.asyncio
async def test_remembered_allow_skips_prompt(bus, perm_service):
    perm_service.check.return_value = "allow"
    gf = GovernanceFilter(bus=bus, permission_decision_service=perm_service)
    event = _make_tool_call("Bash(npm:install)")
    await gf.process(event)
    # Should auto-approve without holding
    assert bus.publish.call_args_list[0][0][0].kind == AgentEventKind.TOOL_CALL
```

**Step 2: Implement `PermissionDecisionService`** (~60 lines wrapping `ACPSessionsDB`)

**Step 3: Add `permission_decisions` table** (SQL from design doc)

**Step 4: Wire into `GovernanceFilter.process()`** — check at step 5 of hierarchy (after snapshot, before admin policies)

**Step 5: Wire into `on_permission_response()`** — persist when response includes `remember: true`

**Step 6: Add CRUD endpoints in `acp_permissions.py`**

**Step 7: Run tests, commit**

```bash
git commit -m "feat(acp): add permission decision persistence with 'remember' pattern"
```

---

## Phase 2: Streaming & Async

### Task 2.1: SSE Consumer (Item 2.1)

**Files:**
- Create: `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/sse_consumer.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` — add SSE endpoint
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_sse_consumer.py`

**Key implementation:** `SSEConsumer(EventConsumer)` with `asyncio.Queue` and `iter_sse_lines()` generator. Uses `replay_utils.replay_events()` for catch-up. Endpoint returns `StreamingResponse(media_type="text/event-stream")` with 15s heartbeat.

```python
class SSEConsumer(EventConsumer):
    consumer_id: str = "sse"

    def __init__(self, from_sequence: int = 0) -> None:
        self._queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._from_sequence = from_sequence
        self._bus: SessionEventBus | None = None

    async def on_event(self, event: AgentEvent) -> None:
        await self._queue.put(event)

    async def start(self, bus: SessionEventBus) -> None:
        self._bus = bus
        bus.subscribe(self.consumer_id, from_sequence=self._from_sequence)
        # Replay buffered events
        if self._from_sequence > 0:
            await replay_events(bus, self._from_sequence, self._queue.put)

    async def stop(self) -> None:
        if self._bus:
            self._bus.unsubscribe(self.consumer_id)

    async def iter_sse_lines(self) -> AsyncGenerator[str, None]:
        while True:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=15.0)
                yield f"event: {event.kind.value}\ndata: {json.dumps(event.to_dict())}\n\n"
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
```

---

### Task 2.2: Event Sink Unification (Item 2.3)

**Files:**
- Create: `tldw_Server_API/app/core/Sandbox/event_bridge.py`
- Test: `tldw_Server_API/tests/sandbox/test_event_bridge.py`

**Key implementation:** `SandboxEventBridge` subscribes to `RunStreamHub.subscribe()` and translates frames to `AgentEvent` objects published on `SessionEventBus`.

Frame mapping: `stdout/stderr → TERMINAL_OUTPUT`, `file_change → FILE_CHANGE`, `exit → COMPLETION`, `error → ERROR`.

---

### Task 2.3: `acp_run` Scheduler Handler (Item 3.1)

**Files:**
- Create: `tldw_Server_API/app/core/Scheduler/handlers/acp.py`
- Modify: `tldw_Server_API/app/services/workflows_scheduler.py` — add import for registration
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_run_handler.py`

**Key implementation:**

```python
from tldw_Server_API.app.core.Scheduler.base.registry import task

@task(name="acp_run", max_retries=1, timeout=7200, queue="acp")
async def acp_run(payload: dict[str, Any]) -> dict[str, Any]:
    user_id = payload["user_id"]
    prompt = payload["prompt"]
    cwd = payload.get("cwd", ".")
    # ... create session, send prompt, close session, return result
```

Registration in `workflows_scheduler.py`:
```python
from tldw_Server_API.app.core.Scheduler.handlers import acp as _ensure_acp_handlers  # noqa: F401
```

---

### Task 2.4: Async Fire-and-Forget API (Item 3.2)

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` — add 2 endpoints
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_async_api.py`

**Endpoints:**
- `POST /api/v1/acp/sessions/prompt-async` → submit `acp_run` to Scheduler → return `{task_id, poll_url}`
- `GET /api/v1/acp/tasks/{task_id}` → return `{status, result, usage, error}` from Scheduler

---

## Phase 3: Automation

### Task 3.1: ACP Schedule CRUD (Item 3.3)

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py` — add `acp_config_json` column + migration
- Modify: `tldw_Server_API/app/services/workflows_scheduler.py` — guard `_load_all()`, route ACP schedules
- Create endpoints in: `tldw_Server_API/app/api/v1/endpoints/acp_schedules.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_schedules.py`

**Critical guard in `_load_all()`:**
```python
for s in items:
    if not s.enabled:
        continue
    # Guard: skip ACP schedules
    if hasattr(s, 'acp_config_json') and s.acp_config_json:
        self._add_acp_job(s, uid)  # new method for ACP schedules
    else:
        self._add_job(s, uid)  # existing workflow_run path
```

---

### Task 3.2: Run History + Cost Tracking API (Item 3.4)

**Files:**
- Modify: `tldw_Server_API/app/services/admin_acp_sessions_service.py` — add query methods
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` — add 2 endpoints
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_run_history.py`

**Endpoints:**
- `GET /api/v1/acp/runs` — query `SessionRecord` with filters
- `GET /api/v1/acp/runs/aggregate` — sum token usage via `compute_token_cost()`

---

### Task 3.3: Webhook/Event Trigger System (Item 3.5)

**Files:**
- Create: `tldw_Server_API/app/core/Agent_Client_Protocol/triggers.py` (~500 lines)
- Create endpoints in: `tldw_Server_API/app/api/v1/endpoints/acp_triggers.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_webhook_triggers.py`

**Key security implementation:**

```python
import hmac
import hashlib
import time

def verify_github_signature(payload_body: bytes, signature: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), payload_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

def verify_slack_signature(payload_body: bytes, timestamp: str, signature: str, secret: str) -> bool:
    if abs(time.time() - float(timestamp)) > 300:
        return False  # Replay attack prevention
    sig_basestring = f"v0:{timestamp}:{payload_body.decode()}"
    expected = "v0=" + hmac.new(
        secret.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

**Trigger CRUD:** `POST/GET/PUT/DELETE /api/v1/acp/triggers` with encrypted secrets, rate limiting (60 req/min per trigger_id), and `owner_user_id` mapping.

---

## Phase 4: Sandbox & Permissions

### Task 4.1: Git Worktree Runner (Item 1.1)

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/models.py` — add `RuntimeType.worktree`
- Create: `tldw_Server_API/app/core/Sandbox/runners/worktree_runner.py` (~400 lines)
- Modify: `tldw_Server_API/app/core/Sandbox/runners/seatbelt_policy.py` — add worktree profile
- Modify: `tldw_Server_API/app/core/Sandbox/runtime_capabilities.py` — add preflight
- Modify: `tldw_Server_API/app/core/Sandbox/service.py` — route in `start_run_scaffold()` (~line 1337), `cancel_run()` (~line 1747), and `collect_runtime_preflights()`
- Test: `tldw_Server_API/tests/sandbox/test_worktree_runner.py`

**Key security constraints:**
- Repo path allowlist validation (no `/etc`, `~/.ssh`, etc.)
- Linux: require `unshare` — refuse if unavailable
- macOS: delegate to `SeatbeltRunner` with worktree-specific profile
- Restrict env var passthrough (`HOME`, `SSH_AUTH_SOCK` stripped)

**Core lifecycle:**
```python
class WorktreeRunner:
    async def create_session(self, repo_path: str, branch: str = "HEAD") -> str:
        self._validate_repo_path(repo_path)
        worktree_path = tempfile.mkdtemp(prefix="tldw_wt_")
        subprocess.check_call(["git", "worktree", "add", worktree_path, "--detach"], cwd=repo_path)
        return worktree_path

    async def run(self, worktree_path: str, command: list[str], **kwargs) -> RunResult:
        if sys.platform == "darwin":
            return await self._run_with_seatbelt(worktree_path, command, **kwargs)
        return await self._run_with_unshare(worktree_path, command, **kwargs)

    async def destroy_session(self, worktree_path: str) -> None:
        subprocess.check_call(["git", "worktree", "remove", "--force", worktree_path])
```

---

### Task 4.2: Permission Policy Templates (Item 4.4)

**Files:**
- Modify: `tldw_Server_API/app/services/acp_runtime_policy_service.py` — add template loading
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/config.py` — add template definitions
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_policy_templates.py`

**4 presets:** `read-only`, `developer`, `admin`, `lockdown` — loaded as base layers in `build_snapshot()`, merged with user customizations.

---

### Task 4.3: MCP Tool @Mentions (Item 4.5)

**Files:**
- Create: `tldw_Server_API/app/core/Agent_Client_Protocol/prompt_utils.py` (~180 lines)
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` — wire at line 597
- Modify: `tldw_Server_API/app/core/Workflows/adapters/integration/acp.py` — wire in `_render_prompt()`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_prompt_utils.py`

**Key implementation:**
```python
import re
_AT_MENTION_RE = re.compile(r"@([\w.-]+)")

async def preprocess_mentions(
    messages: list[dict[str, Any]],
    tool_registry: Any,
    cache: dict[str, bool] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse @tool_name in messages, resolve, return (messages, tool_hints)."""
    tool_hints = []
    if cache is None:
        cache = {}
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            for match in _AT_MENTION_RE.finditer(content):
                name = match.group(1)
                if name not in cache:
                    cache[name] = await tool_registry.tool_exists(name)
                if cache[name]:
                    tool_hints.append(name)
    return messages, list(set(tool_hints))
```

---

## Phase 5: Advanced

### Task 5.1: Docker Warm Container Pool (Item 1.3)

**Files:**
- Create: `tldw_Server_API/app/core/Sandbox/pool.py` (~250 lines)
- Modify: `tldw_Server_API/app/core/Sandbox/runners/docker_runner.py` — add `docker exec` path
- Test: `tldw_Server_API/tests/sandbox/test_warm_pool.py`

**Key design:** Pre-create containers with `sleep infinity` entrypoint. On claim, use `docker exec` instead of `docker start`. File injection via `docker cp` still works.

```python
class DockerWarmPool:
    def __init__(self, pool_size: int = 3, images: list[str] | None = None):
        self._pool: dict[str, list[str]] = {}  # image -> [container_ids]
        self._pool_size = pool_size
        self._images = images or ["python:3.12-slim"]
        self._lock = threading.Lock()

    def claim(self, image: str) -> str | None:
        with self._lock:
            pool = self._pool.get(image, [])
            return pool.pop(0) if pool else None

    def _replenish(self, image: str) -> None:
        cmd = ["docker", "create", "--entrypoint", "/bin/sh", image, "-c", "sleep infinity"]
        cid = subprocess.check_output(cmd, text=True).strip()
        subprocess.check_call(["docker", "start", cid])
        with self._lock:
            self._pool.setdefault(image, []).append(cid)
```

Register with `health_monitor.py` and add shutdown hook for cleanup.

---

### Task 5.2: Checkpoint-Based Rollback (Item 4.3)

**Files:**
- Create: `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/checkpoint_consumer.py` (~200 lines)
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` — add rollback endpoint
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_checkpoint_rollback.py`

**Key implementation:** `CheckpointConsumer(EventConsumer)` auto-snapshots via `SandboxService.create_snapshot(session_id)` before `FILE_CHANGE` events. Handles active-run guard (skip + log warning). Rollback endpoint calls `SandboxService.restore_snapshot()`.

```python
class CheckpointConsumer(EventConsumer):
    consumer_id = "checkpoint"

    def __init__(self, sandbox_service: SandboxService) -> None:
        self._service = sandbox_service
        self._checkpoints: dict[int, str] = {}  # sequence -> snapshot_id

    async def on_event(self, event: AgentEvent) -> None:
        if event.kind != AgentEventKind.FILE_CHANGE:
            return
        try:
            result = self._service.create_snapshot(event.session_id)
            self._checkpoints[event.sequence] = result["snapshot_id"]
        except Exception:
            logger.warning("Checkpoint skipped for seq {}: active run or error", event.sequence)
```

---

## Cross-Cutting: Apply Throughout All Phases

After each major task, apply:

1. **Observability:** Add `increment_counter()` / `observe_histogram()` for key operations
2. **Config validation:** Add new env vars to `validate_acp_config()` pattern
3. **Health checks:** Register background components with `health_monitor.py`
4. **Graceful shutdown:** Add shutdown hooks for background threads (pool, scheduler)

---

## Verification Checklist

After all phases:

- [ ] `python -m pytest tldw_Server_API/tests/ -x -q` — full test suite passes
- [ ] Server starts cleanly: `python -m uvicorn tldw_Server_API.app.main:app`
- [ ] SSE endpoint streams typed events with sequence numbers
- [ ] WS reconnect replays missed events
- [ ] Async API returns task_id and polls for results
- [ ] Scheduled ACP run fires on cron trigger
- [ ] Webhook trigger with HMAC enqueues `acp_run`
- [ ] `GovernanceFilter` checks MCPHub snapshot → persisted decisions → admin policies → heuristics
- [ ] "Remember" button persists and auto-applies decisions
- [ ] Git worktree runner creates isolated worktree and cleans up
- [ ] Warm pool claims container in <1s vs cold start
- [ ] Rollback restores sandbox state to checkpoint
