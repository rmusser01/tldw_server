import importlib.machinery
import json
import sys
import types
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPResponseError
from tldw_Server_API.app.services.acp_runtime_policy_service import ACPRuntimePolicySnapshot

pytestmark = pytest.mark.unit


# Stub heavyweight audio deps before app import in shared fixtures.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.nn = types.SimpleNamespace(Module=object)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf


class StubRunnerClient:
    def __init__(self) -> None:
        self.agent_capabilities = {"promptCapabilities": {"image": False}}
        self.cancelled = []
        self.closed = []
        self.prompt_calls = []
        self.create_session_calls = []
        self.denied_sessions = set()
        self._updates = {
            "session-123": [
                {"sessionId": "session-123", "event": "message", "content": "hello"}
            ]
        }

    async def create_session(
        self,
        cwd: str,
        mcp_servers=None,
        agent_type: str | None = None,
        user_id: int | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
    ) -> str:
        self.create_session_calls.append(
            {
                "cwd": cwd,
                "mcp_servers": mcp_servers,
                "agent_type": agent_type,
                "user_id": user_id,
                "persona_id": persona_id,
                "workspace_id": workspace_id,
                "workspace_group_id": workspace_group_id,
                "scope_snapshot_id": scope_snapshot_id,
            }
        )
        return "session-123"

    async def verify_session_access(self, session_id: str, user_id: int) -> bool:
        return session_id not in self.denied_sessions

    async def prompt(self, session_id: str, prompt):
        self.prompt_calls.append((session_id, prompt))
        return {"stopReason": "end", "detail": "ok"}

    async def cancel(self, session_id: str) -> None:
        self.cancelled.append(session_id)

    async def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)

    def pop_updates(self, session_id: str, limit: int = 100):
        updates = list(self._updates.get(session_id, []))
        return updates[:limit]


@pytest.fixture()
def stub_runner_client(monkeypatch, tmp_path):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    stub = StubRunnerClient()

    async def _get_runner_client():
        return stub

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)

    # Provide a fresh session store backed by a temp DB for test isolation
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
    from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore

    _test_db = ACPSessionsDB(db_path=str(tmp_path / "test_acp_sessions.db"))
    _test_store = ACPSessionStore(db=_test_db)

    async def _get_test_store():
        return _test_store

    monkeypatch.setattr(acp_endpoints, "get_acp_session_store", _get_test_store)

    return stub


def test_acp_session_new_success(client_user_only, stub_runner_client, tmp_path, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    captured: dict[str, object] = {}

    class _RuntimePolicyService:
        async def build_snapshot(self, *, session_record, user_id: int, **kwargs):
            del kwargs
            captured["session_id"] = session_record.session_id
            captured["user_id"] = user_id
            return ACPRuntimePolicySnapshot(
                session_id=session_record.session_id,
                user_id=int(user_id),
                policy_snapshot_version="resolved-v1",
                policy_snapshot_fingerprint="snapshot-created-at-session-start",
                policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
                policy_summary={"allowed_tool_count": 1, "approval_mode": "allow"},
                policy_provenance_summary={"source_kinds": ["profile"]},
                resolved_policy_document={"allowed_tools": ["web.search"]},
                approval_summary={"mode": "allow"},
                context_summary={"persona_id": getattr(session_record, "persona_id", None)},
                execution_config={},
            )

        async def persist_snapshot(self, *, session_store, snapshot):
            return await session_store.update_policy_snapshot_state(
                snapshot.session_id,
                policy_snapshot_version=snapshot.policy_snapshot_version,
                policy_snapshot_fingerprint=snapshot.policy_snapshot_fingerprint,
                policy_snapshot_refreshed_at=snapshot.policy_snapshot_refreshed_at,
                policy_summary=snapshot.policy_summary,
                policy_provenance_summary=snapshot.policy_provenance_summary,
                policy_refresh_error=snapshot.refresh_error,
            )

    monkeypatch.setattr(
        acp_endpoints,
        "get_acp_runtime_policy_service",
        lambda: _RuntimePolicyService(),
        raising=False,
    )

    resp = client_user_only.post(
        "/api/v1/acp/sessions/new",
        json={"cwd": str(tmp_path)},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["session_id"] == "session-123"
    assert payload["agent_capabilities"] == {"promptCapabilities": {"image": False}}
    assert payload["policy_snapshot_version"] == "resolved-v1"
    assert payload["policy_snapshot_fingerprint"] == "snapshot-created-at-session-start"
    assert payload["policy_snapshot_refreshed_at"] == "2026-03-14T12:00:00+00:00"
    assert payload["policy_summary"] == {"allowed_tool_count": 1, "approval_mode": "allow"}
    assert payload["policy_provenance_summary"] == {"source_kinds": ["profile"]}
    assert captured == {"session_id": "session-123", "user_id": 1}
    assert stub_runner_client.create_session_calls
    assert isinstance(stub_runner_client.create_session_calls[0]["user_id"], int)


def test_acp_session_new_forwards_tenancy_fields(client_user_only, stub_runner_client, tmp_path):
    resp = client_user_only.post(
        "/api/v1/acp/sessions/new",
        json={
            "cwd": str(tmp_path),
            "agent_type": "codex",
            "persona_id": "persona-abc",
            "workspace_id": "ws-1",
            "workspace_group_id": "wsg-2",
            "scope_snapshot_id": "scope-3",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["persona_id"] == "persona-abc"
    assert payload["workspace_id"] == "ws-1"
    assert payload["workspace_group_id"] == "wsg-2"
    assert payload["scope_snapshot_id"] == "scope-3"
    call = stub_runner_client.create_session_calls[-1]
    assert call["agent_type"] == "codex"
    assert call["persona_id"] == "persona-abc"
    assert call["workspace_id"] == "ws-1"
    assert call["workspace_group_id"] == "wsg-2"
    assert call["scope_snapshot_id"] == "scope-3"
    assert isinstance(call["user_id"], int) and call["user_id"] > 0


def test_acp_session_new_records_sanitized_audit_event(
    client_user_only,
    stub_runner_client,
    tmp_path,
):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    with acp_endpoints._ACP_AUDIT_LOCK:
        acp_endpoints._ACP_AUDIT_EVENTS.clear()

    response = client_user_only.post(
        "/api/v1/acp/sessions/new",
        json={
            "cwd": str(tmp_path / "secret-project"),
            "agent_type": "codex",
            "mcp_servers": [
                {
                    "name": "private-mcp",
                    "type": "stdio",
                    "command": "/private/bin/acp-mcp",
                    "args": ["--token", "sk-should-not-leak"],
                    "env": {"OPENAI_API_KEY": "sk-should-not-leak"},
                }
            ],
        },
    )

    assert response.status_code == 200
    with acp_endpoints._ACP_AUDIT_LOCK:
        events = list(acp_endpoints._ACP_AUDIT_EVENTS)
    event = next(item for item in events if item["action"] == "session_created")
    assert event["session_id"] == response.json()["session_id"]
    assert event["metadata"]["agent_type"] == "codex"
    assert event["metadata"]["mcp_server_count"] == 1
    serialized = json.dumps(event["metadata"])
    assert str(tmp_path) not in serialized
    assert "sk-should-not-leak" not in serialized
    assert "OPENAI_API_KEY" not in serialized
    assert "/private/bin/acp-mcp" not in serialized


def test_acp_agent_registration_records_sanitized_audit_event(
    client_user_only,
    monkeypatch,
):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    import tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry as registry_mod

    class _Registry:
        def register_agent(self, **kwargs):
            return types.SimpleNamespace(type=kwargs["type"], name=kwargs["name"])

    async def _admin_user():
        return types.SimpleNamespace(id=1, is_admin=True)

    monkeypatch.setattr(registry_mod, "get_agent_registry", lambda: _Registry())
    client_user_only.app.dependency_overrides[acp_endpoints.get_request_user] = _admin_user
    with acp_endpoints._ACP_AUDIT_LOCK:
        acp_endpoints._ACP_AUDIT_EVENTS.clear()
    try:
        response = client_user_only.post(
            "/api/v1/acp/agents/register",
            json={
                "agent_type": "audit_agent",
                "name": "Audit Agent",
                "command": "/private/bin/audit-agent",
                "args": ["--api-key", "sk-should-not-leak"],
                "env": {"ANTHROPIC_API_KEY": "sk-should-not-leak"},
                "requires_api_key": "ANTHROPIC_API_KEY",
            },
        )
    finally:
        client_user_only.app.dependency_overrides.pop(acp_endpoints.get_request_user, None)

    assert response.status_code == 200
    with acp_endpoints._ACP_AUDIT_LOCK:
        events = list(acp_endpoints._ACP_AUDIT_EVENTS)
    event = next(item for item in events if item["action"] == "agent_registered")
    assert event["session_id"] == "agent:audit_agent"
    assert event["metadata"]["agent_type"] == "audit_agent"
    assert event["metadata"]["requires_api_key"] is True
    serialized = json.dumps(event["metadata"])
    assert "sk-should-not-leak" not in serialized
    assert "ANTHROPIC_API_KEY" not in serialized
    assert "/private/bin/audit-agent" not in serialized


def test_acp_agent_registration_forwards_entrypoint_kwargs(
    client_user_only,
    monkeypatch,
):
    """Dynamic registration forwards ACP entrypoint metadata into the registry."""
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    import tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry as registry_mod

    captured: dict[str, object] = {}

    class _Registry:
        def register_agent(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(type=kwargs["type"], name=kwargs["name"])

    async def _admin_user():
        return types.SimpleNamespace(id=1, is_admin=True)

    monkeypatch.setattr(registry_mod, "get_agent_registry", lambda: _Registry())
    client_user_only.app.dependency_overrides[acp_endpoints.get_request_user] = _admin_user
    try:
        response = client_user_only.post(
            "/api/v1/acp/agents/register",
            json={
                "agent_type": "adapter_agent",
                "name": "Adapter Agent",
                "entrypoint_strategy": "adapter_acp",
                "acp_command": "adapter-agent-acp",
                "acp_args": ["--stdio"],
                "adapter_source": "https://example.test/adapter",
                "adapter_docs_url": "https://example.test/adapter/docs",
                "certification_blocker": "adapter_missing",
            },
        )
    finally:
        client_user_only.app.dependency_overrides.pop(acp_endpoints.get_request_user, None)

    assert response.status_code == 200
    assert captured["entrypoint_strategy"] == "adapter_acp"
    assert captured["acp_command"] == "adapter-agent-acp"
    assert captured["acp_args"] == ["--stdio"]
    assert captured["adapter_source"] == "https://example.test/adapter"
    assert captured["adapter_docs_url"] == "https://example.test/adapter/docs"
    assert captured["certification_blocker"] == "adapter_missing"


def test_acp_list_audit_events_reads_persisted_rows(tmp_path, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    import tldw_Server_API.app.core.DB_Management.ACP_Audit_DB as audit_db_mod

    audit_db = audit_db_mod.ACPAuditDB(db_path=str(tmp_path / "acp_audit.db"))
    monkeypatch.setattr(audit_db_mod, "_audit_db", audit_db)
    with acp_endpoints._ACP_AUDIT_LOCK:
        acp_endpoints._ACP_AUDIT_EVENTS.clear()
    try:
        audit_db.record_event(
            action="session_created",
            user_id=1,
            session_id="persisted-session",
            metadata={"agent_type": "codex"},
        )
        audit_db.flush()

        events = acp_endpoints._acp_list_audit_events(session_id="persisted-session")
    finally:
        audit_db.close()

    assert len(events) == 1
    assert events[0]["action"] == "session_created"
    assert events[0]["metadata"]["agent_type"] == "codex"


def test_acp_list_audit_events_filters_in_memory_events_by_retention(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    import tldw_Server_API.app.core.DB_Management.ACP_Audit_DB as audit_db_mod

    class _EmptyAuditDB:
        _retention_days = 1

        def flush(self):
            return 0

        def query_events(self, **_kwargs):
            return []

        def get_hot_cache(self, **_kwargs):
            return []

    monkeypatch.setattr(audit_db_mod, "get_acp_audit_db", lambda: _EmptyAuditDB())
    with acp_endpoints._ACP_AUDIT_LOCK:
        acp_endpoints._ACP_AUDIT_EVENTS.clear()
        acp_endpoints._ACP_AUDIT_EVENTS.append(
            {
                "timestamp": "2000-01-01T00:00:00+00:00",
                "action": "old",
                "user_id": 1,
                "session_id": "retention-session",
                "metadata": {},
            }
        )
        acp_endpoints._ACP_AUDIT_EVENTS.append(
            {
                "timestamp": "2999-01-01T00:00:00+00:00",
                "action": "fresh",
                "user_id": 1,
                "session_id": "retention-session",
                "metadata": {},
            }
        )
    try:
        events = acp_endpoints._acp_list_audit_events(session_id="retention-session")
    finally:
        with acp_endpoints._ACP_AUDIT_LOCK:
            acp_endpoints._ACP_AUDIT_EVENTS.clear()

    assert [event["action"] for event in events] == ["fresh"]


def test_agent_audit_scope_is_admin_readable_without_runner_session(
    client_user_only,
    monkeypatch,
):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    import tldw_Server_API.app.core.DB_Management.ACP_Audit_DB as audit_db_mod

    class _EmptyAuditDB:
        def flush(self):
            return 0

        def query_events(self, **_kwargs):
            return []

        def get_hot_cache(self, **_kwargs):
            return []

    async def _admin_user():
        return types.SimpleNamespace(id=1, is_admin=True)

    monkeypatch.setattr(audit_db_mod, "get_acp_audit_db", lambda: _EmptyAuditDB())
    client_user_only.app.dependency_overrides[acp_endpoints.get_request_user] = _admin_user
    with acp_endpoints._ACP_AUDIT_LOCK:
        acp_endpoints._ACP_AUDIT_EVENTS.clear()
        acp_endpoints._ACP_AUDIT_EVENTS.append(
            {
                "timestamp": "2026-05-10T00:00:00+00:00",
                "action": "agent_registered",
                "user_id": 1,
                "session_id": "agent:audit_agent",
                "metadata": {"agent_type": "audit_agent"},
            }
        )
    try:
        response = client_user_only.get("/api/v1/acp/sessions/agent:audit_agent/audit")
    finally:
        client_user_only.app.dependency_overrides.pop(acp_endpoints.get_request_user, None)
        with acp_endpoints._ACP_AUDIT_LOCK:
            acp_endpoints._ACP_AUDIT_EVENTS.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "agent:audit_agent"
    assert payload["events"][0]["action"] == "agent_registered"


def test_acp_session_prompt_success(client_user_only, stub_runner_client):
    resp = client_user_only.post(
        "/api/v1/acp/sessions/prompt",
        json={"session_id": "session-123", "prompt": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["stop_reason"] == "end"
    assert payload["raw_result"]["detail"] == "ok"
    assert stub_runner_client.prompt_calls


def test_acp_session_cancel_and_close(client_user_only, stub_runner_client):
    cancel = client_user_only.post(
        "/api/v1/acp/sessions/cancel",
        json={"session_id": "session-123"},
    )
    assert cancel.status_code == 200
    assert stub_runner_client.cancelled == ["session-123"]

    close = client_user_only.post(
        "/api/v1/acp/sessions/close",
        json={"session_id": "session-123"},
    )
    assert close.status_code == 200
    assert stub_runner_client.closed == ["session-123"]


def test_acp_session_prompt_sanitizes_runtime_errors(client_user_only, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    leaked = "sk-prompt-secret /tmp/acp-prompt-workspace"
    error_logs = []

    class ErrorRunnerClient(StubRunnerClient):
        async def prompt(self, session_id: str, prompt):
            _ = (session_id, prompt)
            raise ACPResponseError(leaked)

    async def _get_runner_client():
        return ErrorRunnerClient()

    def _capture_error(*args, **kwargs):
        error_logs.append((args, kwargs))

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)
    monkeypatch.setattr(acp_endpoints.logger, "error", _capture_error)

    resp = client_user_only.post(
        "/api/v1/acp/sessions/prompt",
        json={"session_id": "session-123", "prompt": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 502
    assert resp.json()["detail"] == "ACP prompt failed"
    assert error_logs
    assert leaked not in str(error_logs)


def test_acp_session_cancel_sanitizes_runtime_errors(client_user_only, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    leaked = "cancel token sk-cancel-secret /Users/example/private"
    error_logs = []

    class ErrorRunnerClient(StubRunnerClient):
        async def cancel(self, session_id: str) -> None:
            _ = session_id
            raise ACPResponseError(leaked)

    async def _get_runner_client():
        return ErrorRunnerClient()

    def _capture_error(*args, **kwargs):
        error_logs.append((args, kwargs))

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)
    monkeypatch.setattr(acp_endpoints.logger, "error", _capture_error)

    resp = client_user_only.post(
        "/api/v1/acp/sessions/cancel",
        json={"session_id": "session-123"},
    )

    assert resp.status_code == 502
    assert resp.json()["detail"] == "ACP session cancel failed"
    assert error_logs
    assert leaked not in str(error_logs)


def test_acp_session_close_sanitizes_runtime_errors(client_user_only, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    leaked = "close token sk-close-secret /var/tmp/acp-close-workspace"
    error_logs = []

    class ErrorRunnerClient(StubRunnerClient):
        async def close_session(self, session_id: str) -> None:
            _ = session_id
            raise ACPResponseError(leaked)

    async def _get_runner_client():
        return ErrorRunnerClient()

    def _capture_error(*args, **kwargs):
        error_logs.append((args, kwargs))

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)
    monkeypatch.setattr(acp_endpoints.logger, "error", _capture_error)

    resp = client_user_only.post(
        "/api/v1/acp/sessions/close",
        json={"session_id": "session-123"},
    )

    assert resp.status_code == 502
    assert resp.json()["detail"] == "ACP session close failed"
    assert error_logs
    assert leaked not in str(error_logs)


def test_acp_session_teardown_sanitizes_runtime_errors(client_user_only, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    class ErrorRunnerClient(StubRunnerClient):
        async def close_session(self, session_id: str) -> None:
            _ = session_id
            raise ACPResponseError("teardown backend exploded at /private/acp.sock")

    async def _get_runner_client():
        return ErrorRunnerClient()

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)

    resp = client_user_only.post("/api/v1/acp/sessions/session-teardown-sanitize/teardown")

    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert detail["status"] == "teardown_failed"
    assert detail["error"] == "ACP session teardown failed"


def test_acp_session_reconcile_sanitizes_runtime_errors(client_user_only, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    class ErrorRunnerClient(StubRunnerClient):
        async def close_session(self, session_id: str) -> None:
            _ = session_id
            raise ACPResponseError("reconcile backend exploded at /private/acp.sock")

    async def _get_runner_client():
        return ErrorRunnerClient()

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)

    resp = client_user_only.post("/api/v1/acp/sessions/session-reconcile-sanitize/reconcile")

    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert detail["status"] == "reconcile_failed"
    assert detail["error"] == "ACP session reconcile failed"


def test_acp_session_updates(client_user_only, stub_runner_client):
    resp = client_user_only.get("/api/v1/acp/sessions/session-123/updates")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["updates"] == [
        {"sessionId": "session-123", "event": "message", "content": "hello"}
    ]


def test_acp_session_new_error(client_user_only, monkeypatch, tmp_path):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    class ErrorRunnerClient(StubRunnerClient):
        async def create_session(
            self,
            cwd: str,
            mcp_servers=None,
            agent_type: str | None = None,
            user_id: int | None = None,
            persona_id: str | None = None,
            workspace_id: str | None = None,
            workspace_group_id: str | None = None,
            scope_snapshot_id: str | None = None,
        ) -> str:
            raise ACPResponseError("boom")

    async def _get_runner_client():
        return ErrorRunnerClient()

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)

    resp = client_user_only.post(
        "/api/v1/acp/sessions/new",
        json={"cwd": str(tmp_path)},
    )
    assert resp.status_code == 502
    assert resp.json()["detail"] == "Failed to create ACP session"


def test_acp_session_prompt_denied_for_unowned_session(client_user_only, stub_runner_client):
    stub_runner_client.denied_sessions.add("session-999")
    resp = client_user_only.post(
        "/api/v1/acp/sessions/prompt",
        json={"session_id": "session-999", "prompt": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "session_not_found"


def test_acp_session_updates_denied_for_unowned_session(client_user_only, stub_runner_client):
    stub_runner_client.denied_sessions.add("session-999")
    resp = client_user_only.get("/api/v1/acp/sessions/session-999/updates")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "session_not_found"


def test_acp_session_fork_creates_runtime_backed_session_and_bootstraps_first_prompt(client_user_only, monkeypatch, tmp_path):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
    from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore

    class ForkRunner:
        def __init__(self) -> None:
            self.agent_capabilities = {"promptCapabilities": {"image": False}}
            self.owners = {"source-session": 1}
            self.create_session_calls = []
            self.prompt_calls = []
            self._counter = 0

        async def create_session(
            self,
            cwd: str,
            mcp_servers=None,
            agent_type: str | None = None,
            user_id: int | None = None,
            persona_id: str | None = None,
            workspace_id: str | None = None,
            workspace_group_id: str | None = None,
            scope_snapshot_id: str | None = None,
        ) -> str:
            self._counter += 1
            session_id = f"fork-runtime-{self._counter}"
            self.owners[session_id] = int(user_id or 0)
            self.create_session_calls.append(
                {
                    "cwd": cwd,
                    "mcp_servers": mcp_servers,
                    "agent_type": agent_type,
                    "user_id": user_id,
                    "persona_id": persona_id,
                    "workspace_id": workspace_id,
                    "workspace_group_id": workspace_group_id,
                    "scope_snapshot_id": scope_snapshot_id,
                }
            )
            return session_id

        async def verify_session_access(self, session_id: str, user_id: int) -> bool:
            return self.owners.get(session_id) == user_id

        async def prompt(self, session_id: str, prompt):
            self.prompt_calls.append((session_id, prompt))
            return {"content": "Bootstrapped reply"}

        def pop_updates(self, session_id: str, limit: int = 100):
            return []

        def has_websocket_connections(self, session_id: str) -> bool:
            return False

    runner = ForkRunner()
    _db = ACPSessionsDB(db_path=str(tmp_path / "fork_bootstrap_test.db"))
    store = ACPSessionStore(db=_db)

    async def _seed() -> None:
        await store.register_session(
            session_id="source-session",
            user_id=1,
            agent_type="codex",
            name="Source Session",
            cwd="/tmp/project",
            mcp_servers=[{"name": "filesystem", "command": "fs-server"}],
            persona_id="persona-1",
            workspace_id="workspace-1",
            workspace_group_id="group-1",
            scope_snapshot_id="scope-1",
        )
        await store.record_prompt(
            "source-session",
            [{"role": "user", "content": "Seed question"}],
            {"content": "Seed answer"},
        )

    import asyncio
    asyncio.run(_seed())

    async def _get_runner_client():
        return runner

    async def _get_store():
        return store

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)
    monkeypatch.setattr(acp_endpoints, "get_acp_session_store", _get_store)

    fork_resp = client_user_only.post(
        "/api/v1/acp/sessions/source-session/fork",
        json={"message_index": 1, "name": "Forked Session"},
    )
    assert fork_resp.status_code == 200
    fork_payload = fork_resp.json()
    fork_session_id = fork_payload["session_id"]
    assert runner.create_session_calls[-1]["mcp_servers"] == [{"name": "filesystem", "command": "fs-server"}]
    assert fork_payload["forked_from"] == "source-session"

    prompt_resp = client_user_only.post(
        "/api/v1/acp/sessions/prompt",
        json={"session_id": fork_session_id, "prompt": [{"role": "user", "content": "Follow-up"}]},
    )
    assert prompt_resp.status_code == 200
    assert runner.prompt_calls[-1][0] == fork_session_id
    assert runner.prompt_calls[-1][1] == [
        {"role": "user", "content": "Seed question"},
        {"role": "assistant", "content": "Seed answer"},
        {"role": "user", "content": "Follow-up"},
    ]

    import asyncio as _asyncio
    fork_record = _asyncio.run(store.get_session(fork_session_id))
    assert fork_record is not None
    assert fork_record.needs_bootstrap is False


def test_acp_session_fork_rejects_non_bootstrappable_source(client_user_only, monkeypatch, tmp_path):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
    from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore

    class ForkRunner:
        def __init__(self) -> None:
            self.agent_capabilities = {"promptCapabilities": {"image": False}}
            self.owners = {"source-session": 1}
            self.create_session_calls = []

        async def create_session(self, *args, **kwargs):
            self.create_session_calls.append((args, kwargs))
            return f"fork-{uuid4()}"

        async def verify_session_access(self, session_id: str, user_id: int) -> bool:
            return self.owners.get(session_id) == user_id

        def has_websocket_connections(self, session_id: str) -> bool:
            return False

    runner = ForkRunner()
    _db = ACPSessionsDB(db_path=str(tmp_path / "fork_test.db"))
    store = ACPSessionStore(db=_db)

    async def _seed() -> None:
        await store.register_session(
            session_id="source-session",
            user_id=1,
            agent_type="codex",
            name="Opaque Source",
            cwd="/tmp/project",
            mcp_servers=[{"name": "filesystem"}],
        )
        await store.record_prompt(
            "source-session",
            [{"role": "user", "content": "Seed question"}],
            {"detail": {"opaque": True}},
        )

    import asyncio
    asyncio.run(_seed())

    async def _get_runner_client():
        return runner

    async def _get_store():
        return store

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)
    monkeypatch.setattr(acp_endpoints, "get_acp_session_store", _get_store)

    fork_resp = client_user_only.post(
        "/api/v1/acp/sessions/source-session/fork",
        json={"message_index": 1, "name": "Forked Session"},
    )
    assert fork_resp.status_code == 409
    assert fork_resp.json()["detail"] == "fork_not_resumable"
    assert runner.create_session_calls == []


def test_acp_session_fork_sanitizes_create_session_errors(client_user_only, monkeypatch, tmp_path):
    import asyncio

    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
    from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore

    class ErrorForkRunner:
        agent_capabilities = {"promptCapabilities": {"image": False}}

        async def create_session(self, *args, **kwargs):
            _ = (args, kwargs)
            raise ACPResponseError("fork backend exploded")

        async def verify_session_access(self, session_id: str, user_id: int) -> bool:
            return session_id == "source-session" and user_id == 1

        def has_websocket_connections(self, session_id: str) -> bool:
            return False

    runner = ErrorForkRunner()
    _db = ACPSessionsDB(db_path=str(tmp_path / "fork_error_test.db"))
    store = ACPSessionStore(db=_db)

    async def _seed() -> None:
        await store.register_session(
            session_id="source-session",
            user_id=1,
            agent_type="codex",
            name="Source Session",
            cwd="/tmp/project",
            mcp_servers=[{"name": "filesystem"}],
        )
        await store.record_prompt(
            "source-session",
            [{"role": "user", "content": "Seed question"}],
            {"content": "Seed answer"},
        )

    asyncio.run(_seed())

    async def _get_runner_client():
        return runner

    async def _get_store():
        return store

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)
    monkeypatch.setattr(acp_endpoints, "get_acp_session_store", _get_store)

    fork_resp = client_user_only.post(
        "/api/v1/acp/sessions/source-session/fork",
        json={"message_index": 1, "name": "Forked Session"},
    )

    assert fork_resp.status_code == 502
    assert fork_resp.json()["detail"] == "Failed to create forked ACP session"
