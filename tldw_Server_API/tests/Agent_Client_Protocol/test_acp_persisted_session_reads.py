"""Stored ACP reads retain exact owner authority without a live runtime."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol as endpoints
from tldw_Server_API.app.core.Agent_Client_Protocol import config as acp_config
from tldw_Server_API.app.core.Agent_Client_Protocol import runner_client
from tldw_Server_API.app.core.Agent_Client_Protocol.config import ACPRunnerConfig, ACPSandboxConfig
from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPRunnerClient
from tldw_Server_API.app.core.Agent_Client_Protocol.sandbox_runner_client import ACPSandboxRunnerManager
from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import _mint
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as _single_user_cookie_client_fixture,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]
single_user_cookie_client = _single_user_cookie_client_fixture
READS = ("detail", "usage", "events", "artifacts", "diagnostics", "audit", "reconciliation")
OWNED = "persisted-owned-session"
FOREIGN = "persisted-foreign-session"


@pytest.fixture(autouse=True)
def private_storage(tmp_path, monkeypatch):
    """Use supported private database and system-log locations."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(tmp_path / "monitoring.db"))
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(tmp_path / "system.log"))
    monkeypatch.setenv("ACP_AUDIT_DB_PATH", str(tmp_path / "audit.db"))


@pytest.fixture
def persisted_sessions(single_user_cookie_client, tmp_path, monkeypatch):
    """Reopen the real store; the actual current runner has no session owners."""
    client, api_key = single_user_cookie_client
    monkeypatch.setenv("ACP_AUDIT_DB_PATH", str(tmp_path / "audit.db"))
    path = str(tmp_path / "sessions.db")

    async def prepare():
        original = ACPSessionStore(ACPSessionsDB(path))
        try:
            await original.register_session(OWNED, user_id=1, agent_type="opencode", name="Saved session", cwd="/tmp/owned-project")  # nosec B108 - stored fixture metadata, no filesystem access
            await original.register_session(FOREIGN, user_id=2, agent_type="opencode", name="Foreign session", cwd="/tmp/foreign-project")  # nosec B108 - stored fixture metadata, no filesystem access
            await original.record_prompt(OWNED, [{"role": "user", "content": "Original saved question"}],
                                        {"content": "Original saved answer", "usage": {"prompt_tokens": 3, "completion_tokens": 5}})
        finally:
            original.get_db().close()
        return ACPSessionStore(ACPSessionsDB(path))

    store = client.portal.call(prepare)
    runtime = ACPRunnerClient(ACPRunnerConfig(command="unused-no-runtime"))
    calls = []

    async def get_store():
        return store

    async def get_runtime():
        calls.append("runtime")
        return runtime

    monkeypatch.setattr(endpoints, "get_acp_session_store", get_store)
    monkeypatch.setattr(endpoints, "get_runner_client", get_runtime)
    monkeypatch.setattr(runner_client, "_runner_client", runtime)
    monkeypatch.setattr(runner_client, "_sandbox_client", None)
    if not any(route.path == "/api/v1/acp/sessions/{session_id}/detail" for route in iter_served_routes(client.app.routes)):
        client.app.include_router(endpoints.router, prefix="/api/v1")
    assert _mint(client, api_key).status_code == 200  # nosec B101 - normal cookie setup
    assert not client.portal.call(runtime.verify_session_access, OWNED, 1)  # nosec B101 - fresh runtime absence
    assert not runtime.is_running  # nosec B101 - no runner launched
    try:
        yield client, store, runtime, calls
    finally:
        client.portal.call(store.get_db().close)


def test_reopened_owner_can_read_every_stored_metadata_family_without_starting_runner(persisted_sessions):
    client, store, _runtime, calls = persisted_sessions
    before = client.portal.call(store.get_session, OWNED)
    responses = {name: client.get(f"/api/v1/acp/sessions/{OWNED}/{name}") for name in READS}
    assert {name: response.status_code for name, response in responses.items()} == dict.fromkeys(READS, 200)  # nosec B101
    detail = responses["detail"].json()
    assert [item["content"] for item in detail["messages"]] == ["Original saved question", "Original saved answer"]  # nosec B101
    assert detail["status"] == "active" and detail["runtime_available"] is False and detail["has_websocket"] is False  # nosec B101
    assert responses["usage"].json()["usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}  # nosec B101
    assert responses["events"].json()["total"] == 2  # nosec B101
    assert responses["reconciliation"].json()["reconciliation"]["status"] == "not_recorded"  # nosec B101
    after = client.portal.call(store.get_session, OWNED)
    assert after.messages == before.messages and after.status == before.status and after.usage == before.usage  # nosec B101
    assert calls == []  # nosec B101 - no runtime startup path


@pytest.mark.parametrize("session_id", [FOREIGN, "missing-session"])
def test_foreign_and_missing_records_refuse_all_stored_reads(persisted_sessions, session_id):
    client, _store, _runtime, calls = persisted_sessions
    for name in READS:
        response = client.get(f"/api/v1/acp/sessions/{session_id}/{name}")
        assert response.status_code == 404 and response.json()["detail"] == "session_not_found"  # nosec B101
    assert calls == []  # nosec B101


def test_owner_list_and_redacted_detail_do_not_require_or_restore_runtime(persisted_sessions):
    client, _store, _runtime, calls = persisted_sessions
    response = client.get("/api/v1/acp/sessions")
    assert response.status_code == 200  # nosec B101
    assert [(item["session_id"], item["runtime_available"]) for item in response.json()["sessions"]] == [(OWNED, False)]  # nosec B101
    detail = client.get(f"/api/v1/acp/sessions/{OWNED}/detail?redacted=true")
    assert detail.status_code == 200  # nosec B101
    assert "Original saved" not in detail.text and "/tmp/owned-project" not in detail.text  # nosec B101 B108 - fixture metadata redaction assertion only
    assert calls == []  # nosec B101


def test_persisted_ownership_does_not_authorize_runtime_requests(persisted_sessions):
    client, _store, _runtime, _calls = persisted_sessions
    csrf = client.cookies.get("csrf_token")
    headers = {"X-CSRF-Token": csrf} if csrf else {}
    for action, payload in (("prompt", {"session_id": OWNED, "prompt": [{"type": "text", "text": "Do not dispatch"}]}),
                            ("cancel", {"session_id": OWNED}), ("close", {"session_id": OWNED})):
        response = client.post(f"/api/v1/acp/sessions/{action}", json=payload, headers=headers)
        assert response.status_code == 404 and response.json()["detail"] == "session_not_found"  # nosec B101
    for action in ("updates", "events/stream", "checkpoints"):
        response = client.get(f"/api/v1/acp/sessions/{OWNED}/{action}")
        assert response.status_code == 404 and response.json()["detail"] == "session_not_found"  # nosec B101
    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect(f"/api/v1/acp/sessions/{OWNED}/stream", headers={"Origin": "http://localhost:3000"}):
            pass
    assert refused.value.code == 4404  # nosec B101 - current runtime owner still required


def test_current_direct_runtime_is_reported_only_for_its_actual_created_owner(persisted_sessions, tmp_path, monkeypatch):
    client, store, _runtime, calls = persisted_sessions
    monkeypatch.setenv("ACP_SANDBOX_ENABLED", "false")
    peer = Path(__file__).resolve().parents[3] / "Helper_Scripts" / "acp_stub_agent.py"
    direct = ACPRunnerClient(ACPRunnerConfig(command=sys.executable, args=[str(peer)],
                                           allowed_session_cwd_roots=[str(tmp_path)]))
    monkeypatch.setattr(runner_client, "_runner_client", direct)

    async def prepare():
        await direct.start()
        session_id = await direct.create_session(str(tmp_path), user_id=1)
        await store.register_session(session_id, user_id=1, agent_type="controlled", cwd=str(tmp_path))
        return session_id

    session_id = client.portal.call(prepare)
    try:
        assert runner_client.get_existing_runner_client() is direct  # nosec B101 - actual running peer
        detail = client.get(f"/api/v1/acp/sessions/{session_id}/detail")
        assert detail.status_code == 200 and detail.json()["runtime_available"] is True  # nosec B101
        assert client.get(f"/api/v1/acp/sessions/{OWNED}/detail").json()["runtime_available"] is False  # nosec B101
        assert calls == []  # nosec B101 - metadata did not request a new runner
    finally:
        client.portal.call(direct.shutdown)
    assert runner_client.get_existing_runner_client() is None  # nosec B101 - terminated transport cannot imply runtime


def test_current_sandbox_accessor_does_not_fabricate_session_ownership(persisted_sessions, monkeypatch):
    client, _store, _runtime, calls = persisted_sessions
    monkeypatch.setenv("ACP_SANDBOX_ENABLED", "true")
    sandbox = ACPSandboxRunnerManager(ACPSandboxConfig(enabled=True))
    monkeypatch.setattr(runner_client, "_sandbox_client", sandbox)
    assert runner_client.get_existing_runner_client() is sandbox  # nosec B101 - actual supported client branch
    assert client.get(f"/api/v1/acp/sessions/{OWNED}/detail").json()["runtime_available"] is False  # nosec B101
    assert calls == []  # nosec B101 - no sandbox start/container/session creation


def test_invalid_runtime_configuration_does_not_hide_persisted_owner_history(persisted_sessions, monkeypatch):
    client, _store, _runtime, calls = persisted_sessions

    def invalid_config():
        raise ValueError("Invalid controlled sandbox configuration")

    monkeypatch.setattr(acp_config, "load_acp_sandbox_config", invalid_config)
    response = client.get(f"/api/v1/acp/sessions/{OWNED}/detail")
    assert response.status_code == 200 and response.json()["runtime_available"] is False  # nosec B101
    assert [item["content"] for item in response.json()["messages"]] == ["Original saved question", "Original saved answer"]  # nosec B101
    assert calls == []  # nosec B101
