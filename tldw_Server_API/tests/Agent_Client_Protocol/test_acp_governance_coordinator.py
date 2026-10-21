from __future__ import annotations

import asyncio
import importlib.machinery
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import (
    ACPGovernanceCoordinator,
    ACPGovernanceDeniedError,
    ACPRunnerClient,
    SessionWebSocketRegistry,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPMessage
from tldw_Server_API.app.services.acp_runtime_policy_service import (
    ACPRuntimePolicySnapshot,
)
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    _mint,
)
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)


@pytest.fixture
def governance_paths(tmp_path, monkeypatch):
    """Opt-in functional paths for only the two real-cookie API controls."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(tmp_path / "system.log"))
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(tmp_path / "alerts.db"))
    monkeypatch.setenv("ACP_SESSIONS_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))


def _cookie_governance_client(fixture):
    """Adapt the official cookie fixture without overriding auth dependencies."""
    from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol
    from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
    client, api_key = fixture
    if not any(route.path == "/api/v1/acp/sessions/prompt" for route in iter_served_routes(client.app.routes)):
        client.app.include_router(agent_client_protocol.router, prefix="/api/v1")
    assert _mint(client, api_key).status_code == 200  # nosec B101 # real cookie setup precondition
    return client

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


class _PromptGovernanceDeniedRunner:
    def __init__(self) -> None:
        self.agent_capabilities = {"promptCapabilities": {"image": False}}

    async def verify_session_access(self, session_id: str, user_id: int) -> bool:
        return True

    async def check_prompt_governance(
        self,
        session_id: str,
        prompt: list[dict[str, Any]],
        *,
        user_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return {
            "action": "deny",
            "status": "deny",
            "category": "acp",
            "rollout_mode": "enforce",
        }

    async def prompt(self, session_id: str, prompt: list[dict[str, Any]], *, on_native_text=None) -> dict[str, Any]:
        raise ACPGovernanceDeniedError(governance={
            "action": "deny",
            "status": "deny",
            "category": "acp",
            "rollout_mode": "enforce",
        })


class _PromptGovernanceShadowDeniedRunner(_PromptGovernanceDeniedRunner):
    async def check_prompt_governance(
        self,
        session_id: str,
        prompt: list[dict[str, Any]],
        *,
        user_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return {
            "action": "deny",
            "status": "deny",
            "category": "acp",
            "rollout_mode": "shadow",
        }

    async def prompt(self, session_id: str, prompt: list[dict[str, Any]], *, on_native_text=None) -> dict[str, Any]:
        return {"stopReason": "end", "detail": "shadow-allowed"}


def _new_runner_client_for_permissions() -> ACPRunnerClient:
    cfg = SimpleNamespace(
        command="echo",
        args=[],
        env={},
        cwd=None,
        startup_timeout_sec=0,
    )
    return ACPRunnerClient(cfg)


@pytest.fixture()
def prompt_governance_denied_runner(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    runner = _PromptGovernanceDeniedRunner()

    async def _get_runner_client():
        return runner

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)
    return runner


@pytest.fixture()
def prompt_governance_shadow_denied_runner(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    runner = _PromptGovernanceShadowDeniedRunner()

    async def _get_runner_client():
        return runner

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)
    return runner


def test_prompt_denied_by_governance_returns_blocked_error(
    governance_paths,
    single_user_cookie_client,
    prompt_governance_denied_runner,
):
    client_user_only = _cookie_governance_client(single_user_cookie_client)
    response = client_user_only.post(
        "/api/v1/acp/sessions/prompt",
        json={
            "session_id": "session-gov-prompt",
            "prompt": [{"role": "user", "content": "ship this change"}],
        },
        headers={"X-CSRF-Token": client_user_only.cookies["csrf_token"]},
    )

    assert response.status_code == 403
    body = response.json()
    assert body["detail"]["code"] == "governance_blocked"
    assert body["detail"]["governance"]["action"] == "deny"


def test_prompt_shadow_deny_governance_is_not_blocked_at_endpoint(
    governance_paths,
    single_user_cookie_client,
    prompt_governance_shadow_denied_runner,
):
    client_user_only = _cookie_governance_client(single_user_cookie_client)
    response = client_user_only.post(
        "/api/v1/acp/sessions/prompt",
        json={
            "session_id": "session-gov-shadow",
            "prompt": [{"role": "user", "content": "ship this change"}],
        },
        headers={"X-CSRF-Token": client_user_only.cookies["csrf_token"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "end"
    assert body["raw_result"]["detail"] == "shadow-allowed"


@pytest.mark.asyncio
async def test_permission_request_uses_single_unified_approval_path(monkeypatch, admitted_permission_prompt):
    client = _new_runner_client_for_permissions()
    session_id = "session-gov-auto"

    async def _send(_payload: dict[str, Any]) -> None:
        return None

    registry = SessionWebSocketRegistry(session_id=session_id)
    registry.websockets.add(_send)
    client._ws_registry[session_id] = registry

    async def _fake_check_permission_governance(
        sid: str,
        tool_name: str,
        tool_arguments: dict[str, Any],
        *,
        tier: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return {
            "action": "require_approval",
            "status": "require_approval",
            "category": "acp",
        }

    monkeypatch.setattr(client, "check_permission_governance", _fake_check_permission_governance, raising=False)

    async def _fake_snapshot(
        sid: str,
        *,
        force_refresh: bool = False,
    ) -> ACPRuntimePolicySnapshot | None:
        del sid, force_refresh
        return ACPRuntimePolicySnapshot(
            session_id=session_id,
            user_id=7,
            policy_snapshot_version="resolved-v1",
            policy_snapshot_fingerprint="snapshot-unified",
            policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
            policy_summary={"approval_mode": "require_approval"},
            policy_provenance_summary={"source_kinds": ["profile"]},
            resolved_policy_document={
                "allowed_tools": ["fs.read"],
                "approval_mode": "require_approval",
            },
            approval_summary={"mode": "require_approval"},
            context_summary={},
            execution_config={},
        )

    monkeypatch.setattr(client, "_get_runtime_policy_snapshot", _fake_snapshot, raising=False)

    prompts: list[dict[str, Any]] = []

    async def _fake_broadcast(sid: str, message: dict[str, Any]) -> None:
        if message.get("type") == "permission_request":
            prompts.append(message)
            await client.respond_to_permission(sid, str(message["request_id"]), True)

    monkeypatch.setattr(client, "_broadcast_to_session", _fake_broadcast)

    await admitted_permission_prompt(client, session_id)
    response = await client._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="perm-1",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "fs.read", "input": {"path": "README.md"}},
            },
        )
    )

    assert response.result == {"outcome": {"outcome": "approved"}}
    assert len(prompts) == 1


@pytest.mark.asyncio
async def test_permission_request_uses_runtime_policy_snapshot_authority(monkeypatch, admitted_permission_prompt):
    client = _new_runner_client_for_permissions()
    session_id = "session-policy-prompt"

    async def _send(_payload: dict[str, Any]) -> None:
        return None

    registry = SessionWebSocketRegistry(session_id=session_id)
    registry.websockets.add(_send)
    client._ws_registry[session_id] = registry

    async def _fake_snapshot(
        sid: str,
        *,
        force_refresh: bool = False,
    ) -> ACPRuntimePolicySnapshot | None:
        del sid, force_refresh
        return ACPRuntimePolicySnapshot(
            session_id=session_id,
            user_id=7,
            policy_snapshot_version="resolved-v1",
            policy_snapshot_fingerprint="snapshot-123",
            policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
            policy_summary={"approval_mode": "require_approval"},
            policy_provenance_summary={"source_kinds": ["profile"]},
            resolved_policy_document={
                "allowed_tools": ["fs.read"],
                "approval_mode": "require_approval",
            },
            approval_summary={"mode": "require_approval"},
            context_summary={"persona_id": "persona-1"},
            execution_config={},
        )

    monkeypatch.setattr(client, "_get_runtime_policy_snapshot", _fake_snapshot, raising=False)
    async def _fake_check_permission_governance(*args, **kwargs) -> dict[str, Any]:
        del args, kwargs
        return {"action": "allow", "status": "allow"}

    monkeypatch.setattr(client, "check_permission_governance", _fake_check_permission_governance, raising=False)

    prompts: list[dict[str, Any]] = []

    async def _fake_broadcast(sid: str, message: dict[str, Any]) -> None:
        if message.get("type") == "permission_request":
            prompts.append(message)
            await client.respond_to_permission(sid, str(message["request_id"]), True)

    monkeypatch.setattr(client, "_broadcast_to_session", _fake_broadcast)

    await admitted_permission_prompt(client, session_id)
    response = await client._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="perm-policy-1",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "fs.read", "input": {"path": "README.md"}},
            },
        )
    )

    assert response.result == {"outcome": {"outcome": "approved"}}
    assert len(prompts) == 1
    assert prompts[0]["approval_requirement"] == "approval_required"
    assert prompts[0]["policy_snapshot_fingerprint"] == "snapshot-123"
    assert prompts[0]["provenance_summary"] == {"source_kinds": ["profile"]}


@pytest.mark.asyncio
async def test_permission_request_governance_deny_enforced_blocks_runtime_allow(monkeypatch, admitted_permission_prompt):
    client = _new_runner_client_for_permissions()

    async def _fake_snapshot(
        sid: str,
        *,
        force_refresh: bool = False,
    ) -> ACPRuntimePolicySnapshot | None:
        del sid, force_refresh
        return ACPRuntimePolicySnapshot(
            session_id="session-governance-deny",
            user_id=7,
            policy_snapshot_version="resolved-v1",
            policy_snapshot_fingerprint="snapshot-allow",
            policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
            policy_summary={"approval_mode": "allow"},
            policy_provenance_summary={"source_kinds": ["profile"]},
            resolved_policy_document={"allowed_tools": ["fs.read"]},
            approval_summary={"mode": "allow"},
            context_summary={},
            execution_config={},
        )

    async def _fake_check_permission_governance(*args, **kwargs):
        del args, kwargs
        return {
            "action": "deny",
            "status": "deny",
            "rollout_mode": "enforce",
            "category": "acp",
        }

    monkeypatch.setattr(client, "_get_runtime_policy_snapshot", _fake_snapshot, raising=False)
    monkeypatch.setattr(client, "check_permission_governance", _fake_check_permission_governance, raising=False)

    await admitted_permission_prompt(client, "session-governance-deny")
    response = await client._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="perm-governance-deny",
            method="session/request_permission",
            params={
                "sessionId": "session-governance-deny",
                "tool": {"name": "fs.read", "input": {"path": "README.md"}},
            },
        )
    )

    assert response.result == {
        "outcome": {
            "outcome": "denied",
            "deny_reason": "governance_denied",
            "policy_snapshot_fingerprint": "snapshot-allow",
            "provenance_summary": {"source_kinds": ["profile"]},
            "governance": {
                "action": "deny",
                "status": "deny",
                "rollout_mode": "enforce",
                "category": "acp",
            },
        }
    }


@pytest.mark.asyncio
async def test_permission_request_governance_require_approval_prompts_runtime_allow(monkeypatch, admitted_permission_prompt):
    client = _new_runner_client_for_permissions()
    session_id = "session-governance-prompt"

    async def _send(_payload: dict[str, Any]) -> None:
        return None

    registry = SessionWebSocketRegistry(session_id=session_id)
    registry.websockets.add(_send)
    client._ws_registry[session_id] = registry

    async def _fake_snapshot(
        sid: str,
        *,
        force_refresh: bool = False,
    ) -> ACPRuntimePolicySnapshot | None:
        del sid, force_refresh
        return ACPRuntimePolicySnapshot(
            session_id=session_id,
            user_id=7,
            policy_snapshot_version="resolved-v1",
            policy_snapshot_fingerprint="snapshot-allow",
            policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
            policy_summary={"approval_mode": "allow"},
            policy_provenance_summary={"source_kinds": ["profile"]},
            resolved_policy_document={"allowed_tools": ["fs.read"]},
            approval_summary={"mode": "allow"},
            context_summary={},
            execution_config={},
        )

    async def _fake_check_permission_governance(*args, **kwargs):
        del args, kwargs
        return {
            "action": "require_approval",
            "status": "require_approval",
            "rollout_mode": "enforce",
            "category": "acp",
        }

    prompts: list[dict[str, Any]] = []

    async def _fake_broadcast(sid: str, message: dict[str, Any]) -> None:
        if message.get("type") == "permission_request":
            prompts.append(message)
            await client.respond_to_permission(sid, str(message["request_id"]), True)

    monkeypatch.setattr(client, "_get_runtime_policy_snapshot", _fake_snapshot, raising=False)
    monkeypatch.setattr(client, "check_permission_governance", _fake_check_permission_governance, raising=False)
    monkeypatch.setattr(client, "_broadcast_to_session", _fake_broadcast)

    await admitted_permission_prompt(client, session_id)
    response = await client._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="perm-governance-prompt",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "fs.read", "input": {"path": "README.md"}},
            },
        )
    )

    assert response.result == {"outcome": {"outcome": "approved"}}
    assert len(prompts) == 1
    assert prompts[0]["approval_requirement"] == "approval_required"
    assert prompts[0]["governance_reason"] == "governance_approval_required"
    assert prompts[0]["policy_snapshot_fingerprint"] == "snapshot-allow"
    assert prompts[0]["governance"]["action"] == "require_approval"


@pytest.mark.asyncio
async def test_denied_tool_from_runtime_snapshot_returns_denied_without_prompt(monkeypatch, admitted_permission_prompt):
    client = _new_runner_client_for_permissions()

    async def _fake_snapshot(
        sid: str,
        *,
        force_refresh: bool = False,
    ) -> ACPRuntimePolicySnapshot | None:
        del sid, force_refresh
        return ACPRuntimePolicySnapshot(
            session_id="session-policy-deny",
            user_id=7,
            policy_snapshot_version="resolved-v1",
            policy_snapshot_fingerprint="snapshot-deny",
            policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
            policy_summary={},
            policy_provenance_summary={"source_kinds": ["profile"]},
            resolved_policy_document={"denied_tools": ["fs.delete"]},
            approval_summary={},
            context_summary={},
            execution_config={},
        )

    monkeypatch.setattr(client, "_get_runtime_policy_snapshot", _fake_snapshot, raising=False)

    async def _fake_check_permission_governance(*args, **kwargs):
        del args, kwargs
        return None

    monkeypatch.setattr(client, "check_permission_governance", _fake_check_permission_governance, raising=False)

    await admitted_permission_prompt(client, "session-policy-deny")
    response = await client._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="perm-policy-2",
            method="session/request_permission",
            params={
                "sessionId": "session-policy-deny",
                "tool": {"name": "fs.delete", "input": {"path": "README.md"}},
            },
        )
    )

    assert response.result == {
        "outcome": {
            "outcome": "denied",
            "deny_reason": "tool_denied_by_policy",
            "policy_snapshot_fingerprint": "snapshot-deny",
            "provenance_summary": {"source_kinds": ["profile"]},
        }
    }


@pytest.mark.asyncio
async def test_runtime_policy_snapshot_refresh_is_singleflight(monkeypatch):
    client = _new_runner_client_for_permissions()
    calls = {"count": 0}

    class _Store:
        async def get_session(self, session_id: str):
            from tldw_Server_API.app.services.admin_acp_sessions_service import SessionRecord, SessionTokenUsage

            return SessionRecord(
                session_id=session_id,
                user_id=7,
                usage=SessionTokenUsage(),
            )

        async def update_policy_snapshot_state(self, session_id: str, **kwargs):
            from tldw_Server_API.app.services.admin_acp_sessions_service import SessionRecord, SessionTokenUsage

            return SessionRecord(
                session_id=session_id,
                user_id=7,
                usage=SessionTokenUsage(),
                policy_snapshot_version=kwargs.get("policy_snapshot_version"),
                policy_snapshot_fingerprint=kwargs.get("policy_snapshot_fingerprint"),
                policy_snapshot_refreshed_at=kwargs.get("policy_snapshot_refreshed_at"),
                policy_summary=kwargs.get("policy_summary"),
                policy_provenance_summary=kwargs.get("policy_provenance_summary"),
                policy_refresh_error=kwargs.get("policy_refresh_error"),
            )

    class _RuntimePolicyService:
        async def build_snapshot(self, **kwargs):
            del kwargs
            calls["count"] += 1
            await asyncio.sleep(0)
            return ACPRuntimePolicySnapshot(
                session_id="session-refresh",
                user_id=7,
                policy_snapshot_version="resolved-v1",
                policy_snapshot_fingerprint="snapshot-refresh",
                policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
                policy_summary={},
                policy_provenance_summary={},
                resolved_policy_document={"allowed_tools": ["fs.read"]},
                approval_summary={},
                context_summary={},
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

    async def _get_store():
        return _Store()

    monkeypatch.setattr(client, "_get_acp_session_store", _get_store, raising=False)
    monkeypatch.setattr(client, "_runtime_policy_service", _RuntimePolicyService(), raising=False)

    first, second = await asyncio.gather(
        client._get_runtime_policy_snapshot("session-refresh", force_refresh=True),
        client._get_runtime_policy_snapshot("session-refresh", force_refresh=True),
    )

    assert calls["count"] == 1
    assert first is second


@pytest.mark.asyncio
async def test_governance_require_approval_plus_batch_tier_creates_one_prompt(monkeypatch, admitted_permission_prompt):
    client = _new_runner_client_for_permissions()
    session_id = "session-gov-batch"

    async def _send(_payload: dict[str, Any]) -> None:
        return None

    registry = SessionWebSocketRegistry(session_id=session_id)
    registry.websockets.add(_send)
    registry.batch_approved_tiers.add("batch")
    client._ws_registry[session_id] = registry

    async def _fake_check_permission_governance(
        sid: str,
        tool_name: str,
        tool_arguments: dict[str, Any],
        *,
        tier: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return {
            "action": "require_approval",
            "status": "require_approval",
            "category": "acp",
        }

    monkeypatch.setattr(client, "check_permission_governance", _fake_check_permission_governance, raising=False)

    async def _fake_snapshot(
        sid: str,
        *,
        force_refresh: bool = False,
    ) -> ACPRuntimePolicySnapshot | None:
        del sid, force_refresh
        return ACPRuntimePolicySnapshot(
            session_id=session_id,
            user_id=7,
            policy_snapshot_version="resolved-v1",
            policy_snapshot_fingerprint="snapshot-batch",
            policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
            policy_summary={"approval_mode": "require_approval"},
            policy_provenance_summary={"source_kinds": ["profile"]},
            resolved_policy_document={
                "allowed_tools": ["fs.write"],
                "approval_mode": "require_approval",
            },
            approval_summary={"mode": "require_approval"},
            context_summary={},
            execution_config={},
        )

    monkeypatch.setattr(client, "_get_runtime_policy_snapshot", _fake_snapshot, raising=False)

    prompts: list[dict[str, Any]] = []

    async def _fake_broadcast(sid: str, message: dict[str, Any]) -> None:
        if message.get("type") == "permission_request":
            prompts.append(message)
            await client.respond_to_permission(sid, str(message["request_id"]), True)

    monkeypatch.setattr(client, "_broadcast_to_session", _fake_broadcast)

    await admitted_permission_prompt(client, session_id)
    response = await client._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="perm-2",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "fs.write", "input": {"path": "README.md", "content": "ok"}},
            },
        )
    )

    assert response.result == {"outcome": {"outcome": "approved"}}
    assert len(prompts) == 1


def test_permission_outcome_shadow_deny_falls_back_to_tier_logic():
    outcome = ACPGovernanceCoordinator.resolve_permission_outcome(
        tier="auto",
        batch_tier_approved=False,
        governance={"action": "deny", "status": "deny", "rollout_mode": "shadow"},
    )

    assert outcome == "approve"


def test_permission_outcome_off_mode_ignores_require_approval():
    outcome = ACPGovernanceCoordinator.resolve_permission_outcome(
        tier="auto",
        batch_tier_approved=False,
        governance={"action": "require_approval", "status": "require_approval", "rollout_mode": "off"},
    )

    assert outcome == "approve"


@pytest.mark.asyncio
async def test_prompt_shadow_rollout_deny_is_not_blocked(monkeypatch):
    client = _new_runner_client_for_permissions()

    async def _fake_check_prompt_governance(
        session_id: str,
        prompt: list[dict[str, Any]],
        *,
        user_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return {
            "action": "deny",
            "status": "deny",
            "category": "acp",
            "rollout_mode": "shadow",
        }

    monkeypatch.setattr(client, "check_prompt_governance", _fake_check_prompt_governance, raising=False)

    async def _fake_call(method: str, payload: dict[str, Any], *, timeout_sec: float | None = None) -> ACPMessage:
        assert method == "session/prompt"
        assert payload["sessionId"] == "session-shadow"
        return ACPMessage(jsonrpc="2.0", id="acp-shadow", result={"stopReason": "end"})

    client._client = SimpleNamespace(call=_fake_call, rpc_timeout_sec=60)

    response = await client.prompt(
        "session-shadow",
        [{"role": "user", "content": "continue"}],
    )

    assert response == {"stopReason": "end"}


@pytest.mark.asyncio
async def test_check_permission_governance_records_metrics(monkeypatch):
    client = _new_runner_client_for_permissions()

    class _GovernanceStub:
        async def validate_change(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["surface"] == "acp_permission"
            return {
                "action": "allow",
                "status": "allow",
                "category": "acp",
            }

    client._governance = _GovernanceStub()  # type: ignore[assignment]
    monkeypatch.setattr(client, "_resolve_governance_rollout_mode", lambda _metadata=None: "enforce", raising=False)

    metric_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        client,
        "_record_governance_check",
        lambda **kwargs: metric_calls.append({k: str(v) for k, v in kwargs.items()}),
        raising=False,
    )

    governance = await client.check_permission_governance(
        "session-metric",
        "fs.read",
        {"path": "README.md"},
        tier="auto",
    )

    assert isinstance(governance, dict)
    assert metric_calls
    assert metric_calls[-1]["surface"] == "acp_permission"
    assert metric_calls[-1]["status"] == "allow"
    assert metric_calls[-1]["rollout_mode"] == "enforce"
