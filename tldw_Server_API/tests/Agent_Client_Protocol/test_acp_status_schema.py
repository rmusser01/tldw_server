import importlib.machinery
import sys
import types

import pytest

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


class _StubUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class _StubSessionRecord:
    def __init__(self) -> None:
        self.session_id = "session-123"
        self.user_id = 1
        self.agent_type = "codex"
        self.name = "Test Session"
        self.status = "active"
        self.created_at = "2026-02-26T00:00:00+00:00"
        self.last_activity_at = "2026-02-26T00:05:00+00:00"
        self.message_count = 3
        self.tags = ["test", "acp"]
        self.persona_id = "persona-abc"
        self.workspace_id = "ws-1"
        self.workspace_group_id = "wsg-2"
        self.scope_snapshot_id = "scope-3"
        self.forked_from = "session-root"
        self.policy_snapshot_version = "resolved-v1"
        self.policy_snapshot_fingerprint = "snapshot-123"
        self.policy_snapshot_refreshed_at = "2026-03-14T12:00:00+00:00"
        self.policy_summary = {"allowed_tool_count": 2, "approval_mode": "require_approval"}
        self.policy_provenance_summary = {"source_kinds": ["governance_pack", "capability_mapping"]}
        self.cwd = "/tmp/test-project"
        self.messages = [
            {"role": "user", "content": {"text": "hi"}, "timestamp": "2026-02-26T00:00:01+00:00"},
            {
                "role": "assistant",
                "content": {
                    "text": "done",
                    "artifacts": [
                        {"id": "artifact-1", "type": "summary", "uri": "artifact://summary/1"},
                    ],
                },
                "timestamp": "2026-02-26T00:00:02+00:00",
            },
        ]
        self.usage = _StubUsage(prompt_tokens=10, completion_tokens=20)

    def to_info_dict(self, *, has_websocket: bool = False) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_type": self.agent_type,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
            "message_count": self.message_count,
            "usage": self.usage.to_dict(),
            "tags": list(self.tags),
            "has_websocket": has_websocket,
            "persona_id": self.persona_id,
            "workspace_id": self.workspace_id,
            "workspace_group_id": self.workspace_group_id,
            "scope_snapshot_id": self.scope_snapshot_id,
            "forked_from": self.forked_from,
            "policy_snapshot_version": self.policy_snapshot_version,
            "policy_snapshot_fingerprint": self.policy_snapshot_fingerprint,
            "policy_snapshot_refreshed_at": self.policy_snapshot_refreshed_at,
            "policy_summary": self.policy_summary,
            "policy_provenance_summary": self.policy_provenance_summary,
        }

    def to_detail_dict(self, *, has_websocket: bool = False, fork_lineage: list[str] | None = None) -> dict:
        payload = self.to_info_dict(has_websocket=has_websocket)
        payload["messages"] = list(self.messages)
        payload["cwd"] = self.cwd
        payload["fork_lineage"] = fork_lineage or []
        return payload


class _StubSessionStore:
    def __init__(self) -> None:
        self.record = _StubSessionRecord()

    async def list_sessions(
        self,
        *,
        user_id: int | None = None,
        status: str | None = None,
        agent_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        if user_id is not None and user_id != self.record.user_id:
            return [], 0
        return [self.record], 1

    async def list_runs(
        self,
        *,
        user_id: int | None = None,
        status: str | None = None,
        agent_type: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        if user_id is not None and user_id != self.record.user_id:
            return {"items": [], "total": 0, "limit": limit, "offset": offset}
        return {
            "items": [self.record.to_info_dict()],
            "total": 2,
            "limit": limit,
            "offset": offset,
        }

    async def get_session(self, session_id: str):
        if session_id != self.record.session_id:
            return None
        return self.record

    async def get_fork_lineage(self, session_id: str, *, max_depth: int = 50):
        return []


class _StubRunnerClient:
    async def verify_session_access(self, session_id: str, user_id: int) -> bool:
        return session_id == "session-123" and user_id == 1

    def has_websocket_connections(self, session_id: str) -> bool:
        return session_id == "session-123"


@pytest.fixture()
def stub_acp_store(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    store = _StubSessionStore()
    runner = _StubRunnerClient()

    async def _get_store():
        return store

    async def _get_runner():
        return runner

    monkeypatch.setattr(acp_endpoints, "get_acp_session_store", _get_store)
    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner)
    return store


def test_acp_list_sessions_status_schema_includes_tenancy(client_user_only, stub_acp_store):
    resp = client_user_only.get("/api/v1/acp/sessions")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 1
    assert payload["pagination"]["total"] == 1
    assert payload["pagination"]["limit"] == 100
    assert payload["pagination"]["offset"] == 0
    assert payload["pagination"]["has_more"] is False
    assert payload["pagination"]["next_offset"] is None
    assert payload["has_more"] is False
    assert payload["next_offset"] is None
    session = payload["sessions"][0]
    assert session["status"] == "active"
    assert session["has_websocket"] is True
    assert session["persona_id"] == "persona-abc"
    assert session["workspace_id"] == "ws-1"
    assert session["workspace_group_id"] == "wsg-2"
    assert session["scope_snapshot_id"] == "scope-3"
    assert session["forked_from"] == "session-root"
    assert session["policy_snapshot_version"] == "resolved-v1"
    assert session["policy_snapshot_fingerprint"] == "snapshot-123"
    assert session["policy_snapshot_refreshed_at"] == "2026-03-14T12:00:00+00:00"
    assert session["policy_summary"] == {"allowed_tool_count": 2, "approval_mode": "require_approval"}
    assert session["policy_provenance_summary"] == {
        "source_kinds": ["governance_pack", "capability_mapping"]
    }
    assert session["usage"]["total_tokens"] == 30


def test_acp_session_detail_status_schema_includes_tenancy(client_user_only, stub_acp_store):
    resp = client_user_only.get("/api/v1/acp/sessions/session-123/detail")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["session_id"] == "session-123"
    assert payload["status"] == "active"
    assert len(payload["messages"]) == 2
    assert payload["cwd"] == "/tmp/test-project"
    assert payload["persona_id"] == "persona-abc"
    assert payload["workspace_id"] == "ws-1"
    assert payload["workspace_group_id"] == "wsg-2"
    assert payload["scope_snapshot_id"] == "scope-3"
    assert payload["forked_from"] == "session-root"
    assert payload["policy_snapshot_version"] == "resolved-v1"
    assert payload["policy_snapshot_fingerprint"] == "snapshot-123"
    assert payload["policy_snapshot_refreshed_at"] == "2026-03-14T12:00:00+00:00"
    assert payload["policy_summary"] == {"allowed_tool_count": 2, "approval_mode": "require_approval"}
    assert payload["policy_provenance_summary"] == {
        "source_kinds": ["governance_pack", "capability_mapping"]
    }


def test_permission_request_schema_keeps_tier_and_adds_policy_fields():
    from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
        ACPPermissionTier,
        ACPWSPermissionRequestMessage,
    )

    payload = ACPWSPermissionRequestMessage(
        request_id="perm-1",
        session_id="session-123",
        tool_name="web.search",
        tool_arguments={"query": "opa acp"},
        tier=ACPPermissionTier.INDIVIDUAL,
        approval_requirement="approval_required",
        governance_reason="policy_approval_required",
        deny_reason=None,
        provenance_summary={"source_kinds": ["capability_mapping"]},
        runtime_narrowing_reason="workspace_trust_required",
        policy_snapshot_fingerprint="snapshot-123",
    )

    assert payload.tier == ACPPermissionTier.INDIVIDUAL
    assert payload.approval_requirement == "approval_required"
    assert payload.governance_reason == "policy_approval_required"
    assert payload.provenance_summary == {"source_kinds": ["capability_mapping"]}
    assert payload.runtime_narrowing_reason == "workspace_trust_required"
    assert payload.policy_snapshot_fingerprint == "snapshot-123"


def test_acp_session_usage_schema(client_user_only, stub_acp_store):
    resp = client_user_only.get("/api/v1/acp/sessions/session-123/usage")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["session_id"] == "session-123"
    assert payload["agent_type"] == "codex"
    assert payload["usage"]["prompt_tokens"] == 10
    assert payload["usage"]["completion_tokens"] == 20
    assert payload["usage"]["total_tokens"] == 30


def test_acp_session_events_query_schema(client_user_only, stub_acp_store):
    resp = client_user_only.get("/api/v1/acp/sessions/session-123/events")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["session_id"] == "session-123"
    assert payload["total"] == 2
    assert len(payload["events"]) == 2
    assert payload["pagination"] == {
        "mode": "offset",
        "limit": 100,
        "offset": 0,
        "total": 2,
        "has_more": False,
        "next_offset": None,
    }
    assert payload["has_more"] is False
    assert payload["next_offset"] is None
    assert payload["events"][0]["event_type"] == "message"
    assert payload["events"][0]["role"] == "user"


def test_acp_list_runs_query_schema_includes_pagination(client_user_only, stub_acp_store):
    resp = client_user_only.get("/api/v1/acp/runs?limit=1&offset=0")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 2
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert len(payload["items"]) == 1
    assert payload["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": 0,
        "total": 2,
        "has_more": True,
        "next_offset": 1,
    }
    assert payload["has_more"] is True
    assert payload["next_offset"] == 1


def test_acp_session_artifacts_query_schema(client_user_only, stub_acp_store):
    resp = client_user_only.get("/api/v1/acp/sessions/session-123/artifacts")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["session_id"] == "session-123"
    assert payload["total"] == 1
    assert payload["artifacts"][0]["id"] == "artifact-1"
    assert payload["artifacts"][0]["type"] == "summary"
