from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.services.admin_acp_sessions_service import (
    SessionRecord,
    SessionTokenUsage,
)


class _StubPolicyResolver:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def resolve_for_context(
        self,
        *,
        user_id: int | str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self.calls.append({"user_id": user_id, "metadata": dict(metadata or {})})
        if not self._responses:
            raise AssertionError("No stub responses left")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_build_snapshot_uses_normalized_context_and_profile_hints():
    from tldw_Server_API.app.services.acp_runtime_policy_service import (
        ACPRuntimePolicyService,
    )

    session = SessionRecord(
        session_id="session-1",
        user_id=42,
        agent_type="codex",
        name="Policy Session",
        cwd="/workspace/project",
        usage=SessionTokenUsage(),
        mcp_servers=[{"name": "filesystem", "type": "stdio"}],
        persona_id="persona-1",
        workspace_id="workspace-1",
        workspace_group_id="group-1",
        scope_snapshot_id="scope-1",
    )
    resolver = _StubPolicyResolver(
        [
            {
                "policy_document": {
                    "allowed_tools": ["web.search"],
                    "denied_tools": ["shell.exec"],
                    "approval_mode": "require_approval",
                    "capabilities": ["tool.invoke.research"],
                },
                "allowed_tools": ["web.search"],
                "denied_tools": ["shell.exec"],
                "sources": [{"source_kind": "profile", "profile_id": 7}],
                "provenance": [{"source_kind": "capability_mapping", "field": "allowed_tools"}],
            }
        ]
    )
    service = ACPRuntimePolicyService(policy_resolver=resolver)

    snapshot = await service.build_snapshot(
        session_record=session,
        user_id=42,
        acp_profile={
            "id": 7,
            "profile": {
                "execution_config": {"sandbox_mode": "workspace-write"},
                "policy_hints": {"tags": ["researcher"]},
            },
        },
        extra_metadata={"team_id": 9, "org_id": 4},
    )

    assert snapshot.context_summary["persona_id"] == "persona-1"
    assert snapshot.execution_config == {"sandbox_mode": "workspace-write"}
    assert snapshot.policy_provenance_summary["source_kinds"] == [
        "capability_mapping",
        "profile",
    ]
    assert snapshot.policy_summary["allowed_tool_count"] == 1
    assert resolver.calls[0]["metadata"]["mcp_policy_context_enabled"] is True
    assert resolver.calls[0]["metadata"]["acp_profile_id"] == 7
    assert resolver.calls[0]["metadata"]["acp_profile_hint_tags"] == ["researcher"]


@pytest.mark.asyncio
async def test_build_snapshot_fingerprint_changes_when_effective_policy_changes():
    from tldw_Server_API.app.services.acp_runtime_policy_service import (
        ACPRuntimePolicyService,
    )

    session = SessionRecord(
        session_id="session-2",
        user_id=7,
        usage=SessionTokenUsage(),
    )
    resolver = _StubPolicyResolver(
        [
            {"policy_document": {"allowed_tools": ["web.search"]}, "sources": [], "provenance": []},
            {"policy_document": {"allowed_tools": ["docs.search"]}, "sources": [], "provenance": []},
        ]
    )
    service = ACPRuntimePolicyService(policy_resolver=resolver)

    first = await service.build_snapshot(session_record=session, user_id=7)
    second = await service.build_snapshot(session_record=session, user_id=7)

    assert first.policy_snapshot_fingerprint != second.policy_snapshot_fingerprint


@pytest.mark.asyncio
async def test_build_snapshot_logs_and_falls_back_when_db_template_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.Agent_Client_Protocol.templates as templates_module
    import tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB as sessions_db_module
    import tldw_Server_API.app.services.acp_runtime_policy_service as runtime_policy_service_module
    from tldw_Server_API.app.services.acp_runtime_policy_service import ACPRuntimePolicyService

    session = SessionRecord(
        session_id="session-3",
        user_id=1,
        usage=SessionTokenUsage(),
    )
    resolver = _StubPolicyResolver([{"policy_document": {}, "sources": [], "provenance": []}])
    service = ACPRuntimePolicyService(policy_resolver=resolver)
    log_exception = MagicMock()

    def _raise_resolution_error(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("db template lookup failed")

    monkeypatch.setattr(templates_module, "resolve_for_session", _raise_resolution_error)
    monkeypatch.setattr(sessions_db_module, "ACPSessionsDB", lambda: object())
    monkeypatch.setattr(
        runtime_policy_service_module,
        "logger",
        SimpleNamespace(exception=log_exception),
    )

    snapshot = await service.build_snapshot(
        session_record=session,
        user_id=1,
        template_name="lockdown",
    )

    assert snapshot.resolved_policy_document["tool_tier_overrides"] == {"*": "individual"}
    assert log_exception.call_count == 1
