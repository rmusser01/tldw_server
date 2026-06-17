from __future__ import annotations

import pytest

from mcp_unified.gateway.policy_explain import (
    GatewayPolicyExplainError,
    GatewayPolicyExplainService,
    PolicyExplainRequest,
    ProfileToolPreviewRequest,
)
from mcp_unified.policy_grants import InMemoryPolicyGrantStore
from mcp_unified.profiles import MCPProfile, ProfilePolicy
from mcp_unified.storage.models import AuditEvent


class _MemoryAuditStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[AuditEvent] = []

    async def append_event(self, event: AuditEvent) -> None:
        if self.fail:
            raise RuntimeError("audit backend failed")
        self.events.append(event)


def _profile() -> MCPProfile:
    return MCPProfile(
        id="backend-engineer",
        name="Backend Engineer",
        policy_document=ProfilePolicy(
            allowed_tools=["fs.patch"],
            denied_tools=["shell.exec"],
        ),
    )


def _profile_with_rules(*, permission_rules: list[dict[str, str]]) -> MCPProfile:
    return MCPProfile(
        id="backend-engineer",
        name="Backend Engineer",
        policy_document=ProfilePolicy(
            allowed_tools=["fs.patch"],
            permission_rules=permission_rules,
        ),
    )


@pytest.mark.asyncio
async def test_explain_tool_call_redacts_subjects_and_audits() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
        actor_id="operator-1",
    )

    response = await service.explain_tool_call(
        PolicyExplainRequest(
            profile_id="backend-engineer",
            tool_name="fs.patch",
            arguments={"path": "/Users/example/project/src/app.py"},
        )
    )

    assert response.ok is True
    assert response.final_outcome == "allow"
    assert response.subjects[0].redaction_state in {"sanitized", "redacted"}
    rendered = response.model_dump_json()
    assert "/Users/example" not in rendered
    assert audit.events[0].event_type == "policy.explain.requested"
    assert "/Users/example" not in str(audit.events[0].payload)


@pytest.mark.asyncio
async def test_explain_tool_call_uses_runtime_permission_rule_denial_as_final_outcome() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile_with_rules(
            permission_rules=[
                {"pattern": "Edit(/Users/example/**)", "outcome": "deny"},
            ]
        ),
        audit_store=audit,
        actor_id="operator-1",
    )

    response = await service.explain_tool_call(
        PolicyExplainRequest(
            profile_id="backend-engineer",
            tool_name="fs.patch",
            arguments={"path": "/Users/example/project/src/app.py"},
        )
    )

    assert response.final_outcome == "deny"
    assert response.reason_code == "permission_rule_denied"
    assert response.visibility == "hidden"
    assert response.call_state == "blocked"
    assert audit.events[0].payload["final_outcome"] == "deny"


@pytest.mark.asyncio
async def test_explain_tool_call_redacts_shell_shaped_tool_name_from_response_and_audit() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: MCPProfile(
            id="shell-profile",
            name="Shell Profile",
            policy_document=ProfilePolicy(
                allowed_tools=["Bash(echo TOPSECRET)"],
            ),
        ),
        audit_store=audit,
        actor_id="operator-1",
    )

    response = await service.explain_tool_call(
        PolicyExplainRequest(
            profile_id="shell-profile",
            tool_name="Bash(echo TOPSECRET)",
        )
    )

    assert "TOPSECRET" not in response.model_dump_json()
    assert "TOPSECRET" not in str(audit.events[0].payload)
    assert "TOPSECRET" not in str(audit.events[0].target_id)
    assert response.tool_name == "Bash([redacted-command])"


@pytest.mark.asyncio
async def test_explain_tool_call_static_policy_only_ignores_runtime_approval_grants() -> None:
    grant_store = InMemoryPolicyGrantStore()
    grant_store.create_grant(
        profile_id="backend-engineer",
        grant_type="approval",
        subject_type="path",
        value="/Users/example/project/src/app.py",
        ttl_seconds=300,
        session_id="session-1",
    )
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile_with_rules(
            permission_rules=[
                {"pattern": "Edit(/Users/example/**)", "outcome": "ask"},
            ]
        ),
        audit_store=audit,
        actor_id="operator-1",
        policy_grant_store=grant_store,
    )

    runtime_response = await service.explain_tool_call(
        PolicyExplainRequest(
            profile_id="backend-engineer",
            tool_name="fs.patch",
            arguments={"path": "/Users/example/project/src/app.py"},
            session_id="session-1",
        )
    )
    static_response = await service.explain_tool_call(
        PolicyExplainRequest(
            profile_id="backend-engineer",
            tool_name="fs.patch",
            arguments={"path": "/Users/example/project/src/app.py"},
            session_id="session-1",
            mode="static_policy_only",
        )
    )

    assert runtime_response.final_outcome == "allow"
    assert static_response.mode == "static_policy_only"
    assert static_response.final_outcome == "ask"
    assert static_response.reason_code == "approval_required"
    assert static_response.call_state == "approval_required"


@pytest.mark.asyncio
async def test_explain_tool_call_redacts_file_uri_paths() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
        actor_id="operator-1",
    )

    response = await service.explain_tool_call(
        PolicyExplainRequest(
            profile_id="backend-engineer",
            tool_name="fs.patch",
            arguments={
                "path": "file:///Users/example/project/src/app.py?token=secret#fragment",
            },
        )
    )

    rendered = response.model_dump_json()
    assert "/Users/example" not in rendered
    assert "token=secret" not in rendered
    assert "fragment" not in rendered
    assert response.subjects[0].value == ".../app.py"


@pytest.mark.asyncio
async def test_explain_tool_call_fails_closed_when_audit_append_fails() -> None:
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=_MemoryAuditStore(fail=True),
        actor_id="operator-1",
    )

    with pytest.raises(GatewayPolicyExplainError) as exc_info:
        await service.explain_tool_call(
            PolicyExplainRequest(profile_id="backend-engineer", tool_name="fs.patch")
        )

    assert exc_info.value.reason_code == "audit_store_unavailable"


@pytest.mark.asyncio
async def test_preview_requires_catalog_for_complete_denied_counts() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
        actor_id="operator-1",
    )

    response = await service.preview_profile_tools(
        ProfileToolPreviewRequest(profile_id="backend-engineer")
    )

    assert response.degraded is True
    assert "catalog_unavailable" in response.degraded_reasons
    assert response.redacted is True
    assert audit.events[0].event_type == "policy.preview_tools.requested"
