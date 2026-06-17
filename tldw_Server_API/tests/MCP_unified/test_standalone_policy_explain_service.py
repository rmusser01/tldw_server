from __future__ import annotations

import pytest

from mcp_unified.gateway.policy_explain import (
    GatewayPolicyExplainError,
    GatewayPolicyExplainService,
    PolicyExplainRequest,
    ProfileToolPreviewRequest,
)
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
