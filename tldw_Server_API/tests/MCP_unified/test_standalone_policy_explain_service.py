from __future__ import annotations

import pytest

import mcp_unified.gateway.policy_explain as policy_explain
from mcp_unified.gateway.policy_explain import (
    GatewayPolicyExplainError,
    GatewayPolicyExplainService,
    PolicyExplainErrorResponse,
    PolicyExplainRequest,
    ProfileToolPreviewRequest,
    _redact_subject_value,
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


def test_subject_redaction_states_use_approved_contract() -> None:
    assert _redact_subject_value("tool", "fs.patch")[1] == "raw_safe"
    assert _redact_subject_value("path", "")[1] == "omitted"


def test_error_response_uses_message_and_details_contract() -> None:
    payload = PolicyExplainErrorResponse(
        message="Policy evaluation failed",
        reason_code="policy_evaluation_failed",
        details={"field": "safe"},
    ).model_dump()

    assert payload["ok"] is False
    assert payload["message"] == "Policy evaluation failed"
    assert payload["reason_code"] == "policy_evaluation_failed"
    assert payload["details"] == {"field": "safe"}
    assert "error" not in payload


def test_tool_names_from_catalog_reads_tool_id_and_mapping_payload() -> None:
    catalog = {
        "tools": [
            {"tool_id": "fs.patch"},
            {"name": "shell.exec"},
            {"tool_name": "db.query"},
            "cache.clear",
        ],
    }

    assert policy_explain._tool_names_from_catalog(catalog) == {
        "fs.patch",
        "shell.exec",
        "db.query",
        "cache.clear",
    }


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
    assert response.visibility == "visible"
    assert response.subjects[0].redaction_state in {"sanitized", "redacted"}
    rendered = response.model_dump_json()
    assert "/Users/example" not in rendered
    assert audit.events[0].event_type == "policy.explain.requested"
    assert "/Users/example" not in str(audit.events[0].payload)
    dumped = response.model_dump(mode="json")
    assert dumped["evaluated_at"]
    assert dumped["truncated"] is False
    assert dumped["installation_status"] == "unknown"
    assert dumped["runtime_availability"] == "unknown"


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
async def test_explain_tool_call_preserves_tool_level_ask_as_final_outcome() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: MCPProfile(
            id="backend-engineer",
            name="Backend Engineer",
            policy_document=ProfilePolicy(
                tool_rules=[{"pattern": "fs.patch", "outcome": "ask"}],
            ),
        ),
        audit_store=audit,
        actor_id="operator-1",
    )

    response = await service.explain_tool_call(
        PolicyExplainRequest(profile_id="backend-engineer", tool_name="fs.patch")
    )

    assert response.final_outcome == "ask"
    assert response.reason_code == "approval_required"
    assert response.visibility == "visible"
    assert response.call_state == "approval_required"
    assert response.tool_policy_outcome == "ask"
    assert audit.events[0].payload["final_outcome"] == "ask"


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
    assert set(static_response.skipped_contributors) >= {
        "session_grants",
        "approval_grants",
        "runtime_availability",
    }


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
async def test_explain_tool_call_redacts_url_shaped_path_subjects() -> None:
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
                "path": "https://user:pass@api.example.test/path?token=secret#frag",
            },
        )
    )

    rendered = response.model_dump_json()
    audit_payload = str(audit.events[0].payload)
    assert "token=secret" not in rendered
    assert "token=secret" not in audit_payload
    assert "frag" not in rendered
    assert "frag" not in audit_payload
    assert "user:pass" not in rendered
    assert "user:pass" not in audit_payload
    assert response.subjects[0].value == "https://api.example.test"


@pytest.mark.asyncio
async def test_explain_tool_call_redacts_file_uri_domain_subject() -> None:
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
                "uri": "file:///Users/example/project/src/app.py?token=secret#fragment",
            },
        )
    )

    rendered = response.model_dump_json()
    audit_payload = str(audit.events[0].payload)
    assert "/Users/example" not in rendered
    assert "/Users/example" not in audit_payload
    assert "token=secret" not in rendered
    assert "token=secret" not in audit_payload
    assert "fragment" not in rendered
    assert "fragment" not in audit_payload
    assert response.subjects[0].type == "domain"
    assert response.subjects[0].value == ".../app.py"


@pytest.mark.asyncio
async def test_explain_tool_call_redacts_file_uri_domain_list_subjects() -> None:
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
                "uris": [
                    "file:///Users/example/project/src/app.py?token=secret#fragment",
                ],
            },
        )
    )

    rendered = response.model_dump_json()
    audit_payload = str(audit.events[0].payload)
    assert "/Users/example" not in rendered
    assert "/Users/example" not in audit_payload
    assert "token=secret" not in rendered
    assert "token=secret" not in audit_payload
    assert "fragment" not in rendered
    assert "fragment" not in audit_payload
    assert response.subjects[0].type == "domain"
    assert response.subjects[0].value == ".../app.py"


@pytest.mark.asyncio
async def test_explain_tool_call_redacts_absolute_uri_path_like_domain_subject() -> None:
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
            arguments={"uri": "/Users/example/project/src/app.py"},
        )
    )

    rendered = response.model_dump_json()
    audit_payload = str(audit.events[0].payload)
    assert "/Users/example" not in rendered
    assert "/Users/example" not in audit_payload
    assert response.subjects[0].type == "domain"
    assert response.subjects[0].value == ".../app.py"


@pytest.mark.asyncio
async def test_explain_tool_call_redacts_windows_uri_path_like_domain_subject() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
        actor_id="operator-1",
    )

    raw_path = r"C:\Users\example\project\src\app.py"
    response = await service.explain_tool_call(
        PolicyExplainRequest(
            profile_id="backend-engineer",
            tool_name="fs.patch",
            arguments={"uri": raw_path},
        )
    )

    rendered = response.model_dump_json()
    audit_payload = str(audit.events[0].payload)
    assert "C:/Users/example" not in rendered
    assert "C:/Users/example" not in audit_payload
    assert raw_path not in rendered
    assert raw_path not in audit_payload
    assert response.subjects[0].type == "domain"
    assert response.subjects[0].value == ".../app.py"


@pytest.mark.asyncio
async def test_explain_tool_call_unknown_simulator_status_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
        actor_id="operator-1",
    )

    def _unknown_policy_status(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "profile_id": "backend-engineer",
            "tool_name": "fs.patch",
            "legacy_policy": {"status": "resolved", "reason_code": "resolved"},
            "subjects": [],
            "approval_grant_markers": [],
            "path_scopes": [],
            "denial": None,
            "overall": {"status": "mystery", "reason_code": "mystery_status"},
        }

    monkeypatch.setattr(
        policy_explain,
        "simulate_tool_call_policy",
        _unknown_policy_status,
    )

    response = await service.explain_tool_call(
        PolicyExplainRequest(profile_id="backend-engineer", tool_name="fs.patch")
    )

    assert response.final_outcome == "deny"
    assert response.reason_code == "policy_status_unknown"
    assert response.degraded is True
    assert "unknown_policy_status" in response.degraded_reasons
    assert audit.events[0].payload["final_outcome"] == "deny"


@pytest.mark.asyncio
async def test_explain_tool_call_audits_policy_evaluation_failure_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
        actor_id="operator-1",
    )

    def _raise_policy_error(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("invalid policy internals")

    monkeypatch.setattr(
        policy_explain,
        "simulate_tool_call_policy",
        _raise_policy_error,
    )

    with pytest.raises(GatewayPolicyExplainError) as exc_info:
        await service.explain_tool_call(
            PolicyExplainRequest(
                profile_id="backend-engineer",
                tool_name="Bash(echo TOPSECRET)",
                arguments={"path": "/Users/example/project/src/app.py"},
            )
        )

    assert exc_info.value.reason_code == "policy_evaluation_failed"
    assert audit.events[0].event_type == "policy.explain.requested"
    assert audit.events[0].payload["reason_code"] == "policy_evaluation_failed"
    assert "TOPSECRET" not in str(audit.events[0].payload)
    assert "/Users/example" not in str(audit.events[0].payload)


@pytest.mark.asyncio
async def test_explain_tool_call_audits_profile_not_found_before_raising() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: None,
        audit_store=audit,
        actor_id="operator-1",
    )

    with pytest.raises(GatewayPolicyExplainError) as exc_info:
        await service.explain_tool_call(
            PolicyExplainRequest(
                profile_id="missing-profile",
                tool_name="Bash(echo TOPSECRET)",
                arguments={"path": "/Users/example/project/src/app.py"},
            )
        )

    assert exc_info.value.reason_code == "profile_not_found"
    assert audit.events[0].event_type == "policy.explain.requested"
    assert audit.events[0].payload["reason_code"] == "profile_not_found"
    assert "TOPSECRET" not in str(audit.events[0].payload)
    assert "/Users/example" not in str(audit.events[0].payload)


@pytest.mark.asyncio
async def test_preview_profile_tools_audits_profile_not_found_before_raising() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: None,
        audit_store=audit,
        actor_id="operator-1",
    )

    with pytest.raises(GatewayPolicyExplainError) as exc_info:
        await service.preview_profile_tools(
            ProfileToolPreviewRequest(profile_id="missing-profile")
        )

    assert exc_info.value.reason_code == "profile_not_found"
    assert audit.events[0].event_type == "policy.preview_tools.requested"
    assert audit.events[0].payload["reason_code"] == "profile_not_found"
    assert audit.events[0].payload["profile_id"] == "missing-profile"


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
    dumped = response.model_dump(mode="json")
    assert "tools" in dumped
    assert "entries" not in dumped
    assert dumped["evaluated_at"]
    assert dumped["truncated"] is False


@pytest.mark.asyncio
async def test_preview_catalog_provider_failure_degrades_and_audits() -> None:
    audit = _MemoryAuditStore()

    def _broken_catalog() -> list[str]:
        raise RuntimeError("catalog backend failed")

    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
        actor_id="operator-1",
        catalog_provider=_broken_catalog,
    )

    response = await service.preview_profile_tools(
        ProfileToolPreviewRequest(profile_id="backend-engineer")
    )

    assert response.degraded is True
    assert "catalog_unavailable" in response.degraded_reasons
    assert [entry.tool_name for entry in response.tools] == ["fs.patch", "shell.exec"]
    assert audit.events[0].event_type == "policy.preview_tools.requested"
    assert audit.events[0].payload["degraded"] is True
    assert "catalog_unavailable" in audit.events[0].payload["degraded_reasons"]
