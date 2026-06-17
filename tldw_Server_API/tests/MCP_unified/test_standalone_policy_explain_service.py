from __future__ import annotations

import pytest

import mcp_unified.gateway.policy_explain as policy_explain
from mcp_unified.gateway.tool_discovery import (
    AdminToolCatalogEntry,
    list_profile_tools,
    search_profile_tools,
)
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


def _profile_with_preview_recommendations() -> MCPProfile:
    return MCPProfile(
        id="backend-engineer",
        name="Backend Engineer",
        policy_document=ProfilePolicy(
            allowed_tools=["cache.clear", "fs.patch", "fs.read"],
            denied_tools=["shell.exec"],
        ),
        metadata={
            "tooling": {
                "recommended_tools": [
                    {
                        "id": "fs.inspect",
                        "display_name": "Inspect Files",
                        "category": "filesystem",
                    },
                ],
            },
        },
    )


def _installed_preview_backend_tools() -> list[dict[str, object]]:
    return [
        {
            "name": "fs.patch",
            "description": "Patch files",
            "metadata": {"category": "filesystem"},
        },
        {
            "name": "shell.exec",
            "description": "Execute shell commands",
            "metadata": {"category": "shell"},
        },
    ]


def _filter_preview_backend_tools() -> list[dict[str, object]]:
    return [
        {
            "name": "cache.clear",
            "description": "Clear cache entries",
            "metadata": {"category": "cache"},
        },
        {
            "name": "fs.patch",
            "description": "Patch files",
            "metadata": {"category": "filesystem"},
        },
        {
            "name": "fs.read",
            "description": "Read files",
            "metadata": {"category": "filesystem"},
        },
        {
            "name": "shell.exec",
            "description": "Execute shell commands",
            "metadata": {"category": "shell"},
        },
    ]


def _profile_with_rules(*, permission_rules: list[dict[str, str]]) -> MCPProfile:
    return MCPProfile(
        id="backend-engineer",
        name="Backend Engineer",
        policy_document=ProfilePolicy(
            allowed_tools=["fs.patch"],
            permission_rules=permission_rules,
        ),
    )


def _raise_resolver_failure(_profile_id: str) -> MCPProfile:
    raise RuntimeError("resolver backend failed")


async def _raise_async_resolver_failure(_profile_id: str) -> MCPProfile:
    raise RuntimeError("async resolver backend failed")


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
async def test_explain_tool_call_uses_default_audit_actor() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
    )

    await service.explain_tool_call(
        PolicyExplainRequest(profile_id="backend-engineer", tool_name="fs.patch")
    )

    assert audit.events[0].actor_id == "local-cli"


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
async def test_explain_tool_call_redacts_path_subject_query_fragment_and_userinfo() -> None:
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
                "path": "user:pass@/Users/example/project/src/app.py?token=secret#frag",
            },
        )
    )

    rendered = response.model_dump_json()
    audit_payload = str(audit.events[0].payload)
    for leaked_value in ("token=secret", "frag", "user:pass", "/Users/example"):
        assert leaked_value not in rendered
        assert leaked_value not in audit_payload
    assert response.subjects[0].value == ".../app.py"


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


def test_effective_decision_keeps_explicit_denial_over_legacy_approval_status() -> None:
    decision = policy_explain._effective_decision_from_simulation(
        {
            "legacy_policy": {
                "status": "approval_required",
                "reason_code": "approval_required",
            },
            "overall": {
                "status": "denied",
                "reason_code": "denied_by_hook",
            },
            "denial": {
                "reason_code": "denied_by_hook",
            },
        },
        tool_policy_outcome="ask",
        tool_policy_reason_code="approval_required",
    )

    assert decision["final_outcome"] == "deny"
    assert decision["reason_code"] == "denied_by_hook"
    assert decision["visibility"] == "hidden"
    assert decision["call_state"] == "blocked"


def test_effective_decision_preserves_approval_required_without_denial_payload() -> None:
    decision = policy_explain._effective_decision_from_simulation(
        {
            "legacy_policy": {
                "status": "approval_required",
                "reason_code": "approval_required",
            },
            "overall": {
                "status": "denied",
                "reason_code": "approval_required",
            },
            "denial": None,
        },
        tool_policy_outcome="ask",
        tool_policy_reason_code="approval_required",
    )

    assert decision["final_outcome"] == "ask"
    assert decision["reason_code"] == "approval_required"
    assert decision["visibility"] == "visible"
    assert decision["call_state"] == "approval_required"


def test_parse_policy_explain_request_redacts_validation_errors() -> None:
    raw_path = "/Users/example/project/src/app.py?token=secret#frag"

    with pytest.raises(GatewayPolicyExplainError) as exc_info:
        policy_explain.parse_policy_explain_request(
            {
                "profile_id": "backend-engineer",
                "tool_name": "fs.patch",
                "arguments": {
                    "path": raw_path,
                    "blob": "x"
                    * (policy_explain.MAX_POLICY_EXPLAIN_ARGUMENT_BYTES + 1),
                },
            }
        )

    assert exc_info.value.reason_code == "invalid_policy_explain_request"
    rendered = exc_info.value.to_payload().model_dump_json()
    for leaked_value in ("/Users/example", "token=secret", "frag"):
        assert leaked_value not in str(exc_info.value)
        assert leaked_value not in rendered


def test_parse_profile_tool_preview_request_redacts_validation_errors() -> None:
    with pytest.raises(GatewayPolicyExplainError) as exc_info:
        policy_explain.parse_profile_tool_preview_request(
            {
                "profile_id": "/Users/example/project?token=secret#frag",
                "limit": 1001,
            }
        )

    assert exc_info.value.reason_code == "invalid_policy_preview_request"
    rendered = exc_info.value.to_payload().model_dump_json()
    for leaked_value in ("/Users/example", "token=secret", "frag"):
        assert leaked_value not in str(exc_info.value)
        assert leaked_value not in rendered


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
async def test_explain_tool_call_audits_sync_profile_resolver_failure() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=_raise_resolver_failure,
        audit_store=audit,
        actor_id="operator-1",
    )

    with pytest.raises(GatewayPolicyExplainError) as exc_info:
        await service.explain_tool_call(
            PolicyExplainRequest(
                profile_id="backend-engineer",
                tool_name="Bash(echo TOPSECRET)",
            )
        )

    assert exc_info.value.reason_code == "profile_resolution_failed"
    assert audit.events[0].event_type == "policy.explain.requested"
    assert audit.events[0].payload["reason_code"] == "profile_resolution_failed"
    assert "TOPSECRET" not in str(audit.events[0].payload)


@pytest.mark.asyncio
async def test_explain_tool_call_audits_async_profile_resolver_failure() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=_raise_async_resolver_failure,
        audit_store=audit,
        actor_id="operator-1",
    )

    with pytest.raises(GatewayPolicyExplainError) as exc_info:
        await service.explain_tool_call(
            PolicyExplainRequest(profile_id="backend-engineer", tool_name="fs.patch")
        )

    assert exc_info.value.reason_code == "profile_resolution_failed"
    assert audit.events[0].payload["reason_code"] == "profile_resolution_failed"


@pytest.mark.asyncio
async def test_explain_tool_call_resolver_failure_preserves_audit_store_failure() -> None:
    service = GatewayPolicyExplainService(
        profile_resolver=_raise_resolver_failure,
        audit_store=_MemoryAuditStore(fail=True),
        actor_id="operator-1",
    )

    with pytest.raises(GatewayPolicyExplainError) as exc_info:
        await service.explain_tool_call(
            PolicyExplainRequest(profile_id="backend-engineer", tool_name="fs.patch")
        )

    assert exc_info.value.reason_code == "audit_store_unavailable"


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
async def test_preview_profile_tools_audits_sync_profile_resolver_failure() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=_raise_resolver_failure,
        audit_store=audit,
        actor_id="operator-1",
    )

    with pytest.raises(GatewayPolicyExplainError) as exc_info:
        await service.preview_profile_tools(
            ProfileToolPreviewRequest(profile_id="backend-engineer")
        )

    assert exc_info.value.reason_code == "profile_resolution_failed"
    assert audit.events[0].event_type == "policy.preview_tools.requested"
    assert audit.events[0].payload["reason_code"] == "profile_resolution_failed"


@pytest.mark.asyncio
async def test_preview_profile_tools_audits_async_profile_resolver_failure() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=_raise_async_resolver_failure,
        audit_store=audit,
        actor_id="operator-1",
    )

    with pytest.raises(GatewayPolicyExplainError) as exc_info:
        await service.preview_profile_tools(
            ProfileToolPreviewRequest(profile_id="backend-engineer")
        )

    assert exc_info.value.reason_code == "profile_resolution_failed"
    assert audit.events[0].payload["reason_code"] == "profile_resolution_failed"


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
    assert dumped["tools"][0]["outcome"] == "allow"
    assert dumped["tools"][0]["installation_status"] == "unknown"
    assert dumped["tools"][0]["runtime_availability"] == "unknown"
    assert "final_outcome" not in dumped["tools"][0]
    assert "tools" not in audit.events[0].payload
    assert audit.events[0].payload["summary"] == {
        "total": 2,
        "allow": 1,
        "ask": 0,
        "deny": 1,
        "visible": 1,
        "hidden": 1,
        "deferred": 0,
        "installed": 0,
        "not_installed": 0,
        "unknown_installation": 2,
    }


@pytest.mark.asyncio
async def test_preview_admin_catalog_includes_denied_installed_tools() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
        actor_id="operator-1",
        admin_tool_catalog_provider=lambda profile: [
            AdminToolCatalogEntry(
                tool_id="fs.patch",
                category="filesystem",
                installation_status="installed",
            ),
            AdminToolCatalogEntry(
                tool_id="shell.exec",
                category="shell",
                installation_status="installed",
            ),
        ],
    )

    response = await service.preview_profile_tools(
        ProfileToolPreviewRequest(profile_id="backend-engineer")
    )

    tools_by_name = {entry.tool_name: entry for entry in response.tools}
    assert response.degraded is False
    assert tools_by_name["fs.patch"].outcome == "allow"
    assert tools_by_name["fs.patch"].visibility == "visible"
    assert tools_by_name["fs.patch"].installation_status == "installed"
    assert tools_by_name["shell.exec"].outcome == "deny"
    assert tools_by_name["shell.exec"].visibility == "hidden"
    assert tools_by_name["shell.exec"].installation_status == "installed"
    assert response.summary.total == 2
    assert response.summary.installed == 2
    assert response.summary.deny == 1
    assert response.summary.hidden == 1


@pytest.mark.asyncio
async def test_preview_installed_catalog_fallback_includes_denied_installed_tools() -> None:
    audit = _MemoryAuditStore()
    backend_tools = _installed_preview_backend_tools()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _profile(),
        audit_store=audit,
        actor_id="operator-1",
        installed_tool_catalog=backend_tools,
    )

    response = await service.preview_profile_tools(
        ProfileToolPreviewRequest(profile_id="backend-engineer")
    )

    tools_by_name = {entry.tool_name: entry for entry in response.tools}
    assert response.degraded is False
    assert tools_by_name["fs.patch"].outcome == "allow"
    assert tools_by_name["fs.patch"].visibility == "visible"
    assert tools_by_name["fs.patch"].installation_status == "installed"
    assert tools_by_name["shell.exec"].outcome == "deny"
    assert tools_by_name["shell.exec"].visibility == "hidden"
    assert tools_by_name["shell.exec"].installation_status == "installed"

    model_catalog = list_profile_tools(_profile(), backend_tools)
    model_tool_ids = {tool["tool_id"] for tool in model_catalog["tools"]}
    assert model_tool_ids == {"fs.patch"}
    search_tool_ids = {
        tool["tool_id"]
        for tool in search_profile_tools(
            _profile(),
            backend_tools,
            query="shell",
            limit=10,
        )
    }
    assert "shell.exec" not in search_tool_ids


@pytest.mark.asyncio
async def test_preview_filters_denied_and_recommendation_rows() -> None:
    audit = _MemoryAuditStore()
    profile = _profile_with_preview_recommendations()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: profile,
        audit_store=audit,
        actor_id="operator-1",
        installed_tool_catalog=_filter_preview_backend_tools(),
    )

    allowed_response = await service.preview_profile_tools(
        ProfileToolPreviewRequest(
            profile_id="backend-engineer",
            include_denied=False,
        )
    )
    assert "shell.exec" not in {entry.tool_name for entry in allowed_response.tools}

    installed_response = await service.preview_profile_tools(
        ProfileToolPreviewRequest(
            profile_id="backend-engineer",
            include_recommendations=False,
        )
    )
    assert "fs.inspect" not in {entry.tool_name for entry in installed_response.tools}
    assert all(
        entry.installation_status == "installed"
        for entry in installed_response.tools
    )


@pytest.mark.asyncio
async def test_preview_category_filter_paginates_after_filtering() -> None:
    audit = _MemoryAuditStore()
    profile = _profile_with_preview_recommendations()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: profile,
        audit_store=audit,
        actor_id="operator-1",
        installed_tool_catalog=_filter_preview_backend_tools(),
    )

    first_page = await service.preview_profile_tools(
        ProfileToolPreviewRequest(
            profile_id="backend-engineer",
            category="filesystem",
            limit=1,
        )
    )
    second_page = await service.preview_profile_tools(
        ProfileToolPreviewRequest(
            profile_id="backend-engineer",
            category="filesystem",
            limit=1,
            cursor=first_page.next_cursor,
        )
    )

    assert [entry.tool_name for entry in first_page.tools] == ["fs.patch"]
    assert first_page.next_cursor == "1"
    assert first_page.truncated is True
    assert [entry.tool_name for entry in second_page.tools] == ["fs.read"]
    assert second_page.next_cursor == "2"
    assert second_page.truncated is True

    all_filesystem = await service.preview_profile_tools(
        ProfileToolPreviewRequest(
            profile_id="backend-engineer",
            category="filesystem",
        )
    )
    assert [entry.tool_name for entry in all_filesystem.tools] == [
        "fs.patch",
        "fs.read",
        "fs.inspect",
    ]


@pytest.mark.asyncio
async def test_preview_defers_argument_sensitive_permission_rules() -> None:
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

    response = await service.preview_profile_tools(
        ProfileToolPreviewRequest(profile_id="backend-engineer")
    )

    entry = next(tool for tool in response.tools if tool.tool_name == "fs.patch")
    assert entry.outcome == "ask"
    assert entry.visibility == "deferred"
    assert entry.call_state == "deferred"
    assert entry.reason_code == "argument_sensitive_policy"
    assert response.summary.ask == 1
    assert response.summary.deferred == 1
    assert response.summary.unknown_installation == 1


@pytest.mark.asyncio
async def test_preview_profile_tools_paginates_and_reports_next_cursor() -> None:
    audit = _MemoryAuditStore()
    tool_names = [f"tool.{index:03d}" for index in range(5)]
    profile = MCPProfile(
        id="catalog-profile",
        name="Catalog Profile",
        policy_document=ProfilePolicy(allowed_tools=tool_names),
    )
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: profile,
        audit_store=audit,
        actor_id="operator-1",
        catalog_provider=lambda: [{"tool_id": tool_name} for tool_name in tool_names],
    )

    response = await service.preview_profile_tools(
        ProfileToolPreviewRequest(
            profile_id="catalog-profile",
            limit=2,
            cursor="2",
        )
    )

    assert [tool.tool_name for tool in response.tools] == ["tool.002", "tool.003"]
    assert response.summary.total == 2
    assert response.summary.visible == 2
    assert response.summary.unknown_installation == 2
    assert response.truncated is True
    assert response.next_cursor == "4"
    assert audit.events[0].payload["truncated"] is True
    assert audit.events[0].payload["next_cursor"] == "4"


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
