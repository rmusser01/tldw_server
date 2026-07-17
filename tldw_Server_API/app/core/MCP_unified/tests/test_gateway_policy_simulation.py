"""Tests for the gateway policy simulation harness and shared subject extraction."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


def _profile(
    profile_id: str,
    *,
    allowed_tools: list[str],
    permission_rules: list[Any] | None = None,
    path_scopes: list[dict[str, Any]] | None = None,
) -> Any:
    from mcp_unified.profiles.models import MCPProfile, ProfilePolicy

    return MCPProfile(
        id=profile_id,
        name=f"Profile {profile_id}",
        policy_document=ProfilePolicy(
            allowed_tools=allowed_tools,
            permission_rules=permission_rules or [],
        ),
        path_scopes=path_scopes or [],
    )


def test_subjects_module_extracts_permission_rule_subjects() -> None:
    from mcp_unified.profiles.subjects import extract_permission_rule_subjects

    subjects = extract_permission_rule_subjects(
        "fs.read_text",
        {
            "path": "docs/a.txt",
            "url": "https://example.com/x",
            "argv": ["git", "status"],
        },
    )
    pairs = {(subject_type, value) for subject_type, value, _argv in subjects}
    assert ("tool", "fs.read_text") in pairs
    assert ("path", "docs/a.txt") in pairs
    assert ("domain", "https://example.com/x") in pairs
    assert ("command", "git status") in pairs

    skill_subjects = extract_permission_rule_subjects(
        "skills.render",
        {"skill_name": "Review-Paper", "arguments": "--formal /* example */"},
    )
    skill_pairs = {(subject_type, value) for subject_type, value, _argv in skill_subjects}
    assert ("skill", "review-paper") in skill_pairs
    assert all(
        subject_type != "skill"
        for subject_type, _, _ in extract_permission_rule_subjects(
            "skills.get", {"name": "Review-Paper"}
        )
    )


def test_skill_permission_subject_extraction_ignores_invalid_values() -> None:
    from mcp_unified.profiles.subjects import extract_permission_rule_subjects

    for arguments in (
        {"skill_name": "   "},
        {"skill_name": 42},
        {"skill_name": {"name": "Review-Paper"}},
        {"skill_name": ["Review-Paper"]},
        {"name": "Review-Paper"},
    ):
        assert all(
            subject_type != "skill"
            for subject_type, _, _ in extract_permission_rule_subjects("skills.render", arguments)
        )


def test_skill_permission_subject_extraction_requires_exact_argument_key() -> None:
    from mcp_unified.profiles.subjects import extract_permission_rule_subjects

    for key in ("SKILL_NAME", " skill_name "):
        assert all(
            subject_type != "skill"
            for subject_type, _, _ in extract_permission_rule_subjects(
                "skills.render", {key: "Review-Paper"}
            )
        )

    subjects = extract_permission_rule_subjects(
        "skills.render", {"skill_name": "Review-Paper"}
    )
    assert ("skill", "review-paper", None) in subjects


def test_subjects_module_enforces_extraction_limits() -> None:
    from mcp_unified.profiles.subjects import (
        MAX_PERMISSION_SUBJECTS,
        PermissionSubjectLimitError,
        extract_permission_rule_subjects,
    )

    with pytest.raises(PermissionSubjectLimitError):
        extract_permission_rule_subjects(
            "fs.read_text",
            {"paths": [f"docs/file-{index}.txt" for index in range(MAX_PERMISSION_SUBJECTS + 1)]},
        )


def test_skill_permission_subject_extraction_enforces_value_limit() -> None:
    from mcp_unified.profiles.subjects import (
        MAX_SUBJECT_VALUE_LENGTH,
        PermissionSubjectLimitError,
        extract_permission_rule_subjects,
    )

    with pytest.raises(PermissionSubjectLimitError, match="max_subject_value_length"):
        extract_permission_rule_subjects(
            "skills.render",
            {"skill_name": "a" * (MAX_SUBJECT_VALUE_LENGTH + 1)},
        )


def test_simulation_reports_allowed_call() -> None:
    from mcp_unified.gateway.policy_simulation import simulate_tool_call_policy

    profile = _profile(
        "reviewer",
        allowed_tools=["fs.read_text"],
        permission_rules=[{"pattern": "Read(docs/private/**)", "outcome": "deny"}],
    )
    result = simulate_tool_call_policy(
        profile,
        "fs.read_text",
        {"path": "docs/public/notes.txt"},
    )
    assert result["overall"]["status"] == "allowed"
    assert result["legacy_policy"]["status"] == "resolved"
    subject_types = {subject["subject_type"] for subject in result["subjects"]}
    assert {"tool", "path"} <= subject_types


def test_simulation_reports_denied_path_rule() -> None:
    from mcp_unified.gateway.policy_simulation import simulate_tool_call_policy

    profile = _profile(
        "reviewer",
        allowed_tools=["fs.read_text"],
        permission_rules=[
            {
                "pattern": "Read(docs/private/**)",
                "outcome": "deny",
                "reason_code": "private_docs_denied",
            }
        ],
    )
    result = simulate_tool_call_policy(
        profile,
        "fs.read_text",
        {"path": "docs/private/secret.txt"},
    )
    assert result["overall"]["status"] == "denied"
    assert result["overall"]["reason_code"] == "private_docs_denied"
    denied_subjects = [
        subject for subject in result["subjects"] if subject["outcome"] == "deny"
    ]
    assert denied_subjects
    assert denied_subjects[0]["subject_type"] == "path"
    assert denied_subjects[0]["matched_rules"]


def test_simulation_reports_legacy_tool_denial() -> None:
    from mcp_unified.gateway.policy_simulation import simulate_tool_call_policy

    profile = _profile("reviewer", allowed_tools=["fs.read_text"])
    result = simulate_tool_call_policy(profile, "admin.delete", {})
    assert result["overall"]["status"] == "denied"
    assert result["legacy_policy"]["status"] != "resolved"


def test_simulation_ask_rule_blocks_then_lease_allows() -> None:
    from mcp_unified.gateway.policy_simulation import simulate_tool_call_policy
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    profile = _profile(
        "researcher",
        allowed_tools=["web.fetch"],
        permission_rules=[{"pattern": "WebFetch(example.com)", "outcome": "ask"}],
    )
    grant_store = InMemoryPolicyGrantStore()
    arguments = {"url": "https://example.com/private"}

    blocked = simulate_tool_call_policy(
        profile,
        "web.fetch",
        arguments,
        policy_grant_store=grant_store,
    )
    assert blocked["overall"]["status"] == "approval_required"

    grant_store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="example.com",
        ttl_seconds=900,
    )
    allowed = simulate_tool_call_policy(
        profile,
        "web.fetch",
        arguments,
        policy_grant_store=grant_store,
    )
    assert allowed["overall"]["status"] == "allowed"
    assert allowed["approval_grant_markers"]


def test_simulation_skill_permission_applies_deny_allow_and_unrelated_rules() -> None:
    from mcp_unified.gateway.policy_simulation import simulate_tool_call_policy

    denied_profile = _profile(
        "reviewer",
        allowed_tools=["skills.render"],
        permission_rules=[{"pattern": "Skill(secret-*)", "outcome": "deny"}],
    )
    allowed_profile = _profile(
        "reviewer",
        allowed_tools=["skills.render"],
        permission_rules=[{"pattern": "Skill(review-*)", "outcome": "allow"}],
    )
    unrelated_profile = _profile(
        "reviewer",
        allowed_tools=["skills.render"],
        permission_rules=[{"pattern": "Skill(secret-*)", "outcome": "deny"}],
    )

    denied = simulate_tool_call_policy(
        denied_profile,
        "skills.render",
        {"skill_name": "secret-plan"},
    )
    allowed = simulate_tool_call_policy(
        allowed_profile,
        "skills.render",
        {"skill_name": "Review-Paper"},
    )
    unrelated = simulate_tool_call_policy(
        unrelated_profile,
        "skills.render",
        {"skill_name": "review-paper"},
    )

    assert denied["overall"]["status"] == "denied"
    assert allowed["overall"]["status"] == "allowed"
    assert unrelated["overall"]["status"] == "allowed"


def test_simulation_skill_permission_approval_lease_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mcp_unified.policy_grants.memory as memory_grants
    from mcp_unified.gateway.policy_simulation import simulate_tool_call_policy
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    profile = _profile(
        "reviewer",
        allowed_tools=["skills.render"],
        permission_rules=[{"pattern": "Skill(review-*)", "outcome": "ask"}],
    )
    grant_store = InMemoryPolicyGrantStore()
    arguments = {"skill_name": "Review-Paper"}

    blocked = simulate_tool_call_policy(
        profile,
        "skills.render",
        arguments,
        policy_grant_store=grant_store,
    )
    assert blocked["overall"]["status"] == "approval_required"

    monkeypatch.setattr(memory_grants.time, "time", lambda: 1_000.0)
    grant_store.create_grant(
        profile_id="reviewer",
        grant_type="approval",
        subject_type="skill",
        value="REVIEW-PAPER",
        ttl_seconds=10,
    )
    allowed = simulate_tool_call_policy(
        profile,
        "skills.render",
        arguments,
        policy_grant_store=grant_store,
    )
    assert allowed["overall"]["status"] == "allowed"
    assert allowed["approval_grant_markers"]

    monkeypatch.setattr(memory_grants.time, "time", lambda: 1_011.0)
    expired = simulate_tool_call_policy(
        profile,
        "skills.render",
        arguments,
        policy_grant_store=grant_store,
    )
    assert expired["overall"]["status"] == "approval_required"


def test_simulation_includes_merged_ttl_path_grants() -> None:
    from mcp_unified.gateway.policy_simulation import simulate_tool_call_policy
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    profile = _profile(
        "reviewer",
        allowed_tools=["fs.read_text"],
        path_scopes=[{"prefix": "docs/manuals", "actions": ["read"], "effect": "allow"}],
    )
    grant_store = InMemoryPolicyGrantStore()
    grant_store.create_grant(
        profile_id="reviewer",
        grant_type="path",
        subject_type="path",
        value="docs/scratch",
        actions=("read", "write"),
        ttl_seconds=900,
    )

    result = simulate_tool_call_policy(
        profile,
        "fs.read_text",
        {"path": "docs/scratch/notes.txt"},
        policy_grant_store=grant_store,
    )
    assert result["overall"]["status"] == "allowed"
    sources = {scope.get("source") for scope in result["path_scopes"]}
    assert "ttl_grant" in sources
    prefixes = {scope["prefix"] for scope in result["path_scopes"]}
    assert {"docs/manuals", "docs/scratch"} <= prefixes


def test_simulation_matches_runtime_outcomes() -> None:
    from mcp_unified.gateway.policy_simulation import simulate_tool_call_policy
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayPolicyDenied, GatewayRequestContext
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore
    from mcp_unified.profiles.store import InMemoryProfileStore

    class _ParityBackend:
        name = "parity-backend"
        version = "0.0-test"

        async def list_tools(self, context: Any) -> list[dict[str, Any]]:
            return []

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
            context: Any,
        ) -> dict[str, Any]:
            return {"content": [{"type": "text", "text": "ok"}]}

    profile = _profile(
        "reviewer",
        allowed_tools=["fs.read_text", "web.fetch"],
        permission_rules=[
            {
                "pattern": "Read(docs/private/**)",
                "outcome": "deny",
                "reason_code": "private_docs_denied",
            },
            {"pattern": "WebFetch(example.com)", "outcome": "ask"},
        ],
    )
    grant_store = InMemoryPolicyGrantStore()
    grant_store.create_grant(
        profile_id="reviewer",
        grant_type="approval",
        subject_type="domain",
        value="approved.example.org",
        ttl_seconds=900,
    )
    profile.policy_document.permission_rules.append(
        {"pattern": "WebFetch(approved.example.org)", "outcome": "ask"}
    )
    runtime = ProfileAwareGatewayRuntime(
        _ParityBackend(),
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="reviewer",
        policy_grant_store=grant_store,
    )

    cases: list[tuple[str, dict[str, Any]]] = [
        ("fs.read_text", {"path": "docs/public/notes.txt"}),
        ("fs.read_text", {"path": "docs/private/secret.txt"}),
        ("web.fetch", {"url": "https://example.com/private"}),
        ("web.fetch", {"url": "https://approved.example.org/data"}),
        ("admin.delete", {}),
    ]

    async def _runtime_status(tool_name: str, arguments: dict[str, Any]) -> str:
        context = GatewayRequestContext(request_id=f"parity-{tool_name}")
        try:
            await runtime.call_tool(tool_name, arguments, context)
        except GatewayPolicyDenied as exc:
            return exc.status
        return "allowed"

    for tool_name, arguments in cases:
        simulated = simulate_tool_call_policy(
            profile,
            tool_name,
            arguments,
            policy_grant_store=grant_store,
        )
        runtime_status = asyncio.run(_runtime_status(tool_name, arguments))
        assert simulated["overall"]["status"] == runtime_status, (
            f"parity mismatch for {tool_name} {arguments}: "
            f"simulated={simulated['overall']['status']} runtime={runtime_status}"
        )


def test_simulation_unmatched_subjects_report_no_reason_code() -> None:
    from mcp_unified.gateway.policy_simulation import simulate_tool_call_policy

    profile = _profile(
        "reviewer",
        allowed_tools=["fs.read_text"],
        permission_rules=[{"pattern": "Read(docs/private/**)", "outcome": "deny"}],
    )
    result = simulate_tool_call_policy(
        profile,
        "fs.read_text",
        {"path": "docs/public/notes.txt"},
    )
    assert result["overall"]["status"] == "allowed"
    for subject in result["subjects"]:
        assert subject["outcome"] == "allow"
        assert subject["reason_code"] is None
        assert subject["matched_rules"] == []
