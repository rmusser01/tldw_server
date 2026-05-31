from __future__ import annotations

from tldw_Server_API.app.core.Workspaces.service_capabilities import (
    build_workspace_service_capability_projection,
)
from tldw_Server_API.app.core.Workspaces.status_projection import (
    build_workspace_capability_projection,
)


def test_service_projection_reports_available_local_research_stack() -> None:
    projection = build_workspace_service_capability_projection(
        workspace_id="ws-1",
        mcp_policy={
            "enabled": True,
            "allowed_tools": ["knowledge.search"],
            "selected_assignment_workspace_ids": ["ws-1"],
        },
        acp_summary={
            "agents": [
                {
                    "agent_type": "codex",
                    "is_configured": True,
                    "setup_blocked": False,
                }
            ],
            "setup_health": {"agent": {"status": "ready"}},
            "compatibility": {"live_certification_required": False},
        },
        sandbox_runtimes=[{"name": "docker", "available": True}],
        provider_health={
            "initialized": True,
            "providers": ["llama.cpp"],
            "health_report": {"llama.cpp": {"status": "healthy"}},
        },
        errors={},
    )

    assert projection["workspace_services"] == {
        "mcp": {
            "state": "available",
            "reason_code": None,
            "management_surface": "mcp_hub",
        },
        "acp": {
            "state": "available",
            "reason_code": None,
            "management_surface": "acp_workspace",
        },
        "sandbox": {
            "state": "available",
            "reason_code": None,
            "management_surface": "sandbox_settings",
        },
        "provider": {
            "state": "available",
            "reason_code": None,
            "management_surface": "model_settings",
        },
    }
    assert projection["allowed_actions"] == {
        "run_mcp_tools": {"allowed": True, "reason_code": None},
        "use_acp_agents": {"allowed": True, "reason_code": None},
        "use_sandbox": {"allowed": True, "reason_code": None},
    }


def test_service_projection_marks_mcp_and_acp_as_needing_approval() -> None:
    projection = build_workspace_service_capability_projection(
        workspace_id="ws-approval",
        mcp_policy={
            "enabled": True,
            "allowed_tools": ["filesystem.read"],
            "approval_mode": "ask",
        },
        acp_summary={
            "agents": [
                {
                    "agent_type": "codex",
                    "is_configured": True,
                    "setup_blocked": False,
                }
            ],
            "setup_health": {"agent": {"status": "ready"}},
        },
        sandbox_runtimes=[{"name": "docker", "available": True}],
        provider_health={
            "initialized": True,
            "providers": ["ollama"],
            "health_report": {"ollama": {"status": "healthy"}},
        },
        errors={},
    )

    assert projection["workspace_services"]["mcp"] == {
        "state": "needs_approval",
        "reason_code": "mcp_approval_required",
        "management_surface": "mcp_hub",
    }
    assert projection["workspace_services"]["acp"] == {
        "state": "needs_approval",
        "reason_code": "acp_approval_required",
        "management_surface": "acp_workspace",
    }
    assert projection["allowed_actions"]["run_mcp_tools"] == {
        "allowed": False,
        "reason_code": "mcp_approval_required",
    }
    assert projection["allowed_actions"]["use_acp_agents"] == {
        "allowed": False,
        "reason_code": "acp_approval_required",
    }


def test_service_projection_fails_closed_for_resolver_and_runtime_failures() -> None:
    projection = build_workspace_service_capability_projection(
        workspace_id="ws-fail",
        mcp_policy=None,
        acp_summary=None,
        sandbox_runtimes=[
            {
                "name": "docker",
                "available": False,
                "normalized_reasons": ["runtime_missing"],
            }
        ],
        provider_health={
            "initialized": True,
            "providers": ["llama.cpp"],
            "health_report": {"llama.cpp": {"status": "circuit_open"}},
        },
        errors={
            "mcp": "mcp_policy_resolution_failed",
            "acp": "acp_status_resolution_failed",
        },
    )

    assert projection["workspace_services"]["mcp"] == {
        "state": "unknown",
        "reason_code": "mcp_policy_resolution_failed",
        "management_surface": "mcp_hub",
    }
    assert projection["workspace_services"]["acp"] == {
        "state": "unknown",
        "reason_code": "acp_status_resolution_failed",
        "management_surface": "acp_workspace",
    }
    assert projection["workspace_services"]["sandbox"] == {
        "state": "blocked",
        "reason_code": "runtime_missing",
        "management_surface": "sandbox_settings",
    }
    assert projection["workspace_services"]["provider"] == {
        "state": "blocked",
        "reason_code": "provider_unavailable",
        "management_surface": "model_settings",
    }
    assert projection["allowed_actions"]["run_mcp_tools"] == {
        "allowed": False,
        "reason_code": "mcp_policy_resolution_failed",
    }


def test_service_projection_treats_unconfigured_acp_agents_as_not_configured() -> None:
    projection = build_workspace_service_capability_projection(
        workspace_id="ws-acp",
        mcp_policy={
            "enabled": True,
            "allowed_tools": ["knowledge.search"],
        },
        acp_summary={
            "agents": [
                {
                    "agent_type": "codex",
                    "is_configured": False,
                    "setup_blocked": True,
                    "primary_blocker": "runner_missing",
                }
            ],
            "setup_health": {"agent": {"status": "blocked"}},
        },
        sandbox_runtimes=[{"name": "docker", "available": True}],
        provider_health={
            "initialized": True,
            "providers": ["ollama"],
            "health_report": {"ollama": {"status": "healthy"}},
        },
        errors={},
    )

    assert projection["workspace_services"]["acp"] == {
        "state": "not_configured",
        "reason_code": "acp_no_agents_configured",
        "management_surface": "acp_workspace",
    }
    assert projection["allowed_actions"]["use_acp_agents"] == {
        "allowed": False,
        "reason_code": "acp_no_agents_configured",
    }


def test_service_projection_warns_when_only_external_providers_are_ready() -> None:
    projection = build_workspace_service_capability_projection(
        workspace_id="ws-external",
        mcp_policy={
            "enabled": True,
            "allowed_tools": ["knowledge.search"],
        },
        acp_summary={"agents": [], "setup_health": {"agent": {"status": "unknown"}}},
        sandbox_runtimes=[{"name": "docker", "available": True}],
        provider_health={
            "initialized": True,
            "providers": ["openai", "anthropic"],
            "health_report": {
                "openai": {"status": "healthy"},
                "anthropic": {"status": "healthy"},
            },
        },
        errors={},
    )

    assert projection["workspace_services"]["provider"] == {
        "state": "degraded",
        "reason_code": "external_provider_only",
        "management_surface": "model_settings",
    }


def test_service_projection_falls_back_to_runtime_providers_when_configured_subset_empty() -> None:
    projection = build_workspace_service_capability_projection(
        workspace_id="ws-runtime-provider",
        provider_health={
            "initialized": True,
            "providers": ["openai", "llama.cpp"],
            "configured_providers": [],
            "health_report": {
                "openai": {"status": "healthy"},
                "llama.cpp": {"status": "healthy"},
            },
        },
        errors={},
    )

    assert projection["workspace_services"]["provider"] == {
        "state": "available",
        "reason_code": None,
        "management_surface": "model_settings",
    }


def test_service_projection_reports_degraded_provider_health() -> None:
    projection = build_workspace_service_capability_projection(
        workspace_id="ws-degraded-provider",
        provider_health={
            "initialized": True,
            "providers": ["ollama"],
            "configured_providers": ["ollama"],
            "health_report": {"ollama": {"status": "degraded"}},
        },
        errors={},
    )

    assert projection["workspace_services"]["provider"] == {
        "state": "degraded",
        "reason_code": "provider_health_degraded",
        "management_surface": "model_settings",
    }


def test_workspace_capabilities_block_grounded_questions_when_provider_is_unavailable() -> None:
    capability_projection = build_workspace_capability_projection(
        workspace={"id": "ws-provider"},
        status_projection={
            "summary": {
                "total": 1,
                "selected": 1,
                "queryable": 1,
                "partially_queryable": 0,
                "processing": 0,
                "failed": 0,
                "missing": 0,
            }
        },
        service_capabilities={
            "workspace_services": {
                "provider": {
                    "state": "not_configured",
                    "reason_code": "provider_not_configured",
                    "management_surface": "model_settings",
                }
            }
        },
    )

    assert capability_projection["allowed_actions"]["ask_grounded_questions"] == {
        "allowed": False,
        "reason_code": "provider_not_configured",
    }


def test_workspace_capabilities_require_selected_queryable_sources() -> None:
    capability_projection = build_workspace_capability_projection(
        workspace={"id": "ws-selection"},
        status_projection={
            "summary": {
                "total": 2,
                "selected": 1,
                "queryable": 1,
                "partially_queryable": 0,
                "processing": 1,
                "failed": 0,
                "missing": 0,
            },
            "sources": [
                {
                    "id": "unselected-ready",
                    "selected": False,
                    "state": "queryable",
                },
                {
                    "id": "selected-indexing",
                    "selected": True,
                    "state": "indexing",
                },
            ],
        },
        service_capabilities={
            "workspace_services": {
                "provider": {
                    "state": "available",
                    "reason_code": None,
                    "management_surface": "model_settings",
                }
            }
        },
    )

    assert capability_projection["allowed_actions"]["ask_grounded_questions"] == {
        "allowed": False,
        "reason_code": "no_queryable_sources",
    }
