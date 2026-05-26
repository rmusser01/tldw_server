"""Service-backed capability projection for Research Workspace."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any

from loguru import logger


LOCAL_PROVIDER_NAMES = frozenset(
    {
        "aphrodite",
        "custom_openai_api",
        "kobold",
        "llama",
        "llama.cpp",
        "mlx",
        "ollama",
        "ooba",
        "tabby",
        "vllm",
    }
)
UNHEALTHY_PROVIDER_STATES = frozenset(
    {"unhealthy", "circuit_open", "down", "failed"}
)
DEGRADED_PROVIDER_STATES = frozenset({"degraded"})
APPROVAL_MODES = frozenset(
    {"approval_required", "ask", "ask_on_broaden", "require_approval"}
)
PROVIDER_CONFIG_SPECS: tuple[tuple[str, str, str | None, str | None, str | None], ...] = (
    ("openai", "API", "openai_api_key", None, None),
    ("bedrock", "API", "bedrock_api_key", None, None),
    ("anthropic", "API", "anthropic_api_key", None, None),
    ("cohere", "API", "cohere_api_key", None, None),
    ("deepseek", "API", "deepseek_api_key", None, None),
    ("qwen", "API", "qwen_api_key", None, None),
    ("google", "API", "google_api_key", None, None),
    ("groq", "API", "groq_api_key", None, None),
    ("mistral", "API", "mistral_api_key", None, None),
    ("huggingface", "API", "huggingface_api_key", None, None),
    ("openrouter", "API", "openrouter_api_key", None, None),
    ("novita", "API", "novita_api_key", None, None),
    ("poe", "API", "poe_api_key", None, None),
    ("together", "API", "together_api_key", None, None),
    ("moonshot", "API", "moonshot_api_key", None, None),
    ("zai", "API", "zai_api_key", None, None),
    ("minimax", "API", "minimax_api_key", None, None),
    ("llama", "Local-API", None, "llama_api_IP", None),
    ("kobold", "Local-API", None, "kobold_api_IP", None),
    ("ooba", "Local-API", None, "ooba_api_IP", None),
    ("tabby", "Local-API", None, "tabby_api_IP", None),
    ("vllm", "Local-API", None, "vllm_api_IP", "vllm_model"),
    ("ollama", "Local-API", None, "ollama_api_IP", "ollama_model"),
    ("aphrodite", "Local-API", None, "aphrodite_api_IP", "aphrodite_model"),
    ("mlx", "MLX", None, None, "mlx_model_path"),
    (
        "custom_openai_api",
        "API",
        None,
        "custom_openai_api_ip",
        "custom_openai_api_model",
    ),
)


async def collect_workspace_service_capabilities(
    *,
    workspace_id: str,
    user_id: int | str | None,
) -> dict[str, Any]:
    """Collect MCP, ACP, sandbox, and provider readiness for a workspace."""
    mcp_policy, mcp_error = await _collect_mcp_policy(
        workspace_id=workspace_id,
        user_id=user_id,
    )
    acp_summary, acp_error = await _to_thread_result(_collect_acp_summary)
    sandbox_runtimes, sandbox_error = await _to_thread_result(_collect_sandbox_runtimes)
    provider_health, provider_error = await _to_thread_result(_collect_provider_health)

    return build_workspace_service_capability_projection(
        workspace_id=workspace_id,
        mcp_policy=mcp_policy,
        acp_summary=acp_summary,
        sandbox_runtimes=sandbox_runtimes,
        provider_health=provider_health,
        errors={
            "mcp": mcp_error,
            "acp": acp_error,
            "sandbox": sandbox_error,
            "provider": provider_error,
        },
    )


def build_workspace_service_capability_projection(
    *,
    workspace_id: str,
    mcp_policy: Mapping[str, Any] | None = None,
    acp_summary: Mapping[str, Any] | None = None,
    sandbox_runtimes: list[Mapping[str, Any]] | None = None,
    provider_health: Mapping[str, Any] | None = None,
    errors: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Build the dynamic service capability slice used by workspace endpoints."""
    error_map = dict(errors or {})
    services = {
        "mcp": _mcp_service(
            mcp_policy,
            workspace_id=workspace_id,
            error=error_map.get("mcp"),
        ),
        "acp": _acp_service(
            acp_summary,
            mcp_policy=mcp_policy,
            error=error_map.get("acp"),
        ),
        "sandbox": _sandbox_service(sandbox_runtimes, error=error_map.get("sandbox")),
        "provider": _provider_service(provider_health, error=error_map.get("provider")),
    }
    return {
        "workspace_services": services,
        "allowed_actions": {
            "run_mcp_tools": _service_action(
                services["mcp"],
                blocked_reason="mcp_not_available",
            ),
            "use_acp_agents": _service_action(
                services["acp"],
                blocked_reason="acp_not_available",
            ),
            "use_sandbox": _service_action(
                services["sandbox"],
                blocked_reason="sandbox_not_available",
            ),
        },
    }


async def _collect_mcp_policy(
    *,
    workspace_id: str,
    user_id: int | str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from tldw_Server_API.app.services.mcp_hub_policy_resolver import (
            get_mcp_hub_policy_resolver,
        )

        resolver = await get_mcp_hub_policy_resolver()
        policy = await resolver.resolve_for_context(
            user_id=user_id,
            metadata={
                "mcp_policy_context_enabled": True,
                "workspace_id": workspace_id,
                "workspace_ids": [workspace_id],
            },
        )
        return dict(policy or {}), None
    except Exception as exc:  # noqa: BLE001 - capability probes fail closed.
        logger.debug("Workspace MCP capability resolution failed: {}", exc)
        return None, "mcp_policy_resolution_failed"


def _collect_acp_summary() -> dict[str, Any]:
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import (
        get_agent_registry,
    )
    from tldw_Server_API.app.core.Agent_Client_Protocol.execution_health import (
        summarize_execution_health,
    )

    agents = get_agent_registry().get_available_agents()
    return summarize_execution_health(sessions=[], agents=agents)


def _collect_sandbox_runtimes() -> list[dict[str, Any]]:
    from tldw_Server_API.app.core.Sandbox.service import SandboxService

    return list(SandboxService(enable_background_tasks=False).feature_discovery())


def _collect_provider_health() -> dict[str, Any]:
    from tldw_Server_API.app.core.Chat.provider_manager import get_provider_manager

    configured_providers = _detect_configured_llm_providers()
    provider_manager = get_provider_manager()
    if provider_manager is None:
        return {
            "initialized": False,
            "providers": [],
            "configured_providers": configured_providers,
            "health_report": {},
        }
    return {
        "initialized": True,
        "providers": list(getattr(provider_manager, "providers", []) or []),
        "configured_providers": configured_providers,
        "primary_provider": getattr(provider_manager, "primary_provider", None),
        "health_report": provider_manager.get_health_report(),
    }


async def _to_thread_result(callable_obj: Any) -> tuple[Any | None, str | None]:
    try:
        return await asyncio.to_thread(callable_obj), None
    except Exception as exc:  # noqa: BLE001 - capability probes fail closed.
        logger.debug("Workspace service capability probe failed: {}", exc)
        return None, f"{_callable_name(callable_obj)}_failed"


def _mcp_service(
    policy: Mapping[str, Any] | None,
    *,
    workspace_id: str,
    error: str | None,
) -> dict[str, Any]:
    if error:
        return _service("unknown", error, "mcp_hub")
    if not isinstance(policy, Mapping) or not bool(policy.get("enabled")):
        return _service("not_configured", "mcp_policy_not_configured", "mcp_hub")

    assignment_workspace_ids = _string_set(
        policy.get("selected_assignment_workspace_ids")
    )
    if assignment_workspace_ids and str(workspace_id) not in assignment_workspace_ids:
        return _service("blocked", "mcp_workspace_not_allowed", "mcp_hub")

    allowed_tools = _string_list(policy.get("allowed_tools"))
    if not allowed_tools:
        return _service("blocked", "mcp_tools_blocked", "mcp_hub")

    if _requires_approval(policy):
        return _service("needs_approval", "mcp_approval_required", "mcp_hub")

    unresolved_capabilities = _string_list(policy.get("unresolved_capabilities"))
    unsupported_requirements = _string_list(
        policy.get("unsupported_environment_requirements")
    )
    capability_warnings = _string_list(policy.get("capability_warnings"))
    if unresolved_capabilities or unsupported_requirements:
        return _service("blocked", "mcp_capability_unresolved", "mcp_hub")
    if capability_warnings:
        return _service("degraded", "mcp_capability_warnings", "mcp_hub")
    return _service("available", None, "mcp_hub")


def _acp_service(
    summary: Mapping[str, Any] | None,
    *,
    mcp_policy: Mapping[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    if error:
        return _service("unknown", error, "acp_workspace")
    if _requires_approval(mcp_policy):
        return _service("needs_approval", "acp_approval_required", "acp_workspace")
    if not isinstance(summary, Mapping):
        return _service("unknown", "acp_status_unknown", "acp_workspace")

    agents = summary.get("agents")
    if not isinstance(agents, list) or not agents:
        return _service("not_configured", "acp_no_agents_configured", "acp_workspace")

    configured_agents = [
        agent
        for agent in agents
        if isinstance(agent, Mapping) and bool(agent.get("is_configured"))
    ]
    if not configured_agents:
        return _service("not_configured", "acp_no_agents_configured", "acp_workspace")
    if any(bool(agent.get("setup_blocked")) for agent in configured_agents):
        return _service("blocked", "acp_agent_setup_blocked", "acp_workspace")

    setup_health = summary.get("setup_health")
    agent_health = setup_health.get("agent") if isinstance(setup_health, Mapping) else None
    agent_status = str((agent_health or {}).get("status") or "").strip().lower()
    if agent_status == "blocked":
        return _service("blocked", "acp_agent_setup_blocked", "acp_workspace")

    compatibility = summary.get("compatibility")
    if isinstance(compatibility, Mapping) and compatibility.get("live_certification_required"):
        return _service("degraded", "acp_live_certification_required", "acp_workspace")

    return _service("available", None, "acp_workspace")


def _sandbox_service(
    runtimes: list[Mapping[str, Any]] | None,
    *,
    error: str | None,
) -> dict[str, Any]:
    if error:
        return _service("unknown", error, "sandbox_settings")
    if not runtimes:
        return _service(
            "not_configured",
            "sandbox_no_runtimes_discovered",
            "sandbox_settings",
        )
    if any(bool(runtime.get("available")) for runtime in runtimes):
        return _service("available", None, "sandbox_settings")

    reason = None
    for runtime in runtimes:
        reasons = _string_list(runtime.get("normalized_reasons")) or _string_list(
            runtime.get("reasons")
        )
        if reasons:
            reason = reasons[0]
            break
    return _service("blocked", reason or "sandbox_runtime_unavailable", "sandbox_settings")


def _provider_service(
    health: Mapping[str, Any] | None,
    *,
    error: str | None,
) -> dict[str, Any]:
    if error:
        return _service("unknown", error, "model_settings")
    if not isinstance(health, Mapping):
        return _service("not_configured", "provider_not_configured", "model_settings")

    has_configured_subset = "configured_providers" in health
    providers = (
        _string_list(health.get("configured_providers"))
        if has_configured_subset
        else _string_list(health.get("providers"))
    )
    if not providers:
        return _service("not_configured", "provider_not_configured", "model_settings")
    if not bool(health.get("initialized")):
        return _service("unknown", "provider_health_unknown", "model_settings")

    report = health.get("health_report")
    if isinstance(report, Mapping):
        provider_states = _provider_health_states(report, providers)
        if provider_states and all(
            state in UNHEALTHY_PROVIDER_STATES for state in provider_states
        ):
            return _service("blocked", "provider_unavailable", "model_settings")
        if any(state in UNHEALTHY_PROVIDER_STATES for state in provider_states):
            return _service("degraded", "provider_health_degraded", "model_settings")
        if any(state in DEGRADED_PROVIDER_STATES for state in provider_states):
            return _service("degraded", "provider_health_degraded", "model_settings")

    if not any(_is_local_provider(provider) for provider in providers):
        return _service("degraded", "external_provider_only", "model_settings")
    return _service("available", None, "model_settings")


def _requires_approval(policy: Mapping[str, Any] | None) -> bool:
    if not isinstance(policy, Mapping):
        return False
    approval_mode = str(policy.get("approval_mode") or "").strip().lower()
    if approval_mode in APPROVAL_MODES:
        return True
    if policy.get("approval_policy_id") is not None:
        return True
    document = policy.get("policy_document")
    if isinstance(document, Mapping):
        document_mode = str(document.get("approval_mode") or "").strip().lower()
        return document_mode in APPROVAL_MODES
    return False


def _service(state: str, reason_code: str | None, management_surface: str) -> dict[str, Any]:
    return {
        "state": state,
        "reason_code": reason_code,
        "management_surface": management_surface,
    }


def _service_action(service: Mapping[str, Any], *, blocked_reason: str) -> dict[str, Any]:
    if str(service.get("state") or "") == "available":
        return {"allowed": True, "reason_code": None}
    return {
        "allowed": False,
        "reason_code": service.get("reason_code") or blocked_reason,
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _string_set(value: Any) -> set[str]:
    return set(_string_list(value))


def _is_local_provider(provider: str) -> bool:
    normalized = provider.strip().lower().replace("-", "_")
    return normalized in LOCAL_PROVIDER_NAMES or normalized == "llama_cpp"


def _provider_health_entry(report: Mapping[str, Any], provider: str) -> Mapping[str, Any] | None:
    for alias in _provider_aliases(provider):
        entry = report.get(alias)
        if isinstance(entry, Mapping):
            return entry
    return None


def _provider_health_states(report: Mapping[str, Any], providers: list[str]) -> list[str]:
    states: list[str] = []
    for provider in providers:
        entry = _provider_health_entry(report, provider)
        if not isinstance(entry, Mapping):
            continue
        states.append(str(entry.get("status") or "").strip().lower())
    return states


def _provider_aliases(provider: str) -> tuple[str, ...]:
    normalized = provider.strip().lower()
    aliases = {
        normalized,
        normalized.replace("-", "_"),
        normalized.replace("_", "-"),
    }
    if normalized == "llama":
        aliases.update({"llama.cpp", "llama_cpp"})
    if normalized == "tabby":
        aliases.add("tabbyapi")
    return tuple(alias for alias in aliases if alias)


def _callable_name(callable_obj: Any) -> str:
    name = str(getattr(callable_obj, "__name__", "service_probe")).strip("_")
    return name or "service_probe"


def _detect_configured_llm_providers() -> list[str]:
    from tldw_Server_API.app.core.config import load_comprehensive_config

    config_parser = load_comprehensive_config()
    configured: list[str] = []
    for (
        provider_name,
        section_name,
        api_key_field,
        endpoint_field,
        model_field,
    ) in PROVIDER_CONFIG_SPECS:
        if provider_name != "mlx" and not config_parser.has_section(section_name):
            continue
        if api_key_field and _valid_config_value(
            _config_option(config_parser, section_name, api_key_field)
            or _provider_env_value(provider_name),
        ):
            configured.append(provider_name)
            continue
        if endpoint_field and _valid_config_value(
            _config_option(config_parser, section_name, endpoint_field),
        ):
            configured.append(provider_name)
            continue
        if model_field and _valid_config_value(
            _config_option(config_parser, section_name, model_field)
            or (os.getenv("MLX_MODEL_PATH") if provider_name == "mlx" else None),
        ):
            configured.append(provider_name)
    return configured


def _config_option(config_parser: Any, section_name: str, option_name: str) -> str | None:
    if not config_parser.has_section(section_name):
        return None
    if not config_parser.has_option(section_name, option_name):
        return None
    return config_parser.get(section_name, option_name, fallback="")


def _provider_env_value(provider_name: str) -> str | None:
    normalized = provider_name.upper().replace(".", "_").replace("-", "_")
    if normalized.endswith("_API"):
        normalized = normalized[: -len("_API")]
    return os.getenv(f"{normalized}_API_KEY")


def _valid_config_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    trimmed = value.strip()
    if not trimmed:
        return False
    if trimmed.startswith("<") and trimmed.endswith(">"):
        return False
    return trimmed.upper() not in {
        "YOUR_API_KEY_HERE",
        "CHANGE_ME",
        "CHANGE-ME",
        "PLACEHOLDER",
    }
