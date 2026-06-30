from __future__ import annotations

import pytest


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_prepare_startup_pre_core_runs_helpers_in_order_and_returns_defer_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import startup_pre_core

    calls: list[tuple[str, dict[str, object]]] = []

    def _record_test_runtime(**kwargs):
        calls.append(("test_runtime", kwargs))

    def _record_transition_gate(**kwargs):
        calls.append(("transition_gate", kwargs))

    async def _record_preflight(**kwargs):
        calls.append(("preflight", kwargs))

    def _record_heavy_policy(**kwargs):
        calls.append(("heavy_policy", kwargs))
        return True

    def _record_bg_tasks(**kwargs):
        calls.append(("bg_tasks", kwargs))

    def _record_prompts_close_worker(**kwargs):
        calls.append(("prompts", kwargs))

    def _record_mcp_validation(**kwargs):
        calls.append(("mcp", kwargs))

    def _record_acp_validation(**kwargs):
        calls.append(("acp", kwargs))

    def _record_content_backend(**kwargs):
        calls.append(("content_backend", kwargs))

    def _record_claims_validation(**kwargs):
        calls.append(("claims_validation", kwargs))

    def _record_evaluations_warmup(**kwargs):
        calls.append(("evaluations", kwargs))

    def _record_telemetry(**kwargs):
        calls.append(("telemetry", kwargs))

    def _record_sentry(**kwargs):
        calls.append(("sentry", kwargs))

    monkeypatch.setattr(startup_pre_core, "_validate_startup_test_runtime", _record_test_runtime)
    monkeypatch.setattr(startup_pre_core, "_apply_startup_transition_gate", _record_transition_gate)
    monkeypatch.setattr(startup_pre_core, "_run_startup_preflight_checks", _record_preflight)
    monkeypatch.setattr(startup_pre_core, "_resolve_deferred_heavy_startup", _record_heavy_policy)
    monkeypatch.setattr(startup_pre_core, "_prepare_startup_bg_tasks", _record_bg_tasks)
    monkeypatch.setattr(startup_pre_core, "_start_prompts_close_worker", _record_prompts_close_worker)
    monkeypatch.setattr(
        startup_pre_core,
        "_validate_startup_mcp_configuration",
        _record_mcp_validation,
    )
    monkeypatch.setattr(
        startup_pre_core,
        "_validate_startup_acp_configuration",
        _record_acp_validation,
    )
    monkeypatch.setattr(
        startup_pre_core,
        "_validate_startup_content_backend",
        _record_content_backend,
    )
    monkeypatch.setattr(
        startup_pre_core,
        "_validate_startup_claims_prompt_validation",
        _record_claims_validation,
    )
    monkeypatch.setattr(
        startup_pre_core,
        "_warm_lazy_evaluations_managers",
        _record_evaluations_warmup,
    )
    monkeypatch.setattr(
        startup_pre_core,
        "_initialize_startup_telemetry",
        _record_telemetry,
    )
    monkeypatch.setattr(startup_pre_core, "_initialize_startup_sentry", _record_sentry)

    defer_heavy = await startup_pre_core.prepare_startup_pre_core(
        app="app",
        logger="logger",
        readiness_state="readiness-state",
        shared_is_truthy="shared-is-truthy",
        route_enabled="route-enabled",
        get_mcp_config="get-mcp-config",
        validate_mcp_config="validate-mcp-config",
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
        test_mode=True,
    )

    assert defer_heavy is True
    assert [name for name, _ in calls] == [
        "test_runtime",
        "transition_gate",
        "preflight",
        "heavy_policy",
        "bg_tasks",
        "prompts",
        "mcp",
        "acp",
        "content_backend",
        "claims_validation",
        "evaluations",
        "telemetry",
        "sentry",
    ]
    assert calls[0][1]["logger"] == "logger"
    assert calls[0][1]["import_exceptions"] == (ImportError,)
    assert calls[1][1]["app"] == "app"
    assert calls[1][1]["readiness_state"] == "readiness-state"
    assert calls[2][1]["startup_guard_exceptions"] == (RuntimeError,)
    assert calls[3][1]["shared_is_truthy"] == "shared-is-truthy"
    assert calls[4][1]["app"] == "app"
    assert calls[6][1]["get_mcp_config"] == "get-mcp-config"
    assert calls[6][1]["validate_mcp_config"] == "validate-mcp-config"
    assert calls[7][1]["route_enabled"] == "route-enabled"
    assert calls[10][1]["test_mode"] is True
    assert calls[11][1]["app"] == "app"
    assert calls[12][1]["import_exceptions"] == (ImportError,)
