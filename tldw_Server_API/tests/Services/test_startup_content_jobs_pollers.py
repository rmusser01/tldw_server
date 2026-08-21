from __future__ import annotations

import asyncio
import importlib
import sys
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerStrategy,
)

pytestmark = pytest.mark.unit


def _import_startup_content_jobs_pollers():
    sys.modules.pop("tldw_Server_API.app.services.startup_content_jobs_pollers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_content_jobs_pollers")


def _context(
    *,
    route_enabled: Callable[..., bool] | None = None,
) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app="app",
        settings={},
        test_mode=True,
        route_enabled=route_enabled or (lambda *_args, **_kwargs: True),
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def _specs_by_name(startup_pollers: Any) -> dict[str, Any]:
    return {spec.name: spec for spec in startup_pollers.provide_content_jobs_worker_specs()}


def _standalone_config(
    *,
    feature_enabled: bool = True,
    egress_enabled: bool = True,
    allowed_models: tuple[str, ...] = ("model-a",),
    max_source_chars: int = 100,
    max_provider_response_bytes: int = 1_000,
    max_output_tokens: int = 100,
    revision_fill: str = "a",
):
    from tldw_Server_API.app.core.Slides.standalone_html_config import (
        ResolvedExecutionTarget,
        ResolvedPrompt,
        SlidesStandaloneHtmlConfig,
        StandaloneHtmlInputLimits,
        StandaloneHtmlOutputLimits,
        StandaloneHtmlProviderLimits,
    )

    def _target(model: str) -> ResolvedExecutionTarget:
        return ResolvedExecutionTarget(
            provider="openai",
            model=model,
            adapter_id="openai_official_chat_v1",
            endpoint_identity="https://api.openai.com:443/v1/chat/completions",
        )

    allowed_targets = tuple(_target(model) for model in allowed_models)
    default_target = _target("model-a")
    enabled = feature_enabled and egress_enabled and default_target in allowed_targets
    reason = None
    if not feature_enabled:
        reason = "feature_disabled"
    elif not egress_enabled:
        reason = "egress_disabled"
    elif not enabled:
        reason = "default_model_not_allowed"
    return SlidesStandaloneHtmlConfig(
        feature_enabled=feature_enabled,
        egress_enabled=egress_enabled,
        enabled=enabled,
        disabled_reason=reason,
        target=default_target if enabled else None,
        prompt=(
            ResolvedPrompt(
                text="Build a standalone presentation.",
                sha256="b" * 64,
                contract_version="slides.standalone_html.v1",
                byte_count=32,
            )
            if enabled
            else None
        ),
        allowed_targets=allowed_targets,
        input_limits=StandaloneHtmlInputLimits(
            max_request_bytes=4_194_304,
            max_source_chars=max_source_chars,
            max_source_tokens=50_000,
            max_audience_chars=500,
            max_source_identifier_bytes=256,
            max_note_ids=100,
            max_rag_query_chars=20_000,
            max_rag_top_k=100,
        ),
        output_limits=StandaloneHtmlOutputLimits(
            max_provider_response_bytes=max_provider_response_bytes,
            max_document_bytes=1_048_576,
        ),
        provider_limits=StandaloneHtmlProviderLimits(
            connect_timeout_seconds=10.0,
            read_timeout_seconds=120.0,
            overall_timeout_seconds=180.0,
            max_output_tokens=max_output_tokens,
        ),
        generation_config_revision=("sha256:" + revision_fill * 64) if enabled else None,
        _revision_manifest="test-manifest" if enabled else "",
    )


@pytest.mark.parametrize(
    "spec_name",
    [
        "audio_jobs_task",
        "audiobook_jobs_task",
        "presentation_render_jobs_task",
        "standalone_html_generation_jobs_task",
        "research_workspace_output_jobs_task",
        "media_ingest_jobs_task",
        "media_ingest_heavy_jobs_task",
        "reading_digest_jobs_task",
        "chat_macros_jobs_task",
        "llamacpp_acquisition_jobs_task",
        "visual_identity_jobs_task",
        "vn_asset_jobs_task",
        "vn_asset_generation_jobs_task",
        "companion_reflection_jobs_task",
        "scheduled_tasks_recurring_question_jobs_task",
    ],
)
def test_content_jobs_worker_specs_match_legacy_worker_contract(
    spec_name: str,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()

    spec = _specs_by_name(startup_pollers)[spec_name]

    assert spec.task_name == spec_name
    assert spec.category == "jobs"
    assert spec.phase is ShutdownPhase.JOB_POLLER_QUIESCE
    assert spec.timeout_sec == (15.0 if spec_name == "standalone_html_generation_jobs_task" else 5.0)
    assert spec.strategy is WorkerStrategy.STOP_EVENT_TASK
    assert spec.factory is not None
    assert callable(spec.factory)


def test_content_jobs_worker_specs_use_expected_names() -> None:
    startup_pollers = _import_startup_content_jobs_pollers()

    assert [spec.name for spec in startup_pollers.provide_content_jobs_worker_specs()] == [
        "audio_jobs_task",
        "audiobook_jobs_task",
        "presentation_render_jobs_task",
        "standalone_html_generation_jobs_task",
        "research_workspace_output_jobs_task",
        "media_ingest_jobs_task",
        "media_ingest_heavy_jobs_task",
        "reading_digest_jobs_task",
        "chat_macros_jobs_task",
        "llamacpp_acquisition_jobs_task",
        "visual_identity_jobs_task",
        "vn_asset_jobs_task",
        "vn_asset_generation_jobs_task",
        "companion_reflection_jobs_task",
        "scheduled_tasks_recurring_question_jobs_task",
    ]


def test_content_jobs_worker_spec_factories_delegate_to_existing_worker_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    calls: list[tuple[str, object]] = []

    for spec_name, factory_name in [
        ("audio_jobs_task", "_run_audio_jobs_worker_service"),
        ("audiobook_jobs_task", "_run_audiobook_jobs_worker_service"),
        ("presentation_render_jobs_task", "_run_presentation_render_jobs_worker_service"),
        ("standalone_html_generation_jobs_task", "_run_standalone_html_generation_jobs_service"),
        ("research_workspace_output_jobs_task", "_run_research_workspace_output_jobs_worker_service"),
        ("media_ingest_jobs_task", "_run_media_ingest_jobs_worker_service"),
        ("media_ingest_heavy_jobs_task", "_run_media_ingest_heavy_jobs_worker_service"),
        ("reading_digest_jobs_task", "_run_reading_digest_jobs_worker_service"),
        ("chat_macros_jobs_task", "_run_chat_macros_jobs_worker_service"),
        ("llamacpp_acquisition_jobs_task", "_run_llamacpp_acquisition_jobs_worker_service"),
        ("visual_identity_jobs_task", "_run_visual_identity_jobs_worker_service"),
        ("vn_asset_jobs_task", "_run_vn_asset_jobs_worker_service"),
        ("vn_asset_generation_jobs_task", "_run_vn_asset_generation_jobs_worker_service"),
        ("companion_reflection_jobs_task", "_run_companion_reflection_jobs_worker_service"),
        (
            "scheduled_tasks_recurring_question_jobs_task",
            "_run_scheduled_tasks_recurring_question_jobs_worker_service",
        ),
    ]:
        monkeypatch.setattr(
            startup_pollers,
            factory_name,
            lambda *args, name=spec_name: calls.append((name, args[-1])) or f"{name}-awaitable",
        )

    specs = _specs_by_name(startup_pollers)

    for spec_name, spec in specs.items():
        assert spec.factory is not None
        assert spec.factory(_context(), f"{spec_name}-stop") == f"{spec_name}-awaitable"

    assert calls == [
        ("audio_jobs_task", "audio_jobs_task-stop"),
        ("audiobook_jobs_task", "audiobook_jobs_task-stop"),
        ("presentation_render_jobs_task", "presentation_render_jobs_task-stop"),
        ("standalone_html_generation_jobs_task", "standalone_html_generation_jobs_task-stop"),
        ("research_workspace_output_jobs_task", "research_workspace_output_jobs_task-stop"),
        ("media_ingest_jobs_task", "media_ingest_jobs_task-stop"),
        ("media_ingest_heavy_jobs_task", "media_ingest_heavy_jobs_task-stop"),
        ("reading_digest_jobs_task", "reading_digest_jobs_task-stop"),
        ("chat_macros_jobs_task", "chat_macros_jobs_task-stop"),
        ("llamacpp_acquisition_jobs_task", "llamacpp_acquisition_jobs_task-stop"),
        ("visual_identity_jobs_task", "visual_identity_jobs_task-stop"),
        ("vn_asset_jobs_task", "vn_asset_jobs_task-stop"),
        ("vn_asset_generation_jobs_task", "vn_asset_generation_jobs_task-stop"),
        ("companion_reflection_jobs_task", "companion_reflection_jobs_task-stop"),
        (
            "scheduled_tasks_recurring_question_jobs_task",
            "scheduled_tasks_recurring_question_jobs_task-stop",
        ),
    ]


def test_content_jobs_worker_spec_predicates_use_route_enabled_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _route_enabled(*args: object, **kwargs: object) -> bool:
        calls.append((args, kwargs))
        return False

    context = _context(route_enabled=_route_enabled)
    specs = _specs_by_name(startup_pollers)
    media_ingest_explicit_flag_specs = {
        "media_ingest_jobs_task",
        "media_ingest_heavy_jobs_task",
    }

    for spec_name in [
        "audio_jobs_task",
        "audiobook_jobs_task",
        "presentation_render_jobs_task",
        "research_workspace_output_jobs_task",
        "media_ingest_jobs_task",
        "media_ingest_heavy_jobs_task",
        "reading_digest_jobs_task",
        "chat_macros_jobs_task",
        "llamacpp_acquisition_jobs_task",
        "visual_identity_jobs_task",
        "vn_asset_jobs_task",
        "vn_asset_generation_jobs_task",
        "companion_reflection_jobs_task",
        "scheduled_tasks_recurring_question_jobs_task",
    ]:
        monkeypatch.setenv(
            {
                "audio_jobs_task": "AUDIO_JOBS_WORKER_ENABLED",
                "audiobook_jobs_task": "AUDIOBOOK_JOBS_WORKER_ENABLED",
                "presentation_render_jobs_task": "PRESENTATION_RENDER_JOBS_WORKER_ENABLED",
                "research_workspace_output_jobs_task": "RESEARCH_WORKSPACE_OUTPUT_JOBS_WORKER_ENABLED",
                "media_ingest_jobs_task": "MEDIA_INGEST_JOBS_WORKER_ENABLED",
                "media_ingest_heavy_jobs_task": "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
                "reading_digest_jobs_task": "READING_DIGEST_JOBS_WORKER_ENABLED",
                "chat_macros_jobs_task": "CHAT_MACROS_JOBS_WORKER_ENABLED",
                "llamacpp_acquisition_jobs_task": "LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED",
                "visual_identity_jobs_task": "VISUAL_IDENTITY_JOBS_WORKER_ENABLED",
                "vn_asset_jobs_task": "VN_ASSET_JOBS_WORKER_ENABLED",
                "vn_asset_generation_jobs_task": "VN_ASSET_GENERATION_JOBS_WORKER_ENABLED",
                "companion_reflection_jobs_task": "COMPANION_REFLECTION_JOBS_WORKER_ENABLED",
                "scheduled_tasks_recurring_question_jobs_task": (
                    "SCHEDULED_TASKS_RECURRING_QUESTION_WORKER_ENABLED"
                ),
            }[spec_name],
            "true",
        )
        assert specs[spec_name].enabled(context) is (spec_name in media_ingest_explicit_flag_specs)

    assert calls == [
        (("audio-jobs",), {}),
        (("audiobooks",), {}),
        (("slides",), {}),
        (("research-workspace-output-jobs",), {"default_stable": True}),
        (("reading",), {}),
        (("chat-macros",), {}),
        (("llamacpp-acquisition",), {}),
        (("visual-identities",), {"default_stable": True}),
        (("vn-assets",), {"default_stable": True}),
        (
            ("vn-assets-generation",),
            {"default_stable": True},
        ),
        (("companion",), {}),
        (("scheduled-tasks-recurring-question",), {"default_stable": False}),
    ]


def test_standalone_html_composite_worker_uses_slides_route_without_a_second_flag() -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _route_enabled(*args: object, **kwargs: object) -> bool:
        calls.append((args, kwargs))
        return True

    spec = _specs_by_name(startup_pollers)["standalone_html_generation_jobs_task"]

    assert spec.enabled(_context(route_enabled=_route_enabled)) is True
    assert calls == [(("slides",), {})]


def test_standalone_html_live_config_can_only_narrow_boot_authority() -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    boot = _standalone_config(
        feature_enabled=False,
        allowed_models=("model-a",),
        max_source_chars=100,
        max_provider_response_bytes=1_000,
        max_output_tokens=100,
    )
    broadened_live = _standalone_config(
        allowed_models=("model-a", "model-b"),
        max_source_chars=200,
        max_provider_response_bytes=2_000,
        max_output_tokens=200,
    )

    restricted = startup_pollers._restrict_standalone_html_config(
        boot,
        broadened_live,
    )

    assert restricted.feature_enabled is False
    assert restricted.egress_enabled is True
    assert [target.model for target in restricted.allowed_targets] == ["model-a"]
    assert restricted.input_limits.max_source_chars == 100
    assert restricted.output_limits.max_provider_response_bytes == 1_000
    assert restricted.provider_limits.max_output_tokens == 100


def test_standalone_html_live_default_removal_disables_restricted_snapshot() -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    boot = _standalone_config(allowed_models=("model-a", "model-b"))
    live = _standalone_config(allowed_models=("model-b",))

    restricted = startup_pollers._restrict_standalone_html_config(boot, live)

    assert restricted.enabled is False
    assert restricted.disabled_reason == "default_model_not_allowed"
    assert [target.model for target in restricted.allowed_targets] == ["model-b"]


@pytest.mark.asyncio
async def test_standalone_html_runtime_reloads_only_narrower_generation_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    from tldw_Server_API.app.core import config as config_module
    from tldw_Server_API.app.core.DB_Management import db_path_utils
    from tldw_Server_API.app.core.Slides import (
        standalone_html_config,
        standalone_html_reconciler,
        standalone_html_registry,
    )

    first_config = _standalone_config(
        allowed_models=("model-a", "model-b"),
        max_source_chars=200,
        max_provider_response_bytes=2_000,
        max_output_tokens=200,
    )
    second_config = _standalone_config(
        egress_enabled=False,
        allowed_models=("model-a",),
        max_source_chars=100,
        max_provider_response_bytes=1_000,
        max_output_tokens=100,
    )
    selected_config = {"value": first_config}
    refresh_calls = {"count": 0}
    keyring = SimpleNamespace(configured_current_key_id="key-one")
    snapshot = SimpleNamespace(
        current_key_id="key-one",
        config_epoch="old-epoch",
        require_generation_ready=lambda: None,
    )

    class _Registry:
        def __init__(self, *, store: object, keyring: object) -> None:
            del store, keyring

        async def snapshot(self):
            return snapshot

        async def activate_configured_current(self, **_kwargs):
            return snapshot

    class _Reconciler:
        def __init__(self, **_kwargs) -> None:
            pass

        def admission_ready(self) -> bool:
            return True

    monkeypatch.setattr(startup_pollers, "_standalone_html_jobs_manager", object)
    monkeypatch.setattr(
        db_path_utils.DatabasePaths,
        "resolve_user_db_base_dir",
        lambda: "/tmp/standalone-html-tests",
    )
    monkeypatch.setattr(config_module, "load_comprehensive_config", lambda: {})
    monkeypatch.setattr(
        config_module,
        "refresh_config_cache",
        lambda: refresh_calls.__setitem__("count", refresh_calls["count"] + 1),
    )
    monkeypatch.setattr(
        standalone_html_config,
        "load_standalone_html_config",
        lambda _config, *, availability: selected_config["value"],
    )
    monkeypatch.setattr(
        standalone_html_registry,
        "StandaloneHtmlHmacKeyring",
        SimpleNamespace(from_env=lambda: keyring),
    )
    monkeypatch.setattr(
        standalone_html_registry,
        "JobManagerDigestKeyRegistryStore",
        lambda _manager: object(),
    )
    monkeypatch.setattr(standalone_html_registry, "StandaloneHtmlKeyRegistry", _Registry)
    monkeypatch.setattr(
        standalone_html_reconciler,
        "FencedStandaloneHtmlReconciler",
        _Reconciler,
    )
    runtime = await startup_pollers._build_standalone_html_generation_runtime(_context())
    selected_config["value"] = second_config

    current = runtime.current_config_loader()

    assert runtime.local_only is False
    assert current.egress_enabled is False
    assert [target.model for target in current.allowed_targets] == ["model-a"]
    assert current.input_limits.max_source_chars == 100
    assert current.output_limits.max_provider_response_bytes == 1_000
    assert current.provider_limits.max_output_tokens == 100
    assert current.generation_config_revision == first_config.generation_config_revision
    assert refresh_calls["count"] == 1


def test_standalone_html_coordination_generation_defaults_to_legacy_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    monkeypatch.delenv("SLIDES_STANDALONE_COORDINATION_GENERATION", raising=False)

    assert startup_pollers._standalone_html_coordination_generation() == 0
    assert startup_pollers._standalone_html_coordination_epoch(
        static_config=_standalone_config(),
        current_key_id="key-one",
    ).startswith("sha256:")


def test_standalone_html_coordination_epoch_encodes_monotonic_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    monkeypatch.setenv("SLIDES_STANDALONE_COORDINATION_GENERATION", "17")

    assert startup_pollers._standalone_html_coordination_generation() == 17
    assert startup_pollers._standalone_html_coordination_epoch(
        static_config=_standalone_config(),
        current_key_id="key-one",
    ).startswith("v1:g17:sha256:")


def test_standalone_html_coordination_epoch_covers_full_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    monkeypatch.setenv("SLIDES_STANDALONE_COORDINATION_GENERATION", "17")
    only_default = _standalone_config(
        allowed_models=("model-a",),
        revision_fill="c",
    )
    with_nondefault = _standalone_config(
        allowed_models=("model-a", "model-b"),
        revision_fill="c",
    )

    assert only_default.generation_config_revision == with_nondefault.generation_config_revision
    assert startup_pollers._standalone_html_coordination_epoch(
        static_config=only_default,
        current_key_id="key-one",
    ) != startup_pollers._standalone_html_coordination_epoch(
        static_config=with_nondefault,
        current_key_id="key-one",
    )


@pytest.mark.parametrize(
    "configured",
    [
        "-1",
        "+1",
        "01",
        " 1",
        "1 ",
        "1.0",
        "not-a-number",
        "9223372036854775808",
    ],
)
def test_standalone_html_coordination_generation_rejects_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    monkeypatch.setenv("SLIDES_STANDALONE_COORDINATION_GENERATION", configured)

    with pytest.raises(ValueError, match="coordination generation is invalid"):
        startup_pollers._standalone_html_coordination_generation()


@pytest.mark.asyncio
async def test_standalone_html_composite_starts_handler_only_after_shared_startup_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    app = FastAPI()
    stop_event = asyncio.Event()
    handler_started = asyncio.Event()

    class _Reconciler:
        def __init__(self) -> None:
            self.calls = 0
            self.released = False

        def run_batch(self):
            self.calls += 1
            return SimpleNamespace(
                startup_ready=self.calls >= 2,
                leader=True,
                completed_pass=self.calls >= 2,
                jobs_available=True,
            )

        def release(self) -> bool:
            self.released = True
            return True

    reconciler = _Reconciler()
    runtime = SimpleNamespace(
        reconciler=reconciler,
        local_only=False,
        admission_gate=SimpleNamespace(open=False),
        validation_pool=None,
    )

    async def _build(_context):
        return runtime

    async def _handler(_runtime, worker_stop_event):
        assert _runtime is runtime
        handler_started.set()
        await worker_stop_event.wait()

    monkeypatch.setattr(startup_pollers, "_build_standalone_html_generation_runtime", _build)
    monkeypatch.setattr(startup_pollers, "_run_standalone_html_generation_handler", _handler)
    monkeypatch.setattr(startup_pollers, "_STANDALONE_RETRY_SECONDS", 0.001)
    context = WorkerLifecycleContext(
        app=app,
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )

    task = asyncio.create_task(startup_pollers._run_standalone_html_generation_jobs_service(context, stop_event))
    await asyncio.wait_for(handler_started.wait(), timeout=1)

    assert reconciler.calls >= 2
    assert app.state.standalone_html_generation_worker_registered is True
    assert app.state.standalone_html_reconciler_admission_ready is True

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)

    assert reconciler.released is True
    assert app.state.standalone_html_generation_worker_registered is False
    assert app.state.standalone_html_reconciler_admission_ready is False


@pytest.mark.asyncio
async def test_standalone_html_composite_retries_after_transient_reconciliation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    app = FastAPI()
    stop_event = asyncio.Event()
    handler_started = asyncio.Event()

    class _Reconciler:
        def __init__(self) -> None:
            self.calls = 0

        def run_batch(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient coordination failure")
            return SimpleNamespace(
                startup_ready=True,
                leader=True,
                completed_pass=True,
                jobs_available=True,
            )

        def release(self) -> bool:
            return True

    reconciler = _Reconciler()
    runtime = SimpleNamespace(
        reconciler=reconciler,
        local_only=False,
        admission_gate=SimpleNamespace(open=False),
        validation_pool=None,
    )

    async def _build(_context):
        return runtime

    async def _handler(_runtime, worker_stop_event):
        handler_started.set()
        await worker_stop_event.wait()

    monkeypatch.setattr(startup_pollers, "_build_standalone_html_generation_runtime", _build)
    monkeypatch.setattr(startup_pollers, "_run_standalone_html_generation_handler", _handler)
    monkeypatch.setattr(startup_pollers, "_STANDALONE_RETRY_SECONDS", 0.001)
    context = WorkerLifecycleContext(
        app=app,
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )

    task = asyncio.create_task(startup_pollers._run_standalone_html_generation_jobs_service(context, stop_event))
    await asyncio.wait_for(handler_started.wait(), timeout=1)

    assert reconciler.calls >= 2
    assert runtime.admission_gate.open is True

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_standalone_html_composite_reopens_admission_after_readiness_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    app = FastAPI()
    stop_event = asyncio.Event()
    handler_started = asyncio.Event()

    class _Reconciler:
        def __init__(self) -> None:
            self.calls = 0

        def run_batch(self):
            self.calls += 1
            return SimpleNamespace(
                startup_ready=self.calls != 2,
                leader=True,
                completed_pass=True,
                jobs_available=True,
            )

        def release(self) -> bool:
            return True

    reconciler = _Reconciler()
    runtime = SimpleNamespace(
        reconciler=reconciler,
        local_only=False,
        admission_gate=SimpleNamespace(open=False),
        validation_pool=None,
    )

    async def _build(_context):
        return runtime

    async def _handler(_runtime, worker_stop_event):
        handler_started.set()
        await worker_stop_event.wait()

    monkeypatch.setattr(startup_pollers, "_build_standalone_html_generation_runtime", _build)
    monkeypatch.setattr(startup_pollers, "_run_standalone_html_generation_handler", _handler)
    monkeypatch.setattr(startup_pollers, "_STANDALONE_RETRY_SECONDS", 0.001)
    context = WorkerLifecycleContext(
        app=app,
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )

    task = asyncio.create_task(startup_pollers._run_standalone_html_generation_jobs_service(context, stop_event))
    await asyncio.wait_for(handler_started.wait(), timeout=1)
    for _ in range(100):
        if reconciler.calls >= 3:
            break
        await asyncio.sleep(0.001)

    assert reconciler.calls >= 3
    assert runtime.admission_gate.open is True
    assert app.state.standalone_html_reconciler_admission_ready is True

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_standalone_html_handler_failure_closes_admission_without_poll_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    app = FastAPI()
    stop_event = asyncio.Event()
    handler_started = asyncio.Event()
    fail_handler = asyncio.Event()

    class _Reconciler:
        def run_batch(self):
            return SimpleNamespace(
                startup_ready=True,
                leader=True,
                completed_pass=True,
                jobs_available=True,
            )

        def release(self) -> bool:
            return True

    runtime = SimpleNamespace(
        reconciler=_Reconciler(),
        local_only=False,
        admission_gate=SimpleNamespace(open=False),
        validation_pool=None,
    )

    async def _build(_context):
        return runtime

    async def _handler(_runtime, _worker_stop_event):
        handler_started.set()
        await fail_handler.wait()
        raise RuntimeError("handler failed")

    async def _pool(_app):
        return object()

    monkeypatch.setattr(startup_pollers, "_build_standalone_html_generation_runtime", _build)
    monkeypatch.setattr(startup_pollers, "_run_standalone_html_generation_handler", _handler)
    monkeypatch.setattr(startup_pollers, "_get_worker_owned_validation_pool", _pool)
    monkeypatch.setattr(startup_pollers, "_STANDALONE_RETRY_SECONDS", 60.0)
    context = WorkerLifecycleContext(
        app=app,
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )
    task = asyncio.create_task(startup_pollers._run_standalone_html_generation_jobs_service(context, stop_event))

    try:
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        assert runtime.admission_gate.open is True
        fail_handler.set()
        for _ in range(20):
            if not runtime.admission_gate.open:
                break
            await asyncio.sleep(0)

        assert runtime.admission_gate.open is False
        assert app.state.standalone_html_generation_worker_registered is False
        assert app.state.standalone_html_reconciler_admission_ready is False
    finally:
        stop_event.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)


@pytest.mark.asyncio
async def test_standalone_html_local_fallback_retries_full_build_each_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    app = FastAPI()
    stop_event = asyncio.Event()
    handler_started = asyncio.Event()
    build_calls = 0

    class _LocalReconciler:
        def __init__(self) -> None:
            self.calls = 0
            self.released = False

        def run_local_expiry_batch(self):
            self.calls += 1
            return SimpleNamespace(
                local_sweep_state=("progressed" if self.calls < 3 else "completed"),
            )

        def release(self) -> bool:
            self.released = True
            return True

    class _FullReconciler:
        def run_batch(self):
            return SimpleNamespace(
                startup_ready=True,
                leader=True,
                completed_pass=True,
                jobs_available=True,
            )

        def release(self) -> bool:
            return True

    local_reconciler = _LocalReconciler()
    local_runtime = SimpleNamespace(
        reconciler=local_reconciler,
        local_only=True,
        admission_gate=None,
    )
    full_runtime = SimpleNamespace(
        reconciler=_FullReconciler(),
        local_only=False,
        admission_gate=SimpleNamespace(open=False),
        validation_pool=None,
    )

    async def _build(_context):
        nonlocal build_calls
        build_calls += 1
        return local_runtime if build_calls == 1 else full_runtime

    async def _handler(_runtime, worker_stop_event):
        handler_started.set()
        await worker_stop_event.wait()

    async def _pool(_app):
        return object()

    monkeypatch.setattr(startup_pollers, "_build_standalone_html_generation_runtime", _build)
    monkeypatch.setattr(startup_pollers, "_run_standalone_html_generation_handler", _handler)
    monkeypatch.setattr(startup_pollers, "_get_worker_owned_validation_pool", _pool)
    monkeypatch.setattr(startup_pollers, "_STANDALONE_RETRY_SECONDS", 0.01)
    context = WorkerLifecycleContext(
        app=app,
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )
    task = asyncio.create_task(startup_pollers._run_standalone_html_generation_jobs_service(context, stop_event))

    try:
        await asyncio.wait_for(handler_started.wait(), timeout=0.3)
        assert build_calls == 2
        assert local_reconciler.calls == 3
        assert local_reconciler.released is True
    finally:
        stop_event.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)


@pytest.mark.asyncio
async def test_standalone_html_jobs_outage_drains_local_pages_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    app = FastAPI()
    stop_event = asyncio.Event()

    class _OutageReconciler:
        def __init__(self) -> None:
            self.run_calls = 0
            self.local_calls = 0

        def run_batch(self):
            self.run_calls += 1
            return SimpleNamespace(
                startup_ready=False,
                leader=True,
                completed_pass=False,
                jobs_available=False,
                local_sweep_state="progressed",
            )

        def run_local_expiry_batch(self):
            self.local_calls += 1
            state = "progressed" if self.local_calls == 1 else "completed"
            return SimpleNamespace(local_sweep_state=state)

        def release(self) -> bool:
            return True

    reconciler = _OutageReconciler()
    runtime = SimpleNamespace(
        reconciler=reconciler,
        local_only=False,
        admission_gate=SimpleNamespace(open=False),
        validation_pool=None,
    )

    async def _build(_context):
        return runtime

    monkeypatch.setattr(startup_pollers, "_build_standalone_html_generation_runtime", _build)
    monkeypatch.setattr(startup_pollers, "_STANDALONE_RETRY_SECONDS", 60.0)
    context = WorkerLifecycleContext(
        app=app,
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )
    task = asyncio.create_task(startup_pollers._run_standalone_html_generation_jobs_service(context, stop_event))

    try:
        for _ in range(60):
            if reconciler.local_calls >= 2:
                break
            await asyncio.sleep(0.005)
        assert reconciler.run_calls == 1
        assert reconciler.local_calls == 2
    finally:
        stop_event.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)


@pytest.mark.asyncio
async def test_standalone_html_cancellation_drains_handler_fence_and_pool_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    app = FastAPI()
    stop_event = asyncio.Event()
    handler_started = asyncio.Event()
    handler_stopping = asyncio.Event()
    allow_handler_finish = asyncio.Event()
    events: list[str] = []

    class _Gate:
        def __init__(self) -> None:
            self._open = False

        @property
        def open(self) -> bool:
            return self._open

        @open.setter
        def open(self, value: bool) -> None:
            if self._open and not value:
                events.append("admission_closed")
            self._open = value

    class _Reconciler:
        def run_batch(self):
            return SimpleNamespace(
                startup_ready=True,
                leader=True,
                completed_pass=True,
                jobs_available=True,
            )

        def release(self) -> bool:
            assert events[-1] == "handler_stopped"
            events.append("fence_released")
            return True

    runtime = SimpleNamespace(
        reconciler=_Reconciler(),
        local_only=False,
        admission_gate=_Gate(),
        validation_pool=None,
    )

    async def _build(_context):
        return runtime

    async def _handler(_runtime, worker_stop_event):
        handler_started.set()
        await worker_stop_event.wait()
        events.append("handler_stop_signaled")
        handler_stopping.set()
        await allow_handler_finish.wait()
        events.append("handler_stopped")

    async def _pool(_app):
        return object()

    async def _close_pool(_app):
        assert events[-1] == "fence_released"
        events.append("pool_closed")

    monkeypatch.setattr(startup_pollers, "_build_standalone_html_generation_runtime", _build)
    monkeypatch.setattr(startup_pollers, "_run_standalone_html_generation_handler", _handler)
    monkeypatch.setattr(startup_pollers, "_get_worker_owned_validation_pool", _pool)
    monkeypatch.setattr(startup_pollers, "_close_worker_owned_validation_pool", _close_pool)
    monkeypatch.setattr(startup_pollers, "_STANDALONE_RETRY_SECONDS", 60.0)
    context = WorkerLifecycleContext(
        app=app,
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )
    task = asyncio.create_task(startup_pollers._run_standalone_html_generation_jobs_service(context, stop_event))

    try:
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(handler_stopping.wait(), timeout=0.2)
        task.cancel()
        allow_handler_finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    finally:
        stop_event.set()
        allow_handler_finish.set()
        await asyncio.gather(task, return_exceptions=True)

    assert events == [
        "admission_closed",
        "handler_stop_signaled",
        "handler_stopped",
        "fence_released",
        "pool_closed",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_call", [1, 2])
async def test_standalone_html_stop_closes_admission_while_reconciliation_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    blocked_call: int,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    app = FastAPI()
    stop_event = asyncio.Event()
    blocked_batch = threading.Event()
    release_batch = threading.Event()
    handler_stopped = asyncio.Event()
    handler_starts = 0

    class _Gate:
        def __init__(self) -> None:
            self._open = False
            self.opened_after_stop = False

        @property
        def open(self) -> bool:
            return self._open

        @open.setter
        def open(self, value: bool) -> None:
            if value and stop_event.is_set():
                self.opened_after_stop = True
            self._open = value

    class _Reconciler:
        def __init__(self) -> None:
            self.calls = 0

        def run_batch(self):
            self.calls += 1
            if self.calls == blocked_call:
                blocked_batch.set()
                release_batch.wait(timeout=5)
            return SimpleNamespace(
                startup_ready=True,
                leader=True,
                completed_pass=False,
                jobs_available=True,
            )

        def release(self) -> bool:
            return True

    runtime = SimpleNamespace(
        reconciler=_Reconciler(),
        local_only=False,
        admission_gate=_Gate(),
        validation_pool=None,
    )

    async def _build(_context):
        return runtime

    async def _handler(_runtime, worker_stop_event):
        nonlocal handler_starts
        handler_starts += 1
        try:
            await worker_stop_event.wait()
        finally:
            handler_stopped.set()

    async def _pool(_app):
        return object()

    async def _close_pool(_app):
        return None

    monkeypatch.setattr(startup_pollers, "_build_standalone_html_generation_runtime", _build)
    monkeypatch.setattr(startup_pollers, "_run_standalone_html_generation_handler", _handler)
    monkeypatch.setattr(startup_pollers, "_get_worker_owned_validation_pool", _pool)
    monkeypatch.setattr(startup_pollers, "_close_worker_owned_validation_pool", _close_pool)
    monkeypatch.setattr(startup_pollers, "_STANDALONE_RETRY_SECONDS", 0.0)
    context = WorkerLifecycleContext(
        app=app,
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )
    task = asyncio.create_task(
        startup_pollers._run_standalone_html_generation_jobs_service(
            context,
            stop_event,
        )
    )

    try:
        for _ in range(200):
            if blocked_batch.is_set() and (blocked_call == 1 or runtime.admission_gate.open):
                break
            await asyncio.sleep(0.005)
        assert blocked_batch.is_set()
        assert runtime.admission_gate.open is (blocked_call == 2)
        assert handler_starts == (0 if blocked_call == 1 else 1)

        stop_event.set()

        if blocked_call == 2:
            await asyncio.wait_for(handler_stopped.wait(), timeout=0.2)
        assert runtime.admission_gate.open is False
        assert app.state.standalone_html_generation_worker_registered is False
        assert app.state.standalone_html_reconciler_admission_ready is False
        assert task.done() is False

        release_batch.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)

        assert handler_starts == (0 if blocked_call == 1 else 1)
        assert runtime.admission_gate.open is False
        assert runtime.admission_gate.opened_after_stop is False
    finally:
        stop_event.set()
        release_batch.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)


@pytest.mark.asyncio
async def test_start_content_jobs_pollers_combines_handles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    calls: list[str] = []

    async def _record_audio(**kwargs):
        del kwargs
        calls.append("audio")
        return ("audio-stop", "audio-task")

    async def _record_audiobook(**kwargs):
        del kwargs
        calls.append("audiobook")
        return ("audiobook-stop", "audiobook-task")

    async def _record_audio_studio(**kwargs):
        del kwargs
        calls.append("audio-studio")
        return ("audio-studio-stop", "audio-studio-task")

    async def _record_presentation(**kwargs):
        del kwargs
        calls.append("presentation")
        return ("presentation-stop", "presentation-task")

    async def _record_research_output(**kwargs):
        del kwargs
        calls.append("research-output")
        return ("research-output-stop", "research-output-task")

    async def _record_media_ingest(**kwargs):
        del kwargs
        calls.append("media-ingest")
        return ("media-stop", "media-task", "media-heavy-stop", "media-heavy-task")

    async def _record_reading_digest(**kwargs):
        del kwargs
        calls.append("reading-digest")
        return ("reading-stop", "reading-task")

    async def _record_chat_macros(**kwargs):
        del kwargs
        calls.append("chat-macros")
        return ("chat-stop", "chat-task")

    async def _record_llamacpp_acquisition(**kwargs):
        del kwargs
        calls.append("llamacpp-acquisition")
        return ("llamacpp-stop", "llamacpp-task")

    async def _record_visual_identity(**kwargs: object) -> tuple[str, str]:
        """Record that the Visual Identity worker starter ran."""

        del kwargs
        calls.append("visual-identity")
        return ("visual-identity-stop", "visual-identity-task")

    async def _record_vn_asset(**kwargs: object) -> tuple[str, str, str, str]:
        """Record that the VN asset worker starter ran."""

        del kwargs
        calls.append("vn-asset")
        return ("vn-asset-stop", "vn-asset-task", "vn-generation-stop", "vn-generation-task")

    async def _record_companion(**kwargs):
        del kwargs
        calls.append("companion")
        return ("companion-stop", "companion-task")

    monkeypatch.setattr(startup_pollers, "_start_audio_jobs_worker", _record_audio)
    monkeypatch.setattr(startup_pollers, "_start_audiobook_jobs_worker", _record_audiobook)
    monkeypatch.setattr(startup_pollers, "_start_audio_studio_jobs_worker", _record_audio_studio)
    monkeypatch.setattr(startup_pollers, "_start_presentation_render_jobs_worker", _record_presentation)
    monkeypatch.setattr(startup_pollers, "_start_research_workspace_output_jobs_worker", _record_research_output)
    monkeypatch.setattr(startup_pollers, "_start_media_ingest_jobs_workers", _record_media_ingest)
    monkeypatch.setattr(startup_pollers, "_start_reading_digest_jobs_worker", _record_reading_digest)
    monkeypatch.setattr(startup_pollers, "_start_chat_macros_jobs_worker", _record_chat_macros)
    monkeypatch.setattr(
        startup_pollers,
        "_start_llamacpp_acquisition_jobs_worker",
        _record_llamacpp_acquisition,
    )
    monkeypatch.setattr(startup_pollers, "_start_visual_identity_jobs_worker", _record_visual_identity)
    monkeypatch.setattr(startup_pollers, "_start_vn_asset_jobs_workers", _record_vn_asset)
    monkeypatch.setattr(startup_pollers, "_start_companion_reflection_jobs_worker", _record_companion)

    handles = await startup_pollers.start_content_jobs_pollers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda *args, **kwargs: False,
    )

    assert calls == [
        "audio",
        "audiobook",
        "audio-studio",
        "presentation",
        "research-output",
        "media-ingest",
        "reading-digest",
        "chat-macros",
        "llamacpp-acquisition",
        "visual-identity",
        "vn-asset",
        "companion",
    ]
    assert handles.audio_jobs_stop_event == "audio-stop"
    assert handles.audio_jobs_task == "audio-task"
    assert handles.audiobook_jobs_stop_event == "audiobook-stop"
    assert handles.audiobook_jobs_task == "audiobook-task"
    assert handles.audio_studio_jobs_stop_event == "audio-studio-stop"
    assert handles.audio_studio_jobs_task == "audio-studio-task"
    assert handles.presentation_render_jobs_stop_event == "presentation-stop"
    assert handles.presentation_render_jobs_task == "presentation-task"
    assert handles.research_workspace_output_jobs_stop_event == "research-output-stop"
    assert handles.research_workspace_output_jobs_task == "research-output-task"
    assert handles.media_ingest_jobs_stop_event == "media-stop"
    assert handles.media_ingest_jobs_task == "media-task"
    assert handles.media_ingest_heavy_jobs_stop_event == "media-heavy-stop"
    assert handles.media_ingest_heavy_jobs_task == "media-heavy-task"
    assert handles.reading_digest_jobs_stop_event == "reading-stop"
    assert handles.reading_digest_jobs_task == "reading-task"
    assert handles.chat_macros_jobs_stop_event == "chat-stop"
    assert handles.chat_macros_jobs_task == "chat-task"
    assert handles.llamacpp_acquisition_jobs_stop_event == "llamacpp-stop"
    assert handles.llamacpp_acquisition_jobs_task == "llamacpp-task"
    assert handles.visual_identity_jobs_stop_event == "visual-identity-stop"
    assert handles.visual_identity_jobs_task == "visual-identity-task"
    assert handles.vn_asset_jobs_stop_event == "vn-asset-stop"
    assert handles.vn_asset_jobs_task == "vn-asset-task"
    assert handles.vn_asset_generation_jobs_stop_event == "vn-generation-stop"
    assert handles.vn_asset_generation_jobs_task == "vn-generation-task"
    assert handles.companion_reflection_jobs_stop_event == "companion-stop"
    assert handles.companion_reflection_jobs_task == "companion-task"


@pytest.mark.asyncio
async def test_start_content_jobs_pollers_passes_inventory_to_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    worker_inventory = object()
    captured_kwargs_by_worker: dict[str, dict[str, object]] = {}

    def _record_worker(label: str, handles: tuple[object, ...]) -> Callable[..., object]:
        """Build a starter stub that captures kwargs for one content worker group."""

        async def _record(**kwargs: object) -> tuple[object, ...]:
            """Capture worker startup kwargs and return deterministic handles."""

            captured_kwargs_by_worker[label] = kwargs
            return handles

        return _record

    monkeypatch.setattr(
        startup_pollers,
        "_start_audio_jobs_worker",
        _record_worker("audio", ("audio-stop", "audio-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_audiobook_jobs_worker",
        _record_worker("audiobook", ("audiobook-stop", "audiobook-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_audio_studio_jobs_worker",
        _record_worker("audio-studio", ("audio-studio-stop", "audio-studio-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_presentation_render_jobs_worker",
        _record_worker("presentation", ("presentation-stop", "presentation-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_research_workspace_output_jobs_worker",
        _record_worker("research-output", ("research-output-stop", "research-output-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_media_ingest_jobs_workers",
        _record_worker("media-ingest", ("media-stop", "media-task", "media-heavy-stop", "media-heavy-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_reading_digest_jobs_worker",
        _record_worker("reading-digest", ("reading-stop", "reading-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_chat_macros_jobs_worker",
        _record_worker("chat-macros", ("chat-stop", "chat-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_llamacpp_acquisition_jobs_worker",
        _record_worker("llamacpp-acquisition", ("llamacpp-stop", "llamacpp-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_visual_identity_jobs_worker",
        _record_worker("visual-identity", ("visual-identity-stop", "visual-identity-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_vn_asset_jobs_workers",
        _record_worker("vn-asset", ("vn-asset-stop", "vn-asset-task", "vn-generation-stop", "vn-generation-task")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_start_companion_reflection_jobs_worker",
        _record_worker("companion", ("companion-stop", "companion-task")),
    )

    await startup_pollers.start_content_jobs_pollers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda *args, **kwargs: False,
        worker_inventory=worker_inventory,
    )

    assert {worker: kwargs["worker_inventory"] for worker, kwargs in captured_kwargs_by_worker.items()} == {
        "audio": worker_inventory,
        "audiobook": worker_inventory,
        "audio-studio": worker_inventory,
        "presentation": worker_inventory,
        "research-output": worker_inventory,
        "media-ingest": worker_inventory,
        "reading-digest": worker_inventory,
        "chat-macros": worker_inventory,
        "llamacpp-acquisition": worker_inventory,
        "visual-identity": worker_inventory,
        "vn-asset": worker_inventory,
        "companion": worker_inventory,
    }


@pytest.mark.parametrize(
    (
        "starter_name",
        "flag_name",
        "route_name",
        "registered_name",
        "factory_name",
        "route_kwargs",
    ),
    [
        (
            "_start_audio_jobs_worker",
            "AUDIO_JOBS_WORKER_ENABLED",
            "audio-jobs",
            "audio_jobs_task",
            "_run_audio_jobs_worker_service",
            {},
        ),
        (
            "_start_audiobook_jobs_worker",
            "AUDIOBOOK_JOBS_WORKER_ENABLED",
            "audiobooks",
            "audiobook_jobs_task",
            "_run_audiobook_jobs_worker_service",
            {},
        ),
        (
            "_start_audio_studio_jobs_worker",
            "AUDIO_STUDIO_JOBS_WORKER_ENABLED",
            "audio-studio",
            "audio_studio_jobs_task",
            "_run_audio_studio_jobs_worker_service",
            {},
        ),
        (
            "_start_presentation_render_jobs_worker",
            "PRESENTATION_RENDER_JOBS_WORKER_ENABLED",
            "slides",
            "presentation_render_jobs_task",
            "_run_presentation_render_jobs_worker_service",
            {},
        ),
        (
            "_start_research_workspace_output_jobs_worker",
            "RESEARCH_WORKSPACE_OUTPUT_JOBS_WORKER_ENABLED",
            "research-workspace-output-jobs",
            "research_workspace_output_jobs_task",
            "_run_research_workspace_output_jobs_worker_service",
            {"default_stable": True},
        ),
        (
            "_start_reading_digest_jobs_worker",
            "READING_DIGEST_JOBS_WORKER_ENABLED",
            "reading",
            "reading_digest_jobs_task",
            "_run_reading_digest_jobs_worker_service",
            {},
        ),
        (
            "_start_chat_macros_jobs_worker",
            "CHAT_MACROS_JOBS_WORKER_ENABLED",
            "chat-macros",
            "chat_macros_jobs_task",
            "_run_chat_macros_jobs_worker_service",
            {},
        ),
        (
            "_start_llamacpp_acquisition_jobs_worker",
            "LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED",
            "llamacpp-acquisition",
            "llamacpp_acquisition_jobs_task",
            "_run_llamacpp_acquisition_jobs_worker_service",
            {},
        ),
        (
            "_start_companion_reflection_jobs_worker",
            "COMPANION_REFLECTION_JOBS_WORKER_ENABLED",
            "companion",
            "companion_reflection_jobs_task",
            "_run_companion_reflection_jobs_worker_service",
            {},
        ),
    ],
)
@pytest.mark.asyncio
async def test_content_jobs_worker_registers_with_worker_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    starter_name: str,
    flag_name: str,
    route_name: str,
    registered_name: str,
    factory_name: str,
    route_kwargs: dict[str, object],
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    registrations: list[dict[str, object]] = []

    class _FakeWorkerInventory:
        """Test double that records custom worker registration calls."""

        async def register_custom(self, **kwargs: object) -> tuple[str, str]:
            """Capture registration kwargs and return deterministic handles."""

            registrations.append(kwargs)
            return f"{registered_name}-task", f"{registered_name}-stop"

    monkeypatch.setattr(
        startup_pollers,
        "_make_event",
        lambda: (_ for _ in ()).throw(AssertionError("legacy event path should not run")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: (_ for _ in ()).throw(AssertionError("legacy task path should not run")),
    )

    def _register_owned_job_poller(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy poller registration should not run")

    stop_event, task = await getattr(startup_pollers, starter_name)(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (flag_name, route_name, route_kwargs),
        worker_inventory=_FakeWorkerInventory(),
    )

    assert stop_event == f"{registered_name}-stop"
    assert task == f"{registered_name}-task"
    assert registrations == [
        {
            "name": registered_name,
            "task_name": registered_name,
            "coroutine_factory": getattr(startup_pollers, factory_name),
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        }
    ]


@pytest.mark.asyncio
async def test_media_ingest_jobs_workers_register_with_worker_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    registrations: list[dict[str, object]] = []

    class _FakeWorkerInventory:
        """Test double that records custom worker registration calls."""

        async def register_custom(self, **kwargs: object) -> tuple[str, str]:
            """Capture registration kwargs and return deterministic handles."""

            registrations.append(kwargs)
            name = str(kwargs["name"])
            return f"{name}-task", f"{name}-stop"

    monkeypatch.setattr(
        startup_pollers,
        "_make_event",
        lambda: (_ for _ in ()).throw(AssertionError("legacy event path should not run")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: (_ for _ in ()).throw(AssertionError("legacy task path should not run")),
    )

    def _register_owned_job_poller(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy poller registration should not run")

    handles = await startup_pollers._start_media_ingest_jobs_workers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda *args, **kwargs: True,
        worker_inventory=_FakeWorkerInventory(),
    )

    assert handles == (
        "media_ingest_jobs_task-stop",
        "media_ingest_jobs_task-task",
        "media_ingest_heavy_jobs_task-stop",
        "media_ingest_heavy_jobs_task-task",
    )
    assert registrations == [
        {
            "name": "media_ingest_jobs_task",
            "task_name": "media_ingest_jobs_task",
            "coroutine_factory": startup_pollers._run_media_ingest_jobs_worker_service,
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        },
        {
            "name": "media_ingest_heavy_jobs_task",
            "task_name": "media_ingest_heavy_jobs_task",
            "coroutine_factory": startup_pollers._run_media_ingest_heavy_jobs_worker_service,
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        },
    ]


@pytest.mark.asyncio
async def test_visual_identity_jobs_worker_registers_with_worker_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    registrations: list[dict[str, object]] = []
    should_start_calls: list[tuple[str, str, dict[str, object]]] = []

    class _FakeWorkerInventory:
        """Test double that records custom worker registration calls."""

        async def register_custom(self, **kwargs: object) -> tuple[str, str]:
            """Capture registration kwargs and return deterministic handles."""

            registrations.append(kwargs)
            return "visual_identity_jobs_task-task", "visual_identity_jobs_task-stop"

    monkeypatch.setattr(
        startup_pollers,
        "_make_event",
        lambda: (_ for _ in ()).throw(AssertionError("legacy event path should not run")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: (_ for _ in ()).throw(AssertionError("legacy task path should not run")),
    )

    def _register_owned_job_poller(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy poller registration should not run")

    def _should_start_worker(flag_key: str, route_key: str, **kwargs: object) -> bool:
        should_start_calls.append((flag_key, route_key, kwargs))
        return True

    stop_event, task = await startup_pollers._start_visual_identity_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=_should_start_worker,
        worker_inventory=_FakeWorkerInventory(),
    )

    assert (stop_event, task) == (
        "visual_identity_jobs_task-stop",
        "visual_identity_jobs_task-task",
    )
    assert should_start_calls == [
        (
            "VISUAL_IDENTITY_JOBS_WORKER_ENABLED",
            "visual-identities",
            {"default_stable": True},
        )
    ]
    assert registrations == [
        {
            "name": "visual_identity_jobs_task",
            "task_name": "visual_identity_jobs_task",
            "coroutine_factory": startup_pollers._run_visual_identity_jobs_worker_service,
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        }
    ]


@pytest.mark.asyncio
async def test_vn_asset_jobs_workers_register_with_worker_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    registrations: list[dict[str, object]] = []

    class _FakeWorkerInventory:
        """Test double that records custom worker registration calls."""

        async def register_custom(self, **kwargs: object) -> tuple[str, str]:
            """Capture registration kwargs and return deterministic handles."""

            registrations.append(kwargs)
            name = str(kwargs["name"])
            return f"{name}-task", f"{name}-stop"

    monkeypatch.setattr(
        startup_pollers,
        "_make_event",
        lambda: (_ for _ in ()).throw(AssertionError("legacy event path should not run")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: (_ for _ in ()).throw(AssertionError("legacy task path should not run")),
    )

    def _register_owned_job_poller(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy poller registration should not run")

    handles = await startup_pollers._start_vn_asset_jobs_workers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda *args, **kwargs: True,
        worker_inventory=_FakeWorkerInventory(),
    )

    assert handles == (
        "vn_asset_jobs_task-stop",
        "vn_asset_jobs_task-task",
        "vn_asset_generation_jobs_task-stop",
        "vn_asset_generation_jobs_task-task",
    )
    assert registrations == [
        {
            "name": "vn_asset_jobs_task",
            "task_name": "vn_asset_jobs_task",
            "coroutine_factory": startup_pollers._run_vn_asset_jobs_worker_service,
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        },
        {
            "name": "vn_asset_generation_jobs_task",
            "task_name": "vn_asset_generation_jobs_task",
            "coroutine_factory": startup_pollers._run_vn_asset_generation_jobs_worker_service,
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        },
    ]


@pytest.mark.asyncio
async def test_start_audio_jobs_worker_registers_owned_poller_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []
    registrations: list[dict[str, object]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "audio-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "audio-task",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_audio_jobs_worker_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "audio-coro",
    )

    def _register_owned_job_poller(app, owned_job_pollers, *, name, task, stop_event):
        registrations.append(
            {
                "app": app,
                "owned_job_pollers": owned_job_pollers,
                "name": name,
                "task": task,
                "stop_event": stop_event,
            }
        )

    owned_job_pollers: list[object] = []
    stop_event, task = await startup_pollers._start_audio_jobs_worker(
        app="app",
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs)
        == (
            "AUDIO_JOBS_WORKER_ENABLED",
            "audio-jobs",
            {},
        ),
    )

    assert stop_event == "audio-stop"
    assert task == "audio-task"
    assert captured_stop_events == ["audio-stop"]
    assert created_coroutines == ["audio-coro"]
    assert registrations == [
        {
            "app": "app",
            "owned_job_pollers": owned_job_pollers,
            "name": "audio_jobs_task",
            "task": "audio-task",
            "stop_event": "audio-stop",
        }
    ]


@pytest.mark.asyncio
async def test_start_llamacpp_acquisition_jobs_worker_registers_owned_poller_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []
    registrations: list[dict[str, object]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "llamacpp-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "llamacpp-task",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_llamacpp_acquisition_jobs_worker_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "llamacpp-coro",
    )

    def _register_owned_job_poller(app, owned_job_pollers, *, name, task, stop_event):
        registrations.append(
            {
                "app": app,
                "owned_job_pollers": owned_job_pollers,
                "name": name,
                "task": task,
                "stop_event": stop_event,
            }
        )

    owned_job_pollers: list[object] = []
    stop_event, task = await startup_pollers._start_llamacpp_acquisition_jobs_worker(
        app="app",
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs)
        == (
            "LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED",
            "llamacpp-acquisition",
            {},
        ),
    )

    assert stop_event == "llamacpp-stop"
    assert task == "llamacpp-task"
    assert captured_stop_events == ["llamacpp-stop"]
    assert created_coroutines == ["llamacpp-coro"]
    assert registrations == [
        {
            "app": "app",
            "owned_job_pollers": owned_job_pollers,
            "name": "llamacpp_acquisition_jobs_task",
            "task": "llamacpp-task",
            "stop_event": "llamacpp-stop",
        }
    ]


@pytest.mark.asyncio
async def test_start_llamacpp_acquisition_jobs_worker_cancels_task_when_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    created_coroutines: list[object] = []

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    task = _FakeTask()
    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "llamacpp-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or task,
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_llamacpp_acquisition_jobs_worker_service",
        lambda stop_event: f"llamacpp-coro:{stop_event}",
    )

    def _register_owned_job_poller(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("registration failed")

    stop_event, returned_task = await startup_pollers._start_llamacpp_acquisition_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs)
        == (
            "LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED",
            "llamacpp-acquisition",
            {},
        ),
    )

    assert stop_event is None
    assert returned_task is None
    assert created_coroutines == ["llamacpp-coro:llamacpp-stop"]
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_start_vn_asset_jobs_workers_use_stable_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    stop_events = iter(["vn-stop", "vn-generation-stop"])
    calls: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: next(stop_events))
    monkeypatch.setattr(startup_pollers, "_create_task", lambda coro: f"task:{coro}")
    monkeypatch.setattr(
        startup_pollers,
        "_run_vn_asset_jobs_worker_service",
        lambda stop_event: f"vn-coro:{stop_event}",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_vn_asset_generation_jobs_worker_service",
        lambda stop_event: f"vn-generation-coro:{stop_event}",
    )

    def _should_start_worker(flag_key: str, route_key: str, **kwargs: object) -> bool:
        calls.append((flag_key, route_key, kwargs))
        return False

    handles = await startup_pollers._start_vn_asset_jobs_workers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=_should_start_worker,
    )

    assert handles == (None, None, None, None)
    assert calls == [
        ("VN_ASSET_JOBS_WORKER_ENABLED", "vn-assets", {"default_stable": True}),
        (
            "VN_ASSET_GENERATION_JOBS_WORKER_ENABLED",
            "vn-assets-generation",
            {"default_stable": True},
        ),
    ]


@pytest.mark.asyncio
async def test_start_media_ingest_jobs_workers_respects_heavy_default_stable_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    created_coroutines: list[object] = []
    registrations: list[dict[str, object]] = []
    calls: list[tuple[str, str, dict[str, object]]] = []
    stop_events = iter(["media-stop", "media-heavy-stop"])

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: next(stop_events))
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or f"task-{len(created_coroutines)}",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_media_ingest_jobs_worker_service",
        lambda stop_event: f"media-coro-{stop_event}",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_media_ingest_heavy_jobs_worker_service",
        lambda stop_event: f"media-heavy-coro-{stop_event}",
    )

    def _register_owned_job_poller(app, owned_job_pollers, *, name, task, stop_event):
        del app, owned_job_pollers
        registrations.append({"name": name, "task": task, "stop_event": stop_event})

    def _should_start_worker(flag, route, **kwargs):
        calls.append((flag, route, kwargs))
        return True

    handles = await startup_pollers._start_media_ingest_jobs_workers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=_should_start_worker,
    )

    assert handles == ("media-stop", "task-1", "media-heavy-stop", "task-2")
    assert created_coroutines == ["media-coro-media-stop", "media-heavy-coro-media-heavy-stop"]
    assert registrations == [
        {"name": "media_ingest_jobs_task", "task": "task-1", "stop_event": "media-stop"},
        {
            "name": "media_ingest_heavy_jobs_task",
            "task": "task-2",
            "stop_event": "media-heavy-stop",
        },
    ]
    assert calls == [
        ("MEDIA_INGEST_JOBS_WORKER_ENABLED", "media", {}),
        (
            "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
            "media-ingest-heavy-jobs",
            {"default_stable": False},
        ),
    ]


@pytest.mark.asyncio
async def test_start_media_ingest_jobs_workers_preserves_light_handles_when_heavy_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    stop_events = iter(["media-stop", "media-heavy-stop"])
    registrations: list[dict[str, object]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: next(stop_events))
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: "media-task" if coro == "media-coro" else (_ for _ in ()).throw(RuntimeError("heavy boom")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_media_ingest_jobs_worker_service",
        lambda stop_event: "media-coro",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_media_ingest_heavy_jobs_worker_service",
        lambda stop_event: "media-heavy-coro",
    )

    def _register_owned_job_poller(app, owned_job_pollers, *, name, task, stop_event):
        del app, owned_job_pollers
        registrations.append({"name": name, "task": task, "stop_event": stop_event})

    handles = await startup_pollers._start_media_ingest_jobs_workers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda *args, **kwargs: True,
    )

    assert handles == ("media-stop", "media-task", None, None)
    assert registrations == [{"name": "media_ingest_jobs_task", "task": "media-task", "stop_event": "media-stop"}]


@pytest.mark.asyncio
async def test_start_companion_reflection_jobs_worker_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "companion-stop")

    def _failing_create_task(coro):
        del coro
        raise RuntimeError("boom")

    monkeypatch.setattr(startup_pollers, "_create_task", _failing_create_task)
    monkeypatch.setattr(
        startup_pollers,
        "_run_companion_reflection_jobs_worker_service",
        lambda stop_event: stop_event,
    )

    stop_event, task = await startup_pollers._start_companion_reflection_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs)
        == (
            "COMPANION_REFLECTION_JOBS_WORKER_ENABLED",
            "companion",
            {},
        ),
    )

    assert stop_event is None
    assert task is None
