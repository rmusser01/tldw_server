from __future__ import annotations

import asyncio
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any, get_args

import pytest

from tldw_Server_API.app.api.v1.schemas.research_workspace_capabilities import (
    ResearchWorkspaceCapabilitiesResponse,
    ResearchWorkspaceCapabilityId,
)
from tldw_Server_API.app.core.Research_Workspace import capabilities
from tldw_Server_API.app.core.Research_Workspace.capabilities import (
    RESEARCH_WORKSPACE_CAPABILITY_IDS,
    ResearchWorkspaceHealthCollectors,
    build_research_workspace_capabilities,
    collect_research_workspace_capabilities,
)

pytestmark = pytest.mark.unit


def _health_inputs(**overrides):
    inputs = {
        "aggregate_health": {
            "status": "ok",
            "checks": {
                "database": {"status": "healthy"},
                "chacha_notes": {"status": "healthy"},
            },
        },
        "rag_health": {"status": "healthy"},
        "llm_health": {
            "status": "healthy",
            "components": {
                "providers": {
                    "initialized": True,
                    "count": 1,
                    "report": {"local": {"status": "healthy"}},
                },
            },
        },
        "slides_health": {"status": "ok"},
        "tts_health": {"status": "healthy", "providers": {"available": 1}},
        "render_health": {"status": "healthy"},
        "image_health": {"status": "healthy", "providers": {"available": 1}},
    }
    inputs.update(overrides)
    return inputs


def test_ready_dependencies_allow_core_research_workspace_actions():
    response = build_research_workspace_capabilities(**_health_inputs())

    assert set(response.capabilities) == set(RESEARCH_WORKSPACE_CAPABILITY_IDS)
    assert response.capabilities["source_browse"].mode == "allow"
    assert response.capabilities["chat"].mode == "allow"
    assert response.capabilities["artifact_text_generation"].mode == "allow"
    assert response.capabilities["slides_generation"].mode == "allow"
    assert response.capabilities["audio_summary"].mode == "allow"
    assert response.capabilities["export_download"].mode == "allow"
    assert response.capabilities["sync_share"].status == "unknown"
    assert response.capabilities["sync_share"].mode == "warn"
    assert response.status == "ready"
    assert response.ttl_seconds == 30


def test_ready_dependencies_allow_media_output_actions():
    response = build_research_workspace_capabilities(
        **_health_inputs(
            render_health={"status": "healthy"},
            image_health={"status": "healthy", "providers": {"available": 1}},
        )
    )

    assert response.capabilities["image_generation"].mode == "allow"
    assert response.capabilities["infographic_generation"].mode == "allow"
    assert response.capabilities["video_overview_generation"].mode == "allow"


def test_image_unavailable_blocks_infographic_only():
    response = build_research_workspace_capabilities(
        **_health_inputs(image_health={"status": "unavailable", "reason_code": "image_backend_unavailable"})
    )

    assert response.capabilities["image_generation"].mode == "block"
    assert response.capabilities["infographic_generation"].mode == "block"
    assert response.capabilities["video_overview_generation"].mode == "allow"
    assert response.capabilities["slides_generation"].mode == "allow"


def test_image_enabled_without_model_reports_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Image_Generation import adapter_registry, listing

    class _Registry:
        def list_backend_names(self, *, include_disabled: bool = False) -> list[str]:
            assert include_disabled is False
            return ["stable_diffusion_cpp"]

    def _get_registry() -> _Registry:
        return _Registry()

    def _list_image_models_for_catalog() -> list[dict[str, Any]]:
        return [{"name": "stable_diffusion_cpp", "is_configured": False}]

    monkeypatch.setattr(adapter_registry, "get_registry", _get_registry)
    monkeypatch.setattr(
        listing,
        "list_image_models_for_catalog",
        _list_image_models_for_catalog,
    )

    image_health = capabilities._collect_image_health()
    response = build_research_workspace_capabilities(**_health_inputs(image_health=image_health))

    assert image_health["reason_code"] == "image_backend_not_configured"
    assert response.capabilities["infographic_generation"].mode == "block"
    assert response.capabilities["infographic_generation"].reason_code == "image_backend_not_configured"


def test_ffmpeg_unavailable_blocks_video_overview_only():
    response = build_research_workspace_capabilities(
        **_health_inputs(
            render_health={"status": "unavailable", "reason_code": "presentation_render_ffmpeg_unavailable"}
        )
    )

    assert response.capabilities["video_overview_generation"].mode == "block"
    assert response.capabilities["video_overview_generation"].reason_code == "presentation_render_ffmpeg_unavailable"
    assert response.capabilities["infographic_generation"].mode == "allow"


def test_tts_unavailable_blocks_audio_and_video_only():
    response = build_research_workspace_capabilities(
        **_health_inputs(tts_health={"status": "error", "providers": {"available": 0}})
    )

    assert response.capabilities["audio_summary"].mode == "block"
    assert response.capabilities["video_overview_generation"].mode == "block"
    assert response.capabilities["video_overview_generation"].reason_code == "tts_unavailable"
    assert response.capabilities["slides_generation"].mode == "allow"
    assert response.capabilities["infographic_generation"].mode == "allow"


def test_tts_without_enabled_providers_skips_runtime_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tts_config = ModuleType("tldw_Server_API.app.core.TTS.tts_config")
    fake_adapter_registry = ModuleType("tldw_Server_API.app.core.TTS.adapter_registry")

    class _Manager:
        def get_config(self) -> SimpleNamespace:
            return SimpleNamespace(providers={"kitten_tts": SimpleNamespace(enabled=False)})

        def get_enabled_providers(self) -> list[str]:
            return []

    def _get_existing_tts_factory() -> None:
        raise AssertionError("runtime TTS probe should be skipped")

    def _get_tts_config_manager() -> _Manager:
        return _Manager()

    fake_tts_config.get_tts_config_manager = _get_tts_config_manager
    fake_adapter_registry.get_existing_tts_factory = _get_existing_tts_factory
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.TTS.tts_config", fake_tts_config)
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.TTS.adapter_registry", fake_adapter_registry)

    tts_health = asyncio.run(capabilities._collect_tts_health())

    assert tts_health == {"status": "unhealthy", "providers": {"total": 1, "available": 0}}


def test_tts_failed_enabled_provider_blocks_media_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tts_config = ModuleType("tldw_Server_API.app.core.TTS.tts_config")
    fake_adapter_registry = ModuleType("tldw_Server_API.app.core.TTS.adapter_registry")

    class _Manager:
        def get_config(self) -> SimpleNamespace:
            return SimpleNamespace(providers={"kitten_tts": SimpleNamespace(enabled=True)})

        def get_enabled_providers(self) -> list[str]:
            return ["kitten_tts"]

    class _Factory:
        def get_status(self) -> dict[str, Any]:
            return {
                "providers": {
                    "kitten_tts": {
                        "failed": True,
                        "status": "error",
                    }
                }
            }

    def _get_existing_tts_factory() -> _Factory:
        return _Factory()

    def _get_tts_config_manager() -> _Manager:
        return _Manager()

    fake_tts_config.get_tts_config_manager = _get_tts_config_manager
    fake_adapter_registry.get_existing_tts_factory = _get_existing_tts_factory
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.TTS.tts_config", fake_tts_config)
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.TTS.adapter_registry", fake_adapter_registry)

    tts_health = asyncio.run(capabilities._collect_tts_health())
    response = build_research_workspace_capabilities(**_health_inputs(tts_health=tts_health))

    assert tts_health["reason_code"] == "tts_provider_failed"
    assert response.capabilities["audio_summary"].mode == "block"
    assert response.capabilities["video_overview_generation"].mode == "block"
    assert response.capabilities["video_overview_generation"].reason_code == "tts_provider_failed"


def test_unknown_source_health_warns_instead_of_overclaiming_or_blocking():
    response = build_research_workspace_capabilities(
        **_health_inputs(
            aggregate_health={
                "status": "ok",
                "checks": {"database": {"status": "healthy"}},
            }
        )
    )

    assert response.capabilities["source_browse"].status == "unknown"
    assert response.capabilities["source_browse"].mode == "warn"
    assert response.capabilities["chat"].mode == "warn"
    assert response.capabilities["artifact_text_generation"].mode == "warn"
    assert response.capabilities["slides_generation"].mode == "warn"
    assert response.capabilities["audio_summary"].mode == "warn"
    assert response.capabilities["export_download"].mode == "allow"


def test_known_source_unavailable_blocks_source_dependent_actions():
    response = build_research_workspace_capabilities(
        **_health_inputs(
            aggregate_health={
                "status": "degraded",
                "checks": {
                    "database": {"status": "healthy"},
                    "chacha_notes": {"status": "unhealthy"},
                },
            }
        )
    )

    assert response.status == "unavailable"
    assert response.capabilities["source_browse"].mode == "block"
    assert response.capabilities["chat"].mode == "block"
    assert response.capabilities["artifact_text_generation"].mode == "block"
    assert response.capabilities["slides_generation"].mode == "block"
    assert response.capabilities["audio_summary"].mode == "block"
    assert response.capabilities["export_download"].mode == "allow"


def test_database_unavailable_blocks_source_dependent_actions():
    response = build_research_workspace_capabilities(
        **_health_inputs(
            aggregate_health={
                "status": "degraded",
                "checks": {
                    "database": {"status": "unhealthy"},
                    "chacha_notes": {"status": "healthy"},
                },
            }
        )
    )

    assert response.status == "unavailable"
    assert response.capabilities["source_browse"].reason_code == "source_store_unavailable"
    assert response.capabilities["source_browse"].mode == "block"
    assert response.capabilities["chat"].mode == "block"
    assert response.capabilities["artifact_text_generation"].mode == "block"


def test_database_degraded_warns_source_dependent_actions():
    response = build_research_workspace_capabilities(
        **_health_inputs(
            aggregate_health={
                "status": "degraded",
                "checks": {
                    "database": {"status": "degraded"},
                    "chacha_notes": {"status": "healthy"},
                },
            }
        )
    )

    assert response.status == "degraded"
    assert response.capabilities["source_browse"].reason_code == "source_store_degraded"
    assert response.capabilities["source_browse"].mode == "warn"
    assert response.capabilities["chat"].mode == "warn"
    assert response.capabilities["artifact_text_generation"].mode == "warn"


def test_llm_unavailable_blocks_generation_but_not_read_only_browsing():
    response = build_research_workspace_capabilities(
        **_health_inputs(
            llm_health={
                "status": "unhealthy",
                "components": {"providers": {"initialized": False, "count": 0}},
            }
        )
    )

    assert response.capabilities["source_browse"].mode == "allow"
    assert response.capabilities["chat"].reason_code == "llm_unavailable"
    assert response.capabilities["chat"].mode == "block"
    assert response.capabilities["artifact_text_generation"].mode == "block"
    assert response.capabilities["slides_generation"].mode == "block"
    assert response.capabilities["audio_summary"].mode == "block"
    assert response.capabilities["export_download"].mode == "allow"


def test_rag_degraded_warns_chat_without_blocking_text_artifacts():
    response = build_research_workspace_capabilities(**_health_inputs(rag_health={"status": "degraded"}))

    assert response.capabilities["chat"].status == "degraded"
    assert response.capabilities["chat"].mode == "warn"
    assert response.capabilities["chat"].reason_code == "rag_degraded"
    assert response.capabilities["artifact_text_generation"].mode == "allow"


def test_rag_unavailable_blocks_chat_as_dependency_failure():
    response = build_research_workspace_capabilities(**_health_inputs(rag_health={"status": "unhealthy"}))

    assert response.capabilities["chat"].status == "unavailable"
    assert response.capabilities["chat"].mode == "block"
    assert response.capabilities["chat"].reason_code == "rag_unavailable"
    assert response.capabilities["artifact_text_generation"].mode == "allow"


def test_unknown_dependency_health_preserves_probe_reason_code():
    response = build_research_workspace_capabilities(
        **_health_inputs(rag_health={"status": "unknown", "reason_code": "rag_health_timeout"})
    )

    assert response.capabilities["chat"].status == "unknown"
    assert response.capabilities["chat"].mode == "warn"
    assert response.capabilities["chat"].reason_code == "rag_health_timeout"
    assert response.capabilities["artifact_text_generation"].mode == "allow"


def test_dependency_health_rejects_unsafe_reason_code_values():
    response = build_research_workspace_capabilities(
        **_health_inputs(rag_health={"status": "unknown", "reason_code": "/tmp/secret"})
    )

    assert response.capabilities["chat"].reason_code == "rag_unknown"
    assert "/tmp/secret" not in response.model_dump_json()


def test_slides_and_tts_health_gate_media_artifact_types():
    response = build_research_workspace_capabilities(
        **_health_inputs(
            slides_health={"status": "unhealthy"},
            tts_health={"status": "error", "providers": {"available": 0}},
        )
    )

    assert response.capabilities["chat"].mode == "allow"
    assert response.capabilities["artifact_text_generation"].mode == "allow"
    assert response.capabilities["slides_generation"].reason_code == "slides_unavailable"
    assert response.capabilities["slides_generation"].mode == "block"
    assert response.capabilities["audio_summary"].reason_code == "tts_unavailable"
    assert response.capabilities["audio_summary"].mode == "block"
    assert response.capabilities["video_overview_generation"].reason_code == "slides_unavailable"
    assert response.capabilities["video_overview_generation"].mode == "block"


def test_slides_db_lookup_failure_blocks_slides_generation(monkeypatch):
    fake_slides_deps = ModuleType("tldw_Server_API.app.api.v1.API_Deps.Slides_DB_Deps")

    def fake_try_get_slides_db_for_user(current_user: object) -> None:
        assert current_user is not None
        return None

    fake_slides_deps.try_get_slides_db_for_user = fake_try_get_slides_db_for_user
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.api.v1.API_Deps.Slides_DB_Deps",
        fake_slides_deps,
    )

    slides_health = capabilities._collect_slides_health(user_id=42)
    response = build_research_workspace_capabilities(**_health_inputs(slides_health=slides_health))

    assert slides_health["status"] == "unavailable"
    assert response.capabilities["slides_generation"].reason_code == "slides_unavailable"
    assert response.capabilities["slides_generation"].mode == "block"
    assert response.capabilities["artifact_text_generation"].mode == "allow"


def test_slides_health_collector_uses_source_free_database_probe(monkeypatch):
    fake_slides_deps = ModuleType("tldw_Server_API.app.api.v1.API_Deps.Slides_DB_Deps")

    class _SlidesDB:
        def __init__(self) -> None:
            self.probe_calls = 0

        def probe_health(self) -> None:
            self.probe_calls += 1

        def list_presentations(self, **_kwargs):
            raise AssertionError("health must not load presentation detail rows")

    db = _SlidesDB()

    def fake_try_get_slides_db_for_user(current_user: object) -> _SlidesDB:
        assert current_user is not None
        return db

    fake_slides_deps.try_get_slides_db_for_user = fake_try_get_slides_db_for_user
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.api.v1.API_Deps.Slides_DB_Deps",
        fake_slides_deps,
    )

    health = capabilities._collect_slides_health(user_id=42)

    assert health == {"status": "ok"}
    assert db.probe_calls == 1


def test_capability_payload_does_not_leak_raw_errors_paths_or_secrets():
    response = build_research_workspace_capabilities(
        **_health_inputs(
            aggregate_health={
                "status": "unhealthy",
                "error": "Traceback /Users/private/key.py",
                "checks": {
                    "database": {"status": "healthy"},
                    "chacha_notes": {
                        "status": "unhealthy",
                        "error": "sqlite path /Users/private/ChaChaNotes.db",
                    },
                },
            },
            rag_health={"status": "unhealthy", "error": "stack trace"},
            llm_health={
                "status": "unhealthy",
                "components": {
                    "providers": {
                        "initialized": True,
                        "count": 1,
                        "report": {"openai": {"api_key": "secret", "status": "unhealthy"}},
                    }
                },
            },
            slides_health={"status": "unhealthy", "detail": "/tmp/secret"},
            tts_health={"status": "error", "message": "provider failed with secret"},
        )
    )

    dumped = response.model_dump_json()
    assert "Traceback" not in dumped
    assert "/Users/" not in dumped
    assert "/tmp/" not in dumped
    assert "secret" not in dumped
    assert "api_key" not in dumped


def test_capability_response_keys_are_typed_as_known_capability_ids():
    annotation = ResearchWorkspaceCapabilitiesResponse.model_fields["capabilities"].annotation
    key_type = get_args(annotation)[0]

    assert get_args(key_type) == get_args(ResearchWorkspaceCapabilityId)


def test_core_capability_collector_does_not_import_api_endpoint_health_functions():
    source = capabilities.__loader__.get_source(capabilities.__name__)

    assert source is not None
    assert "app.api.v1.endpoints.health" not in source
    assert "app.api.v1.endpoints.rag_health" not in source
    assert "app.api.v1.endpoints.llm_providers" not in source


@pytest.mark.asyncio
async def test_tts_health_collection_uses_config_without_provider_initialization(monkeypatch):
    class FakeTtsConfigManager:
        def get_config(self):
            return SimpleNamespace(providers={"openai": object(), "kokoro": object()})

        def get_enabled_providers(self):
            return ["openai"]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_config.get_tts_config_manager",
        lambda: FakeTtsConfigManager(),
    )

    result = await capabilities._collect_tts_health()

    assert result == {"status": "healthy", "providers": {"available": 1, "total": 2}}


@pytest.mark.asyncio
async def test_capability_collection_runs_bounded_independent_probes_concurrently():
    started: list[str] = []

    async def slow_aggregate_health():
        started.append("aggregate")
        await asyncio.sleep(0.2)
        return {
            "status": "ok",
            "checks": {
                "database": {"status": "healthy"},
                "chacha_notes": {"status": "healthy"},
            },
        }

    async def ready_rag_health():
        started.append("rag")
        await asyncio.sleep(0.01)
        return {"status": "healthy"}

    async def ready_llm_health():
        started.append("llm")
        await asyncio.sleep(0.01)
        return {
            "status": "healthy",
            "components": {"providers": {"initialized": True, "count": 1}},
        }

    def ready_slides_health(*, user_id: int | str | None = None):
        started.append(f"slides:{user_id}")
        time.sleep(0.01)
        return {"status": "ok"}

    async def ready_tts_health():
        started.append("tts")
        await asyncio.sleep(0.01)
        return {"status": "healthy", "providers": {"available": 1}}

    async def ready_render_health():
        started.append("render")
        await asyncio.sleep(0.01)
        return {"status": "healthy"}

    async def ready_image_health():
        started.append("image")
        await asyncio.sleep(0.01)
        return {"status": "healthy", "providers": {"available": 1}}

    collectors = ResearchWorkspaceHealthCollectors(
        aggregate_health=slow_aggregate_health,
        rag_health=ready_rag_health,
        llm_health=ready_llm_health,
        slides_health=ready_slides_health,
        tts_health=ready_tts_health,
        render_health=ready_render_health,
        image_health=ready_image_health,
    )

    started_at = time.perf_counter()
    response = await collect_research_workspace_capabilities(
        user_id=42,
        collectors=collectors,
        probe_timeout_seconds=0.05,
    )
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.15
    assert set(started) == {"aggregate", "rag", "llm", "slides:42", "tts", "render", "image"}
    assert response.capabilities["source_browse"].status == "unknown"
    assert response.capabilities["chat"].mode == "warn"


@pytest.mark.asyncio
async def test_slides_probe_timeout_blocks_slides_generation():
    async def ready_aggregate_health():
        return {
            "status": "ok",
            "checks": {
                "database": {"status": "healthy"},
                "chacha_notes": {"status": "healthy"},
            },
        }

    async def ready_rag_health():
        return {"status": "healthy"}

    async def ready_llm_health():
        return {
            "status": "healthy",
            "components": {"providers": {"initialized": True, "count": 1}},
        }

    async def slow_slides_health(*, user_id: int | str | None = None):
        assert user_id == 42
        await asyncio.sleep(0.2)
        return {"status": "ok"}

    async def ready_tts_health():
        return {"status": "healthy", "providers": {"available": 1}}

    async def ready_render_health():
        return {"status": "healthy"}

    async def ready_image_health():
        return {"status": "healthy", "providers": {"available": 1}}

    collectors = ResearchWorkspaceHealthCollectors(
        aggregate_health=ready_aggregate_health,
        rag_health=ready_rag_health,
        llm_health=ready_llm_health,
        slides_health=slow_slides_health,
        tts_health=ready_tts_health,
        render_health=ready_render_health,
        image_health=ready_image_health,
    )

    response = await collect_research_workspace_capabilities(
        user_id=42,
        collectors=collectors,
        probe_timeout_seconds=0.01,
    )

    assert response.capabilities["slides_generation"].status == "unavailable"
    assert response.capabilities["slides_generation"].mode == "block"
    assert response.capabilities["slides_generation"].reason_code == "slides_health_timeout"
    assert response.capabilities["video_overview_generation"].mode == "block"
    assert response.capabilities["video_overview_generation"].reason_code == "slides_health_timeout"
    assert response.capabilities["artifact_text_generation"].mode == "allow"


@pytest.mark.asyncio
async def test_async_callable_collector_instances_are_invoked_without_thread_pool(monkeypatch):
    class AsyncCollector:
        async def __call__(self, *, marker: str):
            return {"status": "healthy", "marker": marker}

    async def forbidden_to_thread(*args, **kwargs):
        raise AssertionError("async callable instances must not be sent to a thread pool")

    monkeypatch.setattr(capabilities.asyncio, "to_thread", forbidden_to_thread)

    result = await capabilities._invoke_health_collector(AsyncCollector(), marker="probe")

    assert result == {"status": "healthy", "marker": "probe"}


@pytest.mark.asyncio
async def test_aggregate_health_collects_chacha_snapshot_off_thread(monkeypatch):
    called: list[str] = []

    class FakePool:
        async def health_check(self):
            return {"status": "healthy"}

    async def get_db_pool():
        return FakePool()

    def get_chacha_health_snapshot():
        return {"status": "healthy"}

    fake_auth_database = ModuleType("tldw_Server_API.app.core.AuthNZ.database")
    fake_auth_database.get_db_pool = get_db_pool
    fake_chacha_deps = ModuleType("tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps")
    fake_chacha_deps.get_chacha_health_snapshot = get_chacha_health_snapshot

    async def tracking_to_thread(func, *args, **kwargs):
        called.append(func.__name__)
        return func(*args, **kwargs)

    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.AuthNZ.database", fake_auth_database)
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps",
        fake_chacha_deps,
    )
    monkeypatch.setattr(capabilities.asyncio, "to_thread", tracking_to_thread)

    result = await capabilities._collect_aggregate_health()

    assert result == {
        "status": "ok",
        "checks": {
            "database": {"status": "healthy"},
            "chacha_notes": {"status": "healthy"},
        },
    }
    assert called == ["get_chacha_health_snapshot"]
