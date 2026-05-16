from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Research_Studio import capabilities
from tldw_Server_API.app.core.Research_Studio.capabilities import (
    RESEARCH_STUDIO_CAPABILITY_IDS,
    build_research_studio_capabilities,
)


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
    }
    inputs.update(overrides)
    return inputs


def test_ready_dependencies_allow_core_research_studio_actions():
    response = build_research_studio_capabilities(**_health_inputs())

    assert set(response.capabilities) == set(RESEARCH_STUDIO_CAPABILITY_IDS)
    assert response.capabilities["source_browse"].mode == "allow"
    assert response.capabilities["chat"].mode == "allow"
    assert response.capabilities["artifact_text_generation"].mode == "allow"
    assert response.capabilities["slides_generation"].mode == "allow"
    assert response.capabilities["audio_summary"].mode == "allow"
    assert response.capabilities["export_download"].mode == "allow"
    assert response.capabilities["sync_share"].status == "unknown"
    assert response.capabilities["sync_share"].mode == "warn"
    assert response.status == "degraded"
    assert response.ttl_seconds == 30


def test_unknown_source_health_warns_instead_of_overclaiming_or_blocking():
    response = build_research_studio_capabilities(
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
    response = build_research_studio_capabilities(
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
    response = build_research_studio_capabilities(
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
    response = build_research_studio_capabilities(
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
    response = build_research_studio_capabilities(
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
    response = build_research_studio_capabilities(
        **_health_inputs(rag_health={"status": "degraded"})
    )

    assert response.capabilities["chat"].status == "degraded"
    assert response.capabilities["chat"].mode == "warn"
    assert response.capabilities["chat"].reason_code == "rag_degraded"
    assert response.capabilities["artifact_text_generation"].mode == "allow"


def test_rag_unavailable_blocks_chat_as_dependency_failure():
    response = build_research_studio_capabilities(
        **_health_inputs(rag_health={"status": "unhealthy"})
    )

    assert response.capabilities["chat"].status == "unavailable"
    assert response.capabilities["chat"].mode == "block"
    assert response.capabilities["chat"].reason_code == "rag_unavailable"
    assert response.capabilities["artifact_text_generation"].mode == "allow"


def test_slides_and_tts_health_only_gate_their_artifact_types():
    response = build_research_studio_capabilities(
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


def test_capability_payload_does_not_leak_raw_errors_paths_or_secrets():
    response = build_research_studio_capabilities(
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


@pytest.mark.asyncio
async def test_tts_health_collection_uses_config_without_provider_initialization(monkeypatch):
    class FakeTtsConfigManager:
        def get_config(self):
            return SimpleNamespace(providers={"openai": object(), "kokoro": object()})

        def get_enabled_providers(self):
            return ["openai"]

    async def forbidden_setup_health():
        raise AssertionError("Research Studio capability collection must not initialize TTS providers")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_config.get_tts_config_manager",
        lambda: FakeTtsConfigManager(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.audio.audio_health.collect_setup_tts_health",
        forbidden_setup_health,
        raising=False,
    )

    result = await capabilities._collect_tts_health()

    assert result == {"status": "healthy", "providers": {"available": 1, "total": 2}}
