"""Gateway speech fallback policy tests."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.TTS.gateway_config import normalize_gateway_specs
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAudioQualityError,
    TTSAuthenticationError,
    TTSModelNotFoundError,
    TTSNetworkError,
    TTSProviderError,
    TTSQuotaExceededError,
    TTSRateLimitError,
    TTSTimeoutError,
    TTSValidationError,
)
from tldw_Server_API.tests.TTS_NEW.unit.service.test_tts_gateway_execution import (
    MP3,
    FakeCircuitManager,
    FakeRegistry,
    collect,
    executor,
    gateway_config,
    request,
    specs,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_primary_policy_order_ignores_target_policy_and_skips_do_not_consume_posts() -> None:
    gateway_specs = normalize_gateway_specs(
        {},
        {
            "primary": gateway_config(
                "Primary/Model",
                "PrimaryVoice",
                fallback={
                    "on": ["timeout"],
                    "max_attempts": 2,
                    "targets": [
                        {"backend": "disabled", "model": "Disabled/Model", "voice": "DisabledVoice"},
                        {"backend": "target", "model": "Target/Model", "voice": "TargetVoice"},
                    ],
                },
            ),
            "disabled": gateway_config("Disabled/Model", "DisabledVoice", enabled=False),
            "target": gateway_config(
                "Target/Model",
                "TargetVoice",
                fallback={
                    "on": ["timeout"],
                    "max_attempts": 2,
                    "targets": [{"backend": "last", "model": "Last/Model", "voice": "LastVoice"}],
                },
            ),
            "last": gateway_config("Last/Model", "LastVoice"),
        },
    )
    primary_error = TTSTimeoutError("private primary detail")
    registry = FakeRegistry(
        {
            "gateway:primary": [(primary_error,)],
            "gateway:disabled": [(MP3,)],
            "gateway:target": [(MP3,)],
            "gateway:last": [(MP3,)],
        }
    )
    result = await executor(registry, gateway_specs).execute(request(stream=False), user_id=1)

    assert await collect(result) == MP3
    assert [backend for backend, _overrides in registry.calls] == [
        "gateway:primary",
        "gateway:target",
    ]
    assert sum(adapter.posts for adapter in registry.adapters) == 2
    assert "gateway:last" not in [backend for backend, _overrides in registry.calls]


@pytest.mark.asyncio
async def test_attempts_get_fresh_requests_credentials_and_target_drops_primary_options() -> None:
    gateway_specs = specs(
        primary_fallback={
            "on": ["timeout"],
            "max_attempts": 2,
            "targets": [{"backend": "target", "model": "Target/Model", "voice": "TargetVoice"}],
        }
    )
    registry = FakeRegistry(
        {
            "gateway:primary": [(TTSTimeoutError("private"),)],
            "gateway:target": [(MP3,)],
        }
    )
    touches: list[str] = []
    original = request(
        stream=False,
        speed=1.25,
        language="en-GB",
        target_sample_rate=24000,
        extra_params={"provider": {"style": "primary-only"}},
    )
    result = await executor(registry, gateway_specs, touches=touches).execute(original, user_id=9)

    assert await collect(result) == MP3
    first, second = [adapter.requests[0] for adapter in registry.adapters]
    assert first is not second and first is not original and second is not original
    assert (first.model, first.voice) == ("Primary/Model", "PrimaryVoice")
    assert (second.model, second.voice) == ("Target/Model", "TargetVoice")
    assert first.extra_params == {"provider": {"style": "primary-only"}}
    assert second.extra_params == {}
    assert second.speed == 1.25 and second.language == "en-GB"
    assert second.target_sample_rate == 24000
    assert registry.calls[0][1]["api_key"] == "primary-key"
    assert registry.calls[1][1]["api_key"] == "target-key"
    assert touches == ["gateway:primary", "gateway:target"]


@pytest.mark.asyncio
async def test_all_skipped_targets_reraises_original_primary_error_object() -> None:
    gateway_specs = specs(
        primary_fallback={
            "on": ["timeout"],
            "max_attempts": 4,
            "targets": [
                {"backend": "target", "model": "Target/Model", "voice": "TargetVoice"},
                {"backend": "last", "model": "Last/Model", "voice": "LastVoice"},
            ],
        },
        target_enabled=False,
    )
    original_error = TTSTimeoutError("private original")
    registry = FakeRegistry(
        {
            "gateway:primary": [(original_error,)],
            "gateway:target": [(MP3,)],
            "gateway:last": [(MP3,)],
        }
    )
    result = await executor(
        registry,
        gateway_specs,
        keys={
            "gateway:primary": "primary-key",
            "gateway:target": "target-key",
            "gateway:last": None,
        },
    ).execute(request(stream=False), user_id=1)

    with pytest.raises(TTSTimeoutError) as raised:
        await collect(result)
    assert raised.value is original_error
    assert [adapter.posts for adapter in registry.adapters] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "error"),
    [
        ("timeout", TTSTimeoutError("private")),
        ("network_error", TTSNetworkError("private")),
        ("upstream_5xx", TTSProviderError("private", error_code="503")),
        ("rate_limited", TTSRateLimitError("private")),
        ("quota_exceeded", TTSQuotaExceededError("private")),
        ("authentication_failed", TTSAuthenticationError("private")),
        ("model_not_found", TTSModelNotFoundError("private")),
        ("invalid_audio", TTSAudioQualityError("private", error_code="INVALID_AUDIO")),
    ],
)
async def test_each_attempted_stable_category_can_advance(category: str, error: BaseException) -> None:
    gateway_specs = specs(
        primary_fallback={
            "on": [category],
            "max_attempts": 2,
            "targets": [{"backend": "target", "model": "Target/Model", "voice": "TargetVoice"}],
        }
    )
    registry = FakeRegistry(
        {"gateway:primary": [(error,)], "gateway:target": [(MP3,)]}
    )
    result = await executor(registry, gateway_specs).execute(request(stream=False), user_id=1)

    assert await collect(result) == MP3
    assert result.metadata["failure_category"] == category
    assert result.metadata["attempt_count"] == 2


@pytest.mark.asyncio
async def test_primary_circuit_open_can_fallback_without_consuming_post_and_open_target_skips() -> None:
    gateway_specs = specs(
        primary_fallback={
            "on": ["circuit_open"],
            "max_attempts": 1,
            "targets": [
                {"backend": "target", "model": "Target/Model", "voice": "TargetVoice"},
                {"backend": "last", "model": "Last/Model", "voice": "LastVoice"},
            ],
        }
    )
    registry = FakeRegistry(
        {
            "gateway:primary": [(MP3,)],
            "gateway:target": [(MP3,)],
            "gateway:last": [(MP3,)],
        }
    )
    circuit = FakeCircuitManager(opened={"gateway:primary", "gateway:target"})
    result = await executor(registry, gateway_specs, circuit=circuit).execute(
        request(stream=False),
        user_id=1,
    )

    assert await collect(result) == MP3
    assert [backend for backend, _overrides in registry.calls] == ["gateway:last"]
    assert registry.adapters[0].posts == 1
    assert result.metadata["failure_category"] == "circuit_open"


@pytest.mark.asyncio
async def test_primary_local_authorization_is_terminal_and_target_incompatibility_is_skip() -> None:
    gateway_specs = specs(
        primary_fallback={
            "on": ["model_not_found", "timeout"],
            "max_attempts": 3,
            "targets": [
                {"backend": "target", "model": "Target/Model", "voice": "TargetVoice"},
                {"backend": "last", "model": "Last/Model", "voice": "LastVoice"},
            ],
        },
        target_supports_speed=False,
    )
    local_registry = FakeRegistry(
        {"gateway:primary": [(MP3,)], "gateway:target": [(MP3,)], "gateway:last": [(MP3,)]}
    )
    local = request(model="Not/Authorized")
    local_result = await executor(local_registry, gateway_specs).execute(local, user_id=1)
    with pytest.raises(TTSValidationError):
        await collect(local_result)
    assert local_registry.adapters == []

    registry = FakeRegistry(
        {
            "gateway:primary": [(TTSTimeoutError("private"),)],
            "gateway:target": [(MP3,)],
            "gateway:last": [(MP3,)],
        }
    )
    result = await executor(registry, gateway_specs).execute(
        request(stream=False, speed=1.25),
        user_id=1,
    )
    assert await collect(result) == MP3
    assert [backend for backend, _overrides in registry.calls] == [
        "gateway:primary",
        "gateway:last",
    ]


@pytest.mark.asyncio
async def test_user_auth_failure_uses_only_separate_target_credential() -> None:
    gateway_specs = specs(
        primary_fallback={
            "on": ["authentication_failed"],
            "max_attempts": 2,
            "targets": [{"backend": "target", "model": "Target/Model", "voice": "TargetVoice"}],
        }
    )
    registry = FakeRegistry(
        {
            "gateway:primary": [(TTSAuthenticationError("private"),)],
            "gateway:target": [(MP3,)],
        }
    )
    result = await executor(
        registry,
        gateway_specs,
        keys={"gateway:primary": "user-key", "gateway:target": "target-admin-key"},
        sources={"gateway:primary": "user", "gateway:target": "server_default"},
    ).execute(request(stream=False), user_id=11)

    assert await collect(result) == MP3
    assert [overrides["api_key"] for _backend, overrides in registry.calls] == [
        "user-key",
        "target-admin-key",
    ]
    assert [backend for backend, _overrides in registry.calls].count("gateway:primary") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("private internal detail"),
        TTSValidationError("private validation detail"),
        TTSAudioQualityError(
            "private size detail",
            error_code="RESPONSE_SIZE_EXCEEDED",
        ),
    ],
)
async def test_unknown_local_and_response_size_errors_never_fallback(error: BaseException) -> None:
    gateway_specs = specs(
        primary_fallback={
            "on": ["invalid_audio", "upstream_5xx"],
            "max_attempts": 2,
            "targets": [{"backend": "target", "model": "Target/Model", "voice": "TargetVoice"}],
        }
    )
    registry = FakeRegistry(
        {"gateway:primary": [(error,)], "gateway:target": [(MP3,)]}
    )
    result = await executor(registry, gateway_specs).execute(request(stream=False), user_id=1)

    with pytest.raises(type(error)) as raised:
        await collect(result)
    assert raised.value is error
    assert len(registry.adapters) == 1


@pytest.mark.asyncio
async def test_circuit_records_only_network_timeout_and_upstream_5xx() -> None:
    for error, expected_failure_count in (
        (TTSNetworkError("private"), 1),
        (TTSTimeoutError("private"), 1),
        (TTSProviderError("private", error_code="500"), 1),
        (TTSAuthenticationError("private"), 0),
        (TTSRateLimitError("private"), 0),
        (TTSAudioQualityError("private", error_code="INVALID_AUDIO"), 0),
    ):
        gateway_specs = specs()
        registry = FakeRegistry({"gateway:primary": [(error,)]})
        circuit = FakeCircuitManager()
        result = await executor(registry, gateway_specs, circuit=circuit).execute(
            request(stream=False),
            user_id=1,
        )
        with pytest.raises(type(error)):
            await collect(result)

        breaker = circuit.breakers["gateway:primary"]
        assert len(breaker.failures) == expected_failure_count
        assert breaker.releases == (0 if expected_failure_count else 1)
