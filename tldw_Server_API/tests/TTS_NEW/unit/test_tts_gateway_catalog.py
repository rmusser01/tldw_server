"""Tests for bounded, credential-scoped TTS gateway model discovery."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from tldw_Server_API.app.core.http_client import RetryPolicy
from tldw_Server_API.app.core.TTS import gateway_catalog as catalog_module
from tldw_Server_API.app.core.TTS.gateway_catalog import (
    MAX_DISCOVERY_BYTES,
    MAX_DISCOVERY_MODEL_ID_LENGTH,
    MAX_DISCOVERY_MODELS,
    GatewayCatalog,
)
from tldw_Server_API.app.core.TTS.gateway_config import normalize_gateway_specs


class FakeClock:
    """Small controllable wall clock for TTL tests."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _spec(
    slug: str = "company",
    *,
    discovery_enabled: bool = True,
    models_path: str | None = "models",
    ttl_seconds: int = 10,
    stale_ttl_seconds: int = 30,
    allowed_models: list[str] | None = None,
    allow_discovered_models: bool = True,
    default_model: str | None = "Configured/Default",
    model_overrides: dict[str, Any] | None = None,
    display_name: str | None = None,
):
    gateway: dict[str, Any] = {
        "enabled": True,
        "display_name": display_name or f"{slug} speech",
        "base_url": f"https://{slug}.example/v1/",
        "speech_path": "audio/speech",
        "headers": {"X-Route": f"{slug}-route"},
        "api_key": "admin-secret",
        "allow_user_api_key": True,
        "default_model": default_model,
        "default_voice": "Narrator",
        "allowed_models": allowed_models,
        "allow_discovered_models": allow_discovered_models,
        "model_overrides": model_overrides or {"Configured/Overlay": {"default_voice": "Guide"}},
        "discovery": {
            "enabled": discovery_enabled,
            "models_path": models_path,
            "query": {"output_modalities": "speech", "region": "west"},
            "ttl_seconds": ttl_seconds,
            "stale_ttl_seconds": stale_ttl_seconds,
            "timeout_seconds": 3.5,
        },
    }
    return normalize_gateway_specs({}, {slug: gateway})[f"gateway:{slug}"]


@pytest.mark.asyncio
async def test_fresh_hit_uses_one_safe_bounded_discovery_request(monkeypatch):
    calls: list[dict[str, Any]] = []

    async def fake_fetch_json(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return {"data": [{"id": "Vendor/Exact-TTS"}]}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    clock = FakeClock()
    catalog = GatewayCatalog(max_entries=4, clock=clock)
    spec = _spec()

    first = await catalog.get(
        spec,
        credential_scope_token="credential-record:7:revision:1",
        api_key="user-secret",
    )
    second = await catalog.get(
        spec,
        credential_scope_token="credential-record:7:revision:1",
        api_key="user-secret",
    )

    assert first == second
    assert first.models == (
        "Vendor/Exact-TTS",
        "Configured/Default",
        "Configured/Overlay",
    )
    assert first.discovery_status == "fresh"
    assert first.source == "discovery"
    assert first.fetched_at == 1000.0
    assert first.fresh_until == 1010.0
    assert first.stale_until == 1030.0
    assert len(calls) == 1
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == "https://company.example/v1/models"
    assert calls[0]["params"] == {"output_modalities": "speech", "region": "west"}
    assert calls[0]["headers"] == {
        "X-Route": "company-route",
        "Authorization": "Bearer user-secret",
    }
    assert calls[0]["timeout"] == 3.5
    assert isinstance(calls[0]["retry"], RetryPolicy)
    assert "GET" in calls[0]["retry"].retry_on_methods
    assert calls[0]["require_json_ct"] is True
    assert calls[0]["max_bytes"] == MAX_DISCOVERY_BYTES


@pytest.mark.asyncio
async def test_expired_fresh_entry_refreshes_without_sleep(monkeypatch):
    responses = iter(
        [
            {"data": [{"id": "Vendor/First"}]},
            {"data": [{"id": "Vendor/Second"}]},
        ]
    )
    calls = 0

    async def fake_fetch_json(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    clock = FakeClock()
    catalog = GatewayCatalog(max_entries=2, clock=clock)
    spec = _spec()

    first = await catalog.get(spec, credential_scope_token="scope", api_key="key")
    clock.advance(11)
    second = await catalog.get(spec, credential_scope_token="scope", api_key="key")

    assert first.models[0] == "Vendor/First"
    assert second.models[0] == "Vendor/Second"
    assert second.fetched_at == 1011.0
    assert calls == 2


@pytest.mark.asyncio
async def test_zero_ttl_refreshes_sequential_calls_at_same_time(monkeypatch):
    calls = 0

    async def fake_fetch_json(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return {"data": [{"id": f"Vendor/Call-{calls}"}]}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    catalog = GatewayCatalog(max_entries=2, clock=FakeClock())
    spec = _spec(ttl_seconds=0)

    first = await catalog.get(spec, credential_scope_token="scope", api_key="key")
    second = await catalog.get(spec, credential_scope_token="scope", api_key="key")

    assert first.models[0] == "Vendor/Call-1"
    assert second.models[0] == "Vendor/Call-2"
    assert calls == 2


@pytest.mark.asyncio
async def test_positive_ttl_refreshes_at_exact_fresh_boundary(monkeypatch):
    calls = 0

    async def fake_fetch_json(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return {"data": [{"id": f"Vendor/Call-{calls}"}]}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    clock = FakeClock()
    catalog = GatewayCatalog(max_entries=2, clock=clock)
    spec = _spec(ttl_seconds=10)

    await catalog.get(spec, credential_scope_token="scope", api_key="key")
    clock.advance(10)
    result = await catalog.get(spec, credential_scope_token="scope", api_key="key")

    assert result.models[0] == "Vendor/Call-2"
    assert calls == 2


@pytest.mark.asyncio
async def test_discovery_error_uses_stale_entry_only_inside_stale_window(monkeypatch):
    failing = False

    async def fake_fetch_json(**_kwargs: Any) -> Any:
        if failing:
            raise RuntimeError("upstream body containing secret")
        return {"data": [{"id": "Vendor/Cached"}]}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    clock = FakeClock()
    catalog = GatewayCatalog(max_entries=2, clock=clock)
    spec = _spec()
    await catalog.get(spec, credential_scope_token="scope", api_key="key")

    failing = True
    clock.advance(11)
    stale = await catalog.get(spec, credential_scope_token="scope", api_key="key")
    clock.advance(19)
    stale_at_boundary = await catalog.get(
        spec,
        credential_scope_token="scope",
        api_key="key",
    )
    clock.advance(0.001)
    unavailable = await catalog.get(spec, credential_scope_token="scope", api_key="key")

    assert stale.models[0] == "Vendor/Cached"
    assert stale.discovery_status == "stale"
    assert stale.source == "stale_cache"
    assert stale.stale is True
    assert stale_at_boundary.models[0] == "Vendor/Cached"
    assert stale_at_boundary.discovery_status == "stale"
    assert unavailable.models == ("Configured/Default", "Configured/Overlay")
    assert unavailable.discovery_status == "unavailable"
    assert unavailable.source == "static"
    assert unavailable.fetched_at is None
    assert unavailable.fresh_until is None
    assert unavailable.stale_until is None


@pytest.mark.asyncio
async def test_lru_access_updates_recency_and_evicts_oldest(monkeypatch):
    calls: list[str] = []

    async def fake_fetch_json(**kwargs: Any) -> Any:
        calls.append(kwargs["url"])
        host = kwargs["url"].split("//", 1)[1].split(".", 1)[0]
        return {"data": [{"id": f"Vendor/{host}"}]}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    catalog = GatewayCatalog(max_entries=2, clock=FakeClock())
    one, two, three = _spec("one"), _spec("two"), _spec("three")

    await catalog.get(one, credential_scope_token="scope", api_key="key")
    await catalog.get(two, credential_scope_token="scope", api_key="key")
    await catalog.get(one, credential_scope_token="scope", api_key="key")
    await catalog.get(three, credential_scope_token="scope", api_key="key")
    await catalog.get(two, credential_scope_token="scope", api_key="key")

    assert len(calls) == 4
    assert calls[-1] == "https://two.example/v1/models"
    assert len(catalog._cache) == 2


@pytest.mark.asyncio
async def test_concurrent_expired_requests_coalesce_to_one_refresh(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fake_fetch_json(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"data": [{"id": "Vendor/Shared"}]}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    catalog = GatewayCatalog(max_entries=2, clock=FakeClock())
    spec = _spec()

    tasks = [asyncio.create_task(catalog.get(spec, credential_scope_token="scope", api_key="key")) for _ in range(3)]
    await started.wait()
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*tasks)

    assert calls == 1
    assert all(result == results[0] for result in results)
    assert catalog._inflight == {}


@pytest.mark.asyncio
async def test_shared_refresh_failure_is_sanitized_and_inflight_is_reusable(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    should_fail = True
    calls = 0

    async def fake_fetch_json(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        if should_fail:
            raise RuntimeError("secret upstream error")
        return {"data": [{"id": "Vendor/Recovered"}]}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    catalog = GatewayCatalog(max_entries=2, clock=FakeClock())
    spec = _spec()
    tasks = [asyncio.create_task(catalog.get(spec, credential_scope_token="scope", api_key="key")) for _ in range(2)]
    await started.wait()
    release.set()
    results = await asyncio.gather(*tasks)

    assert calls == 1
    assert all(result.discovery_status == "unavailable" for result in results)
    assert "secret upstream error" not in repr(results)
    assert catalog._inflight == {}

    should_fail = False
    release.clear()
    release.set()
    recovered = await catalog.get(spec, credential_scope_token="scope", api_key="key")
    assert recovered.models[0] == "Vendor/Recovered"
    assert calls == 2


@pytest.mark.asyncio
async def test_cancelled_caller_does_not_cancel_shared_refresh_or_leave_waiters(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fake_fetch_json(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"data": [{"id": "Vendor/Survived"}]}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    catalog = GatewayCatalog(max_entries=2, clock=FakeClock())
    spec = _spec()
    leader = asyncio.create_task(catalog.get(spec, credential_scope_token="scope", api_key="key"))
    await started.wait()
    waiter = asyncio.create_task(catalog.get(spec, credential_scope_token="scope", api_key="key"))
    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader
    release.set()

    result = await waiter
    assert result.models[0] == "Vendor/Survived"
    assert calls == 1
    assert catalog._inflight == {}


@pytest.mark.asyncio
async def test_cancelled_refresh_cleans_waiters_and_allows_retry(monkeypatch):
    cancel_refresh = True
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fake_fetch_json(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if cancel_refresh:
            started.set()
            await release.wait()
            raise asyncio.CancelledError
        return {"data": [{"id": "Vendor/Retry"}]}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    catalog = GatewayCatalog(max_entries=2, clock=FakeClock())
    spec = _spec()

    first = asyncio.create_task(catalog.get(spec, credential_scope_token="scope", api_key="key"))
    await started.wait()
    second = asyncio.create_task(catalog.get(spec, credential_scope_token="scope", api_key="key"))
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(
        first,
        second,
        return_exceptions=True,
    )
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert catalog._inflight == {}

    cancel_refresh = False
    result = await catalog.get(spec, credential_scope_token="scope", api_key="key")
    assert result.models[0] == "Vendor/Retry"
    assert calls == 2


@pytest.mark.asyncio
async def test_credential_rotation_and_config_generation_partition_cache(monkeypatch):
    calls = 0

    async def fake_fetch_json(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return {"data": [{"id": f"Vendor/Call-{calls}"}]}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    catalog = GatewayCatalog(max_entries=8, clock=FakeClock())
    spec = _spec()
    changed_spec = replace(spec, config_generation="changed-generation")

    first = await catalog.get(spec, credential_scope_token="revision-1", api_key="old-key")
    rotated = await catalog.get(spec, credential_scope_token="revision-2", api_key="new-key")
    changed = await catalog.get(
        changed_spec,
        credential_scope_token="revision-2",
        api_key="new-key",
    )

    assert first.models[0] == "Vendor/Call-1"
    assert rotated.models[0] == "Vendor/Call-2"
    assert changed.models[0] == "Vendor/Call-3"
    assert calls == 3


@pytest.mark.asyncio
async def test_cache_and_public_result_contain_no_raw_scope_key_user_or_error(monkeypatch):
    async def fake_fetch_json(**_kwargs: Any) -> Any:
        return {"data": [{"id": "Vendor/Safe"}]}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    catalog = GatewayCatalog(max_entries=2, clock=FakeClock())
    spec = _spec()
    scope = "user-id-alice:credential-record-99:revision-4"
    key = "super-secret-api-key"

    result = await catalog.get(spec, credential_scope_token=scope, api_key=key)
    rendered = repr((catalog._cache, catalog._inflight, result))

    assert scope not in rendered
    assert key not in rendered
    assert "user-id-alice" not in rendered
    assert "credential-record-99" not in rendered
    assert spec.base_url not in repr(result)
    assert spec.headers[0][0] not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("discovery_enabled", "models_path", "expected_status"),
    [(False, "models", "disabled"), (True, None, "unavailable")],
)
async def test_disabled_or_missing_endpoint_returns_static_without_network(
    monkeypatch,
    discovery_enabled: bool,
    models_path: str | None,
    expected_status: str,
):
    async def unexpected_fetch(**_kwargs: Any) -> Any:
        raise AssertionError("discovery must not run")

    monkeypatch.setattr(catalog_module, "afetch_json", unexpected_fetch)
    catalog = GatewayCatalog(max_entries=2, clock=FakeClock())
    spec = _spec(discovery_enabled=discovery_enabled, models_path=models_path)

    result = await catalog.get(spec, credential_scope_token="scope", api_key=None)

    assert result.models == ("Configured/Default", "Configured/Overlay")
    assert result.discovery_status == expected_status
    assert result.source == "static"
    assert result.fetched_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"data": {}},
        {"data": ["Vendor/Bad"]},
        {"data": [{}]},
        {"data": [{"id": 42}]},
        {"data": [{"id": "   "}]},
        {"data": [{"id": "x" * (MAX_DISCOVERY_MODEL_ID_LENGTH + 1)}]},
        {"data": [{"id": f"Vendor/{index}"} for index in range(MAX_DISCOVERY_MODELS + 1)]},
    ],
)
async def test_malformed_or_oversized_payload_is_safe_unavailable(monkeypatch, payload: Any):
    async def fake_fetch_json(**_kwargs: Any) -> Any:
        return payload

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    catalog = GatewayCatalog(max_entries=2, clock=FakeClock())
    result = await catalog.get(_spec(), credential_scope_token="scope", api_key="key")

    assert result.models == ("Configured/Default", "Configured/Overlay")
    assert result.discovery_status == "unavailable"
    assert result.discovered_model_count is None


@pytest.mark.asyncio
async def test_discovery_dedupes_exact_ids_but_preserves_case_and_order(monkeypatch):
    async def fake_fetch_json(**_kwargs: Any) -> Any:
        return {
            "data": [
                {"id": "Vendor/Exact"},
                {"id": "Vendor/Exact"},
                {"id": "vendor/exact"},
                {"id": "Vendor/Other"},
            ]
        }

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    catalog = GatewayCatalog(max_entries=2, clock=FakeClock())
    result = await catalog.get(_spec(), credential_scope_token="scope", api_key="key")

    assert result.models == (
        "Vendor/Exact",
        "vendor/exact",
        "Vendor/Other",
        "Configured/Default",
        "Configured/Overlay",
    )
    assert result.discovered_model_count == 3


@pytest.mark.asyncio
async def test_authorization_is_exact_and_allowed_models_are_authoritative(monkeypatch):
    async def fake_fetch_json(**_kwargs: Any) -> Any:
        return {
            "data": [
                {"id": "Allowed/Discovered"},
                {"id": "allowed/discovered"},
                {"id": "Denied/Upstream"},
            ]
        }

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    spec = _spec(
        allowed_models=[
            "Allowed/Discovered",
            "Configured/Default",
            "Configured/Overlay",
            "Allowed/Static",
        ],
        allow_discovered_models=False,
    )
    result = await GatewayCatalog(max_entries=2, clock=FakeClock()).get(
        spec,
        credential_scope_token="scope",
        api_key="key",
    )

    assert result.models == (
        "Allowed/Discovered",
        "Configured/Default",
        "Configured/Overlay",
        "Allowed/Static",
    )
    assert result.discovered_model_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allow_discovered", "expected"),
    [
        (False, ("Configured/Default", "Configured/Overlay")),
        (
            True,
            ("Vendor/Dynamic", "Configured/Default", "Configured/Overlay"),
        ),
    ],
)
async def test_discovered_models_require_explicit_admission(
    monkeypatch,
    allow_discovered: bool,
    expected: tuple[str, ...],
):
    async def fake_fetch_json(**_kwargs: Any) -> Any:
        return {"data": [{"id": "Vendor/Dynamic"}]}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    spec = _spec(allow_discovered_models=allow_discovered)
    result = await GatewayCatalog(max_entries=2, clock=FakeClock()).get(
        spec,
        credential_scope_token="scope",
        api_key="key",
    )

    assert result.models == expected


@pytest.mark.asyncio
async def test_no_effective_key_sends_only_fixed_server_headers(monkeypatch):
    calls: list[dict[str, Any]] = []

    async def fake_fetch_json(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return {"data": []}

    monkeypatch.setattr(catalog_module, "afetch_json", fake_fetch_json)
    await GatewayCatalog(max_entries=2, clock=FakeClock()).get(
        _spec(),
        credential_scope_token="anonymous-admin-scope",
        api_key=None,
    )

    assert calls[0]["headers"] == {"X-Route": "company-route"}
    assert "Authorization" not in calls[0]["headers"]


def test_catalog_requires_a_positive_bound():
    with pytest.raises(ValueError, match="max_entries"):
        GatewayCatalog(max_entries=0)
