from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import (
    CustomOpenAIAdapter,
    CustomOpenAIAdapter2,
    NovitaAdapter,
    PoeAdapter,
    TogetherAdapter,
    make_custom_openai_adapter_class,
)

_LATE_ENV = "https://late-env.example/v1"
_SNAPSHOT_A = "https://snapshot-a.example/v1"
_SNAPSHOT_B = "https://snapshot-b.example/v1"


_DEFAULTED_CASES = (
    (CustomOpenAIAdapter, "http://127.0.0.1:11434/v1"),
    (NovitaAdapter, "https://api.novita.ai/openai"),
    (PoeAdapter, "https://api.poe.com/v1"),
    (TogetherAdapter, "https://api.together.xyz/v1"),
)

_FAIL_CLOSED_CASES = (
    CustomOpenAIAdapter2,
    make_custom_openai_adapter_class(3),
    make_custom_openai_adapter_class(99),
)


def _set_late_endpoint_env(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    for name in adapter.default_base_url_env:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(adapter.default_base_url_env[0], _LATE_ENV)


def _request(
    adapter: Any,
    endpoint: str | None,
    *,
    include_section: bool = True,
) -> dict[str, Any]:
    section = {"api_base_url": endpoint} if endpoint is not None else {}
    return {
        "credentials_resolved": True,
        "app_config": {adapter.config_section: section} if include_section else {},
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "include_section",
    (True, False),
    ids=("empty-provider-section", "empty-config-map"),
)
@pytest.mark.parametrize("adapter_type,canonical_endpoint", _DEFAULTED_CASES)
def test_resolved_defaulted_custom_adapter_ignores_late_environment(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
    canonical_endpoint: str,
    include_section: bool,
) -> None:
    adapter = adapter_type()
    _set_late_endpoint_env(monkeypatch, adapter)

    assert (
        adapter._resolve_base(
            _request(adapter, None, include_section=include_section),
        )
        == canonical_endpoint
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "include_section",
    (True, False),
    ids=("empty-provider-section", "empty-config-map"),
)
@pytest.mark.parametrize("adapter_type", _FAIL_CLOSED_CASES)
def test_resolved_numbered_custom_adapter_fails_closed_when_snapshot_has_no_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
    include_section: bool,
) -> None:
    adapter = adapter_type()
    _set_late_endpoint_env(monkeypatch, adapter)

    with pytest.raises(RuntimeError, match="requires an explicit base URL"):
        adapter._resolve_base(
            _request(adapter, None, include_section=include_section),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter_type",
    tuple(case[0] for case in _DEFAULTED_CASES) + _FAIL_CLOSED_CASES,
)
def test_resolved_custom_adapter_keeps_snapshot_when_environment_rotates(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
) -> None:
    adapter = adapter_type()
    _set_late_endpoint_env(monkeypatch, adapter)

    assert adapter._resolve_base(_request(adapter, _SNAPSHOT_A)) == _SNAPSHOT_A


@pytest.mark.unit
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "adapter_type",
    tuple(case[0] for case in _DEFAULTED_CASES) + _FAIL_CLOSED_CASES,
)
def test_concurrent_resolved_custom_adapters_keep_request_snapshots_isolated(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
) -> None:
    adapter = adapter_type()
    _set_late_endpoint_env(monkeypatch, adapter)
    gate = threading.Barrier(2)

    def _resolve_after_both_started(request: dict[str, Any]) -> str:
        gate.wait(timeout=5)
        return adapter._resolve_base(request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_resolve_after_both_started, _request(adapter, _SNAPSHOT_A))
        second = executor.submit(_resolve_after_both_started, _request(adapter, _SNAPSHOT_B))

    assert {first.result(), second.result()} == {_SNAPSHOT_A, _SNAPSHOT_B}


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter_type",
    tuple(case[0] for case in _DEFAULTED_CASES) + _FAIL_CLOSED_CASES,
)
def test_unmarked_custom_adapter_keeps_legacy_environment_behavior(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
) -> None:
    adapter = adapter_type()
    _set_late_endpoint_env(monkeypatch, adapter)

    assert adapter._resolve_base({"app_config": {adapter.config_section: {}}}) == _LATE_ENV
