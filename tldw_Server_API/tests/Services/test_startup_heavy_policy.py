from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_heavy_policy():
    sys.modules.pop("tldw_Server_API.app.services.startup_heavy_policy", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_heavy_policy")


def test_resolve_deferred_heavy_startup_forces_sync_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_policy = _import_startup_heavy_policy()

    monkeypatch.setattr(
        startup_policy,
        "_getenv",
        lambda key: {"DISABLE_HEAVY_STARTUP": "1", "DEFER_HEAVY_STARTUP": "1"}.get(key),
    )

    result = startup_policy.resolve_deferred_heavy_startup(
        shared_is_truthy=lambda value: str(value).strip().lower() in {"1", "true", "yes", "on"},
        startup_guard_exceptions=(RuntimeError,),
    )

    assert result is False


def test_resolve_deferred_heavy_startup_returns_true_when_defer_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_policy = _import_startup_heavy_policy()

    monkeypatch.setattr(
        startup_policy,
        "_getenv",
        lambda key: {"DISABLE_HEAVY_STARTUP": None, "DEFER_HEAVY_STARTUP": "true"}.get(key),
    )

    result = startup_policy.resolve_deferred_heavy_startup(
        shared_is_truthy=lambda value: str(value).strip().lower() in {"1", "true", "yes", "on"},
        startup_guard_exceptions=(RuntimeError,),
    )

    assert result is True


def test_resolve_deferred_heavy_startup_defaults_false_without_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_policy = _import_startup_heavy_policy()

    monkeypatch.setattr(startup_policy, "_getenv", lambda key: None)

    result = startup_policy.resolve_deferred_heavy_startup(
        shared_is_truthy=lambda value: str(value).strip().lower() in {"1", "true", "yes", "on"},
        startup_guard_exceptions=(RuntimeError,),
    )

    assert result is False


def test_resolve_deferred_heavy_startup_falls_back_to_false_on_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_policy = _import_startup_heavy_policy()

    def _raise_runtime_error(_key: str):
        raise RuntimeError("env boom")

    monkeypatch.setattr(startup_policy, "_getenv", _raise_runtime_error)

    result = startup_policy.resolve_deferred_heavy_startup(
        shared_is_truthy=lambda value: str(value).strip().lower() in {"1", "true", "yes", "on"},
        startup_guard_exceptions=(RuntimeError,),
    )

    assert result is False
