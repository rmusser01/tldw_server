"""Mechanism tests for the singleton-guard plugin.

These prove the boundary-snapshot leak detector works, independent of which
real globals are on the production watchlist — a synthetic module is placed in
``sys.modules`` and a synthetic ``WatchedGlobal`` points at it.
"""
from __future__ import annotations

import sys
import types

import pytest

from tldw_Server_API.app.core.DB_Management import media_db  # noqa: F401 - import sanity only
from tldw_Server_API.tests._plugins import singleton_guard as sg

pytestmark = pytest.mark.unit


def _fake_module_with_cache(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module._cache = {}  # type: ignore[attr-defined]
    return module


def _install_watch(monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
    watch = sg.WatchedGlobal(
        label="synthetic._cache",
        module=module_name,
        reader=sg._len_attr("_cache"),
        why="synthetic test cache",
    )
    monkeypatch.setattr(sg, "WATCHLIST", [watch])


def test_guard_detects_leak_when_module_grows_a_watched_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "tldw_singleton_guard_fake_leaky"
    fake = _fake_module_with_cache(module_name)
    monkeypatch.setitem(sys.modules, module_name, fake)
    _install_watch(monkeypatch, module_name)

    guard = sg.SingletonGuard(mode="warn")
    guard.enter_module("mod_a")  # baseline: cache empty
    fake._cache["polluted"] = 1  # mod_a leaves state behind
    guard.enter_module("mod_b")  # boundary: mod_a finalized here
    guard.finish()

    assert guard.leaks, "guard failed to detect a cross-module cache leak"
    assert "mod_a" in guard.leaks[0]
    assert "synthetic._cache" in guard.leaks[0]


def test_guard_detects_singleton_becoming_live_from_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None -> set is the #2580/#2581 signal (DB backend bound, drain singleton
    became live) and must be flagged, not silently skipped."""
    module_name = "tldw_singleton_guard_fake_none_to_set"
    module = types.ModuleType(module_name)
    module._instance = None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    watch = sg.WatchedGlobal(
        label="synthetic._instance",
        module=module_name,
        reader=sg._is_set_attr("_instance"),
        why="synthetic singleton",
    )
    monkeypatch.setattr(sg, "WATCHLIST", [watch])

    guard = sg.SingletonGuard(mode="warn")
    guard.enter_module("mod_a")  # baseline: _instance is None (False)
    module._instance = object()  # mod_a leaves a live singleton  # type: ignore[attr-defined]
    guard.enter_module("mod_b")
    guard.finish()

    assert guard.leaks, "guard missed a None->set singleton leak"
    assert "synthetic._instance" in guard.leaks[0]


def test_guard_is_quiet_when_module_clears_state_it_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """set -> None (a module clearing its own singleton) is hygiene, not a leak."""
    module_name = "tldw_singleton_guard_fake_set_to_none"
    module = types.ModuleType(module_name)
    module._instance = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    watch = sg.WatchedGlobal(
        label="synthetic._instance",
        module=module_name,
        reader=sg._is_set_attr("_instance"),
        why="synthetic singleton",
    )
    monkeypatch.setattr(sg, "WATCHLIST", [watch])

    guard = sg.SingletonGuard(mode="warn")
    guard.enter_module("mod_a")  # baseline: live
    module._instance = None  # mod_a cleaned up after itself  # type: ignore[attr-defined]
    guard.enter_module("mod_b")
    guard.finish()

    assert not guard.leaks, f"clearing state was wrongly flagged: {guard.leaks}"


def test_is_regression_only_flags_the_dangerous_direction() -> None:
    """Unit coverage for the leak-direction rule."""
    assert sg._is_regression(False, True) is True  # became live
    assert sg._is_regression(True, False) is False  # cleared
    assert sg._is_regression(0, 3) is True  # grew
    assert sg._is_regression(3, 0) is False  # shrank
    assert sg._is_regression(None, 12345) is True  # became set (id)
    assert sg._is_regression(111, 222) is True  # rebound to a different object
    assert sg._is_regression(12345, None) is False  # cleared (id -> None)
    assert sg._is_regression(5, 5) is False  # unchanged


def test_guard_is_quiet_when_module_cleans_up_after_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "tldw_singleton_guard_fake_clean"
    fake = _fake_module_with_cache(module_name)
    monkeypatch.setitem(sys.modules, module_name, fake)
    _install_watch(monkeypatch, module_name)

    guard = sg.SingletonGuard(mode="warn")
    guard.enter_module("mod_a")
    fake._cache["temp"] = 1
    del fake._cache["temp"]  # cleaned up before the boundary
    guard.enter_module("mod_b")
    guard.finish()

    assert not guard.leaks, f"guard reported a false positive: {guard.leaks}"


def test_guard_ignores_modules_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A watched module absent from sys.modules contributes no snapshot/leak."""
    _install_watch(monkeypatch, "tldw_module_that_is_not_imported_anywhere")

    guard = sg.SingletonGuard(mode="warn")
    guard.enter_module("mod_a")
    guard.enter_module("mod_b")
    guard.finish()

    assert not guard.leaks


def test_error_mode_forces_nonzero_exit_via_sessionfinish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "tldw_singleton_guard_fake_error"
    fake = _fake_module_with_cache(module_name)
    monkeypatch.setitem(sys.modules, module_name, fake)
    _install_watch(monkeypatch, module_name)

    guard = sg.SingletonGuard(mode="error")
    guard.enter_module("mod_a")
    fake._cache["x"] = 1
    guard.enter_module("mod_b")

    class _FakeConfig:
        _singleton_guard = guard

    class _FakeSession:
        config = _FakeConfig()
        exitstatus = 0

    session = _FakeSession()
    sg.pytest_sessionfinish(session, exitstatus=0)  # type: ignore[arg-type]

    assert session.exitstatus == 1, "error mode did not force a nonzero exit"


def test_reader_helpers_snapshot_expected_shapes() -> None:
    module = types.ModuleType("tldw_reader_helper_probe")
    module._instance = None  # type: ignore[attr-defined]
    module._cache = {"a": 1}  # type: ignore[attr-defined]

    assert sg._is_set_attr("_instance")(module) is False
    module._instance = object()  # type: ignore[attr-defined]
    assert sg._is_set_attr("_instance")(module) is True
    assert sg._len_attr("_cache")(module) == 1
    assert sg._len_attr("_missing")(module) is None
