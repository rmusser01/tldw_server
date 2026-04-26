from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_transition_gate():
    sys.modules.pop("tldw_Server_API.app.services.startup_transition_gate", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_transition_gate")


def test_apply_startup_transition_gate_marks_lifecycle_then_disables_job_acquire_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_transition = _import_startup_transition_gate()
    app = object()
    readiness_state = {"ready": False}
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        startup_transition,
        "_mark_lifecycle_startup",
        lambda seen_app, seen_readiness_state: calls.append(("mark", seen_app, seen_readiness_state)),
    )
    monkeypatch.setattr(
        startup_transition,
        "_disable_job_acquire_gate",
        lambda: calls.append(("gate",)),
    )

    startup_transition.apply_startup_transition_gate(
        app=app,
        readiness_state=readiness_state,
        import_exceptions=(ImportError,),
    )

    assert calls == [("mark", app, readiness_state), ("gate",)]


def test_apply_startup_transition_gate_swallows_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_transition = _import_startup_transition_gate()
    app = object()
    readiness_state = {"ready": False}
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        startup_transition,
        "_mark_lifecycle_startup",
        lambda seen_app, seen_readiness_state: calls.append(("mark", seen_app, seen_readiness_state)),
    )

    def _raise_import_error() -> None:
        raise ImportError("jobs unavailable")

    monkeypatch.setattr(
        startup_transition,
        "_disable_job_acquire_gate",
        _raise_import_error,
    )

    startup_transition.apply_startup_transition_gate(
        app=app,
        readiness_state=readiness_state,
        import_exceptions=(ImportError,),
    )

    assert calls == [("mark", app, readiness_state)]


def test_apply_startup_transition_gate_runs_gate_when_lifecycle_marker_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_transition = _import_startup_transition_gate()
    app = object()
    readiness_state = {"ready": False}
    calls: list[str] = []

    def _raise_import_error(_seen_app, _seen_readiness_state) -> None:
        raise ImportError("lifecycle unavailable")

    monkeypatch.setattr(startup_transition, "_mark_lifecycle_startup", _raise_import_error)
    monkeypatch.setattr(
        startup_transition,
        "_disable_job_acquire_gate",
        lambda: calls.append("gate"),
    )

    startup_transition.apply_startup_transition_gate(
        app=app,
        readiness_state=readiness_state,
        import_exceptions=(ImportError,),
    )

    assert calls == ["gate"]


def test_apply_startup_transition_gate_reraises_non_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_transition = _import_startup_transition_gate()

    monkeypatch.setattr(
        startup_transition,
        "_mark_lifecycle_startup",
        lambda seen_app, seen_readiness_state: None,
    )

    def _raise_runtime_error() -> None:
        raise RuntimeError("gate failure")

    monkeypatch.setattr(
        startup_transition,
        "_disable_job_acquire_gate",
        _raise_runtime_error,
    )

    with pytest.raises(RuntimeError, match="gate failure"):
        startup_transition.apply_startup_transition_gate(
            app=object(),
            readiness_state={"ready": False},
            import_exceptions=(ImportError,),
        )
