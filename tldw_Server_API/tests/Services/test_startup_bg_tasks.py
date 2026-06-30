from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_bg_tasks():
    sys.modules.pop("tldw_Server_API.app.services.startup_bg_tasks", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_bg_tasks")


def test_prepare_startup_bg_tasks_sets_empty_container() -> None:
    startup_bg_tasks = _import_startup_bg_tasks()
    app = SimpleNamespace(state=SimpleNamespace(bg_tasks={"old": object()}))

    startup_bg_tasks.prepare_startup_bg_tasks(
        app=app,
        startup_guard_exceptions=(OSError,),
    )

    assert app.state.bg_tasks == {}


def test_prepare_startup_bg_tasks_swallows_guard_failures() -> None:
    startup_bg_tasks = _import_startup_bg_tasks()

    class _FailingState:
        def __setattr__(self, name: str, value: object) -> None:
            raise OSError("state boom")

    app = SimpleNamespace(state=_FailingState())

    startup_bg_tasks.prepare_startup_bg_tasks(
        app=app,
        startup_guard_exceptions=(OSError,),
    )


def test_prepare_startup_bg_tasks_reraises_non_guard_failures() -> None:
    startup_bg_tasks = _import_startup_bg_tasks()

    class _FailingState:
        def __setattr__(self, name: str, value: object) -> None:
            raise RuntimeError("state boom")

    app = SimpleNamespace(state=_FailingState())

    with pytest.raises(RuntimeError, match="state boom"):
        startup_bg_tasks.prepare_startup_bg_tasks(
            app=app,
            startup_guard_exceptions=(OSError,),
        )
