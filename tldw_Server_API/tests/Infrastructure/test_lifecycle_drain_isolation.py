"""Drain state must not survive the test that produced it.

``DrainGateMiddleware`` answers 503 to every non-control-plane request while an
app is draining, and a ``TestClient`` lifespan exit drains whichever app object
it was handed. That object is not always the one ``app.main`` currently
exports: suites reload ``app.main``, while test modules that ran
``from ...main import app`` at collection time keep routing through the object
they pinned. Resetting only the current module's app therefore left the pinned
one draining for the rest of the process, and every later request through it
came back as::

    {"status": "not_ready", "reason": "shutdown_in_progress"}

which surfaced as ~29 unrelated Setup failures depending on suite order.
"""

from __future__ import annotations

import ast
import gc
import weakref
from pathlib import Path

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.services.app_lifecycle import (
    is_lifecycle_draining,
    mark_lifecycle_shutdown,
    reset_all_lifecycle_states,
)

ROOT_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"
FIXTURE_NAME = "_reset_main_app_lifecycle_state_between_tests"


@pytest.mark.unit
def test_reset_all_clears_an_app_that_is_not_the_current_module_app() -> None:
    """The app being reset is usually not the one ``app.main`` exports."""
    pinned = FastAPI()
    current = FastAPI()
    mark_lifecycle_shutdown(pinned)
    assert is_lifecycle_draining(pinned)

    reset_all_lifecycle_states()

    assert not is_lifecycle_draining(pinned), (
        "a drained app stayed drained because it was no longer the app "
        "reachable through app.main; every request through it would 503"
    )
    assert not is_lifecycle_draining(current)


@pytest.mark.unit
def test_reset_all_clears_several_drained_apps_at_once() -> None:
    """Reloading repeatedly leaves more than one pinned app behind."""
    apps = [FastAPI() for _ in range(3)]
    for app in apps:
        mark_lifecycle_shutdown(app)
    assert all(is_lifecycle_draining(app) for app in apps)

    reset_all_lifecycle_states()

    assert not any(is_lifecycle_draining(app) for app in apps)


@pytest.mark.unit
def test_registry_does_not_keep_apps_alive() -> None:
    """Tracking apps must not turn every test app into a permanent leak."""
    app = FastAPI()
    mark_lifecycle_shutdown(app)
    # Reaching it here proves it is tracked, so the collection check below is
    # about a real entry rather than passing vacuously.
    reset_all_lifecycle_states()
    assert not is_lifecycle_draining(app)
    ref = weakref.ref(app)

    del app
    gc.collect()

    assert ref() is None, "the app registry is holding a strong reference"


@pytest.mark.unit
def test_root_fixture_resets_every_app_not_just_the_current_one() -> None:
    """Guard the wiring: the semantics above only help if the fixture calls them.

    Checks for the *call*, inside the fixture, rather than for the name
    anywhere in the file -- an import left behind after the call was deleted
    would satisfy a substring check while resetting nothing.
    """
    tree = ast.parse(ROOT_CONFTEST.read_text(encoding="utf-8"))

    fixture = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == FIXTURE_NAME
        ),
        None,
    )
    assert fixture is not None, (
        f"{ROOT_CONFTEST.name} no longer defines {FIXTURE_NAME}(), so nothing "
        "clears drain state between tests."
    )

    called = {
        node.func.id
        for node in ast.walk(fixture)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "reset_all_lifecycle_states" in called, (
        f"{FIXTURE_NAME}() no longer calls reset_all_lifecycle_states(). "
        "Resetting only the app exported by app.main leaves apps pinned by "
        "earlier imports draining, which 503s unrelated tests downstream."
    )
