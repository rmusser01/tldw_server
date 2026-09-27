from __future__ import annotations

import weakref
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException

_LIFECYCLE_EVENTS_ATTR = "_tldw_lifecycle_events"
_LIFECYCLE_STATE_ATTR = "_tldw_lifecycle_state"
_LifecycleEvent = Literal["startup", "shutdown"]

# Every app that has lifecycle state, so all of them can be reset together.
#
# More than one app object can be live at once: the test suite replaces
# ``sys.modules["tldw_Server_API.app.main"]`` to get a freshly built app,
# but modules that already did ``from ... .main import app`` keep routing
# through the object they pinned at import time. Draining one of those
# pinned objects is invisible to anything that looks only at the current
# module, so the registry keeps hold of them. It is weak, so an app that
# goes out of scope is not kept alive by being listed here.
_LIFECYCLE_APPS: weakref.WeakSet[FastAPI] = weakref.WeakSet()


@dataclass
class AppLifecycleState:
    phase: Literal["starting", "ready", "draining", "stopped"] = "starting"
    ready: bool = False
    draining: bool = False


def _append_lifecycle_event(app: FastAPI, event: _LifecycleEvent) -> None:
    events = getattr(app.state, _LIFECYCLE_EVENTS_ATTR, None)
    if events is None:
        events = []
        app.state._tldw_lifecycle_events = events
    events.append(event)


def get_or_create_lifecycle_state(app: FastAPI) -> AppLifecycleState:
    state = getattr(app.state, _LIFECYCLE_STATE_ATTR, None)
    if state is None:
        state = AppLifecycleState()
        app.state._tldw_lifecycle_state = state
        _LIFECYCLE_APPS.add(app)
    return state


def reset_lifecycle_state(app: FastAPI) -> AppLifecycleState:
    state = AppLifecycleState()
    app.state._tldw_lifecycle_state = state
    _LIFECYCLE_APPS.add(app)
    return state


def reset_all_lifecycle_states() -> None:
    """Reset every app that has lifecycle state.

    Resetting only the app reachable through the current ``app.main`` module is
    not enough. A ``TestClient`` lifespan exit marks whichever app object it was
    given as draining, and that object may no longer be the one the module
    exports -- in which case it stays drained for the rest of the process and
    ``DrainGateMiddleware`` answers 503 to every request routed through it.
    """
    for app in list(_LIFECYCLE_APPS):
        reset_lifecycle_state(app)


def is_lifecycle_draining(app: FastAPI) -> bool:
    """Return True when the app is actively draining shutdown traffic."""
    state = get_or_create_lifecycle_state(app)
    return state.draining or state.phase == "draining"


def assert_may_start_work(app: FastAPI, kind: str) -> None:
    """Raise a 503 if the app is in draining mode and work should not start."""
    if is_lifecycle_draining(app):
        raise HTTPException(
            status_code=503,
            detail={"message": "Shutdown in progress", "kind": kind},
        )


def mark_lifecycle_startup(
    app: FastAPI,
    readiness_state: MutableMapping[str, bool] | None = None,
) -> AppLifecycleState:
    """Record startup transition and mark readiness true."""
    state = get_or_create_lifecycle_state(app)
    state.phase = "ready"
    state.ready = True
    state.draining = False
    if readiness_state is not None:
        readiness_state["ready"] = True
    _append_lifecycle_event(app, "startup")
    return state


def mark_lifecycle_shutdown(
    app: FastAPI,
    readiness_state: MutableMapping[str, bool] | None = None,
) -> AppLifecycleState:
    """Record shutdown transition and mark readiness false."""
    state = get_or_create_lifecycle_state(app)
    state.phase = "draining"
    state.ready = False
    state.draining = True
    if readiness_state is not None:
        readiness_state["ready"] = False
    _append_lifecycle_event(app, "shutdown")
    return state
