"""The Prompt Studio WebSocket connection manager, as core sees it (TASK-13950.20).

The manager stays in its endpoint module, because accepting a connection needs that module's
helpers and the app lifecycle. The endpoint publishes its manager here when it loads, and core
(the MCTS optimizer, the optimization engine) reads it at call time. Before the endpoint loads,
and in a process that never loads it, ``current()`` returns None: there is no client to reach.
"""

from __future__ import annotations

from typing import Any, TypeVar

_T = TypeVar("_T")

_published: dict[str, Any] = {}


def publish(manager: _T) -> _T:
    """Make ``manager`` the one core broadcasts through, replacing any earlier one, and return it."""
    _published["manager"] = manager
    return manager


def current() -> Any | None:
    """The published connection manager, or None before the WebSocket module has loaded."""
    return _published.get("manager")
