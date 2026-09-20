"""Runtime contracts and adapters for the staged Web_Scraping refactor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .browser import (
    BrowserLaunchOptions,
    RuntimeBrowserContext,
    RuntimeBrowserLauncher,
    RuntimeBrowserLocator,
    RuntimeBrowserPage,
    RuntimeBrowserRequest,
    RuntimeBrowserRoute,
    RuntimeWebSocketRoute,
)
from .cancellation import is_cancellation
from .policy import OutboundPolicyChecker, ProbeEgressDecision, ProbeEgressGuard
from .requests import FetchRequest, RuntimeRequestContext
from .responses import FetchResponse, PolicyDecision
from .sessions import RuntimeCookie, RuntimeSessionState
from .timeouts import RuntimeTimeouts

if TYPE_CHECKING:
    from .fetch import DefaultFetchClient, FetchClient

__all__ = [
    "BrowserLaunchOptions",
    "DefaultFetchClient",
    "FetchClient",
    "FetchRequest",
    "FetchResponse",
    "OutboundPolicyChecker",
    "PolicyDecision",
    "ProbeEgressDecision",
    "ProbeEgressGuard",
    "RuntimeBrowserContext",
    "RuntimeBrowserLauncher",
    "RuntimeBrowserLocator",
    "RuntimeBrowserPage",
    "RuntimeBrowserRequest",
    "RuntimeBrowserRoute",
    "RuntimeCookie",
    "RuntimeRequestContext",
    "RuntimeSessionState",
    "RuntimeTimeouts",
    "RuntimeWebSocketRoute",
    "is_cancellation",
]


def __getattr__(name: str) -> Any:
    if name in {"DefaultFetchClient", "FetchClient"}:
        from .fetch import DefaultFetchClient, FetchClient

        fetch_exports = {
            "DefaultFetchClient": DefaultFetchClient,
            "FetchClient": FetchClient,
        }
        globals().update(fetch_exports)
        return fetch_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
