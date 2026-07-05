"""Runtime contracts and adapters for the staged Web_Scraping refactor."""

from __future__ import annotations

from .browser import BrowserLaunchOptions, RuntimeBrowserContext, RuntimeBrowserLauncher, RuntimeBrowserPage
from .cancellation import is_cancellation
from .policy import OutboundPolicyChecker
from .requests import FetchRequest, RuntimeRequestContext
from .responses import FetchResponse, PolicyDecision
from .sessions import RuntimeCookie, RuntimeSessionState
from .timeouts import RuntimeTimeouts

__all__ = [
    "BrowserLaunchOptions",
    "FetchRequest",
    "FetchResponse",
    "OutboundPolicyChecker",
    "PolicyDecision",
    "RuntimeBrowserContext",
    "RuntimeBrowserLauncher",
    "RuntimeBrowserPage",
    "RuntimeCookie",
    "RuntimeRequestContext",
    "RuntimeSessionState",
    "RuntimeTimeouts",
    "is_cancellation",
]
