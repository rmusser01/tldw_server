"""Runtime contracts and adapters for the staged Web_Scraping refactor."""

from __future__ import annotations

from .browser import BrowserLaunchOptions, RuntimeBrowserContext, RuntimeBrowserLauncher, RuntimeBrowserPage
from .cancellation import is_cancellation
from .fetch import DefaultFetchClient, FetchClient
from .policy import OutboundPolicyChecker, ProbeEgressDecision, ProbeEgressGuard
from .requests import FetchRequest, RuntimeRequestContext
from .responses import FetchResponse, PolicyDecision
from .sessions import RuntimeCookie, RuntimeSessionState
from .timeouts import RuntimeTimeouts

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
    "RuntimeBrowserPage",
    "RuntimeCookie",
    "RuntimeRequestContext",
    "RuntimeSessionState",
    "RuntimeTimeouts",
    "is_cancellation",
]
