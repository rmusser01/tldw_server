"""Route the LLM gateway's adapter lookup to a test's fake adapters (TASK-13950.27 PR H).

Prompt Studio and quiz generation used to fetch an adapter themselves, so their tests patched that lookup in
the module. They now call the gateway, so a test supplies its adapter where the gateway looks it up.

A test may fake two boundaries at once, the executor's and the test runner's. The gateway has one lookup, so a
route registered with ``caller="prompt_executor"`` answers calls made from ``prompt_executor.py`` and the
``"any"`` route answers every other call.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import pytest

from tldw_Server_API.app.core.LLM_Calls import adapter_utils, gateway

_ROUTES = "_gateway_adapter_routes"


class _Registry:
    def __init__(self, routes: dict[str, Callable[[str], Any]]) -> None:
        self._routes = routes

    def get_adapter(self, provider: str) -> Any:
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_filename.endswith("prompt_executor.py") and "prompt_executor" in self._routes:
                return self._routes["prompt_executor"](provider)
            frame = frame.f_back
        route = self._routes.get("any") or self._routes["prompt_executor"]
        return route(provider)

    def get_audited_call_policy_transport(self, _provider: str) -> None:
        return None

    def capabilities(self) -> dict[str, Any]:
        return {}


def route_gateway_adapters(
    monkeypatch: pytest.MonkeyPatch, adapter_for: Callable[[str], Any], *, caller: str = "any"
) -> None:
    """Make the gateway call ``adapter_for(provider)``'s adapter; ``None`` means no adapter."""
    routes = monkeypatch.__dict__.setdefault(_ROUTES, {})
    routes[caller] = adapter_for
    monkeypatch.setattr(gateway, "_get_llm_registry", lambda: _Registry(routes))
    # Bypasses the gateway's runtime-issued-handle authenticity check (adapter_utils), because these tests hand the
    # call a fake handle and already patch the module-level check the same way; real handles are covered elsewhere.
    monkeypatch.setattr(
        adapter_utils,
        "is_runtime_issued_provider_call_credentials",
        lambda value, *, provider=None: value is not None and (provider is None or getattr(value, "provider", provider) == provider),
    )
