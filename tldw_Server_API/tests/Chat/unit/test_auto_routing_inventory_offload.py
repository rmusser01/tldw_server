"""Slow provider discovery must not block either auto-routing event loop."""

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions, chat

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["chat", "character"])
async def test_auto_routing_discovery_runs_off_event_loop(
    monkeypatch: pytest.MonkeyPatch, surface: str,
) -> None:
    module = chat if surface == "chat" else character_chat_sessions
    event_loop_thread = threading.get_ident()
    discovered = []

    def listing() -> dict[str, Any]:
        discovered.append(threading.get_ident())
        assert discovered[-1] != event_loop_thread
        return {"providers": [], "default_provider": "openai"}

    class StopAfterDiscovery(Exception):
        pass

    def stop(**kwargs: Any) -> None:
        raise StopAfterDiscovery

    monkeypatch.setattr(module, "get_llm_provider_overrides_snapshot", lambda: {})
    monkeypatch.setattr(module, "get_configured_providers", listing)
    monkeypatch.setattr(module, "resolve_routing_policy", stop)
    request = SimpleNamespace(model="auto", routing=None)
    with pytest.raises(StopAfterDiscovery):
        if surface == "chat":
            await chat._resolve_auto_chat_routing_decision(
                request, request=None, sticky_store=None, current_user=None,
                request_id=None, credential_runtime=None,
            )
        else:
            await character_chat_sessions._resolve_auto_character_chat_routing_decision(
                chat_id="synthetic", body=request, raw_provider=None,
                formatted_messages=[], sticky_store=None, current_user=None,
                credential_runtime=None,
            )
    await asyncio.sleep(0)
    assert len(discovered) == 1
