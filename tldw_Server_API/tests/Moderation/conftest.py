"""Fixtures for the gateway moderation hooks (TASK-13950.27 PR C)."""

from __future__ import annotations

import json

import pytest

import tldw_Server_API.app.core.Moderation.gateway_hooks as hooks_mod
import tldw_Server_API.app.core.Moderation.moderation_service as ms_mod
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.app.core.Moderation.moderation_service import ModerationService

PROVIDER = "e2e-fake"  # outside the gateway's supported set, so the stub registry answers
USAGE = {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# Frame builders copied from the adapters (anthropic_adapter.py:544-547, google_adapter.py:484,556,
# cohere_adapter.py:317) and the OpenAI compact shape (normalize_provider_line passes it through).
def openai_frame(text: str, index: int | None = 0, finish: str | None = None) -> str:
    choice: dict = {"delta": {"content": text} if text else {}}
    if index is not None:
        choice["index"] = index
    if finish:
        choice["finish_reason"] = finish
    return f"data: {json.dumps({'choices': [choice]}, separators=(',', ':'), ensure_ascii=False)}\n\n"


def anthropic_frame(text: str, original: str | None = None) -> str:
    event = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": original or text}}
    return sse({"choices": [{"delta": {"content": text}}], "provider_response": event})


def google_frame(text: str) -> str:
    event = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return sse({"choices": [{"delta": {"content": text}}], "provider_response": event})


def cohere_frame(text: str) -> str:
    event = {"event_type": "text-generation", "text": text}
    return sse({"choices": [{"delta": {"content": text}}], "provider_response": event})


def finish_frame(reason: str = "stop", index: int = 0) -> str:
    return sse({"choices": [{"index": index, "delta": {}, "finish_reason": reason}]})


def usage_frame() -> str:
    return sse({"choices": [], "usage": dict(USAGE)})


DONE = "data: [DONE]\n\n"


def reply(text: str, **extra) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "model": "m",
        "usage": dict(USAGE),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        **extra,
    }


class _Stream:
    """A provider stream that records whether it was closed."""

    def __init__(self, items: list, owner: "StubRegistry") -> None:
        self._it = iter(items)
        self._owner = owner

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self) -> None:
        self._owner.closed += 1


class StubRegistry:
    """Answers every call with ``reply`` (dict) or ``items`` (SSE list) and records the requests."""

    def __init__(self) -> None:
        self.reply: dict = reply("provider reply")
        self.items: list = [openai_frame("provider reply"), finish_frame(), usage_frame(), DONE]
        self.requests: list[dict] = []
        self.closed = 0

    def get_adapter(self, name):
        return self

    def get_audited_call_policy_transport(self, name):
        return None

    def capabilities(self):
        return {}

    def chat(self, request, *, timeout=None):
        self.requests.append(dict(request))
        return json.loads(json.dumps(self.reply))

    def stream(self, request, *, timeout=None):
        self.requests.append(dict(request))
        return _Stream(list(self.items), self)

    async def astream(self, request, *, timeout=None):
        self.requests.append(dict(request))
        for item in self.items:
            yield item


@pytest.fixture
def stub_llm_registry(monkeypatch):
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    stub = StubRegistry()
    monkeypatch.setattr(gateway, "_get_llm_registry", lambda: stub)
    return stub


@pytest.fixture
def make_service(monkeypatch, tmp_path):
    """Install a real ModerationService (block ``forbidden``, redact ``secret``) as the singleton."""

    def make(enabled: bool = True, **extra) -> ModerationService:
        blocklist = tmp_path / "bl.txt"
        blocklist.write_text("forbidden -> block\nsecret -> redact:[X]\n")
        cfg = {
            "enabled": str(enabled).lower(),
            "blocklist_file": str(blocklist),
            "user_overrides_file": str(tmp_path / "uo.json"),
            "runtime_overrides_file": str(tmp_path / "ro.json"),
            **{k: str(v).lower() if isinstance(v, bool) else v for k, v in extra.items()},
        }
        monkeypatch.setattr(ms_mod, "load_and_log_configs", lambda: {"moderation": cfg})
        service = ModerationService()
        monkeypatch.setattr(ms_mod, "_moderation_service", service)
        return service

    return make


@pytest.fixture(autouse=True)
def _clean_hook_state():
    hooks_mod._GUARDIAN_ENGINES.clear()
    getattr(hooks_mod._stream_buffer_chars, "cache_clear", lambda: None)()  # may be monkeypatched
    yield
    hooks_mod._unregister()
    hooks_mod._GUARDIAN_ENGINES.clear()
    getattr(hooks_mod._stream_buffer_chars, "cache_clear", lambda: None)()  # may be monkeypatched
    assert gateway._POLICY_HOOKS == []


@pytest.fixture
def moderation_hooks(make_service):
    """Moderation on (block ``forbidden``, redact ``secret``) with the hooks registered; removed after."""
    service = make_service(enabled=True)
    hooks_mod.register_moderation_gateway_hooks()
    return service


@pytest.fixture
def buffer_chars(monkeypatch):
    def set_(n: int) -> None:
        monkeypatch.setattr(hooks_mod, "_stream_buffer_chars", lambda: n)

    return set_
