"""With moderation off, every moderated surface makes zero evaluator calls and changes nothing
(TASK-13950.27 PR C, part a). The /chat/completions variant joins in part b, with the app wiring."""

from __future__ import annotations

import pytest

import tldw_Server_API.app.core.Moderation.gateway_hooks as hooks_mod
from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator
from tldw_Server_API.app.core.Moderation.surfaces import MODERATED_SURFACES
from tldw_Server_API.tests.Moderation.conftest import DONE, PROVIDER, finish_frame, openai_frame, reply

pytestmark = pytest.mark.unit

SURFACES = sorted(MODERATED_SURFACES)


@pytest.fixture
def evaluator_spy(monkeypatch):
    calls: list[str] = []
    for name in ("evaluate_text", "redact_text", "redact_text_with_count"):
        real = getattr(PolicyEvaluator, name)

        def spy(self, *a, _real=real, _name=name, **k):
            calls.append(_name)
            return _real(self, *a, **k)

        monkeypatch.setattr(PolicyEvaluator, name, spy)
    return calls


@pytest.fixture
def guardian_files(monkeypatch, tmp_path):
    """Guardian is on, but the user has no DB: nothing may be created."""
    import tldw_Server_API.app.core.feature_flags as flags

    monkeypatch.setattr(flags, "is_guardian_enabled", lambda: True)
    return tmp_path


def _call(surface: str, stream: bool):
    return gateway.perform_chat_api_call(
        api_endpoint=PROVIDER, model="m", api_key="k", streaming=stream, llm_surface=surface,
        messages_payload=[{"role": "user", "content": "forbidden secret"}],
    )


@pytest.mark.parametrize("surface_key", [True, None], ids=["key-on", "no-key"])
@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize("surface", SURFACES)
def test_global_off_makes_zero_evaluator_calls(
    make_service, stub_llm_registry, evaluator_spy, surface, stream, surface_key, guardian_files
):
    make_service(enabled=False, **({f"surface.{surface}": True} if surface_key else {}))
    hooks_mod.register_moderation_gateway_hooks()
    stub_llm_registry.reply = reply("forbidden secret")
    stub_llm_registry.items = [openai_frame("forbidden secret"), finish_frame(), DONE]
    result = _call(surface, stream)
    if stream:
        assert list(result) == stub_llm_registry.items
    else:
        assert result == reply("forbidden secret")
    assert stub_llm_registry.requests[0]["messages"][0]["content"] == "forbidden secret"
    assert evaluator_spy == []
    assert list(guardian_files.rglob("*guardian*")) == []


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize("surface", SURFACES)
def test_global_on_makes_evaluator_calls(make_service, stub_llm_registry, evaluator_spy, surface, stream):
    make_service(enabled=True)
    hooks_mod.register_moderation_gateway_hooks()
    stub_llm_registry.reply = reply("fine")
    stub_llm_registry.items = [openai_frame("fine"), finish_frame(), DONE]
    with pytest.raises(LLMPolicyRefusal):
        _call(surface, stream)  # the input holds "forbidden"
    assert len(evaluator_spy) > 0
