"""With moderation off, /chat/completions makes zero evaluator calls and the provider's reply is untouched
(TASK-13950.27 PR C, part b)."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator
from tldw_Server_API.tests.Moderation.conftest import DONE, finish_frame, openai_frame, reply
from tldw_Server_API.tests.Moderation.conftest import make_service as make_service  # noqa: F401
from tldw_Server_API.tests.Moderation.conftest import stub_llm_registry as stub_llm_registry  # noqa: F401

pytestmark = pytest.mark.integration

_BODY = {"model": "claude-3-5-haiku-20241022", "api_provider": "anthropic",
         "messages": [{"role": "user", "content": "forbidden secret"}]}


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_chat_completions_global_off_zero_moderation_calls(
    authenticated_client, mock_chacha_db, setup_dependencies, make_service, stub_llm_registry, monkeypatch, stream
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    make_service(enabled=False, **{"surface.chat.completions": True})
    calls: list[str] = []
    for name in ("evaluate_text", "redact_text", "redact_text_with_count"):
        real = getattr(PolicyEvaluator, name)
        monkeypatch.setattr(PolicyEvaluator, name, lambda self, *a, _r=real, _n=name, **k: (calls.append(_n), _r(self, *a, **k))[1])
    stub_llm_registry.reply = reply("forbidden secret reply")
    stub_llm_registry.items = [openai_frame("forbidden secret reply"), finish_frame(), DONE]
    response = authenticated_client.post("/api/v1/chat/completions", json={**_BODY, "stream": stream})
    assert response.status_code == 200, response.text
    assert "forbidden secret reply" in response.text
    assert "forbidden secret" in str(stub_llm_registry.requests[0]["messages"][-1]["content"])
    assert calls == []
