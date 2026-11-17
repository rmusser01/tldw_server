"""First-chat verification neither records nor is gated (LLM usage accounting spec §1)."""

import pytest

from tldw_Server_API.app.core.Setup import first_chat_verifier
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context

pytestmark = pytest.mark.unit


async def test_first_chat_verification_runs_exempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """The verification call sees exempt=True."""
    seen: list[bool] = []

    async def _probe(**kwargs: object) -> dict:
        """Report the flag, then return a minimal completion."""
        seen.append(current_llm_call_context().exempt)
        return {"choices": [{"message": {"content": "hello"}}]}

    monkeypatch.setattr(first_chat_verifier, "perform_chat_api_call_async", _probe)
    await first_chat_verifier._call_chat_completion(provider="openai", model="m", prompt="hi")
    assert seen == [True]
