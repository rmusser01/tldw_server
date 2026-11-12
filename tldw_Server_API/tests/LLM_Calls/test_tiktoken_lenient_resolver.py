"""A lenient tiktoken resolver, and the three sites that crashed without one.

resolve_tiktoken_encoding is strict BY DESIGN -- it raises TokenizerUnavailable so the
strict-token-counting machinery can refuse an approximate count. That is precisely why
eleven sites re-implemented the encoding_for_model -> cl100k_base fallback instead of
adopting it. The fix is a lenient twin, not an adoption campaign.

Three of those eleven caught only KeyError, but tiktoken raises AttributeError for a
non-string model (verified on 0.14.0). A workflow step written as `model: null` yields
None from config.get("model", "gpt-4") -- the default does not apply to an explicit null --
so the step crashed where its bare-except siblings degraded to cl100k_base.
"""

from __future__ import annotations

import pytest


def test_lenient_resolver_falls_back_for_unknown_model() -> None:
    from tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver import (
        resolve_tiktoken_encoding_or_default,
    )

    enc = resolve_tiktoken_encoding_or_default("definitely-not-a-real-model")
    assert enc is not None
    assert enc.encode("hello")


@pytest.mark.parametrize("bad", [None, 123, "", "   "])
def test_lenient_resolver_tolerates_non_string_models(bad) -> None:
    from tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver import (
        resolve_tiktoken_encoding_or_default,
    )

    enc = resolve_tiktoken_encoding_or_default(bad)
    assert enc.encode("hello")


def test_lenient_resolver_still_honours_a_known_model() -> None:
    import tiktoken

    from tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver import (
        resolve_tiktoken_encoding_or_default,
    )

    assert (
        resolve_tiktoken_encoding_or_default("gpt-4").name
        == tiktoken.encoding_for_model("gpt-4").name
    )


def test_strict_twin_is_unchanged() -> None:
    """The strict contract must survive: it exists so callers can refuse to guess."""
    from tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver import (
        TokenizerUnavailable,
        resolve_tiktoken_encoding,
    )

    with pytest.raises(TokenizerUnavailable):
        resolve_tiktoken_encoding("")
    with pytest.raises(TokenizerUnavailable):
        resolve_tiktoken_encoding("definitely-not-a-real-model")


def test_rag_token_counter_survives_a_none_model() -> None:
    from tldw_Server_API.app.core.RAG.rag_service.utils import TokenCounter

    assert TokenCounter(model=None).count("hello world") > 0  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_workflow_token_count_adapter_survives_model_null() -> None:
    from tldw_Server_API.app.core.Workflows.adapters.text.nlp import (
        run_token_count_adapter,
    )

    out = await run_token_count_adapter({"model": None}, {"text": "hello world"})
    assert isinstance(out, dict)
