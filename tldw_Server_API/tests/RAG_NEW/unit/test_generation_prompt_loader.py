"""Unit tests for RAG generation prompt-template loading behavior."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.RAG.rag_service import generation as generation_mod
from tldw_Server_API.app.core.RAG.rag_service.generation import PromptTemplates


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.debug_kwargs: list[dict[str, object]] = []
        self.errors: list[str] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)
        self.debug_kwargs.append(dict(kwargs))

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.errors.append(message)


def test_prompt_templates_load_switchable_profile_prompt_keys() -> None:
    text = PromptTemplates.get_template("instruction_tuned")

    assert "Use the provided context" in text
    assert "{context}" in text
    assert "{question}" in text


def test_prompt_templates_falls_back_to_default_for_unknown_key() -> None:
    unknown = PromptTemplates.get_template("does_not_exist")

    assert "Context:" in unknown
    assert "Question:" in unknown


def test_prompt_templates_sanitizes_loader_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    def _fail_load_prompt(_category: str, _name: str) -> str:
        raise RuntimeError(
            "prompt loader failed at /private/prompts/rag.yaml "
            "api_key=sk-test-private-token"
        )

    PromptTemplates._load_rag_prompt_cached.cache_clear()
    monkeypatch.setattr(generation_mod, "load_prompt", _fail_load_prompt)
    monkeypatch.setattr(generation_mod, "logger", logger_stub)

    try:
        text = PromptTemplates.get_template("instruction_tuned")
    finally:
        PromptTemplates._load_rag_prompt_cached.cache_clear()

    assert text == PromptTemplates.DEFAULT
    assert logger_stub.debugs == ["Prompt loader failed for rag prompt 'instruction_tuned'"]
    rendered = "\n".join(logger_stub.debugs)
    assert "prompt loader failed at" not in rendered
    assert "/private/prompts/rag.yaml" not in rendered
    assert "sk-test-private-token" not in rendered


@pytest.mark.asyncio
async def test_generate_streaming_response_warms_prompt_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmed: list[str] = []

    async def _fake_warm(name: str) -> None:
        warmed.append(name)

    class _StubGenerator:
        async def generate_stream(self, context: Any, query: str) -> AsyncIterator[str]:
            if False:
                yield ""  # pragma: no cover

    monkeypatch.setattr(PromptTemplates, "warm_template_async", _fake_warm)
    monkeypatch.setattr(generation_mod, "create_generator", lambda _config: _StubGenerator())

    ctx = SimpleNamespace(
        config={"generation": {"prompt_template": "instruction_tuned"}},
        query="warm prompt",
        metadata={},
    )

    await generation_mod.generate_streaming_response(ctx)

    assert warmed == ["instruction_tuned"]


@pytest.mark.asyncio
async def test_generate_streaming_response_ignores_non_generator_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_configs: list[dict[str, Any]] = []

    class _StubGenerator:
        async def generate_stream(self, context: Any, query: str) -> AsyncIterator[str]:
            if False:
                yield ""  # pragma: no cover

    def _fake_create_generator(config: dict[str, Any]) -> _StubGenerator:
        captured_configs.append(dict(config))
        return _StubGenerator()

    monkeypatch.setattr(generation_mod, "create_generator", _fake_create_generator)

    ctx = SimpleNamespace(
        config={"generation": {"provider": "openai", "model": "gpt-4o-mini"}},
        query="stream without config blowups",
        metadata={},
    )

    await generation_mod.generate_streaming_response(
        ctx,
        enable_claims=True,
        claims_top_k=5,
        claims_concurrency=4,
    )

    assert captured_configs == [
        {"provider": "openai", "model": "gpt-4o-mini", "streaming": True}
    ]


@pytest.mark.asyncio
async def test_llm_generator_error_log_sanitizes_provider_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    generator = generation_mod.LLMGenerator(
        generation_mod.GenerationConfig(
            provider="openai",
            model="gpt-4o-mini",
            fallback_enabled=False,
        )
    )

    async def _fail_call_llm(_prompt: str, **_kwargs: Any) -> str:
        raise RuntimeError(
            "provider failed at /private/rag-generation/provider.log "
            "api_key=sk-test-private-token"
        )

    monkeypatch.setattr(generator, "_call_llm", _fail_call_llm)
    monkeypatch.setattr(generation_mod, "logger", logger_stub)

    with pytest.raises(RuntimeError):
        await generator.generate(
            SimpleNamespace(documents=[]),
            "query containing sk-test-private-token",
        )

    assert logger_stub.errors == ["Error generating response"]
    rendered = "\n".join(logger_stub.errors)
    assert "provider failed" not in rendered
    assert "/private/rag-generation/provider.log" not in rendered
    assert "sk-test-private-token" not in rendered


@pytest.mark.asyncio
async def test_streaming_generator_ignores_finish_only_openai_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _stream() -> AsyncIterator[dict[str, Any]]:
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Machine learning "},
                    "finish_reason": None,
                }
            ]
        }
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "id": "chatcmpl-terminal",
            "object": "chat.completion.chunk",
        }

    async def _fake_call_llm(_prompt: str, **_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        return _stream()

    generator = generation_mod.StreamingGenerator(
        generation_mod.GenerationConfig(provider="openai", model="gpt-4o-mini")
    )
    monkeypatch.setattr(generator, "_call_llm", _fake_call_llm)

    ctx = SimpleNamespace(documents=[], query="What is machine learning?")
    chunks = [chunk async for chunk in generator.generate_stream(ctx, ctx.query)]

    assert chunks == ["Machine learning "]


@pytest.mark.asyncio
async def test_streaming_claims_overlay_debug_log_omits_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    class _StubGenerator:
        async def generate_stream(self, _context: Any, _query: str) -> AsyncIterator[str]:
            yield (
                "Sentence one has enough padding to start the overlay buffer "
                "with a complete sentence and a meaningful amount of content. "
            )
            yield (
                "Sentence two also has enough padding to force the claims overlay path "
                "without exposing private tokens. Additional neutral padding ensures the "
                "buffer exceeds the overlay threshold used by streaming generation."
            )

    class _ExplodingClaimsEngine:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            raise RuntimeError(
                "claims overlay failed at /private/rag/claims.db token=secret-token"
            )

    monkeypatch.setattr(generation_mod, "create_generator", lambda _config: _StubGenerator())
    monkeypatch.setattr(generation_mod, "ClaimsEngine", _ExplodingClaimsEngine)
    monkeypatch.setattr(generation_mod, "logger", logger_stub)

    ctx = SimpleNamespace(
        config={"generation": {"provider": "openai", "model": "gpt-4o-mini"}},
        query="claims overlay sanitizer",
        metadata={},
        documents=[],
    )

    result = await generation_mod.generate_streaming_response(ctx, enable_claims=True)
    chunks = [chunk async for chunk in result.stream_generator]

    assert "".join(chunks).startswith("Sentence one")
    assert "claims_overlay" not in result.metadata
    assert logger_stub.debugs == ["Claims overlay enrichment failed during streaming generation"]
    assert logger_stub.debug_kwargs == [{}]
    rendered = "\n".join(logger_stub.debugs) + repr(logger_stub.debug_kwargs)
    assert "/private/" not in rendered
    assert "secret-token" not in rendered
    assert "claims overlay failed" not in rendered
