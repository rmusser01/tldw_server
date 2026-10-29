"""The gateway's pre-call and post-call policy hooks (TASK-13950.13, PR D)."""

import gc
import threading
import warnings

import pytest

from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
from tldw_Server_API.app.core.MCP_unified.modules.implementations.quizzes_module import QuizzesModule
from tldw_Server_API.app.core.RAG.rag_service.generation import GenerationConfig, LLMGenerator
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id, llm_call_context
from tldw_Server_API.app.core.Workflows.adapters.llm import run_llm_adapter

pytestmark = pytest.mark.unit


class _Adapter:
    """Answers every call with one fixed completion, or a two-chunk stream."""

    def chat(self, request, *, timeout=None):
        return {"choices": [{"message": {"role": "assistant", "content": '{"intent": "factual"}'}}],
                "seen_request": dict(request)}

    def stream(self, request, *, timeout=None):
        return iter(["data: a\n\n", "data: b\n\n"])

    async def astream(self, request, *, timeout=None):
        for chunk in ("data: a\n\n", "data: b\n\n"):
            yield chunk


class _Registry:
    def get_adapter(self, name):
        return _Adapter()


class _Refused(Exception):
    pass


_CALL = {"api_endpoint": "e2e-fake", "model": "m", "api_key": "k", "messages_payload": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def fake_provider(monkeypatch):
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    monkeypatch.setattr(gateway, "_get_llm_registry", lambda: _Registry())


@pytest.fixture
def recorded():
    calls = {"pre": [], "post": []}
    remove = gateway.add_policy_hooks(
        pre_call=lambda call: calls["pre"].append((call.surface, call.provider)),
        post_call=lambda call, result: (calls["post"].append(call.surface), result)[1],
    )
    yield calls
    remove()


def test_no_hook_is_registered_by_default_and_calls_are_unchanged(fake_provider):
    assert gateway._POLICY_HOOKS == []
    assert gateway.perform_chat_api_call(**_CALL)["choices"][0]["message"]["content"] == '{"intent": "factual"}'


async def _workflow():
    return await run_llm_adapter({"provider": "e2e-fake", "model": "m", "prompt": "hi", "api_key": "k"}, {"user_id": "1"})


async def _rag():  # RAG answer generation, the surface .27 moderates (rag_service/generation.py, perform_chat_api_call_async)
    config = GenerationConfig(provider="e2e-fake", model="m", api_key="k", streaming=False)
    return await LLMGenerator(config)._call_llm("hi")


async def _mcp():  # the weakest driver: a private method on an instance built without __init__
    return await QuizzesModule.__new__(QuizzesModule)._call_llm("make a quiz", provider="e2e-fake", model="m")


@pytest.mark.asyncio
@pytest.mark.parametrize("drive", [_workflow, _rag, _mcp], ids=["workflows", "rag", "mcp"])
async def test_a_registered_hook_fires_for_workflow_rag_and_mcp_calls(fake_provider, recorded, drive):
    result = await drive()

    assert result is not None
    assert recorded["pre"] == [(None, "e2e-fake")]
    assert recorded["post"] == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_a_pre_call_refusal_reaches_the_caller_unchanged_and_skips_the_provider(fake_provider, monkeypatch, mode):
    refusal = _Refused("refused by policy")

    def refuse(call):
        raise refusal

    monkeypatch.setattr(_Adapter, "chat", lambda self, request, **kw: pytest.fail("provider called after a refusal"))
    remove = gateway.add_policy_hooks(pre_call=refuse)
    try:
        with pytest.raises(_Refused) as raised:
            kwargs = {**_CALL, "call_policy": ProviderCallPolicy(privacy_safe_errors=True)}
            if mode == "sync":
                gateway.perform_chat_api_call(**kwargs)
            else:
                await gateway.perform_chat_api_call_async(**kwargs)
    finally:
        remove()

    assert raised.value is refusal


def test_a_post_call_hook_can_wrap_a_sync_stream(fake_provider):
    remove = gateway.add_policy_hooks(post_call=lambda call, result: (chunk.upper() for chunk in result))
    try:
        assert list(gateway.perform_chat_api_call(**_CALL, streaming=True)) == ["DATA: A\n\n", "DATA: B\n\n"]
    finally:
        remove()


@pytest.mark.asyncio
async def test_a_post_call_hook_can_wrap_an_async_stream(fake_provider):
    def upper(call, stream):
        async def wrapper():
            async for chunk in stream:
                yield chunk.upper()

        return wrapper()

    remove = gateway.add_policy_hooks(post_call=upper)
    try:
        stream = await gateway.perform_chat_api_call_async(**_CALL, streaming=True)
        assert [c async for c in stream] == ["DATA: A\n\n", "DATA: B\n\n"]
    finally:
        remove()


def test_the_surface_id_reaches_hooks_but_not_the_provider(fake_provider):
    seen = {}

    def redact(call):
        seen["surface"] = call.surface
        call.request["messages"] = [{"role": "user", "content": "[redacted]"}]

    remove = gateway.add_policy_hooks(pre_call=redact)
    try:
        result = gateway.perform_chat_api_call(**_CALL, llm_surface="chat.completions")
    finally:
        remove()

    assert result["seen_request"]["messages"] == [{"role": "user", "content": "[redacted]"}]
    assert "llm_surface" not in result["seen_request"]
    assert seen["surface"] == "chat.completions"


def test_an_async_hook_on_a_sync_call_is_a_type_error(fake_provider):
    async def async_pre(call):
        return None

    remove = gateway.add_policy_hooks(pre_call=async_pre)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(TypeError, match="async_pre"):
                gateway.perform_chat_api_call(**_CALL)
            gc.collect()
    finally:
        remove()

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning) and "never awaited" in str(w.message)]


def test_a_stream_wrapper_consumed_in_another_thread_sees_the_user(fake_provider):
    seen = []

    def wrap(call, stream):
        def wrapper():
            for chunk in stream:
                seen.append((call.user_id, current_llm_user_id()))
                yield chunk

        return wrapper()

    remove = gateway.add_policy_hooks(post_call=wrap)
    try:
        with llm_call_context(user_id=41):
            stream = gateway.perform_chat_api_call(**_CALL, streaming=True)
        chunks = []
        consumer = threading.Thread(target=lambda: chunks.extend(stream))  # a plain thread carries no context
        consumer.start()
        consumer.join(timeout=10)
    finally:
        remove()

    assert chunks == ["data: a\n\n", "data: b\n\n"]
    assert seen == [(41, None), (41, None)]  # the call kept the user; the consuming thread's context has none
