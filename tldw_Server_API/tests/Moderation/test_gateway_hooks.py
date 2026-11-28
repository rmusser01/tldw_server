"""Gateway input/output moderation hooks (TASK-13950.27 PR C, part a)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import types

import pytest
from loguru import logger

import tldw_Server_API.app.core.Moderation.gateway_hooks as hooks_mod
from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.app.core.LLM_Calls.gateway import GatewayCall
from tldw_Server_API.app.core.Moderation.execution_runner import ModerationExecutionRunner
from tldw_Server_API.app.core.Moderation.gateway_hooks import (
    INPUT_REFUSAL,
    INPUT_UNCHECKED_REFUSAL,
    OUTPUT_REFUSAL,
    StreamModerator,
    moderate_input_messages_for_surface,
    moderate_output_text_for_surface,
    resolve_policy,
)
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context
from tldw_Server_API.tests.Moderation.conftest import (
    DONE,
    PROVIDER,
    USAGE,
    anthropic_frame,
    cohere_frame,
    finish_frame,
    google_frame,
    openai_frame,
    reply,
    sse,
    usage_frame,
)

pytestmark = pytest.mark.unit

SURFACE = "rag.generation"


def _call(messages=None, *, surface=SURFACE, stream=False):
    return gateway.perform_chat_api_call(
        api_endpoint=PROVIDER,
        model="m",
        api_key="k",
        messages_payload=messages if messages is not None else [{"role": "user", "content": "hi"}],
        streaming=stream,
        llm_surface=surface,
    )


def _parse(item: str) -> dict:
    return json.loads(item.split("data:", 1)[1])


def _text(items) -> str:
    out = ""
    for item in items:
        if isinstance(item, str) and item.startswith("data:") and "[DONE]" not in item:
            for choice in _parse(item).get("choices", []):
                out += (choice.get("delta") or {}).get("content") or ""
    return out


# ------------------------------------------------------------------ input


def test_input_block_raises_and_provider_not_called(moderation_hooks, stub_llm_registry):
    with pytest.raises(LLMPolicyRefusal) as raised:
        _call([{"role": "user", "content": "this is forbidden"}])
    assert raised.value.detail == INPUT_REFUSAL
    assert raised.value.status_code == 400
    assert stub_llm_registry.requests == []


def test_input_redact_reaches_provider_and_caller_list_unchanged(moderation_hooks, stub_llm_registry):
    messages = [{"role": "user", "content": "my secret here"}]
    _call(messages)
    assert stub_llm_registry.requests[0]["messages"] == [{"role": "user", "content": "my [X] here"}]
    assert messages == [{"role": "user", "content": "my secret here"}]


def test_input_moderates_only_user_and_tool_text_parts(moderation_hooks, stub_llm_registry):
    image = {"type": "image_url", "image_url": {"url": "secret"}}
    messages = [
        {"role": "system", "content": "secret"},
        {"role": "assistant", "content": "secret"},
        {"role": "user", "content": [{"type": "text", "text": "a secret"}, image]},
        {"role": "tool", "content": "secret"},
        {"role": "user", "content": None},
    ]
    _call(messages)
    sent = stub_llm_registry.requests[0]["messages"]
    assert sent[0]["content"] == "secret" and sent[1]["content"] == "secret"
    assert sent[2]["content"] == [{"type": "text", "text": "a [X]"}, image]
    assert sent[3]["content"] == "[X]"
    assert sent[4]["content"] is None
    assert messages[2]["content"][0]["text"] == "a secret"


def test_input_text_function_same_list_when_unchanged(moderation_hooks):
    messages = [{"role": "user", "content": "fine"}]
    assert moderate_input_messages_for_surface(messages, surface=SURFACE, user_id=None) is messages
    out = moderate_input_messages_for_surface([{"role": "user", "content": "secret"}], surface=SURFACE, user_id=None)
    assert out == [{"role": "user", "content": "[X]"}]
    with pytest.raises(LLMPolicyRefusal):
        moderate_input_messages_for_surface([{"role": "tool", "content": "forbidden"}], surface=SURFACE, user_id=None)


# ------------------------------------------------------------------ output, non-stream


def test_output_block_non_stream_content_filter_keeps_usage(moderation_hooks, stub_llm_registry):
    stub_llm_registry.reply = reply("a forbidden answer", provider_response={"raw": "forbidden"})
    stub_llm_registry.reply["choices"][0]["message"].update(
        tool_calls=[{"id": "t"}], reasoning_content="forbidden reasoning", unknown_field="forbidden extra"
    )
    result = _call()
    choice = result["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": OUTPUT_REFUSAL}
    assert choice["finish_reason"] == "content_filter"
    assert result["usage"] == USAGE
    assert "forbidden" not in json.dumps(result)


def test_output_redact_non_stream(moderation_hooks, stub_llm_registry):
    stub_llm_registry.reply = reply("the secret is out")
    result = _call()
    assert result["choices"][0]["message"]["content"] == "the [X] is out"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == USAGE


@pytest.mark.parametrize("shape", ["google", "cohere"])
def test_non_stream_redact_drops_provider_response(moderation_hooks, stub_llm_registry, shape):
    raw = (
        {"candidates": [{"content": {"parts": [{"text": "a secret"}]}}]}
        if shape == "google"
        else {"text": "a secret", "meta": {}}
    )
    stub_llm_registry.reply = reply("a secret", provider_response=raw)
    result = _call()
    assert result["choices"][0]["message"]["content"] == "a [X]"
    assert "provider_response" not in result and "secret" not in json.dumps(result)


def test_non_stream_unchanged_result_is_same_content(moderation_hooks, stub_llm_registry):
    stub_llm_registry.reply = reply("fine", provider_response={"raw": 1})
    result = _call()
    assert result["provider_response"] == {"raw": 1}
    stub_llm_registry.reply = {"choices": [{"message": {"role": "assistant", "content": None}}]}
    assert _call()["choices"][0]["message"]["content"] is None


# ------------------------------------------------------------------ output, stream


def test_stream_redact_across_chunk_boundary(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(16)
    stub_llm_registry.items = [openai_frame("keep the sec"), openai_frame("ret safe now, please"), finish_frame(), DONE]
    out = list(_call(stream=True))
    assert _text(out) == "keep the [X] safe now, please"
    assert "secret" not in "".join(out) and "sec" not in _text(out).replace("[X]", "")


def test_stream_redact_drops_provider_response(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(0)
    stub_llm_registry.items = [anthropic_frame("a secret"), anthropic_frame("clean"), finish_frame(), DONE]
    out = list(_call(stream=True))
    first, second = _parse(out[0]), out[1]
    assert first["choices"][0]["delta"]["content"] == "a [X]"
    assert "provider_response" not in first
    # a content chunk with a provider_response is rebuilt without it, even when its text is unchanged
    assert _parse(second)["choices"] == [{"delta": {"content": "clean"}}] and "provider_response" not in _parse(second)
    assert "secret" not in "".join(out)


def test_stream_block_emits_content_filter_done_and_closes_provider(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(8)
    stub_llm_registry.items = [openai_frame("fine so far "), openai_frame("now forbidden text"), openai_frame("later"), usage_frame(), DONE]
    out = list(_call(stream=True))
    assert out[-1] == DONE and out.count(DONE) == 1
    assert _parse(out[-2])["choices"][0]["finish_reason"] == "content_filter"
    assert _text(out).endswith(OUTPUT_REFUSAL)
    assert "forbidden" not in "".join(out) and "later" not in "".join(out)
    assert stub_llm_registry.closed >= 1


def test_unchanged_items_are_the_same_objects(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(0)
    items = [
        openai_frame("héllo wörld — ok"),
        openai_frame("two"),
        finish_frame(),
        usage_frame(),
        sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"name": "f"}}]}}]}),
        DONE,
    ]
    stub_llm_registry.items = items
    out = list(_call(stream=True))
    assert len(out) == len(items) and all(a is b for a, b in zip(out, items))


def test_flush_precedes_finish_reason_chunk(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(1024)
    stub_llm_registry.items = [openai_frame("short"), finish_frame("stop"), usage_frame(), DONE]
    out = list(_call(stream=True))
    assert _parse(out[0])["choices"][0]["delta"] == {"content": "short"}
    assert out[1] is stub_llm_registry.items[1]
    assert out[2] is stub_llm_registry.items[2] and out[3] == DONE and len(out) == 4


def test_non_content_items_pass_through(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(1024)
    items = [
        ": heartbeat\n\n",
        "event: ping\n\n",
        'data: {"error": {"message": "x", "code": "provider_error"}}\n\n',
        sse({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}),
        sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "secret"}}]}}]}),
        'event: message\ndata: {"choices":[{"index":0,"delta":{"reasoning_content":"r"}}]}\n\n',
        "data: not json\n\n",
        usage_frame(),
        b"raw-bytes",
        None,
    ]
    stub_llm_registry.items = items
    out = list(_call(stream=True))
    assert len(out) == len(items) and all(a is b for a, b in zip(out, items))


def test_n2_and_missing_index_keep_choices_separate(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(1024)
    stub_llm_registry.items = [
        openai_frame("a-sec", index=0),
        openai_frame("b-fine", index=1),
        openai_frame("ret", index=0),
        openai_frame(" no index"),  # missing index is choice 0
        finish_frame(index=0),
        finish_frame(index=1),
        DONE,
    ]
    out = list(_call(stream=True))
    by_choice: dict[int, str] = {0: "", 1: ""}
    for item in out:
        if "[DONE]" in item:
            continue
        for choice in _parse(item)["choices"]:
            by_choice[choice.get("index", 0)] += (choice["delta"].get("content") or "")
    assert by_choice == {0: "a-[X] no index", 1: "b-fine"}


def test_changed_chunk_with_finish_reason_keeps_it(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(1024)
    stub_llm_registry.items = [sse({"choices": [{"index": 0, "delta": {"content": "a secret"}, "finish_reason": "stop"}]}), DONE]
    out = list(_call(stream=True))
    chunk = _parse(out[0])
    assert chunk["choices"][0]["delta"] == {"content": "a [X]"} and chunk["choices"][0]["finish_reason"] == "stop"


def test_changed_content_drops_logprobs(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(0)
    lp = {"content": [{"token": "secret"}]}
    stub_llm_registry.items = [sse({"choices": [{"index": 0, "delta": {"content": "secret"}, "logprobs": lp}]}), DONE]
    stub_llm_registry.reply = reply("secret")
    stub_llm_registry.reply["choices"][0]["logprobs"] = lp
    assert "secret" not in "".join(_call(stream=True)) and "secret" not in json.dumps(_call())


def test_stream_without_done_flushes_held_text(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(1024)
    stub_llm_registry.items = [openai_frame("held text")]
    out = list(_call(stream=True))
    assert _text(out) == "held text" and DONE not in out


def test_done_flushes_held_text_before_done(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(1024)
    stub_llm_registry.items = [google_frame("tail"), cohere_frame(" more"), DONE]
    out = list(_call(stream=True))
    assert _text(out[:-1]) == "tail more" and out[-1] == DONE
    assert "provider_response" not in "".join(out[:-1])


def test_after_flush_text_never_leaks_through_provider_response(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(1024)
    stub_llm_registry.items = [anthropic_frame("a secret"), DONE]
    out = list(_call(stream=True))
    assert "secret" not in "".join(out)


@pytest.fixture
def runner_threads(monkeypatch):
    checks: set[tuple[str | None, int]] = set()
    execute = ModerationExecutionRunner.execute

    def observe(self, request, scope):
        checks.add((request.phase, threading.get_ident()))
        return execute(self, request, scope)

    monkeypatch.setattr(ModerationExecutionRunner, "execute", observe)
    return checks


@pytest.mark.asyncio
async def test_async_entry_stream_is_wrapped(moderation_hooks, stub_llm_registry, buffer_chars, runner_threads):
    buffer_chars(0)
    stub_llm_registry.items = [anthropic_frame("a secret"), finish_frame(), usage_frame(), DONE]
    stream = await gateway.perform_chat_api_call_async(
        api_endpoint=PROVIDER, model="m", api_key="k", messages_payload=[{"role": "user", "content": "hi"}],
        streaming=True, llm_surface=SURFACE,
    )
    out = [item async for item in stream]
    assert _text(out) == "a [X]" and out[-1] == DONE and "secret" not in "".join(out)
    assert {(phase, tid != threading.get_ident()) for phase, tid in runner_threads} == {
        ("input", True), ("output", True),
    }


async def test_async_stream_without_text_passes_through_when_moderation_is_full(
    moderation_hooks, stub_llm_registry, monkeypatch,
):
    execute = hooks_mod.run_moderation_async

    async def check(call, *, metadata):
        if metadata.phase == "output":
            raise ModerationExecutionError("capacity_exhausted")
        return await execute(call, metadata=metadata)

    monkeypatch.setattr(hooks_mod, "run_moderation_async", check)
    tool = sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}}]})
    items = [tool, finish_frame("tool_calls"), usage_frame(), DONE]
    stub_llm_registry.items = items
    stream = await gateway.perform_chat_api_call_async(
        api_endpoint=PROVIDER, model="m", api_key="k", messages_payload=[{"role": "user", "content": "hi"}],
        streaming=True, llm_surface=SURFACE,
    )
    assert [item async for item in stream] == items


@pytest.mark.parametrize("cancel", [False, True], ids=["admission-failure", "cancelled-check"])
async def test_async_stream_abandoned_during_check_closes_adapter(moderation_hooks, stub_llm_registry, monkeypatch, cancel):
    execute = hooks_mod.run_moderation_async
    checking = asyncio.Event()
    closed = False

    async def check(call, *, metadata):
        if metadata.phase == "output":
            if not cancel:
                raise ModerationExecutionError("capacity_exhausted")
            checking.set()
            await asyncio.Future()
        return await execute(call, metadata=metadata)

    async def source():
        nonlocal closed
        try:
            yield openai_frame("provider reply")
            yield DONE
        finally:
            closed = True

    monkeypatch.setattr(hooks_mod, "run_moderation_async", check)
    async with contextlib.aclosing(source()) as upstream:
        monkeypatch.setattr(stub_llm_registry, "astream", lambda *_a, **_kw: upstream)
        stream = await gateway.perform_chat_api_call_async(
            api_endpoint=PROVIDER, model="m", api_key="k", messages_payload=[{"role": "user", "content": "hi"}],
            streaming=True, llm_surface=SURFACE,
        )
        async with contextlib.aclosing(stream):
            if cancel:
                read = asyncio.create_task(anext(stream))
                try:
                    await asyncio.wait_for(checking.wait(), timeout=5)
                finally:
                    read.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await read
            else:
                assert _text([await anext(stream)]) == OUTPUT_REFUSAL
            assert closed


@pytest.mark.asyncio
async def test_async_entry_input_block_and_non_stream_redact(moderation_hooks, stub_llm_registry, runner_threads):
    kwargs = {"api_endpoint": PROVIDER, "model": "m", "api_key": "k", "llm_surface": SURFACE}
    with pytest.raises(LLMPolicyRefusal):
        await gateway.perform_chat_api_call_async(**kwargs, messages_payload=[{"role": "user", "content": "forbidden"}])
    assert stub_llm_registry.requests == []
    stub_llm_registry.reply = reply("a secret")
    result = await gateway.perform_chat_api_call_async(**kwargs, messages_payload=[{"role": "user", "content": "hi"}])
    assert result["choices"][0]["message"]["content"] == "a [X]"
    assert {(phase, tid != threading.get_ident()) for phase, tid in runner_threads} == {
        ("input", True), ("output", True),
    }


@pytest.mark.asyncio
async def test_async_entry_cancellation_signals_running_check(moderation_hooks, stub_llm_registry, monkeypatch):
    loop = asyncio.get_running_loop()
    caller_thread = threading.get_ident()
    started, finished = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    scopes = []
    execute = ModerationExecutionRunner.execute

    def pause(self, request, scope):
        scopes.append(scope)
        loop.call_soon_threadsafe(started.set)
        try:
            if threading.get_ident() != caller_thread:
                release.wait(timeout=5)
            return execute(self, request, scope)
        finally:
            loop.call_soon_threadsafe(finished.set)

    monkeypatch.setattr(ModerationExecutionRunner, "execute", pause)
    task = asyncio.create_task(gateway.perform_chat_api_call_async(
        api_endpoint=PROVIDER, model="m", api_key="k", llm_surface=SURFACE,
        messages_payload=[{"role": "user", "content": "hello"}],
    ))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert scopes[0].cancelled.is_set()
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=5)
    assert stub_llm_registry.requests == []


def test_stream_moderator_unit(moderation_hooks):
    policy = moderation_hooks.get_effective_policy(None, surface=SURFACE)
    mod = StreamModerator(policy, holdback=4)
    assert mod.feed("abcdefgh") == ("abcd", False)
    assert mod.held == "efgh" and mod.flush() == "efgh" and mod.flush() == ""
    assert StreamModerator(policy, holdback=4).feed("xx forbidden")[1] is True


# ------------------------------------------------------------------ skips and policy


@pytest.mark.parametrize("surface", ["chat.completions", "internal.eval_judge", "moderation.classifier", None])
def test_caller_moderated_exempt_and_none_surfaces_skip(moderation_hooks, stub_llm_registry, surface):
    stub_llm_registry.reply = reply("forbidden secret")
    result = _call([{"role": "user", "content": "forbidden secret"}], surface=surface)
    assert result["choices"][0]["message"]["content"] == "forbidden secret"
    assert stub_llm_registry.requests[0]["messages"][0]["content"] == "forbidden secret"
    assert resolve_policy(surface, None) is None


def test_policy_resolved_once_per_call(moderation_hooks, stub_llm_registry, monkeypatch):
    calls = []
    real = moderation_hooks.get_effective_policy
    monkeypatch.setattr(
        moderation_hooks, "get_effective_policy", lambda uid, *, surface: (calls.append(surface), real(uid, surface=surface))[1]
    )
    result = _call()
    assert calls == [SURFACE]
    assert result["choices"][0]["message"]["content"] == "provider reply"


def test_post_call_resolves_when_no_pre_hook_ran(moderation_hooks):
    call = GatewayCall(surface=SURFACE, provider=PROVIDER, request={}, stream=False, user_id=None)
    out = hooks_mod.post_call(call, reply("a secret"))
    assert out["choices"][0]["message"]["content"] == "a [X]"


def _guardian(monkeypatch, *, exists=True):
    import tldw_Server_API.app.core.DB_Management.guardian_db_resolver as resolver
    import tldw_Server_API.app.core.feature_flags as flags
    import tldw_Server_API.app.core.Moderation.supervised_policy as sp

    seen = {"resolve": 0, "engines": 0}

    class Engine:
        def build_moderation_policy_overlay(self, uid, base, chat_type=None):
            import dataclasses
            import re

            from tldw_Server_API.app.core.Moderation.models import PatternRule

            seen["chat_type"] = chat_type
            rule = PatternRule(regex=re.compile("guarded"), action="block")
            return dataclasses.replace(base, enabled=True, block_patterns=[*base.block_patterns, rule])

    def resolve(user_id):
        seen["resolve"] += 1
        if not exists:
            raise FileNotFoundError("Guardian DB does not exist")
        return object()

    def engine_for(db):
        seen["engines"] += 1
        return Engine()

    monkeypatch.setattr(flags, "is_guardian_enabled", lambda: True)
    monkeypatch.setattr(resolver, "resolve_existing_guardian_db_for_user_id", resolve)
    monkeypatch.setattr(sp, "get_supervised_policy_engine", engine_for)
    return seen


def test_guardian_overlay_enables_for_hook_surface(make_service, stub_llm_registry, monkeypatch):
    make_service(enabled=False)
    seen = _guardian(monkeypatch)
    hooks_mod.register_moderation_gateway_hooks()
    with llm_call_context(user_id=7):
        with pytest.raises(LLMPolicyRefusal):
            _call([{"role": "user", "content": "guarded words"}])
        _call()
        _call()
    assert seen["chat_type"] == "regular"
    assert seen["engines"] == 1  # one engine per user, cached across calls
    assert stub_llm_registry.requests and len(stub_llm_registry.requests) == 2


def test_no_guardian_db_means_no_overlay_and_no_file_created(make_service, stub_llm_registry, monkeypatch, tmp_path):
    make_service(enabled=False)
    seen = _guardian(monkeypatch, exists=False)
    hooks_mod.register_moderation_gateway_hooks()
    before = sorted(p for p in tmp_path.rglob("*"))
    with llm_call_context(user_id=7):
        result = _call([{"role": "user", "content": "guarded forbidden"}])
    assert result["choices"][0]["message"]["content"] == "provider reply"
    assert seen["resolve"] == 1 and seen["engines"] == 0
    assert sorted(tmp_path.rglob("*")) == before


def test_guardian_failure_falls_back_to_base_policy(make_service, stub_llm_registry, monkeypatch):
    make_service(enabled=True)
    seen = _guardian(monkeypatch)
    import tldw_Server_API.app.core.Moderation.supervised_policy as sp

    monkeypatch.setattr(sp, "get_supervised_policy_engine", lambda db: types.SimpleNamespace(
        build_moderation_policy_overlay=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))))
    hooks_mod.register_moderation_gateway_hooks()
    with llm_call_context(user_id=7):
        with pytest.raises(LLMPolicyRefusal):
            _call([{"role": "user", "content": "forbidden"}])
    assert seen["resolve"] == 1


def test_user_id_none_uses_default_policy(make_service, monkeypatch):
    service = make_service(enabled=True)
    seen = []
    real = service.get_effective_policy
    monkeypatch.setattr(service, "get_effective_policy", lambda uid, *, surface: (seen.append(uid), real(uid, surface=surface))[1])
    assert resolve_policy(SURFACE, None) is not None
    assert resolve_policy(SURFACE, 5) is not None
    assert seen == [None, "5"]


def test_disabled_policy_and_phase_skip(make_service):
    make_service(enabled=False)
    assert resolve_policy(SURFACE, None) is None
    text, action = moderate_output_text_for_surface("forbidden", surface=SURFACE, user_id=None)
    assert (text, action) == ("forbidden", "pass")


def test_output_text_function_actions(moderation_hooks):
    assert moderate_output_text_for_surface("ok", surface=SURFACE, user_id=None) == ("ok", "pass")
    assert moderate_output_text_for_surface("a secret", surface=SURFACE, user_id=None) == ("a [X]", "redact")
    assert moderate_output_text_for_surface("forbidden", surface=SURFACE, user_id=None) == (OUTPUT_REFUSAL, "block")


def test_reentrant_call_is_not_moderated(moderation_hooks, stub_llm_registry, monkeypatch):
    nested = {}
    real = moderation_hooks.evaluate_action

    def evaluate(text, policy, phase):
        if "nested" not in nested:  # a gateway call made while a hook evaluates
            nested["nested"] = _call([{"role": "user", "content": "forbidden"}])
        return real(text, policy, phase)

    monkeypatch.setattr(moderation_hooks, "evaluate_action", evaluate)
    _call()
    assert nested["nested"]["choices"][0]["message"]["content"] == "provider reply"


def test_hooks_log_no_message_text(moderation_hooks, stub_llm_registry):
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), level="WARNING")
    try:
        stub_llm_registry.reply = reply("the secret is out")
        _call([{"role": "user", "content": "my secret"}])
    finally:
        logger.remove(sink)
    joined = "".join(lines)
    assert "moderation input redact surface=rag.generation provider=e2e-fake" in joined
    assert "my secret" not in joined and "the secret" not in joined and "k" != ""


def test_evaluator_failure_on_output_blocks(moderation_hooks, stub_llm_registry, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    stub_llm_registry.reply = reply("anything")
    monkeypatch.setattr(moderation_hooks, "evaluate_action", boom)
    with pytest.raises(LLMPolicyRefusal) as raised:  # input fails closed, as a refusal and not a provider failure
        _call([{"role": "user", "content": "x"}])
    assert raised.value.detail == INPUT_UNCHECKED_REFUSAL and isinstance(raised.value.__cause__, RuntimeError)
    assert stub_llm_registry.requests == []
    with pytest.raises(LLMPolicyRefusal):
        moderate_input_messages_for_surface([{"role": "user", "content": "x"}], surface=SURFACE, user_id=None)
    call = GatewayCall(surface=SURFACE, provider=PROVIDER, request={}, stream=False, user_id=None)
    call._moderation_policy = moderation_hooks.get_effective_policy(None, surface=SURFACE)
    assert hooks_mod.post_call(call, reply("anything"))["choices"][0]["message"]["content"] == OUTPUT_REFUSAL


def test_bad_stream_buffer_config_warns_and_uses_default(monkeypatch):
    import tldw_Server_API.app.core.config_sections.moderation as cfg

    def bad(parser, env=None):
        raise ValueError("bad")

    monkeypatch.setattr(cfg, "load_moderation_surface_config", bad)
    assert hooks_mod._stream_buffer_chars() == 1024


# ------------------------------------------------------------------ registration


def test_register_is_idempotent_remove_resets_and_reregisters(make_service):
    make_service()
    remove = hooks_mod.register_moderation_gateway_hooks()
    again = hooks_mod.register_moderation_gateway_hooks()
    assert len(gateway._POLICY_HOOKS) == 1
    again()
    assert gateway._POLICY_HOOKS == []
    remove()  # a second remove is harmless
    hooks_mod.register_moderation_gateway_hooks()
    assert len(gateway._POLICY_HOOKS) == 1


def test_zero_holdback_multi_part_google_event_leaks_nothing(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(0)
    event = {"candidates": [{"content": {"parts": [{"text": "fine part "}, {"text": "a secret part"}]}}]}
    stub_llm_registry.items = [
        sse({"choices": [{"delta": {"content": "fine part "}}], "provider_response": event}),
        sse({"choices": [{"delta": {"content": "a secret part"}}], "provider_response": event}),
        DONE,
    ]
    out = list(_call(stream=True))
    assert "secret" not in "".join(out) and "provider_response" not in "".join(out)
    assert _text(out) == "fine part a [X] part"


def test_post_call_resolution_failure_fails_closed(moderation_hooks, monkeypatch, stub_llm_registry):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(moderation_hooks, "get_effective_policy", boom)
    call = GatewayCall(surface=SURFACE, provider=PROVIDER, request={}, stream=False, user_id=None)
    out = hooks_mod.post_call(call, reply("a secret"))
    assert out["choices"][0]["message"]["content"] == OUTPUT_REFUSAL and out["usage"] == USAGE
    call = GatewayCall(surface=SURFACE, provider=PROVIDER, request={}, stream=True, user_id=None)
    stream = list(hooks_mod.post_call(call, iter([openai_frame("a secret"), DONE])))
    assert _text(stream) == OUTPUT_REFUSAL and stream[-1] == DONE and "secret" not in "".join(stream)


def test_changed_multi_line_item_keeps_non_data_lines(moderation_hooks, stub_llm_registry, buffer_chars):
    buffer_chars(0)
    item = 'event: message\nid: 7\ndata: {"choices":[{"index":0,"delta":{"content":"a secret"}}]}\n\n'
    stub_llm_registry.items = [item, DONE]
    out = list(_call(stream=True))
    lines = out[0].splitlines()
    assert lines[:2] == ["event: message", "id: 7"] and lines[2].startswith("data: ")
    assert json.loads(lines[2][6:])["choices"][0]["delta"]["content"] == "a [X]" and out[0].endswith("\n\n")


def test_closing_the_sync_wrapper_before_first_next_closes_the_source(moderation_hooks):
    closed = []

    class Source:
        def __iter__(self):
            return iter([openai_frame("x")])

        def close(self):
            closed.append(1)

    call = GatewayCall(surface=SURFACE, provider=PROVIDER, request={}, stream=True, user_id=None)
    source = Source()
    wrapper = hooks_mod.post_call(call, source)
    assert wrapper is not source  # moderation is on: the stream is wrapped, not passed through
    wrapper.close()
    assert closed == [1]


def test_off_path_returns_the_same_object(make_service):
    make_service(enabled=False)
    call = GatewayCall(surface=SURFACE, provider=PROVIDER, request={}, stream=False, user_id=None)
    result = reply("a secret", provider_response={"raw": 1})
    assert hooks_mod.post_call(call, result) is result
    items = [anthropic_frame("a secret"), finish_frame(), DONE]
    source = iter(items)
    stream_call = GatewayCall(surface=SURFACE, provider=PROVIDER, request={}, stream=True, user_id=None)
    out = hooks_mod.post_call(stream_call, source)
    assert out is source
    assert all(a is b for a, b in zip(list(out), items, strict=True))


@pytest.mark.asyncio
async def test_async_policy_resolution_failure_refuses(moderation_hooks, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(moderation_hooks, "get_effective_policy", boom)

    async def source():
        yield openai_frame("a secret")
        yield DONE

    call = GatewayCall(surface=SURFACE, provider=PROVIDER, request={}, stream=True, user_id=None)
    out = [item async for item in hooks_mod.post_call(call, source())]
    assert _text(out) == OUTPUT_REFUSAL and out[-1] == DONE and "secret" not in "".join(out)
