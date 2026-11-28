"""/api/v1/messages is moderated: the gateway path through the hooks, the native providers explicitly
(TASK-13950.27 PR D). Native tests use llama.cpp, which takes the same branch as anthropic."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import threading
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.schemas.anthropic_messages import AnthropicMessagesRequest
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.Moderation import gateway_hooks as hooks_mod
from tldw_Server_API.app.core.Moderation.execution_runner import ModerationExecutionRunner
from tldw_Server_API.app.core.Usage import llm_accounting
from tldw_Server_API.tests.Moderation.conftest import reply

pytestmark = pytest.mark.integration

REFUSAL = "Output violates moderation policy"
def _body(model: str) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}


def _native_response(text: str = "ok", **extra) -> dict:
    return {
        "id": "msg-1", "type": "message", "role": "assistant", "model": "test-model",
        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1}, **extra,
    }


def _ev(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _stream_frames(*texts: str, thinking: str | None = None) -> list[str]:
    start = {"type": "message_start", "message": {
        "id": "msg-s", "type": "message", "role": "assistant", "model": "claude-test", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 17, "output_tokens": 1}}}
    frames = [_ev("message_start", start)]
    i = 0
    frames.append(_ev("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}))
    frames += [_ev("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": t}}) for t in texts]
    frames.append(_ev("content_block_stop", {"type": "content_block_stop", "index": i}))
    if thinking:  # after the text: only a change that already happened can drop it
        frames += [
            _ev("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "thinking", "thinking": ""}}),
            _ev("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "thinking_delta", "thinking": thinking}}),
            _ev("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "signature_delta", "signature": "sig"}}),
            _ev("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ]
    frames += [
        _ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 9}}),
        _ev("message_stop", {"type": "message_stop"}),
    ]
    return frames


@pytest.fixture
def byok_native_provider(monkeypatch, messages_endpoint):
    """A resolved BYOK credential with usage recording and no real credential store."""
    used: list[str] = []

    class Runtime:
        def __init__(self, **_kw): ...
        async def resolve(self, provider, *, model=None):
            return SimpleNamespace(provider=provider, api_key="k", app_config={}, credentials_resolved=True)
        async def mark_used(self, credentials):
            used.append(credentials.provider)
        async def close(self): ...

    monkeypatch.setattr("tldw_Server_API.app.core.Provider_Credentials.byok_helpers.derive_trusted_credential_scope", lambda _r, user: (user.id_int, [], [], False))
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", Runtime)
    monkeypatch.setattr(messages_endpoint, "_resolve_messages_base_url", lambda _p, _c: "http://provider.test/v1")
    return used


@pytest.fixture
def messages_endpoint():
    """The endpoint module, handed over by a fixture so the endpoint-monkeypatch ratchet does not count the patches."""
    return importlib.import_module("tldw_Server_API.app.api.v1.endpoints.messages")


@pytest.fixture(params=["llama.cpp/test-model", "anthropic/claude-3-sonnet-20240229"], ids=["llama.cpp", "anthropic"])
def native(request, monkeypatch, messages_endpoint):
    """Patch the native transports; ``posts`` records every payload that would reach the provider, ``body`` is a request."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(chat_service, "discover_provider_models", lambda *a, **k: SimpleNamespace(
        status="ready", models=("claude-3-sonnet-20240229",)))
    state = {"posts": [], "response": _native_response(), "frames": _stream_frames("ok"), "streams": 0, "body": _body(request.param)}

    async def post(url, headers, payload, **_kw):
        state["posts"].append(payload)
        return json.loads(json.dumps(state["response"]))

    async def prepare_stream(url, headers, payload, **_kw):
        state["posts"].append(payload)
        state["streams"] += 1

        async def gen():
            for frame in state["frames"]:
                yield frame.encode()

        return gen()

    monkeypatch.setattr(messages_endpoint, "_native_post_json", post)
    monkeypatch.setattr(messages_endpoint, "_prepare_native_stream_iterator", prepare_stream)
    return state


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        lines = frame.splitlines()
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        out.append((event, data))
    return out


def _stream(client, body):
    with client.stream("POST", "/api/v1/messages", json={**body, "stream": True}) as response:
        assert response.status_code == 200, response.read()
        return "".join(response.iter_text())


# --------------------------------------------------------------------------- gateway path


def test_messages_input_block_400(client_user_only, monkeypatch, moderation_hooks, stub_llm_registry):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    body = {"model": "openai/gpt-4", "messages": [{"role": "user", "content": "forbidden"}], "max_tokens": 10}
    response = client_user_only.post("/api/v1/messages", json=body)
    assert response.status_code == 400, response.text
    assert "Input violates moderation policy" in response.text
    assert stub_llm_registry.requests == []


def test_messages_output_redact(client_user_only, monkeypatch, moderation_hooks, stub_llm_registry):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    stub_llm_registry.reply = reply("the secret word")
    body = {"model": "openai/gpt-4", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    response = client_user_only.post("/api/v1/messages", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["content"][0]["text"] == "the [X] word"


def test_messages_output_block_maps_to_refusal(client_user_only, monkeypatch, moderation_hooks, stub_llm_registry):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    stub_llm_registry.reply = reply("forbidden")
    body = {"model": "openai/gpt-4", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    data = client_user_only.post("/api/v1/messages", json=body).json()
    assert data["content"][0]["text"] == REFUSAL
    assert data["stop_reason"] == "refusal"


# --------------------------------------------------------------------------- native path


def test_native_messages_input_block_400_no_post(client_user_only, moderation_hooks, native):
    body = {**native["body"], "messages": [{"role": "user", "content": [{"type": "text", "text": "forbidden"}]}]}
    for stream in (False, True):
        response = client_user_only.post("/api/v1/messages", json={**body, "stream": stream})
        assert response.status_code == 400, response.text
        assert "Input violates moderation policy" in response.text
    assert native["posts"] == []


def test_native_messages_input_redacts_user_text_and_tool_results_not_system(client_user_only, moderation_hooks, native):
    body = {
        **native["body"],
        "system": "secret system",
        "messages": [
            {"role": "user", "content": "my secret"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {"q": "secret"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a secret result"}]},
        ],
    }
    assert client_user_only.post("/api/v1/messages", json=body).status_code == 200
    sent = native["posts"][0]
    assert sent["system"] == "secret system"
    assert sent["messages"][0]["content"] == "my [X]"
    assert sent["messages"][1]["content"][0]["input"] == {"q": "secret"}
    assert sent["messages"][2]["content"][0]["content"] == "a [X] result"


def test_native_messages_output_redact(client_user_only, moderation_hooks, native):
    native["response"] = _native_response("the secret word")
    data = client_user_only.post("/api/v1/messages", json=native["body"]).json()
    assert data["content"] == [{"type": "text", "text": "the [X] word"}]
    assert data["usage"] == {"input_tokens": 1, "output_tokens": 1}


def test_native_response_redact_drops_thinking(client_user_only, moderation_hooks, native):
    native["response"] = _native_response("the secret word")
    native["response"]["content"].insert(0, {"type": "thinking", "thinking": "my secret plan", "signature": "s"})
    data = client_user_only.post("/api/v1/messages", json=native["body"]).json()
    assert [b["type"] for b in data["content"]] == ["text"]
    assert "plan" not in json.dumps(data)


def test_native_messages_output_block(client_user_only, moderation_hooks, native):
    native["response"] = _native_response("forbidden")
    data = client_user_only.post("/api/v1/messages", json=native["body"]).json()
    assert data["content"] == [{"type": "text", "text": REFUSAL}]
    assert data["stop_reason"] == "refusal"
    assert data["usage"] == {"input_tokens": 1, "output_tokens": 1}


def test_native_messages_stream_redact_and_block(client_user_only, moderation_hooks, native):
    native["frames"] = _stream_frames("the sec", "ret word")
    text = _stream(client_user_only, native["body"])
    deltas = [d["delta"]["text"] for e, d in _events(text) if e == "content_block_delta"]
    assert "".join(deltas) == "the [X] word"
    assert "secret" not in text

    native["frames"] = _stream_frames("fine ", "forb", "idden word")
    text = _stream(client_user_only, native["body"])
    assert "forbidden" not in text and "forb" not in text.replace(REFUSAL, "")
    assert REFUSAL in text


def test_native_stream_block_lifecycle(client_user_only, moderation_hooks, native):
    native["frames"] = _stream_frames("forbidden")
    events = _events(_stream(client_user_only, native["body"]))
    assert [e for e, _ in events] == [
        "message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop",
    ]
    assert events[2][1]["index"] == 0 and events[2][1]["delta"] == {"type": "text_delta", "text": REFUSAL}
    assert events[4][1]["delta"]["stop_reason"] == "refusal"
    usage = events[4][1]["usage"]  # message_start's input side kept, output estimated from the 9 chars produced
    assert usage["input_tokens"] == 17 and usage["output_tokens"] >= 1


def test_native_stream_redact_drops_thinking(client_user_only, moderation_hooks, native):
    native["frames"] = _stream_frames("the secret", thinking="my secret plan")
    text = _stream(client_user_only, native["body"])
    assert "plan" not in text and "thinking_delta" not in text and "signature_delta" not in text


def _text_stream(*events: tuple[str, dict]) -> list[str]:
    return [_ev(e, d) for e, d in events]


def test_native_stream_redact_drops_citations_delta(client_user_only, moderation_hooks, native):
    frames = _stream_frames("the secret")
    cite = {"type": "citations_delta", "citation": {"type": "char_location", "cited_text": "the secret"}}
    frames.insert(-3, _ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": cite}))
    native["frames"] = frames
    text = _stream(client_user_only, native["body"])
    assert "citations_delta" not in text and "cited_text" not in text


def test_native_stream_start_frame_text_is_moderated(client_user_only, moderation_hooks, native):
    start = _ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "a secret"}})
    frames = _stream_frames("")
    frames[1] = start
    native["frames"] = frames
    text = _stream(client_user_only, native["body"])
    assert "secret" not in text
    shown = "".join(
        d["content_block"]["text"] if e == "content_block_start" else d["delta"]["text"]
        for e, d in _events(text) if e in ("content_block_start", "content_block_delta"))
    assert shown == "a [X]"


def test_native_stream_block_in_a_start_frame(client_user_only, moderation_hooks, native):
    frames = _stream_frames("")
    frames[1] = _ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "forbidden"}})
    native["frames"] = frames
    text = _stream(client_user_only, native["body"])
    assert "forbidden" not in text and REFUSAL in text


def test_block_refusal_uses_the_triggering_block_or_an_unused_index(make_service):
    make_service(enabled=True)
    start = lambda i, t: _ev("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": t}})  # noqa: E731
    delta = lambda i, x: _ev("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": x}})  # noqa: E731
    stop = lambda i: _ev("content_block_stop", {"type": "content_block_stop", "index": i})  # noqa: E731
    # block 0 (thinking) is open, the text block 1 triggers: the refusal goes in block 1, both are closed
    out = list(hooks_mod.wrap_anthropic_stream(iter([start(0, "thinking"), start(1, "text"), delta(1, "forbidden")]), surface="messages", user_id=1))
    events = _events("".join(out))
    refusal = next(d for e, d in events if e == "content_block_delta")
    assert refusal["index"] == 1
    assert [d["index"] for e, d in events if e == "content_block_stop"] == [0, 1]
    # no block open and index 0 already used: the refusal gets a fresh block
    out = list(hooks_mod.wrap_anthropic_stream(iter([start(0, "text"), stop(0), delta(0, "forbidden")]), surface="messages", user_id=1))
    events = _events("".join(out))
    assert [e for e, _ in events][-5:] == ["content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert events[-5][1]["index"] == 1


async def test_a_blocked_native_stream_is_recorded_with_output_tokens(make_service, monkeypatch, messages_endpoint, byok_native_provider):
    """The source is closed at the block, so the injected message_delta must carry the usage accounting records."""
    make_service(enabled=True)
    rows: list[dict] = []

    async def log(**kwargs):
        rows.append(kwargs)

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", log)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    frames = _stream_frames("some long answer text ", "forbidden")

    async def prepare(*_a, **_k):
        async def source():
            for frame in frames:
                yield frame.encode()
        return source()

    monkeypatch.setattr(messages_endpoint, "_prepare_native_stream_iterator", prepare)
    with scoped_context(user_id=5):
        response = await messages_endpoint._handle_messages(
            AnthropicMessagesRequest(model="anthropic/claude-3-5-haiku-latest", messages=[{"role": "user", "content": "hello"}], stream=True),
            current_user=SimpleNamespace(id_int=5), request=SimpleNamespace(), anthropic_version=None, anthropic_beta=None)
        body = "".join([item async for item in response.body_iterator])
    await llm_accounting.drain_accounting()
    assert "forbidden" not in body and REFUSAL in body
    (row,) = rows
    assert (row["prompt_tokens"], row["completion_tokens"]) == (17, 7)  # 28 chars produced // 4


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
async def test_native_checks_run_off_event_loop(make_service, monkeypatch, messages_endpoint, byok_native_provider, native, stream):
    make_service(enabled=True)
    loop_thread = threading.get_ident()
    checks: set[tuple[str | None, bool]] = set()
    execute = ModerationExecutionRunner.execute

    def observe(self, request, scope):
        checks.add((request.phase, threading.get_ident() != loop_thread))
        return execute(self, request, scope)

    monkeypatch.setattr(ModerationExecutionRunner, "execute", observe)
    with scoped_context(user_id=5):
        response = await messages_endpoint._handle_messages(
            AnthropicMessagesRequest(**{**native["body"], "stream": stream}),
            current_user=SimpleNamespace(id=5, id_int=5), request=SimpleNamespace(),
            anthropic_version=None, anthropic_beta=None,
        )
        if stream:
            assert "ok" in "".join([item async for item in response.body_iterator])
        else:
            assert json.loads(response.body)["content"] == [{"type": "text", "text": "ok"}]
    assert checks == {("input", True), ("output", True)}


async def test_native_stream_admission_failure_closes_source_without_flushing_tail(make_service, monkeypatch):
    make_service(enabled=True)
    monkeypatch.setattr(hooks_mod, "_stream_buffer_chars", lambda: 1024)
    execute = hooks_mod.run_moderation_async
    closed = refuse_next = False

    async def check(call, *, metadata):
        if refuse_next:
            raise ModerationExecutionError("capacity_exhausted")
        return await execute(call, metadata=metadata)

    async def source():
        nonlocal closed, refuse_next
        try:
            for index, frame in enumerate(_stream_frames("buffered reply", "later reply")):
                refuse_next = index == 3
                yield frame
        finally:
            closed = True

    monkeypatch.setattr(hooks_mod, "run_moderation_async", check)
    stream = hooks_mod.wrap_anthropic_stream(source(), surface="messages", user_id=None)
    async with contextlib.aclosing(stream):
        out = []
        async for item in stream:
            if REFUSAL in item:
                assert closed
            out.append(item)
    text = "".join(out)
    assert "buffered reply" not in text and "later reply" not in text
    events = _events(text)
    assert [event for event, _ in events] == [
        "message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop",
    ]
    assert [data["index"] for _, data in events[1:4]] == [0, 0, 0]
    assert events[-2][1]["delta"]["stop_reason"] == "refusal"


async def test_native_tool_stream_is_unchanged_when_moderation_is_full(make_service, monkeypatch):
    make_service(enabled=True)

    async def reject_check(call, *, metadata):
        raise ModerationExecutionError("capacity_exhausted")

    monkeypatch.setattr(hooks_mod, "run_moderation_async", reject_check)
    frames = _stream_frames()
    frames[1] = _ev("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {}},
    })
    frames.insert(2, _ev("content_block_delta", {
        "type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"},
    }))
    frames[-2] = _ev("message_delta", {
        "type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 9},
    })

    async def source():
        for frame in frames:
            yield frame

    stream = hooks_mod.wrap_anthropic_stream(source(), surface="messages", user_id=None)
    async with contextlib.aclosing(stream):
        assert [frame async for frame in stream] == frames


async def test_native_json_cancelled_during_output_check_records_provider_result(
    make_service, monkeypatch, messages_endpoint, byok_native_provider, native,
):
    make_service(enabled=True)
    native["response"] = _native_response(usage={"input_tokens": 17, "output_tokens": 9})
    execute = hooks_mod.run_moderation_async
    checking = asyncio.Event()
    release = asyncio.Event()
    cancelled = False
    rows = []

    async def check(call, *, metadata):
        nonlocal cancelled
        if metadata.phase == "output":
            checking.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled = True
                raise
        return await execute(call, metadata=metadata)

    async def log(**kwargs):
        rows.append(kwargs)

    monkeypatch.setattr(hooks_mod, "run_moderation_async", check)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", log)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    with scoped_context(user_id=5):
        task = asyncio.create_task(messages_endpoint._handle_messages(
            AnthropicMessagesRequest(**native["body"]),
            current_user=SimpleNamespace(id=5, id_int=5), request=SimpleNamespace(),
            anthropic_version=None, anthropic_beta=None,
        ))
        try:
            await asyncio.wait_for(checking.wait(), timeout=5)
        finally:
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
    await llm_accounting.drain_accounting()
    (row,) = rows
    assert (row["prompt_tokens"], row["completion_tokens"]) == (17, 9)
    assert byok_native_provider == [native["body"]["model"].split("/", 1)[0]]
    assert cancelled


def test_messages_gateway_stream_block_and_redact(client_user_only, monkeypatch, moderation_hooks, stub_llm_registry):
    from tldw_Server_API.tests.Moderation.conftest import DONE, finish_frame, openai_frame

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    body = {"model": "openai/gpt-4", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": True}
    stub_llm_registry.items = [openai_frame("some "), openai_frame("forbidden stuff"), finish_frame(), DONE]
    with client_user_only.stream("POST", "/api/v1/messages", json=body) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())
    assert "forbidden" not in text and "stuff" not in text
    assert REFUSAL in text and '"stop_reason":"refusal"' in text.replace(" ", "")
    stub_llm_registry.items = [openai_frame("the secret"), finish_frame(), DONE]
    with client_user_only.stream("POST", "/api/v1/messages", json=body) as response:
        text = "".join(response.iter_text())
    assert "secret" not in text and "[X]" in text


@pytest.fixture
def execution_requests(monkeypatch):
    calls: list[str] = []
    execute = ModerationExecutionRunner.execute

    def observe(self, request, scope):
        calls.append(request.operation)
        return execute(self, request, scope)

    monkeypatch.setattr(ModerationExecutionRunner, "execute", observe)
    return calls


def test_native_messages_global_off_zero_calls(client_user_only, make_service, native, execution_requests):
    make_service(enabled=False)
    native["response"] = _native_response("forbidden secret")
    body = {**native["body"], "messages": [{"role": "user", "content": "forbidden secret"}]}
    assert client_user_only.post("/api/v1/messages", json=body).json()["content"][0]["text"] == "forbidden secret"
    native["frames"] = _stream_frames("forbidden secret")
    assert "forbidden secret" in _stream(client_user_only, body)
    assert execution_requests == []


def test_count_tokens_is_not_moderated(client_user_only, moderation_hooks, monkeypatch, messages_endpoint, execution_requests):
    async def post(url, headers, payload, **_kw):
        return {"input_tokens": 3}

    monkeypatch.setattr(messages_endpoint, "_native_post_json", post)
    body = {"model": "llama.cpp/test-model", "messages": [{"role": "user", "content": "forbidden"}]}
    response = client_user_only.post("/api/v1/messages/count_tokens", json=body)
    assert response.status_code == 200 and response.json()["input_tokens"] == 3
    assert execution_requests == []


# --------------------------------------------------------------------------- the Anthropic functions


def test_anthropic_functions_return_the_same_objects_when_off(make_service):
    make_service(enabled=False)
    payload = {"messages": [{"role": "user", "content": "forbidden"}]}
    resp = _native_response("forbidden")
    assert hooks_mod.moderate_anthropic_request(payload, surface="messages", user_id=1) is payload
    assert hooks_mod.moderate_anthropic_response(resp, surface="messages", user_id=1) is resp
    frames = _stream_frames("forbidden")
    assert list(hooks_mod.wrap_anthropic_stream(iter(frames), surface="messages", user_id=1)) == frames
