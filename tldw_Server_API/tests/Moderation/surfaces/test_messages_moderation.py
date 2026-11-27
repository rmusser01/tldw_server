"""/api/v1/messages is moderated: the gateway path through the hooks, the native providers explicitly
(TASK-13950.27 PR D). Native tests use llama.cpp, which takes the same branch as anthropic."""

from __future__ import annotations

import importlib
import json

import pytest

from tldw_Server_API.app.core.Moderation import gateway_hooks as hooks_mod
from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator
from tldw_Server_API.tests.Moderation.conftest import reply

pytestmark = pytest.mark.integration

REFUSAL = "Output violates moderation policy"
NATIVE_BODY = {"model": "llama.cpp/test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}


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
def messages_endpoint():
    """The endpoint module, handed over by a fixture so the endpoint-monkeypatch ratchet does not count the patches."""
    return importlib.import_module("tldw_Server_API.app.api.v1.endpoints.messages")


@pytest.fixture
def native(monkeypatch, messages_endpoint):
    """Patch the native transports; ``posts`` records every payload that would reach the provider."""
    state = {"posts": [], "response": _native_response(), "frames": _stream_frames("ok"), "streams": 0}

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
    body = {**NATIVE_BODY, "messages": [{"role": "user", "content": [{"type": "text", "text": "forbidden"}]}]}
    for stream in (False, True):
        response = client_user_only.post("/api/v1/messages", json={**body, "stream": stream})
        assert response.status_code == 400, response.text
        assert "Input violates moderation policy" in response.text
    assert native["posts"] == []


def test_native_messages_input_redacts_user_text_and_tool_results_not_system(client_user_only, moderation_hooks, native):
    body = {
        **NATIVE_BODY,
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
    data = client_user_only.post("/api/v1/messages", json=NATIVE_BODY).json()
    assert data["content"] == [{"type": "text", "text": "the [X] word"}]
    assert data["usage"] == {"input_tokens": 1, "output_tokens": 1}


def test_native_response_redact_drops_thinking(client_user_only, moderation_hooks, native):
    native["response"] = _native_response("the secret word")
    native["response"]["content"].insert(0, {"type": "thinking", "thinking": "my secret plan", "signature": "s"})
    data = client_user_only.post("/api/v1/messages", json=NATIVE_BODY).json()
    assert [b["type"] for b in data["content"]] == ["text"]
    assert "plan" not in json.dumps(data)


def test_native_messages_output_block(client_user_only, moderation_hooks, native):
    native["response"] = _native_response("forbidden")
    data = client_user_only.post("/api/v1/messages", json=NATIVE_BODY).json()
    assert data["content"] == [{"type": "text", "text": REFUSAL}]
    assert data["stop_reason"] == "refusal"
    assert data["usage"] == {"input_tokens": 1, "output_tokens": 1}


def test_native_messages_stream_redact_and_block(client_user_only, moderation_hooks, native):
    native["frames"] = _stream_frames("the sec", "ret word")
    text = _stream(client_user_only, NATIVE_BODY)
    deltas = [d["delta"]["text"] for e, d in _events(text) if e == "content_block_delta"]
    assert "".join(deltas) == "the [X] word"
    assert "secret" not in text

    native["frames"] = _stream_frames("fine ", "forb", "idden word")
    text = _stream(client_user_only, NATIVE_BODY)
    assert "forbidden" not in text and "forb" not in text.replace(REFUSAL, "")
    assert REFUSAL in text


def test_native_stream_block_lifecycle(client_user_only, moderation_hooks, native):
    native["frames"] = _stream_frames("forbidden")
    events = _events(_stream(client_user_only, NATIVE_BODY))
    assert [e for e, _ in events] == [
        "message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop",
    ]
    assert events[2][1]["delta"] == {"type": "text_delta", "text": REFUSAL}
    assert events[4][1]["delta"]["stop_reason"] == "refusal"


def test_native_stream_redact_drops_thinking(client_user_only, moderation_hooks, native):
    native["frames"] = _stream_frames("the secret", thinking="my secret plan")
    text = _stream(client_user_only, NATIVE_BODY)
    assert "plan" not in text and "thinking_delta" not in text and "signature_delta" not in text


def test_native_messages_global_off_zero_calls(client_user_only, make_service, native, monkeypatch):
    make_service(enabled=False)
    calls: list[str] = []
    for name in ("evaluate_text", "redact_text", "redact_text_with_count"):
        real = getattr(PolicyEvaluator, name)
        monkeypatch.setattr(PolicyEvaluator, name, lambda self, *a, _r=real, _n=name, **k: (calls.append(_n), _r(self, *a, **k))[1])
    native["response"] = _native_response("forbidden secret")
    body = {**NATIVE_BODY, "messages": [{"role": "user", "content": "forbidden secret"}]}
    assert client_user_only.post("/api/v1/messages", json=body).json()["content"][0]["text"] == "forbidden secret"
    native["frames"] = _stream_frames("forbidden secret")
    assert "forbidden secret" in _stream(client_user_only, body)
    assert calls == []


def test_count_tokens_is_not_moderated(client_user_only, moderation_hooks, monkeypatch, messages_endpoint):
    calls: list[str] = []
    real = PolicyEvaluator.evaluate_text
    monkeypatch.setattr(PolicyEvaluator, "evaluate_text", lambda self, *a, **k: (calls.append("e"), real(self, *a, **k))[1])

    async def post(url, headers, payload, **_kw):
        return {"input_tokens": 3}

    monkeypatch.setattr(messages_endpoint, "_native_post_json", post)
    body = {"model": "llama.cpp/test-model", "messages": [{"role": "user", "content": "forbidden"}]}
    response = client_user_only.post("/api/v1/messages/count_tokens", json=body)
    assert response.status_code == 200 and response.json()["input_tokens"] == 3
    assert calls == []


# --------------------------------------------------------------------------- the Anthropic functions


def test_anthropic_functions_return_the_same_objects_when_off(make_service):
    make_service(enabled=False)
    payload = {"messages": [{"role": "user", "content": "forbidden"}]}
    resp = _native_response("forbidden")
    assert hooks_mod.moderate_anthropic_request(payload, surface="messages", user_id=1) is payload
    assert hooks_mod.moderate_anthropic_response(resp, surface="messages", user_id=1) is resp
    frames = _stream_frames("forbidden")
    assert list(hooks_mod.wrap_anthropic_stream(iter(frames), surface="messages", user_id=1)) == frames
