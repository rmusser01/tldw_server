from __future__ import annotations

import threading
import types

import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatRateLimitError, ChatBadRequestError
from tldw_Server_API.app.core.LLM_Calls.providers import mlx_provider as mp


def _fake_mlx_module():
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, chat_template=None):
            parts = []
            for m in messages:
                parts.append(f"{m.get('role')}:{m.get('content')}")
            return " | ".join(parts) + (" <gen>" if add_generation_prompt else "")

    def load(model_path, **kwargs):
        return ("model", FakeTokenizer())

    def generate(model, tokenizer, prompt, stream=False, verbose=False, **kwargs):
        return f"out:{prompt}"

    def generate_stream(model, tokenizer, prompt, verbose=False, **kwargs):
        yield "hi"
        yield "there"

    def embed(model, tokenizer, text):
        return [0.1, 0.2, 0.3]

    mod = types.SimpleNamespace()
    mod.load = load
    mod.generate = generate
    mod.generate_stream = generate_stream
    mod.embed = embed
    return mod


def _patch_mlx(monkeypatch):
    fake = _fake_mlx_module()
    monkeypatch.setattr(mp.MLXSessionRegistry, "_import_mlx", lambda self: fake)
    mp._registry = None  # reset global registry
    return fake


def test_load_and_unload(monkeypatch):
    _patch_mlx(monkeypatch)
    reg = mp.get_mlx_registry()
    status = reg.load(model_path="fake-model", overrides={"max_concurrent": 1})
    assert status["active"] is True
    assert status["model"] == "fake-model"
    assert status["max_concurrent"] == 1
    reg.unload()
    assert reg.status()["active"] is False


def test_load_blank_model_path_raises(monkeypatch):
    _patch_mlx(monkeypatch)
    reg = mp.get_mlx_registry()
    with pytest.raises(ChatBadRequestError):
        reg.load(model_path="   ", overrides={"max_concurrent": 1})


def test_overflow_raises_rate_limit(monkeypatch):
    _patch_mlx(monkeypatch)
    reg = mp.get_mlx_registry()
    reg.load(model_path="fake-model", overrides={"max_concurrent": 1})
    with reg.session_scope():
        with pytest.raises(ChatRateLimitError):
            with reg.session_scope():
                pass


def test_chat_and_embeddings(monkeypatch):
    _patch_mlx(monkeypatch)
    reg = mp.get_mlx_registry()
    reg.load(model_path="fake-model", overrides={"max_concurrent": 1})
    chat_adapter = mp.MLXChatAdapter()
    emb_adapter = mp.MLXEmbeddingsAdapter()

    chat_resp = chat_adapter.chat({"messages": [{"role": "user", "content": "hi"}]})
    assert chat_resp["model"] == "fake-model"
    assert chat_resp["choices"][0]["message"]["role"] == "assistant"
    assert chat_resp["choices"][0]["message"]["content"].startswith("out:")
    stream_chunks = list(chat_adapter.stream({"messages": [{"role": "user", "content": "hi"}], "stream": True}))
    assert len(stream_chunks) >= 2
    assert stream_chunks[0].startswith("data: ")
    assert stream_chunks[-1].strip() == "data: [DONE]"

    emb_resp = emb_adapter.embed({"input": "hello", "model": "fake-model"})
    assert emb_resp["data"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_load_reports_unapplied_runtime_overrides(monkeypatch):
    _patch_mlx(monkeypatch)
    reg = mp.get_mlx_registry()
    status = reg.load(
        model_path="fake-model",
        overrides={
            "max_concurrent": 1,
            "quantization": "4bit",
            "max_kv_cache_size": 4096,
        },
    )

    unapplied = status.get("config", {}).get("unapplied_runtime_overrides", {})
    assert unapplied.get("quantization") == "4bit"
    assert unapplied.get("max_kv_cache_size") == 4096


def test_failed_older_load_does_not_restore_over_newer_success(monkeypatch):
    started = threading.Event()
    release_failure = threading.Event()

    class FakeTokenizer:
        pass

    def load(model_path, **kwargs):
        if model_path == "slow-fail":
            started.set()
            assert release_failure.wait(timeout=2)
            raise RuntimeError("load failed")
        return (f"model:{model_path}", FakeTokenizer())

    fake = types.SimpleNamespace(
        load=load,
        generate=lambda *args, **kwargs: "ok",
        generate_stream=lambda *args, **kwargs: iter(["ok"]),
        embed=lambda *args, **kwargs: [1.0],
    )
    monkeypatch.setattr(mp.MLXSessionRegistry, "_import_mlx", lambda self: fake)

    reg = mp.MLXSessionRegistry()
    load_overrides = {"max_concurrent": 1, "warmup": False, "compile": False}
    reg.load(model_path="old", overrides=load_overrides)

    errors = []

    def load_slow_failure():
        try:
            reg.load(model_path="slow-fail", overrides=load_overrides)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=load_slow_failure)
    worker.start()
    assert started.wait(timeout=2)

    reg.load(model_path="new", overrides=load_overrides)
    release_failure.set()
    worker.join(timeout=2)

    assert errors
    assert reg.status()["model"] == "new"


def test_embeddings_response_uses_active_session_model(monkeypatch):
    _patch_mlx(monkeypatch)
    reg = mp.get_mlx_registry()
    reg.load(model_path="fake-model", overrides={"max_concurrent": 1})
    emb_adapter = mp.MLXEmbeddingsAdapter()

    resp = emb_adapter.embed({"input": "hello", "model": "wrong-model"})
    assert resp["model"] == "fake-model"


def test_session_scope_without_load_raises():
    reg = mp.MLXSessionRegistry()
    with pytest.raises(ChatBadRequestError):
        with reg.session_scope():
            pass


def test_embeddings_missing_input_raises(monkeypatch):
    _patch_mlx(monkeypatch)
    reg = mp.get_mlx_registry()
    reg.load(model_path="fake-model", overrides={"max_concurrent": 1})
    emb_adapter = mp.MLXEmbeddingsAdapter()
    with pytest.raises(ChatBadRequestError):
        emb_adapter.embed({"model": "fake-model"})


@pytest.mark.asyncio
async def test_async_chat_handler(monkeypatch):
    _patch_mlx(monkeypatch)
    reg = mp.get_mlx_registry()
    reg.load(model_path="fake-model", overrides={"max_concurrent": 1})
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

    stream = await perform_chat_api_call_async(
        api_provider="mlx",
        messages=[{"role": "user", "content": "hi"}],
        streaming=True,
        model="fake-model",
    )
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    assert len(chunks) >= 2
    assert chunks[-1].strip() == "data: [DONE]"
