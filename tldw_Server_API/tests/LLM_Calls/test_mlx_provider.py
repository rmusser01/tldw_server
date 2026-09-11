from __future__ import annotations

import asyncio
import gc
import threading
import types
import weakref

import pytest
from loguru import logger

from tldw_Server_API.app.core.Chat.bounded_daemon import (
    BoundedDaemonPool,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatBadRequestError,
    ChatProviderError,
    ChatRateLimitError,
)
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


@pytest.mark.asyncio
async def test_chat_timeout_keeps_generation_capacity_and_session_until_worker_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class _CredentialMarker:
        pass

    class _ObservedPool(BoundedDaemonPool):
        def __init__(self) -> None:
            super().__init__(1)
            self.released: threading.Event | None = None

        def start(self, target, *, name, released_event=None):
            assert released_event is not None
            thread = super().start(
                target,
                name=name,
                released_event=released_event,
            )
            self.released = released_event
            return thread

    def blocking_generate(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=2)
        return "ok"

    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="fake-model",
        model=object(),
        tokenizer=object(),
        generate_fn=blocking_generate,
        generate_stream_fn=None,
        embed_fn=None,
        supports_embeddings=False,
        config={},
    )
    pool = _ObservedPool()
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(mp, "get_mlx_registry", lambda: registry)
    monkeypatch.setattr(registry, "_worker_pool", pool, raising=False)
    adapter = mp.MLXChatAdapter()
    api_key_marker = _CredentialMarker()
    app_config_marker = _CredentialMarker()
    api_key_ref = weakref.ref(api_key_marker)
    app_config_ref = weakref.ref(app_config_marker)
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "api_key": api_key_marker,
        "app_config": app_config_marker,
    }

    def capture_timeout(request_arg):
        try:
            adapter.chat(request_arg, timeout=0.05)
        except TimeoutError as exc:
            return str(exc), exc.__cause__, exc.__context__
        raise AssertionError("MLX generation unexpectedly completed before its deadline")

    try:
        timeout_message, timeout_cause, timeout_context = await asyncio.wait_for(
            asyncio.to_thread(capture_timeout, request),
            timeout=1,
        )

        assert timeout_message == "MLX generation timed out"
        assert timeout_cause is None
        assert timeout_context is None
        assert started.is_set()
        assert pool.active_count == 1
        assert registry._inflight == 1

        request.pop("api_key")
        request.pop("app_config")
        del api_key_marker, app_config_marker
        gc.collect()
        assert api_key_ref() is None
        assert app_config_ref() is None

        with pytest.raises(ChatRateLimitError) as exc_info:
            await asyncio.wait_for(
                asyncio.to_thread(adapter.chat, request, timeout=0.05),
                timeout=1,
            )
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
    finally:
        release.set()

    assert pool.released is not None
    assert await asyncio.to_thread(pool.released.wait, 2)
    assert pool.active_count == 0
    assert registry._inflight == 0
    assert adapter.chat(request, timeout=1)["choices"][0]["message"]["content"] == "ok"
    assert pool.active_count == 0


@pytest.mark.parametrize("timeout", [False, 0, -1, float("nan"), float("inf")])
def test_chat_rejects_non_positive_or_non_finite_timeout_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    timeout: float,
) -> None:
    generated = threading.Event()
    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="fake-model",
        model=object(),
        tokenizer=object(),
        generate_fn=lambda *_args, **_kwargs: generated.set() or "ok",
        generate_stream_fn=None,
        embed_fn=None,
        supports_embeddings=False,
        config={},
    )
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(registry, "_worker_pool", BoundedDaemonPool(1), raising=False)

    with pytest.raises(ValueError, match="positive finite"):
        mp.MLXChatAdapter().chat(
            {"messages": [{"role": "user", "content": "hello"}]},
            timeout=timeout,
        )

    assert generated.is_set() is False


@pytest.mark.asyncio
async def test_chat_worker_capacity_tracks_registry_max_concurrent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Barrier(3)
    release = threading.Event()
    results: list[dict[str, object]] = []

    def cooperative_generate(*_args, **_kwargs):
        entered.wait(timeout=2)
        assert release.wait(timeout=2)
        return "ok"

    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="fake-model",
        model=object(),
        tokenizer=object(),
        generate_fn=cooperative_generate,
        generate_stream_fn=None,
        embed_fn=None,
        supports_embeddings=False,
        config={},
    )
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(mp, "set_gauge", lambda *_args, **_kwargs: None)
    registry._set_concurrency(2)
    monkeypatch.setattr(mp, "get_mlx_registry", lambda: registry)
    adapter = mp.MLXChatAdapter()
    request = {"messages": [{"role": "user", "content": "hello"}]}

    def invoke() -> None:
        results.append(adapter.chat(request, timeout=1))

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    second.start()
    entered.wait(timeout=2)

    try:
        with pytest.raises(ChatRateLimitError) as exc_info:
            adapter.chat(request, timeout=0.1)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
    finally:
        release.set()
        first.join(timeout=2)
        second.join(timeout=2)

    assert len(results) == 2
    assert registry._inflight == 0


@pytest.mark.asyncio
async def test_stream_timeout_retains_capacity_until_noncooperative_worker_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_stream(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=2)
        yield "late"

    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="fake-model",
        model=object(),
        tokenizer=object(),
        generate_fn=lambda *_args, **_kwargs: "fallback",
        generate_stream_fn=blocking_stream,
        embed_fn=None,
        supports_embeddings=False,
        config={},
    )
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(mp, "get_mlx_registry", lambda: registry)
    adapter = mp.MLXChatAdapter()

    def consume() -> tuple[str, BaseException | None, BaseException | None]:
        try:
            list(
                adapter.stream(
                    {"messages": [{"role": "user", "content": "hello"}]},
                    timeout=0.05,
                )
            )
        except TimeoutError as exc:
            return str(exc), exc.__cause__, exc.__context__
        raise AssertionError("MLX stream unexpectedly completed before its deadline")

    try:
        message, cause, context = await asyncio.wait_for(
            asyncio.to_thread(consume),
            timeout=1,
        )
        assert message == "MLX streaming timed out"
        assert cause is None
        assert context is None
        assert started.is_set()
        assert registry._worker_pool.active_count == 1
        assert registry._inflight == 1
    finally:
        release.set()

    for _ in range(200):
        if registry._worker_pool.active_count == 0 and registry._inflight == 0:
            break
        await asyncio.sleep(0.01)
    assert registry._worker_pool.active_count == 0
    assert registry._inflight == 0


def test_stream_failure_after_partial_output_never_replays_nonstream_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "mlx-partial-stream-private-sentinel"
    fallback_called = threading.Event()

    def fail_after_partial(*_args, **_kwargs):
        yield "partial"
        raise RuntimeError(sentinel)

    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="fake-model",
        model=object(),
        tokenizer=object(),
        generate_fn=lambda *_args, **_kwargs: fallback_called.set() or "full-replay",
        generate_stream_fn=fail_after_partial,
        embed_fn=None,
        supports_embeddings=False,
        config={},
    )
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(mp, "get_mlx_registry", lambda: registry)
    chunks: list[str] = []

    with pytest.raises(ChatProviderError) as exc_info:
        for chunk in mp.MLXChatAdapter().stream(
            {"messages": [{"role": "user", "content": "hello"}]},
            timeout=1,
        ):
            chunks.append(chunk)

    assert len(chunks) == 1
    assert "partial" in chunks[0]
    assert "full-replay" not in "".join(chunks)
    assert fallback_called.is_set() is False
    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_stream_close_drops_credentials_and_suppresses_late_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocking_next_started = threading.Event()
    release = threading.Event()

    class _CredentialMarker:
        pass

    def blocking_after_first(*_args, **_kwargs):
        yield "first"
        blocking_next_started.set()
        assert release.wait(timeout=2)
        yield "late"

    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="fake-model",
        model=object(),
        tokenizer=object(),
        generate_fn=lambda *_args, **_kwargs: "fallback",
        generate_stream_fn=blocking_after_first,
        embed_fn=None,
        supports_embeddings=False,
        config={},
    )
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(mp, "get_mlx_registry", lambda: registry)
    api_key_marker = _CredentialMarker()
    app_config_marker = _CredentialMarker()
    api_key_ref = weakref.ref(api_key_marker)
    app_config_ref = weakref.ref(app_config_marker)
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "api_key": api_key_marker,
        "app_config": app_config_marker,
    }
    stream = mp.MLXChatAdapter().stream(request, timeout=1)

    assert next(stream).startswith("data: ")
    assert await asyncio.to_thread(blocking_next_started.wait, 1)
    stream.close()
    request.pop("api_key")
    request.pop("app_config")
    del api_key_marker, app_config_marker
    gc.collect()

    assert api_key_ref() is None
    assert app_config_ref() is None
    assert registry._worker_pool.active_count == 1
    assert registry._inflight == 1
    with pytest.raises(StopIteration):
        next(stream)

    release.set()
    for _ in range(200):
        if registry._worker_pool.active_count == 0 and registry._inflight == 0:
            break
        await asyncio.sleep(0.01)
    assert registry._worker_pool.active_count == 0
    assert registry._inflight == 0


@pytest.mark.asyncio
async def test_embeddings_timeout_drops_credentials_and_retains_capacity_until_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class _CredentialMarker:
        pass

    def blocking_embed(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=2)
        return [1.0]

    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="fake-model",
        model=object(),
        tokenizer=object(),
        generate_fn=None,
        generate_stream_fn=None,
        embed_fn=blocking_embed,
        supports_embeddings=True,
        config={},
    )
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(mp, "get_mlx_registry", lambda: registry)
    adapter = mp.MLXEmbeddingsAdapter()
    api_key_marker = _CredentialMarker()
    app_config_marker = _CredentialMarker()
    api_key_ref = weakref.ref(api_key_marker)
    app_config_ref = weakref.ref(app_config_marker)
    request = {
        "input": "hello",
        "api_key": api_key_marker,
        "app_config": app_config_marker,
    }

    def capture_timeout(request_arg):
        try:
            adapter.embed(request_arg, timeout=0.05)
        except TimeoutError as exc:
            return str(exc), exc.__cause__, exc.__context__
        raise AssertionError("MLX embedding unexpectedly completed before its deadline")

    try:
        message, cause, context = await asyncio.wait_for(
            asyncio.to_thread(capture_timeout, request),
            timeout=1,
        )
        assert message == "MLX embeddings timed out"
        assert cause is None
        assert context is None
        assert started.is_set()
        assert registry._worker_pool.active_count == 1
        assert registry._inflight == 1

        request.pop("api_key")
        request.pop("app_config")
        del api_key_marker, app_config_marker
        gc.collect()
        assert api_key_ref() is None
        assert app_config_ref() is None

        with pytest.raises(ChatRateLimitError):
            adapter.embed(request, timeout=0.05)
    finally:
        release.set()

    for _ in range(200):
        if registry._worker_pool.active_count == 0 and registry._inflight == 0:
            break
        await asyncio.sleep(0.01)
    assert registry._worker_pool.active_count == 0
    assert registry._inflight == 0


@pytest.mark.parametrize("surface", ["stream", "embed"])
@pytest.mark.parametrize("timeout", [False, 0, -1, float("nan"), float("inf")])
def test_stream_and_embeddings_reject_invalid_deadlines_before_work(
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    timeout: float,
) -> None:
    invoked = threading.Event()
    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="fake-model",
        model=object(),
        tokenizer=object(),
        generate_fn=lambda *_args, **_kwargs: invoked.set() or "fallback",
        generate_stream_fn=lambda *_args, **_kwargs: invoked.set() or iter(["item"]),
        embed_fn=lambda *_args, **_kwargs: invoked.set() or [1.0],
        supports_embeddings=True,
        config={},
    )
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(mp, "get_mlx_registry", lambda: registry)

    with pytest.raises(ValueError, match="positive finite"):
        if surface == "stream":
            next(
                mp.MLXChatAdapter().stream(
                    {"messages": [{"role": "user", "content": "hello"}]},
                    timeout=timeout,
                )
            )
        else:
            mp.MLXEmbeddingsAdapter().embed({"input": "hello"}, timeout=timeout)

    assert invoked.is_set() is False


@pytest.mark.parametrize("surface", ["stream", "embed"])
def test_stream_and_embeddings_apply_default_deadlines(
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=2)
        return [1.0]

    def blocking_stream(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=2)
        yield "late"

    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="fake-model",
        model=object(),
        tokenizer=object(),
        generate_fn=blocking,
        generate_stream_fn=blocking_stream,
        embed_fn=blocking,
        supports_embeddings=True,
        config={},
    )
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(mp, "get_mlx_registry", lambda: registry)
    monkeypatch.setattr(mp, "_MLX_STREAM_TIMEOUT_SECONDS", 0.05, raising=False)
    monkeypatch.setattr(mp, "_MLX_EMBEDDINGS_TIMEOUT_SECONDS", 0.05, raising=False)

    try:
        with pytest.raises(TimeoutError):
            if surface == "stream":
                list(
                    mp.MLXChatAdapter().stream(
                        {"messages": [{"role": "user", "content": "hello"}]}
                    )
                )
            else:
                mp.MLXEmbeddingsAdapter().embed({"input": "hello"})
        assert started.is_set()
    finally:
        release.set()

    for _ in range(200):
        if registry._worker_pool.active_count == 0:
            break
        threading.Event().wait(0.01)
    assert registry._worker_pool.active_count == 0


@pytest.mark.parametrize("surface", ["chat", "stream", "embed"])
def test_mlx_worker_errors_are_sanitized_and_detached(
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    sentinel = "mlx-worker-secret-sentinel"

    def fail(*_args, **_kwargs):
        raise RuntimeError(sentinel)

    def fail_stream(*_args, **_kwargs):
        raise RuntimeError(sentinel)
        yield  # pragma: no cover - keeps this a generator function

    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="fake-model",
        model=object(),
        tokenizer=object(),
        generate_fn=fail,
        generate_stream_fn=fail_stream,
        embed_fn=fail,
        supports_embeddings=True,
        config={},
    )
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(mp, "get_mlx_registry", lambda: registry)

    with pytest.raises(ChatProviderError) as exc_info:
        if surface == "chat":
            mp.MLXChatAdapter().chat(
                {"messages": [{"role": "user", "content": "hello"}]},
                timeout=1,
            )
        elif surface == "stream":
            list(
                mp.MLXChatAdapter().stream(
                    {"messages": [{"role": "user", "content": "hello"}]},
                    timeout=1,
                )
            )
        else:
            mp.MLXEmbeddingsAdapter().embed({"input": "hello"}, timeout=1)

    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_mlx_prompt_template_failure_does_not_log_raw_worker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "mlx-template-secret-sentinel"
    captured: list[str] = []

    class _FailingTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            raise RuntimeError(sentinel)

    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="fake-model",
        model=object(),
        tokenizer=_FailingTokenizer(),
        generate_fn=lambda *_args, **_kwargs: "ok",
        generate_stream_fn=None,
        embed_fn=None,
        supports_embeddings=False,
        config={},
    )
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(mp, "get_mlx_registry", lambda: registry)
    sink_id = logger.add(lambda message: captured.append(str(message)), level="DEBUG")
    try:
        response = mp.MLXChatAdapter().chat(
            {"messages": [{"role": "user", "content": "hello"}]},
            timeout=1,
        )
    finally:
        logger.remove(sink_id)

    assert response["choices"][0]["message"]["content"] == "ok"
    assert sentinel not in "".join(captured)


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


def test_same_limit_reload_preserves_capacity_owned_by_late_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_generate(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=2)
        return "old"

    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="old-model",
        model=object(),
        tokenizer=object(),
        generate_fn=blocking_generate,
        generate_stream_fn=None,
        embed_fn=None,
        supports_embeddings=False,
        config={},
    )
    fake = _fake_mlx_module()
    monkeypatch.setattr(registry, "_import_mlx", lambda: fake)
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(mp, "get_mlx_registry", lambda: registry)
    monkeypatch.setattr(mp, "observe_histogram", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mp, "increment_counter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mp, "set_gauge", lambda *_args, **_kwargs: None)
    adapter = mp.MLXChatAdapter()
    request = {"messages": [{"role": "user", "content": "hello"}]}

    original_pool = registry._worker_pool
    original_semaphore = registry._sema
    try:
        with pytest.raises(TimeoutError, match="MLX generation timed out"):
            adapter.chat(request, timeout=0.05)
        assert started.is_set()

        registry.load(
            model_path="new-model",
            overrides={"max_concurrent": 1, "warmup": False, "compile": False},
        )

        assert registry._worker_pool is original_pool
        assert registry._sema is original_semaphore
        with pytest.raises(ChatRateLimitError) as exc_info:
            adapter.chat(request, timeout=0.05)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
    finally:
        release.set()

    for _ in range(200):
        if original_pool.active_count == 0 and registry._inflight == 0:
            break
        threading.Event().wait(0.01)
    assert original_pool.active_count == 0
    assert registry._inflight == 0
    assert registry.status()["model"] == "new-model"
    assert "out:" in adapter.chat(request, timeout=1)["choices"][0]["message"]["content"]


def test_limit_change_during_active_worker_fails_closed_then_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_generate(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=2)
        return "old"

    registry = mp.MLXSessionRegistry()
    registry._session = mp.MLXSession(
        model_id="old-model",
        model=object(),
        tokenizer=object(),
        generate_fn=blocking_generate,
        generate_stream_fn=None,
        embed_fn=None,
        supports_embeddings=False,
        config={},
    )
    fake = _fake_mlx_module()
    monkeypatch.setattr(registry, "_import_mlx", lambda: fake)
    monkeypatch.setattr(registry, "_ensure_metrics", lambda: None)
    monkeypatch.setattr(mp, "get_mlx_registry", lambda: registry)
    monkeypatch.setattr(mp, "observe_histogram", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mp, "increment_counter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mp, "set_gauge", lambda *_args, **_kwargs: None)
    adapter = mp.MLXChatAdapter()
    request = {"messages": [{"role": "user", "content": "hello"}]}

    original_pool = registry._worker_pool
    original_semaphore = registry._sema
    try:
        with pytest.raises(TimeoutError, match="MLX generation timed out"):
            adapter.chat(request, timeout=0.05)
        assert started.is_set()

        with pytest.raises(ChatProviderError) as exc_info:
            registry.load(
                model_path="new-model",
                overrides={"max_concurrent": 2, "warmup": False, "compile": False},
            )
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert registry.status()["model"] == "old-model"
        assert registry.status()["max_concurrent"] == 1
        assert registry._worker_pool is original_pool
        assert registry._sema is original_semaphore
    finally:
        release.set()

    for _ in range(200):
        if original_pool.active_count == 0 and registry._inflight == 0:
            break
        threading.Event().wait(0.01)
    assert original_pool.active_count == 0
    assert registry._inflight == 0

    registry.load(
        model_path="new-model",
        overrides={"max_concurrent": 2, "warmup": False, "compile": False},
    )
    assert registry.status()["model"] == "new-model"
    assert registry.status()["max_concurrent"] == 2
    assert registry._worker_pool is not original_pool
    assert registry._sema is not original_semaphore


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
        except Exception as exc:  # noqa: BLE001 - the test records the surfaced load failure
            errors.append(exc)

    worker = threading.Thread(target=load_slow_failure)
    worker.start()
    assert started.wait(timeout=2)

    reg.load(model_path="new", overrides=load_overrides)
    release_failure.set()
    worker.join(timeout=2)

    assert errors
    assert reg.status()["model"] == "new"


def test_superseded_older_load_records_superseded_metric(monkeypatch):
    started = threading.Event()
    release_slow_load = threading.Event()

    class FakeTokenizer:
        pass

    def load(model_path, **kwargs):
        if model_path == "slow-old":
            started.set()
            assert release_slow_load.wait(timeout=2)
        return (f"model:{model_path}", FakeTokenizer())

    fake = types.SimpleNamespace(
        load=load,
        generate=lambda *args, **kwargs: "ok",
        generate_stream=lambda *args, **kwargs: iter(["ok"]),
        embed=lambda *args, **kwargs: [1.0],
    )
    monkeypatch.setattr(mp.MLXSessionRegistry, "_import_mlx", lambda self: fake)

    metric_statuses = []

    def capture_counter(name, labels=None):
        if name == "mlx_load_total":
            metric_statuses.append((labels or {}).copy())

    monkeypatch.setattr(mp, "increment_counter", capture_counter)
    monkeypatch.setattr(mp, "observe_histogram", lambda *args, **kwargs: None)
    monkeypatch.setattr(mp, "set_gauge", lambda *args, **kwargs: None)

    reg = mp.MLXSessionRegistry()
    load_overrides = {"max_concurrent": 1, "warmup": False, "compile": False}

    def load_slow_success():
        reg.load(model_path="slow-old", overrides=load_overrides)

    worker = threading.Thread(target=load_slow_success)
    worker.start()
    assert started.wait(timeout=2)

    reg.load(model_path="new", overrides=load_overrides)
    release_slow_load.set()
    worker.join(timeout=2)

    assert reg.status()["model"] == "new"
    assert {"model": "slow-old", "status": "superseded"} in metric_statuses
    assert {"model": "slow-old", "status": "success"} not in metric_statuses


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
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils
    from tldw_Server_API.tests.provider_credential_test_helpers import (
        resolved_request_fields_async,
    )

    monkeypatch.setattr(
        adapter_utils,
        "resolve_provider_api_key_from_config",
        lambda *_args, **_kwargs: pytest.fail(
            "runtime-bound local dispatch must not reload credentials"
        ),
    )

    credential_fields = await resolved_request_fields_async(
        "mlx",
        api_key=None,
        app_config={"mlx": {}},
        model="fake-model",
    )
    stream = await perform_chat_api_call_async(
        api_provider="mlx",
        messages=[{"role": "user", "content": "hi"}],
        streaming=True,
        model="fake-model",
        **credential_fields,
    )
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    assert len(chunks) >= 2
    assert chunks[-1].strip() == "data: [DONE]"
