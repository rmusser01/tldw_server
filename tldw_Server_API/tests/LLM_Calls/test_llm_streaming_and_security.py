import asyncio
import json

import pytest


def test_google_stream_emits_done_once(monkeypatch):
    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, **kwargs):
            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    return None

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def iter_lines(self):
                    first_chunk = {
                        "candidates": [
                            {"content": {"parts": [{"text": "hello"}]}}
                        ]
                    }
                    return iter(
                        [
                            f"data: {json.dumps(first_chunk)}".encode("utf-8"),
                            b"data: [DONE]",
                        ]
                    )

                def close(self):
                    return None

            return _Resp()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.http_client_factory",
        lambda *a, **k: _Client(),
    )

    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call

    gen = perform_chat_api_call(
        api_provider="google",
        messages=[{"role": "user", "content": "hi"}],
        api_key="test-key",
        model="gemini-2.5-flash",
        streaming=True,
    )
    chunks = list(gen)

    done_count = sum(1 for c in chunks if c.strip().lower() == "data: [done]")
    assert done_count == 1, f"Expected exactly one [DONE], got {done_count}. Chunks: {chunks}"


def test_huggingface_headers_are_masked(monkeypatch):
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call as _perform_chat

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    return None

                def json(self):
                    return {"id": "ok", "choices": [{"message": {"content": "hi"}}]}

                def close(self):
                    return None

            return _Resp()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter.http_client_factory",
        lambda *a, **k: _Client(),
    )

    captured_debug = []

    def _fake_debug(msg, *args, **kwargs):
        rendered = str(msg)
        if args:
            try:
                rendered = rendered.format(*args)
            except Exception:
                rendered = f"{msg} {args}"
        captured_debug.append(rendered)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter.logger.debug",
        _fake_debug,
    )

    secret = "sk-ABCDEF1234567890"
    _perform_chat(
        api_provider="huggingface",
        messages=[{"role": "user", "content": "hi"}],
        api_key=secret,
        streaming=False,
        model="test/Model-Stub",
    )

    joined = "\n".join(captured_debug)
    assert "HuggingFace headers:" in joined
    assert secret not in joined
    assert "***" in joined


def test_http_400_logging_omits_prompt_and_request_body(monkeypatch):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
    from tldw_Server_API.app.core.LLM_Calls import error_utils

    secret_prompt = "SECRET_PROMPT_CONTENT"
    secret_key = "sk-secret-should-not-log"
    body = {
        "error": {
            "message": f"Invalid request for prompt {secret_prompt}",
            "type": "invalid_request_error",
            "code": "bad_request",
        },
        "messages": [{"role": "user", "content": secret_prompt}],
        "api_key": secret_key,
    }

    class _Resp:
        status_code = 400
        text = json.dumps(body)

        def json(self):
            return body

    class _Exc(Exception):
        response = _Resp()

    warnings = []
    errors = []

    class _Logger:
        def warning(self, msg):
            warnings.append(str(msg))

        def error(self, msg):
            errors.append(str(msg))

    monkeypatch.setattr(error_utils, "logger", _Logger())

    exc = _Exc("upstream 400")
    error_utils.log_http_400_body("openai", exc)
    with pytest.raises(ChatBadRequestError):
        error_utils.raise_chat_error_from_http("openai", exc)

    rendered_logs = "\n".join(warnings + errors)
    assert secret_prompt not in rendered_logs
    assert secret_key not in rendered_logs
    assert "messages" not in rendered_logs
    assert "invalid_request_error" in rendered_logs


@pytest.mark.asyncio
async def test_wrap_sync_stream_applies_backpressure_and_closes_on_cancel():
    from tldw_Server_API.app.core.LLM_Calls.streaming import wrap_sync_stream

    class _FastIterator:
        def __init__(self):
            self.yielded = 0
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.closed:
                raise StopIteration
            self.yielded += 1
            return f"chunk-{self.yielded}"

        def close(self):
            self.closed = True

    source = _FastIterator()
    stream = wrap_sync_stream(source, max_queue_size=1)

    assert await stream.__anext__() == "chunk-1"
    await asyncio.sleep(0.1)

    assert source.yielded <= 3
    await stream.aclose()
    await asyncio.sleep(0.1)

    yielded_after_close = source.yielded
    assert source.closed is True
    await asyncio.sleep(0.1)
    assert source.yielded == yielded_after_close


@pytest.mark.asyncio
async def test_wrap_sync_stream_does_not_use_default_executor_for_delivery(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import streaming

    async def fail_to_thread(*args, **kwargs):
        raise AssertionError("wrap_sync_stream should not use the default executor for chunk delivery")

    monkeypatch.setattr(streaming.asyncio, "to_thread", fail_to_thread)

    chunks = []
    async for chunk in streaming.wrap_sync_stream(iter(["chunk-1", "chunk-2"]), max_queue_size=1):
        chunks.append(chunk)

    assert chunks == ["chunk-1", "chunk-2"]


@pytest.mark.asyncio
async def test_wrap_sync_stream_logs_close_errors(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import streaming

    class _ClosingIterator:
        def __init__(self):
            self._done = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._done:
                raise StopIteration
            self._done = True
            return "chunk"

        def close(self):
            raise RuntimeError("close failed")

    debug_messages = []

    def fake_debug(message, *args, **kwargs):
        debug_messages.append(str(message).format(*args))

    monkeypatch.setattr(streaming.logger, "debug", fake_debug)

    chunks = []
    async for chunk in streaming.wrap_sync_stream(_ClosingIterator()):
        chunks.append(chunk)

    assert chunks == ["chunk"]
    assert any("close failed" in message for message in debug_messages)
