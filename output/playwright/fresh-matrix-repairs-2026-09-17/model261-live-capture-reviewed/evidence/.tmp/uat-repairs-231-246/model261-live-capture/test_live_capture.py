"""Behavior tests for the isolated UAT261 live boundary observer."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = Path(__file__).with_name("uat261_capture.py")
_SPEC = importlib.util.spec_from_file_location("uat261_live_capture", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
capture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(capture)


def _envelope(messages):
    return SimpleNamespace(
        fingerprint_version="prompt-v1",
        aggregate_fingerprint="prompt-v1:sha256:messages",
        message_count=len(messages),
        role_counts={"system": 1, "user": 1},
    )


def _request():
    return {
        "api_endpoint": "openai",
        "messages_payload": [
            {"role": "system", "content": "secret system prompt"},
            {"role": "user", "content": "secret user prompt"},
        ],
        "api_key": "secret-api-key",
        "temp": 0.7,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "stop": None,
        "model": "model-x",
        "max_tokens": 128,
        "tools": None,
        "tool_choice": None,
        "billing_prompt_cache_intent": None,
        "inference_prefix_cache_intent": None,
        "streaming": True,
        "app_config": {"credential": "never-store"},
        "credentials_resolved": True,
    }


def _terminal_frame(*, usage=True):
    payload = {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "system_fingerprint": "provider-secret-fingerprint",
    }
    if usage:
        payload["usage"] = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    return f"data: {json.dumps(payload)}\n\n"


def test_installed_observer_forwards_original_frames_and_writes_only_safe_projection(tmp_path):
    frames = [
        "event: completion.chunk\n\n",
        'data: {"choices":[{"delta":{"content":"<think>private</think>answer"},"finish_reason":null}]}\n\n',
        _terminal_frame(),
        "data: [DONE]\n\n",
    ]
    provider_calls = []

    def original(**kwargs):
        provider_calls.append(kwargs)
        return iter(frames)

    app = object()
    sessions = SimpleNamespace(perform_chat_api_call=original)
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)

    observed_app, observer = capture.install_capture(
        app=app,
        sessions_module=sessions,
        capture_dir=capture_dir,
        envelope_builder=_envelope,
    )

    assert observed_app is app
    assert list(sessions.perform_chat_api_call(**_request())) == frames
    observer.close()

    assert provider_calls == [_request()]
    assert sessions.perform_chat_api_call is original
    evidence = json.loads((capture_dir / "uat261_capture.json").read_text(encoding="utf-8"))
    assert evidence["records"][0]["terminal"] == {
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "system_fingerprint_hash": "sha256:9a422071e78d04824ea8f648fed125ddd6f19f0de816d8ab5abb14e898e67fc0",
    }
    assert evidence["records"][0]["final_answer"] is None
    serialized = json.dumps(evidence, sort_keys=True)
    for forbidden in ("secret system prompt", "secret user prompt", "secret-api-key", "never-store", "private</think>answer", "provider-secret-fingerprint"):
        assert forbidden not in serialized


def test_second_capture_restores_original_binding_and_third_call_is_not_captured(tmp_path):
    calls = []

    def original(**kwargs):
        calls.append(kwargs)
        return iter([_terminal_frame(usage=False), "data: [DONE]\n\n"])

    sessions = SimpleNamespace(perform_chat_api_call=original)
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)
    _, observer = capture.install_capture(
        app=object(), sessions_module=sessions, capture_dir=capture_dir, envelope_builder=_envelope
    )

    list(sessions.perform_chat_api_call(**_request()))
    list(sessions.perform_chat_api_call(**_request()))
    assert sessions.perform_chat_api_call is original
    list(sessions.perform_chat_api_call(**_request()))
    observer.close()

    evidence = json.loads((capture_dir / "uat261_capture.json").read_text(encoding="utf-8"))
    assert len(calls) == 3
    assert len(evidence["records"]) == 2
    assert evidence["records"][0]["terminal"]["usage"] is None
    assert evidence["records"][1]["terminal"]["usage"] is None


def test_observer_accepts_the_real_envelope_shape_without_role_counts(tmp_path):
    def real_shape_envelope(messages):
        return SimpleNamespace(
            fingerprint_version="prompt-v1",
            aggregate_fingerprint="prompt-v1:sha256:messages",
            message_count=len(messages),
            segment_token_totals={"static": 1, "user_turn": 2},
        )

    def original(**kwargs):
        return iter([_terminal_frame(usage=False), "data: [DONE]\n\n"])

    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)
    sessions = SimpleNamespace(perform_chat_api_call=original)
    _, observer = capture.install_capture(
        app=object(), sessions_module=sessions, capture_dir=capture_dir, envelope_builder=real_shape_envelope
    )

    list(sessions.perform_chat_api_call(**_request()))
    observer.close()

    record = json.loads((capture_dir / "uat261_capture.json").read_text(encoding="utf-8"))["records"][0]
    assert record["capture_state"] == "terminal_observed"
    assert record["message_role_counts"] == {"system": 1, "user": 1}


def test_observer_preserves_provider_generator_failure_and_marks_record_incomplete(tmp_path):
    def original(**kwargs):
        def frames():
            yield "data: {\"choices\":[{\"delta\":{\"content\":\"<think>not-final</think>\"},\"finish_reason\":null}]}\n\n"
            raise RuntimeError("provider secret detail")

        return frames()

    sessions = SimpleNamespace(perform_chat_api_call=original)
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)
    _, observer = capture.install_capture(
        app=object(), sessions_module=sessions, capture_dir=capture_dir, envelope_builder=_envelope
    )

    stream = sessions.perform_chat_api_call(**_request())
    assert next(stream).startswith("data:")
    with pytest.raises(RuntimeError, match="provider secret detail"):
        next(stream)
    observer.close()

    record = json.loads((capture_dir / "uat261_capture.json").read_text(encoding="utf-8"))["records"][0]
    assert record["capture_state"] == "incomplete"
    assert record["terminal"] == {"finish_reason": None, "usage": None, "system_fingerprint_hash": None}
    assert "provider secret detail" not in json.dumps(record)


def test_observer_reraises_cancellation_without_recording_exception_detail(tmp_path):
    def original(**kwargs):
        def frames():
            raise asyncio.CancelledError()
            yield "unreachable"

        return frames()

    sessions = SimpleNamespace(perform_chat_api_call=original)
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)
    _, observer = capture.install_capture(
        app=object(), sessions_module=sessions, capture_dir=capture_dir, envelope_builder=_envelope
    )

    with pytest.raises(asyncio.CancelledError):
        next(sessions.perform_chat_api_call(**_request()))
    observer.close()

    record = json.loads((capture_dir / "uat261_capture.json").read_text(encoding="utf-8"))["records"][0]
    assert record["capture_state"] == "cancelled"
    assert record["terminal"] == {"finish_reason": None, "usage": None, "system_fingerprint_hash": None}


def test_unstarted_wrapper_close_closes_the_original_sync_provider_stream(tmp_path):
    class ProviderStream:
        def __init__(self):
            self.closed = 0

        def __iter__(self):
            return self

        def __next__(self):
            return "data: [DONE]\n\n"

        def close(self):
            self.closed += 1

    provider = ProviderStream()
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)
    sessions = SimpleNamespace(perform_chat_api_call=lambda **kwargs: provider)
    _, observer = capture.install_capture(
        app=object(), sessions_module=sessions, capture_dir=capture_dir, envelope_builder=_envelope
    )

    sessions.perform_chat_api_call(**_request()).close()
    observer.close()

    assert provider.closed == 1


def test_partially_consumed_wrapper_close_closes_the_original_sync_provider_stream(tmp_path):
    class ProviderStream:
        def __init__(self):
            self.closed = 0
            self.frames = iter(["data: {\"choices\":[{\"delta\":{},\"finish_reason\":null}]}\n\n"])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.frames)

        def close(self):
            self.closed += 1

    provider = ProviderStream()
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)
    sessions = SimpleNamespace(perform_chat_api_call=lambda **kwargs: provider)
    _, observer = capture.install_capture(
        app=object(), sessions_module=sessions, capture_dir=capture_dir, envelope_builder=_envelope
    )

    stream = sessions.perform_chat_api_call(**_request())
    assert next(stream).startswith("data:")
    stream.close()
    observer.close()

    assert provider.closed == 1


def test_wrapper_defers_sync_iterator_acquisition_and_closes_distinct_resources_in_product_order(tmp_path):
    events = []

    class Iterator:
        def __next__(self):
            return "data: [DONE]\n\n"

        def close(self):
            events.append("iterator-close")

    class ProviderStream:
        def __init__(self):
            self.iterator = Iterator()

        def __iter__(self):
            events.append("source-iter")
            return self.iterator

        def close(self):
            events.append("source-close")

    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)
    provider = ProviderStream()
    sessions = SimpleNamespace(perform_chat_api_call=lambda **kwargs: provider)
    _, observer = capture.install_capture(
        app=object(), sessions_module=sessions, capture_dir=capture_dir, envelope_builder=_envelope
    )

    stream = sessions.perform_chat_api_call(**_request())
    assert events == []
    assert next(stream).startswith("data:")
    stream.close()
    observer.close()

    assert events == ["source-iter", "iterator-close", "source-close"]


def test_wrapper_close_reraises_the_original_close_error(tmp_path):
    class ProviderStream:
        def __iter__(self):
            return self

        def __next__(self):
            return "data: [DONE]\n\n"

        def close(self):
            raise RuntimeError("provider close failure")

    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)
    sessions = SimpleNamespace(perform_chat_api_call=lambda **kwargs: ProviderStream())
    _, observer = capture.install_capture(
        app=object(), sessions_module=sessions, capture_dir=capture_dir, envelope_builder=_envelope
    )

    with pytest.raises(RuntimeError, match="provider close failure"):
        sessions.perform_chat_api_call(**_request()).close()
    observer.close()


def test_malformed_frame_limitation_remains_visible_after_a_later_terminal_frame(tmp_path):
    def original(**kwargs):
        return iter(["data: {invalid-json}\n\n", _terminal_frame(), "data: [DONE]\n\n"])

    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)
    sessions = SimpleNamespace(perform_chat_api_call=original)
    _, observer = capture.install_capture(
        app=object(), sessions_module=sessions, capture_dir=capture_dir, envelope_builder=_envelope
    )

    assert list(sessions.perform_chat_api_call(**_request()))[1] == _terminal_frame()
    observer.close()

    record = json.loads((capture_dir / "uat261_capture.json").read_text(encoding="utf-8"))["records"][0]
    assert record["capture_state"] == "terminal_observed"
    assert record["stream_observation_state"] == "incomplete"


@pytest.mark.asyncio
async def test_unstarted_async_wrapper_aclose_closes_the_original_provider_stream(tmp_path):
    class ProviderStream:
        def __init__(self):
            self.closed = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            return "data: [DONE]\n\n"

        async def aclose(self):
            self.closed += 1

    provider = ProviderStream()
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)
    sessions = SimpleNamespace(perform_chat_api_call=lambda **kwargs: provider)
    _, observer = capture.install_capture(
        app=object(), sessions_module=sessions, capture_dir=capture_dir, envelope_builder=_envelope
    )

    await sessions.perform_chat_api_call(**_request()).aclose()
    observer.close()

    assert provider.closed == 1


@pytest.mark.asyncio
async def test_async_wrapper_defers_iterator_acquisition_and_closes_distinct_resources_in_product_order(tmp_path):
    events = []

    class Iterator:
        async def __anext__(self):
            return "data: [DONE]\n\n"

        async def aclose(self):
            events.append("iterator-close")

    class ProviderStream:
        def __init__(self):
            self.iterator = Iterator()

        def __aiter__(self):
            events.append("source-aiter")
            return self.iterator

        async def aclose(self):
            events.append("source-close")

    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)
    provider = ProviderStream()
    sessions = SimpleNamespace(perform_chat_api_call=lambda **kwargs: provider)
    _, observer = capture.install_capture(
        app=object(), sessions_module=sessions, capture_dir=capture_dir, envelope_builder=_envelope
    )

    stream = sessions.perform_chat_api_call(**_request())
    assert events == []
    assert (await stream.__anext__()).startswith("data:")
    await stream.aclose()
    observer.close()

    assert events == ["source-aiter", "iterator-close", "source-close"]


@pytest.mark.parametrize("mode, has_existing_output", [(0o755, False), (0o700, True)])
def test_factory_configuration_fails_closed_for_nonprivate_or_nonfresh_capture_directory(
    monkeypatch, tmp_path, mode, has_existing_output
):
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=mode)
    capture_dir.chmod(mode)
    if has_existing_output:
        (capture_dir / "uat261_capture.json").write_text("prior", encoding="utf-8")
    monkeypatch.setenv("UAT261_CAPTURE_DIR", str(capture_dir))

    with pytest.raises(ValueError):
        capture.create_app()


@pytest.mark.parametrize("filename", ["uat261_capture.json", "uat261_capture_status.json"])
def test_preflight_rejects_a_dangling_fixed_output_symlink(tmp_path, filename):
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir(mode=0o700)
    (capture_dir / filename).symlink_to(tmp_path / "does-not-exist")

    with pytest.raises(ValueError):
        capture._validated_capture_dir(str(capture_dir))
