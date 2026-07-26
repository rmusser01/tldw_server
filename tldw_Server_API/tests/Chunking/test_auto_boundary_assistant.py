from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
)
from tldw_Server_API.app.core.Chat import bounded_daemon
from tldw_Server_API.app.core.Chunking import auto_boundary_assistant as assistant_module
from tldw_Server_API.app.core.Chunking.auto_boundary_assistant import (
    AutoChunkBoundaryAssistantRequest,
    AutoChunkBoundaryAssistantResult,
    ChatAutoChunkBoundaryAssistant,
    append_auto_chunking_fallback,
    extract_bounded_text_excerpt,
    parse_boundary_assistant_response,
)

pytestmark = pytest.mark.unit


async def _wait_for_thread_event(
    event: threading.Event,
    *,
    timeout: float = 1.0,
) -> bool:
    """Poll a thread event without relying on the default executor."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set() and loop.time() < deadline:
        await asyncio.sleep(0.001)
    return event.is_set()


def _runtime_factory(*, api_key="key", resolve_error=None):
    """Return lightweight execution-scoped credential runtimes for unit tests."""

    class Runtime:
        def __init__(self, app_config):
            self.app_config = app_config

        async def resolve(self, provider, *, model=None):
            if resolve_error is not None:
                raise resolve_error
            return SimpleNamespace(
                provider=provider,
                api_key=api_key,
                app_config=self.app_config,
                credentials_resolved=True,
            )

        async def mark_used(self, _handle):
            return None

        async def close(self):
            return None

    return lambda app_config: Runtime(app_config)


def _request(**overrides):
    values = {
        "chunk_options": {
            "method": "structure_aware",
            "max_size": 900,
            "overlap": 120,
            "adaptive": False,
            "multi_level": False,
            "language": "en",
        },
        "chunking_plan": {
            "mode": "auto",
            "goal": "balanced",
            "used_llm": False,
            "method": "structure_aware",
            "max_size": 900,
            "overlap": 120,
            "template_name": None,
            "derived_views": ["section_titles"],
            "fallback_reason": None,
            "rationale": "Detected document structure.",
            "profile": {"media_type": "document", "text_length": 2000},
        },
        "media_type": "document",
        "source_name": "notes.md",
        "extracted_text": "# Intro\n\nSome content.\n\n## Details\n\nMore content.",
        "provider": "openai",
        "model": "gpt-test",
        "timeout_sec": 0.5,
    }
    values.update(overrides)
    return AutoChunkBoundaryAssistantRequest(**values)


def test_boundary_assistant_result_types_represent_success_and_fallback():
    success = AutoChunkBoundaryAssistantResult.success(
        chunk_options={"method": "semantic", "max_size": 800, "overlap": 80},
        derived_views=("topic_sections",),
        rationale="Topic shifts are clearer than headings.",
        provider="openai",
        model="gpt-test",
    )
    fallback = AutoChunkBoundaryAssistantResult.fallback(
        reason="ai_assist_invalid_response",
        rationale="Assistant response did not match the strict schema.",
    )

    assert success.used_llm is True
    assert success.fallback_reason is None
    assert success.provider == "openai"
    assert success.model == "gpt-test"
    assert fallback.used_llm is False
    assert fallback.chunk_options is None
    assert fallback.fallback_reason == "ai_assist_invalid_response"


def test_extract_bounded_text_excerpt_limits_context_without_reordering_text():
    text = "a" * 1200 + "TAIL"

    excerpt = extract_bounded_text_excerpt(text, max_chars=64)

    assert excerpt == "a" * 64
    assert "TAIL" not in excerpt


def test_parse_boundary_assistant_response_accepts_strict_bounded_json():
    request = _request()

    result = parse_boundary_assistant_response(
        '{"method":"semantic","max_size":840,"overlap":84,'
        '"derived_views":["topic_sections","outline"],'
        '"rationale":"The document uses topic shifts more than headings."}',
        request=request,
        provider="openai",
        model="gpt-test",
    )

    assert result.used_llm is True
    assert result.chunk_options == {
        "method": "semantic",
        "max_size": 840,
        "overlap": 84,
        "adaptive": False,
        "multi_level": False,
        "language": "en",
    }
    assert result.derived_views == ("topic_sections", "outline")
    assert result.rationale == "The document uses topic shifts more than headings."


@pytest.mark.parametrize(
    ("response_text", "reason_fragment"),
    [
        ("not-json", "not valid JSON"),
        ('{"method":"shell","max_size":840,"overlap":84}', "method"),
        ('{"method":"semantic","max_size":10,"overlap":1}', "max_size"),
        ('{"method":"semantic","max_size":840,"overlap":840}', "overlap"),
        ('{"method":"semantic","max_size":840,"overlap":84,"derived_views":["bad view"]}', "derived_views"),
        ('{"method":"ebook_chapters","max_size":840,"overlap":84}', "ebook_chapters"),
    ],
)
def test_parse_boundary_assistant_response_rejects_invalid_suggestions(response_text, reason_fragment):
    result = parse_boundary_assistant_response(
        response_text,
        request=_request(),
        provider="openai",
        model="gpt-test",
    )

    assert result.used_llm is False
    assert result.fallback_reason == "ai_assist_invalid_response"
    assert reason_fragment in result.rationale


def test_append_auto_chunking_fallback_preserves_deterministic_plan_and_options():
    request = _request()
    plan = dict(request.chunking_plan)
    options = dict(request.chunk_options)

    updated_options, updated_plan = append_auto_chunking_fallback(
        options,
        plan,
        "ai_assist_timeout",
        "Timed out after 0.5 seconds.",
    )

    assert updated_options == options
    assert updated_plan["method"] == plan["method"]
    assert updated_plan["max_size"] == plan["max_size"]
    assert updated_plan["overlap"] == plan["overlap"]
    assert updated_plan["used_llm"] is False
    assert updated_plan["fallback_reason"] == "ai_assist_timeout"
    assert "Timed out after 0.5 seconds." in updated_plan["rationale"]


@pytest.mark.asyncio
async def test_chat_assistant_returns_unavailable_when_provider_cannot_be_resolved():
    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=lambda **_: pytest.fail("chat call should not run"),
        config_loader=lambda: {},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=_runtime_factory(api_key=None),
        provider_requires_key=lambda _provider: False,
        default_provider=None,
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is False
    assert result.fallback_reason == "ai_assist_unavailable"
    assert "provider" in result.rationale


@pytest.mark.asyncio
async def test_chat_assistant_calls_provider_when_available_and_valid():
    calls = []

    async def chat_call(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=_runtime_factory(api_key="key"),
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is True
    assert result.chunk_options["method"] == "semantic"
    assert result.chunk_options["max_size"] == 820
    assert calls[0]["api_provider"] == "openai"
    assert calls[0]["model"] == "gpt-config"
    assert calls[0]["stream"] is False


@pytest.mark.asyncio
async def test_chat_assistant_runs_availability_checks_off_event_loop_thread():
    main_thread = threading.get_ident()
    loader_threads = []

    async def chat_call(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    def config_loader():
        loader_threads.append(threading.get_ident())
        return {"openai_api": {"model": "gpt-config"}}

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=config_loader,
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=_runtime_factory(api_key="key"),
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is True
    assert loader_threads
    assert loader_threads[0] != main_thread


@pytest.mark.asyncio
async def test_chat_assistant_runs_sync_chat_call_off_event_loop_thread():
    main_thread = threading.get_ident()
    call_threads = []

    def chat_call(**_kwargs):
        call_threads.append(threading.get_ident())
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=_runtime_factory(api_key="key"),
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is True
    assert call_threads
    assert call_threads[0] != main_thread


@pytest.mark.asyncio
async def test_chat_assistant_does_not_offload_native_async_chat_call(
    monkeypatch,
):
    offloaded_callables = []
    real_to_thread = asyncio.to_thread

    async def tracking_to_thread(func, /, *args, **kwargs):
        offloaded_callables.append(func)
        return await real_to_thread(func, *args, **kwargs)

    async def chat_call(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chunking.auto_boundary_assistant.asyncio.to_thread",
        tracking_to_thread,
    )
    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=_runtime_factory(api_key="key"),
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is True
    assert chat_call not in offloaded_callables


@pytest.mark.asyncio
async def test_chat_assistant_cancellation_drains_sync_call_and_mark_before_close(
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    drain_started = asyncio.Event()
    events = []
    pool = bounded_daemon.BoundedDaemonPool(capacity=1)
    real_drain_owned_task = bounded_daemon._drain_owned_task
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-key",
        app_config={"openai_api": {"model": "gpt-config"}},
        credentials_resolved=True,
    )

    class Runtime:
        async def resolve(self, provider, *, model=None):
            assert provider == "openai"
            assert model is None
            return handle

        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle
            assert "close" not in events
            events.append("mark")

        async def close(self):
            events.append("close")

    def chat_call(**_kwargs):
        entered.set()
        release.wait()
        events.append("provider-exit")
        exited.set()
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    async def observe_drain(task):
        drain_started.set()
        return await real_drain_owned_task(task)

    monkeypatch.setattr(bounded_daemon, "_drain_owned_task", observe_drain)
    monkeypatch.setattr(
        assistant_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=lambda _snapshot: Runtime(),
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )
    task = asyncio.create_task(assistant.refine(_request(provider=None, model=None)))
    try:
        assert await _wait_for_thread_event(entered)
        pool_was_active = pool.active_count == 1
        task.cancel()
        assert await asyncio.wait_for(drain_started.wait(), timeout=1.0)
        assert events == []
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert exited.is_set()
    assert events == ["provider-exit", "mark", "close"]
    assert events.count("mark") == 1
    assert pool_was_active
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_chat_assistant_timeout_drains_sync_call_and_mark_before_close(
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    drain_started = asyncio.Event()
    events = []
    pool = bounded_daemon.BoundedDaemonPool(capacity=1)
    real_drain_owned_task = bounded_daemon._drain_owned_task
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-key",
        app_config={"openai_api": {"model": "gpt-config"}},
        credentials_resolved=True,
    )

    class Runtime:
        async def resolve(self, _provider, *, model=None):
            assert model is None
            return handle

        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle
            assert "close" not in events
            events.append("mark")

        async def close(self):
            events.append("close")

    def chat_call(**_kwargs):
        entered.set()
        release.wait()
        events.append("provider-exit")
        exited.set()
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    async def observe_drain(task):
        drain_started.set()
        return await real_drain_owned_task(task)

    monkeypatch.setattr(bounded_daemon, "_drain_owned_task", observe_drain)
    monkeypatch.setattr(
        assistant_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=lambda _snapshot: Runtime(),
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )
    task = asyncio.create_task(
        assistant.refine(_request(provider=None, model=None, timeout_sec=0.001))
    )
    try:
        assert await _wait_for_thread_event(entered)
        pool_was_active = pool.active_count == 1
        assert await asyncio.wait_for(drain_started.wait(), timeout=1.0)
        assert not task.done()
        assert events == []
    finally:
        release.set()

    result = await task
    assert result.used_llm is False
    assert result.fallback_reason == "ai_assist_timeout"
    assert exited.is_set()
    assert events == ["provider-exit", "mark", "close"]
    assert events.count("mark") == 1
    assert pool_was_active
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_chat_assistant_sync_dispatch_bypasses_saturated_default_executor(
    monkeypatch,
):
    """Credential-bearing chat starts outside the default-executor queue."""

    loop = asyncio.get_running_loop()
    previous_default_executor = getattr(loop, "_default_executor", None)
    default_executor = ThreadPoolExecutor(max_workers=1)
    default_entered = threading.Event()
    default_release = threading.Event()
    provider_entered = threading.Event()
    provider_release = threading.Event()
    pool = bounded_daemon.BoundedDaemonPool(capacity=1)
    events: list[str] = []
    default_blocker = None
    task = None
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-key",
        app_config={"openai_api": {"model": "gpt-config"}},
        credentials_resolved=True,
    )

    def block_default_executor() -> None:
        default_entered.set()
        assert default_release.wait(timeout=2.0)

    class Runtime:
        async def resolve(self, provider, *, model=None):
            nonlocal default_blocker
            assert provider == "openai"
            assert model is None
            default_blocker = loop.run_in_executor(None, block_default_executor)
            assert await _wait_for_thread_event(default_entered)
            return handle

        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle
            events.append("mark")

        async def close(self):
            events.append("close")

    def chat_call(**_kwargs):
        events.append("provider-start")
        provider_entered.set()
        assert provider_release.wait(timeout=2.0)
        events.append("provider-exit")
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(
        assistant_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=lambda _snapshot: Runtime(),
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    loop.set_default_executor(default_executor)
    try:
        task = asyncio.create_task(
            assistant.refine(_request(provider=None, model=None))
        )
        assert await _wait_for_thread_event(default_entered)
        assert await _wait_for_thread_event(provider_entered, timeout=0.5)
        assert not default_release.is_set()
        assert pool.active_count == 1

        provider_release.set()
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result.used_llm is True
    finally:
        provider_release.set()
        default_release.set()
        if default_blocker is not None:
            await asyncio.gather(default_blocker, return_exceptions=True)
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        replacement_executor = previous_default_executor or ThreadPoolExecutor()
        loop.set_default_executor(replacement_executor)
        default_executor.shutdown(wait=True, cancel_futures=True)

    assert events == ["provider-start", "provider-exit", "mark", "close"]
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_chat_assistant_pool_saturation_fails_closed_before_dispatch(
    monkeypatch,
):
    """Rejected chat work never starts later or marks its credentials."""

    entered = threading.Event()
    release = threading.Event()
    starts: list[str] = []
    runtimes = []
    pool = bounded_daemon.BoundedDaemonPool(capacity=1)

    class Runtime:
        def __init__(self):
            self.handles = []
            self.events: list[str] = []

        async def resolve(self, provider, *, model=None):
            handle = SimpleNamespace(
                provider=provider,
                api_key=f"{model}-key",
                app_config={"openai_api": {"model": model}},
                credentials_resolved=True,
            )
            self.handles.append(handle)
            return handle

        async def mark_used(self, handle):
            assert handle in self.handles
            self.events.append("mark")

        async def close(self):
            self.events.append("close")

    def runtime_factory(_snapshot):
        runtime = Runtime()
        runtimes.append(runtime)
        return runtime

    def chat_call(**kwargs):
        model = kwargs["model"]
        starts.append(model)
        if model == "model-a":
            entered.set()
            assert release.wait(timeout=2.0)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(
        assistant_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=runtime_factory,
        provider_requires_key=lambda _provider: True,
    )

    admitted = asyncio.create_task(assistant.refine(_request(model="model-a")))
    try:
        assert await _wait_for_thread_event(entered)
        pool_was_active = pool.active_count == 1

        rejected = await assistant.refine(_request(model="model-b"))
        assert rejected.used_llm is False
        assert rejected.fallback_reason == "ai_assist_provider_error"
        assert starts == ["model-a"]
        assert len(runtimes) == 2
        assert runtimes[1].events == ["close"]
        assert pool_was_active
    finally:
        release.set()
        admitted_result = await asyncio.wait_for(admitted, timeout=1.0)

    await asyncio.sleep(0)
    assert admitted_result.used_llm is True
    assert starts == ["model-a"]
    assert runtimes[0].events == ["mark", "close"]
    assert runtimes[1].events == ["close"]
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["cancel", "timeout"])
@pytest.mark.parametrize(
    "raw_response",
    [
        pytest.param("", id="empty"),
        pytest.param("provider error", id="error-string"),
        pytest.param({"error": "provider failed"}, id="error-dict"),
        pytest.param(
            {"choices": [{"message": {"content": "{"}}]},
            id="malformed-json",
        ),
        pytest.param(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"method":"semantic","max_size":10,'
                                '"overlap":1}'
                            )
                        }
                    }
                ]
            },
            id="semantic-fallback",
        ),
    ],
)
async def test_chat_assistant_late_invalid_result_never_marks_credentials(
    monkeypatch,
    completion,
    raw_response,
):
    """Cancellation cleanup marks only strictly valid assistant results."""

    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    drain_started = asyncio.Event()
    events: list[str] = []
    pool = bounded_daemon.BoundedDaemonPool(capacity=1)
    real_drain_owned_task = bounded_daemon._drain_owned_task
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-key",
        app_config={"openai_api": {"model": "gpt-config"}},
        credentials_resolved=True,
    )

    class Runtime:
        async def resolve(self, _provider, *, model=None):
            assert model is None
            return handle

        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle
            events.append("mark")

        async def close(self):
            events.append("close")

    def chat_call(**_kwargs):
        entered.set()
        release.wait()
        events.append("provider-exit")
        exited.set()
        return raw_response

    async def observe_drain(task):
        drain_started.set()
        return await real_drain_owned_task(task)

    monkeypatch.setattr(bounded_daemon, "_drain_owned_task", observe_drain)
    monkeypatch.setattr(
        assistant_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=lambda _snapshot: Runtime(),
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )
    timeout_sec = 0.001 if completion == "timeout" else 0.5
    task = asyncio.create_task(
        assistant.refine(
            _request(provider=None, model=None, timeout_sec=timeout_sec)
        )
    )
    try:
        assert await _wait_for_thread_event(entered)
        pool_was_active = pool.active_count == 1
        if completion == "cancel":
            task.cancel()
        assert await asyncio.wait_for(drain_started.wait(), timeout=1.0)
        assert not task.done()
        assert events == []
    finally:
        release.set()

    if completion == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result.used_llm is False
        assert result.fallback_reason == "ai_assist_timeout"

    assert exited.is_set()
    assert events == ["provider-exit", "close"]
    assert pool_was_active
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_chat_assistant_uses_adapter_canonical_provider_for_alias_availability():
    calls = []

    async def chat_call(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    seen_requires_key = []
    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"local_llm": {"model": "local-model"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: SimpleNamespace(name="local-llm")),
        credential_runtime_factory=_runtime_factory(api_key=None),
        provider_requires_key=lambda provider: seen_requires_key.append(provider) or False,
    )

    result = await assistant.refine(_request(provider="local_llm", model=None))

    assert result.used_llm is True
    assert result.provider == "local-llm"
    assert calls[0]["api_provider"] == "local-llm"
    assert calls[0]["model"] == "local-model"
    assert seen_requires_key == ["local-llm"]


@pytest.mark.asyncio
async def test_chat_assistant_does_not_retry_typeerror_from_runtime_resolution():
    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=lambda **_: pytest.fail("chat call should not run"),
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=_runtime_factory(
            resolve_error=TypeError("inner resolver failure")
        ),
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is False
    assert result.fallback_reason == "ai_assist_provider_error"
    assert "TypeError" in result.rationale


@pytest.mark.asyncio
async def test_chat_assistant_timeout_falls_back_without_raising():
    async def chat_call(**_kwargs):
        await asyncio.sleep(0.05)
        return "{}"

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=_runtime_factory(api_key="key"),
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None, timeout_sec=0.001))

    assert result.used_llm is False
    assert result.fallback_reason == "ai_assist_timeout"


@pytest.mark.asyncio
async def test_chat_assistant_provider_error_falls_back_without_raising():
    async def chat_call(**_kwargs):
        raise RuntimeError("provider exploded")

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "gpt-config"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=_runtime_factory(api_key="key"),
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is False
    assert result.fallback_reason == "ai_assist_provider_error"
    assert "RuntimeError" in result.rationale


@pytest.mark.asyncio
async def test_chat_assistant_uses_runtime_snapshot_at_adapter_boundary():
    calls = []
    handles = []
    marked = []
    closed = []

    class Runtime:
        async def resolve(self, provider, *, model=None):
            assert provider == "openai"
            assert model is None
            handle = SimpleNamespace(
                provider=provider,
                api_key="runtime-key",
                app_config={
                    "openai_api": {
                        "model": "runtime-model",
                        "api_base_url": "https://runtime.example/v1",
                    }
                },
                credentials_resolved=True,
            )
            handles.append(handle)
            return handle

        async def mark_used(self, handle):
            marked.append(handle)

        async def close(self):
            closed.append(True)

    async def chat_call(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {"openai_api": {"model": "stale-model"}},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=lambda _snapshot: Runtime(),
        provider_requires_key=lambda _provider: True,
        default_provider="openai",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is True
    assert calls[0]["api_key"] == "runtime-key"
    assert calls[0]["model"] == "runtime-model"
    assert calls[0]["app_config"]["openai_api"]["api_base_url"] == "https://runtime.example/v1"
    assert calls[0]["credentials_resolved"] is True
    assert calls[0][PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handles[0]
    assert marked == handles
    assert closed == [True]


@pytest.mark.asyncio
async def test_concurrent_chat_assistant_calls_keep_runtime_credentials_isolated():
    entered = {"model-a": threading.Event(), "model-b": threading.Event()}
    release = {"model-a": threading.Event(), "model-b": threading.Event()}
    calls = []
    runtimes = []

    class Runtime:
        def __init__(self):
            self.handles = []
            self.marked = []
            self.closed = False

        async def resolve(self, provider, *, model=None):
            handle = SimpleNamespace(
                provider=provider,
                api_key=f"{model}-key",
                app_config={"openai_api": {"model": model}},
                credentials_resolved=True,
            )
            self.handles.append(handle)
            return handle

        async def mark_used(self, handle):
            self.marked.append(handle)

        async def close(self):
            self.closed = True

    def runtime_factory(_snapshot):
        runtime = Runtime()
        runtimes.append(runtime)
        return runtime

    def chat_call(**kwargs):
        model = kwargs["model"]
        calls.append(
            (
                model,
                kwargs["api_key"],
                kwargs[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY],
            )
        )
        entered[model].set()
        release[model].wait()
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=runtime_factory,
        provider_requires_key=lambda _provider: True,
    )
    first = asyncio.create_task(assistant.refine(_request(model="model-a")))
    second = asyncio.create_task(assistant.refine(_request(model="model-b")))
    try:
        assert await asyncio.to_thread(
            lambda: all(event.wait(1.0) for event in entered.values())
        )
        release["model-b"].set()
        assert (await asyncio.wait_for(second, timeout=1.0)).used_llm is True
        release["model-a"].set()
        assert (await asyncio.wait_for(first, timeout=1.0)).used_llm is True
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(first, second, return_exceptions=True)

    assert {(model, key, handle.api_key) for model, key, handle in calls} == {
        ("model-a", "model-a-key", "model-a-key"),
        ("model-b", "model-b-key", "model-b-key"),
    }
    assert len(runtimes) == 2
    assert all(runtime.marked == runtime.handles for runtime in runtimes)
    assert all(runtime.closed for runtime in runtimes)


@pytest.mark.asyncio
async def test_chat_assistant_accepts_runtime_certified_bedrock_default_chain():
    calls = []
    handle = SimpleNamespace(
        provider="bedrock",
        api_key=None,
        app_config={
            "bedrock_api": {
                "model": "bedrock-model",
                "_runtime_auth_source": "aws_default_chain",
            }
        },
        credentials_resolved=True,
    )

    class Runtime:
        async def resolve(self, provider, *, model=None):
            assert provider == "bedrock"
            assert model is None
            return handle

        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle

        async def close(self):
            return None

    def chat_call(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"method":"semantic","max_size":820,"overlap":82,'
                            '"derived_views":["topic_sections"],'
                            '"rationale":"Clear topic transitions."}'
                        )
                    }
                }
            ]
        }

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=chat_call,
        config_loader=lambda: {},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=lambda _snapshot: Runtime(),
        provider_requires_key=lambda _provider: True,
        default_provider="bedrock",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is True
    assert calls[0]["api_provider"] == "bedrock"
    assert calls[0]["api_key"] is None
    assert calls[0]["credentials_resolved"] is True
    assert calls[0]["app_config"] == handle.app_config


@pytest.mark.asyncio
async def test_chat_assistant_rejects_uncertified_bedrock_default_chain():
    events = []
    handle = SimpleNamespace(
        provider="bedrock",
        api_key=None,
        app_config={
            "bedrock_api": {
                "model": "bedrock-model",
                "_runtime_auth_source": "aws_default_chain",
            }
        },
        credentials_resolved=False,
    )

    class Runtime:
        async def resolve(self, _provider, *, model=None):
            assert model is None
            return handle

        async def mark_used(self, _resolved_handle):
            pytest.fail("rejected credentials must not be marked as used")

        async def close(self):
            events.append("close")

    assistant = ChatAutoChunkBoundaryAssistant(
        chat_call=lambda **_kwargs: pytest.fail(
            "chat must not run with uncertified Bedrock credentials"
        ),
        config_loader=lambda: {},
        registry_getter=lambda: SimpleNamespace(get_adapter=lambda _provider: object()),
        credential_runtime_factory=lambda _snapshot: Runtime(),
        provider_requires_key=lambda _provider: True,
        default_provider="bedrock",
    )

    result = await assistant.refine(_request(provider=None, model=None))

    assert result.used_llm is False
    assert result.fallback_reason == "ai_assist_unavailable"
    assert "API key" in result.rationale
    assert events == ["close"]
