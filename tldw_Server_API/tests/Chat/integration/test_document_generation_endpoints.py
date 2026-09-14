import asyncio
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.chat_documents_deps import get_document_generator_service
from tldw_Server_API.app.api.v1.schemas.document_generator_schemas import GenerateDocumentRequest
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
from tldw_Server_API.app.core.Chat.document_generator import DocumentType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError, InputError
from tldw_Server_API.tests._plugins.chat_fixtures import get_auth_headers

pytestmark = pytest.mark.usefixtures("setup_dependencies")


@pytest.fixture(autouse=True)
def _ensure_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield


def _make_payload(**overrides):


    base = {
        "conversation_id": "chat-42",
        "document_type": "summary",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-test",
        "stream": False,
        "async_generation": False,
    }
    base.update(overrides)
    return base


def _direct_request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/documents/generate", "headers": []})


class _ReleaseTrackingDaemonPool(BoundedDaemonPool):
    """Record each real capacity release for lifecycle assertions."""

    def __init__(self, capacity: int, lifecycle: list, label: str) -> None:
        super().__init__(capacity)
        self.lifecycle = lifecycle
        self.label = label
        self.release_count = 0

    def _release_capacity(self) -> None:
        self.lifecycle.append(self.label)
        self.release_count += 1
        super()._release_capacity()


async def _wait_for_thread_event(
    event: threading.Event,
    *,
    timeout: float = 1.0,
) -> None:
    """Wait for a thread event without consuming the executor under test."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            raise AssertionError("Timed out waiting for thread event")
        await asyncio.sleep(0.001)


def _install_direct_document_runtime(monkeypatch, lifecycle):
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    class RecordingRuntime:
        def __init__(self, **kwargs):
            lifecycle.append(("init", kwargs))

        async def resolve(self, provider, *, model=None):
            lifecycle.append(("resolve", provider, model))
            return SimpleNamespace(
                provider=provider,
                api_key=None,
                app_config={"ollama": {"base_url": "http://generation-a.invalid"}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle):
            lifecycle.append("mark_used")

        async def close(self):
            lifecycle.append("runtime_close")

    monkeypatch.setattr(chat_docs, "ProviderCredentialRuntime", RecordingRuntime)
    monkeypatch.setattr(
        chat_docs,
        "derive_trusted_credential_scope",
        lambda _request, _principal: (1, [11], [22], True),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("result_kind", ["nonstream", "lazy_stream"])
async def test_document_factory_cancellation_hands_off_completed_result_before_runtime_close(
    monkeypatch,
    result_kind,
):
    """Cancellation cannot abandon a sync factory result that still owns credentials."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    lifecycle = []
    entered = threading.Event()
    release = threading.Event()

    class UnconsumedStream:
        def __iter__(self):
            return self

        def __next__(self):
            raise AssertionError("cancelled factory stream must not be consumed")

        def close(self):
            lifecycle.append("upstream_close")

    result = "generated" if result_kind == "nonstream" else UnconsumedStream()

    class BlockingService:
        def __init__(self, _db):
            return None

        def generate_document(self, **_kwargs):
            entered.set()
            release.wait(timeout=5)
            lifecycle.append("factory_exit")
            return result

    _install_direct_document_runtime(monkeypatch, lifecycle)
    task = asyncio.create_task(
        chat_docs.generate_document(
            request=GenerateDocumentRequest(
                **_make_payload(
                    provider="ollama",
                    model="model-a",
                    api_key=None,
                    stream=result_kind == "lazy_stream",
                )
            ),
            http_request=_direct_request(),
            db=object(),
            service_cls=BlockingService,
            principal=AuthPrincipal(kind="user", user_id=1),
        )
    )
    assert await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert "runtime_close" not in lifecycle

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert lifecycle.index("factory_exit") < lifecycle.index("runtime_close")
    if result_kind == "nonstream":
        assert lifecycle.index("mark_used") < lifecycle.index("runtime_close")
        assert "upstream_close" not in lifecycle
    else:
        assert "mark_used" not in lifecycle
        assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
async def test_document_factory_starts_without_default_executor_and_drains_cancel(
    monkeypatch,
):
    """Credential-bearing generation starts directly and owns runtime through exit."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    lifecycle = []
    default_entered = threading.Event()
    default_release = threading.Event()
    factory_entered = threading.Event()
    factory_release = threading.Event()
    factory_starts = 0
    pool = _ReleaseTrackingDaemonPool(1, lifecycle, "capacity_release")
    loop = asyncio.get_running_loop()
    previous_executor = getattr(loop, "_default_executor", None)
    saturated_executor = ThreadPoolExecutor(max_workers=1)
    task = None

    def block_default_executor() -> None:
        default_entered.set()
        assert default_release.wait(timeout=3.0)

    class BlockingService:
        def __init__(self, _db):
            return None

        def generate_document(self, **_kwargs):
            nonlocal factory_starts
            factory_starts += 1
            lifecycle.append("factory_start")
            factory_entered.set()
            assert factory_release.wait(timeout=3.0)
            lifecycle.append("factory_exit")
            return "generated"

    _install_direct_document_runtime(monkeypatch, lifecycle)
    monkeypatch.setattr(chat_docs, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    loop.set_default_executor(saturated_executor)
    default_blocker = loop.run_in_executor(None, block_default_executor)
    try:
        await _wait_for_thread_event(default_entered)
        task = asyncio.create_task(
            chat_docs.generate_document(
                request=GenerateDocumentRequest(
                    **_make_payload(provider="ollama", model="model-a", api_key=None)
                ),
                http_request=_direct_request(),
                db=object(),
                service_cls=BlockingService,
                principal=AuthPrincipal(kind="user", user_id=1),
            )
        )

        await _wait_for_thread_event(factory_entered)
        assert not default_release.is_set()
        assert pool.active_count == 1

        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        assert "runtime_close" not in lifecycle

        factory_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        assert lifecycle.index("factory_exit") < lifecycle.index("capacity_release")
        assert lifecycle.index("capacity_release") < lifecycle.index("mark_used")
        assert lifecycle.index("mark_used") < lifecycle.index("runtime_close")
        assert pool.release_count == 1
        assert pool.active_count == 0
    finally:
        factory_release.set()
        default_release.set()
        await asyncio.gather(default_blocker, return_exceptions=True)
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        replacement_executor = previous_executor or ThreadPoolExecutor()
        loop.set_default_executor(replacement_executor)
        saturated_executor.shutdown(wait=True, cancel_futures=True)

    await asyncio.sleep(0)
    assert factory_starts == 1


@pytest.mark.asyncio
async def test_document_factory_capacity_rejection_is_bounded_and_pre_dispatch(
    monkeypatch,
):
    """Saturated document generation fails with a public 503 before dispatch."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    lifecycle = []
    occupied = threading.Event()
    release = threading.Event()
    dispatched = threading.Event()
    pool = BoundedDaemonPool(1)

    def occupy_pool() -> None:
        occupied.set()
        assert release.wait(timeout=3.0)

    class ForbiddenService:
        def __init__(self, _db):
            return None

        def generate_document(self, **_kwargs):
            dispatched.set()
            return "provider-secret-/srv/private"

        def get_generated_documents(self, **_kwargs):
            return []

    _install_direct_document_runtime(monkeypatch, lifecycle)
    monkeypatch.setattr(chat_docs, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    pool.start(occupy_pool, name="occupied-document-worker")
    try:
        await _wait_for_thread_event(occupied)
        with pytest.raises(HTTPException) as captured:
            await chat_docs.generate_document(
                request=GenerateDocumentRequest(
                    **_make_payload(provider="ollama", model="model-a", api_key=None)
                ),
                http_request=_direct_request(),
                db=object(),
                service_cls=ForbiddenService,
                principal=AuthPrincipal(kind="user", user_id=1),
            )
    finally:
        release.set()
        while pool.active_count:
            await asyncio.sleep(0.001)

    assert captured.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "provider-secret" not in str(captured.value.detail)
    assert not dispatched.is_set()
    assert lifecycle[-1] == "runtime_close"


@pytest.mark.asyncio
async def test_document_stream_missing_credentials_fails_closed_in_test_mode(
    monkeypatch,
):
    """Test flags cannot bypass required server-side provider credentials."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    lifecycle = []
    dispatched = False

    class MissingCredentialRuntime:
        def __init__(self, **_kwargs):
            return None

        async def resolve(self, provider, *, model=None):
            lifecycle.append(("resolve", provider, model))
            return SimpleNamespace(
                provider=provider,
                api_key=None,
                app_config=None,
                credentials_resolved=False,
            )

        async def mark_used(self, _handle):
            lifecycle.append("mark_used")

        async def close(self):
            lifecycle.append("runtime_close")

    class ForbiddenService:
        def __init__(self, _db):
            return None

        def generate_document(self, **_kwargs):
            nonlocal dispatched
            dispatched = True
            return "must-not-run"

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(chat_docs, "ProviderCredentialRuntime", MissingCredentialRuntime)
    monkeypatch.setattr(
        chat_docs,
        "derive_trusted_credential_scope",
        lambda _request, _principal: (1, [], [], False),
    )

    try:
        response = await chat_docs.generate_document(
            request=GenerateDocumentRequest(
                **_make_payload(
                    provider="openai",
                    api_key=None,
                    stream=True,
                )
            ),
            http_request=_direct_request(),
            db=object(),
            service_cls=ForbiddenService,
            principal=AuthPrincipal(kind="user", user_id=1),
        )
    except HTTPException as exc:
        captured = exc
    else:
        if response.background is not None:
            await response.background()
        raise AssertionError("Missing credentials must fail before stream dispatch")

    assert captured.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert captured.detail["error_code"] == "missing_provider_credentials"
    assert dispatched is False
    assert "mark_used" not in lifecycle
    assert lifecycle[-1] == "runtime_close"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory_result", "expected_marks"),
    [
        ({"success": False, "error": "local generation failed"}, 0),
        ({"error": {"message": "terminal provider failure"}}, 0),
        ("", 0),
        ("Error: terminal provider failure", 0),
        ("provider_unavailable", 0),
        ('data: {"error":{"code":"provider_unavailable"}}\n\n', 0),
        ('{"error":{"code":"provider_unavailable"}}', 0),
        ("valid generated document", 1),
    ],
    ids=[
        "failure-dict",
        "terminal-error",
        "empty",
        "error-prefix",
        "canonical-code",
        "sse-error",
        "serialized-error",
        "valid",
    ],
)
async def test_document_cancelled_factory_marks_only_semantic_success(
    monkeypatch,
    factory_result,
    expected_marks,
):
    """Late mechanical completion cannot count an invalid document as provider use."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    lifecycle = []
    entered = threading.Event()
    release = threading.Event()

    class ResultService:
        def __init__(self, _db):
            return None

        def generate_document(self, **_kwargs):
            entered.set()
            assert release.wait(timeout=3.0)
            lifecycle.append("factory_exit")
            return factory_result

    _install_direct_document_runtime(monkeypatch, lifecycle)
    task = asyncio.create_task(
        chat_docs.generate_document(
            request=GenerateDocumentRequest(
                **_make_payload(provider="ollama", model="model-a", api_key=None)
            ),
            http_request=_direct_request(),
            db=object(),
            service_cls=ResultService,
            principal=AuthPrincipal(kind="user", user_id=1),
        )
    )
    try:
        await _wait_for_thread_event(entered)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert lifecycle.count("mark_used") == expected_marks
    assert lifecycle.index("factory_exit") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
async def test_document_provider_exception_never_leaks_to_response_or_logs(
    monkeypatch,
):
    """Raw adapter failures stay outside both the HTTP and observability boundary."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    sentinel = "document-provider-secret-/srv/private/raw-body"
    lifecycle = []

    class FailingService:
        def __init__(self, _db):
            return None

        def generate_document(self, **_kwargs):
            raise RuntimeError(sentinel)

    _install_direct_document_runtime(monkeypatch, lifecycle)
    logs = []
    sink_id = logger.add(logs.append, format="{message}|{exception}")
    try:
        with pytest.raises(HTTPException) as captured:
            await chat_docs.generate_document(
                request=GenerateDocumentRequest(
                    **_make_payload(provider="ollama", model="model-a", api_key=None)
                ),
                http_request=_direct_request(),
                db=object(),
                service_cls=FailingService,
                principal=AuthPrincipal(kind="user", user_id=1),
            )
    finally:
        logger.remove(sink_id)

    assert captured.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert sentinel not in str(captured.value.detail)
    assert sentinel not in "".join(logs)
    assert lifecycle[-1] == "runtime_close"


@pytest.mark.asyncio
async def test_document_sync_stream_disconnect_drains_next_then_marks_and_closes_in_order(
    monkeypatch,
):
    """A disconnect drains sync next() before upstream and credential cleanup."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    lifecycle = []
    entered_second = threading.Event()
    release_second = threading.Event()

    class BlockingStream:
        def __init__(self):
            self.calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.calls += 1
            if self.calls == 1:
                return "first"
            entered_second.set()
            release_second.wait(timeout=5)
            lifecycle.append("next_exit")
            return "second"

        def close(self):
            lifecycle.append("upstream_close")

    class StreamingService:
        def __init__(self, _db):
            return None

        def generate_document(self, **_kwargs):
            return BlockingStream()

        def record_streamed_document(self, **_kwargs):
            lifecycle.append("persist")

    _install_direct_document_runtime(monkeypatch, lifecycle)
    monkeypatch.delenv("STREAMS_UNIFIED", raising=False)
    response = await chat_docs.generate_document(
        request=GenerateDocumentRequest(
            **_make_payload(provider="ollama", model="model-a", api_key=None, stream=True)
        ),
        http_request=_direct_request(),
        db=object(),
        service_cls=StreamingService,
        principal=AuthPrincipal(kind="user", user_id=1),
    )

    first = await response.body_iterator.__anext__()
    assert "first" in str(first)
    consume = asyncio.create_task(response.body_iterator.__anext__())
    assert await asyncio.to_thread(entered_second.wait, 2)
    consume.cancel()
    await asyncio.sleep(0)
    assert "runtime_close" not in lifecycle

    release_second.set()
    with pytest.raises(asyncio.CancelledError):
        await consume

    assert lifecycle.index("next_exit") < lifecycle.index("mark_used")
    assert lifecycle.index("mark_used") < lifecycle.index("upstream_close")
    assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
async def test_document_sync_stream_disconnect_retains_distinct_late_iterator_for_close(
    monkeypatch,
):
    """Cancellation during iter() retains its distinct result through cleanup."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    lifecycle = []
    iter_entered = threading.Event()
    iter_release = threading.Event()
    pool = BoundedDaemonPool(1)

    class DistinctIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise AssertionError("cancelled iterator factory result must not be consumed")

        def close(self):
            lifecycle.append("iterator_close")

    iterator = DistinctIterator()

    class BlockingIterable:
        def __iter__(self):
            lifecycle.append("iter_start")
            iter_entered.set()
            assert iter_release.wait(timeout=3.0)
            lifecycle.append("iter_exit")
            return iterator

    class StreamingService:
        def __init__(self, _db):
            return None

        def generate_document(self, **_kwargs):
            return BlockingIterable()

        def record_streamed_document(self, **_kwargs):
            lifecycle.append("persist")

    _install_direct_document_runtime(monkeypatch, lifecycle)
    monkeypatch.delenv("STREAMS_UNIFIED", raising=False)
    monkeypatch.setattr(chat_docs, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    response = await chat_docs.generate_document(
        request=GenerateDocumentRequest(
            **_make_payload(provider="ollama", model="model-a", api_key=None, stream=True)
        ),
        http_request=_direct_request(),
        db=object(),
        service_cls=StreamingService,
        principal=AuthPrincipal(kind="user", user_id=1),
    )

    consume = asyncio.create_task(response.body_iterator.__anext__())
    try:
        await _wait_for_thread_event(iter_entered)
        consume.cancel()
        await asyncio.sleep(0.03)
        assert not consume.done()
        assert "runtime_close" not in lifecycle
        iter_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consume, timeout=1.0)
    finally:
        iter_release.set()
        if not consume.done():
            consume.cancel()
        await asyncio.gather(consume, return_exceptions=True)

    assert lifecycle.index("iter_exit") < lifecycle.index("iterator_close")
    assert lifecycle.index("iterator_close") < lifecycle.index("runtime_close")
    assert lifecycle.count("iterator_close") == 1
    assert "persist" not in lifecycle
    assert "mark_used" not in lifecycle
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_document_sync_stream_uses_bounded_next_and_reserved_close_when_executor_saturated(
    monkeypatch,
):
    """Disconnect drains direct next/close workers without the loop executor."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs
    from tldw_Server_API.app.core.Chat import bounded_daemon as bounded_daemon_module

    lifecycle = []
    default_entered = threading.Event()
    default_release = threading.Event()
    next_entered = threading.Event()
    next_release = threading.Event()
    close_entered = threading.Event()
    close_release = threading.Event()
    sync_pool = _ReleaseTrackingDaemonPool(1, lifecycle, "sync_capacity_release")
    cleanup_pool = _ReleaseTrackingDaemonPool(1, lifecycle, "cleanup_capacity_release")
    loop = asyncio.get_running_loop()
    previous_executor = getattr(loop, "_default_executor", None)
    saturated_executor = ThreadPoolExecutor(max_workers=1)
    consume = None

    def block_default_executor() -> None:
        default_entered.set()
        assert default_release.wait(timeout=3.0)

    class BlockingStream:
        def __iter__(self):
            lifecycle.append("iter_start")
            lifecycle.append("iter_exit")
            return self

        def __next__(self):
            lifecycle.append("next_start")
            next_entered.set()
            assert next_release.wait(timeout=3.0)
            lifecycle.append("next_exit")
            return "late valid document chunk"

        def close(self):
            lifecycle.append("close_start")
            close_entered.set()
            assert close_release.wait(timeout=3.0)
            lifecycle.append("close_exit")

    class StreamingService:
        def __init__(self, _db):
            return None

        def generate_document(self, **_kwargs):
            return BlockingStream()

        def record_streamed_document(self, **_kwargs):
            lifecycle.append("persist")

    _install_direct_document_runtime(monkeypatch, lifecycle)
    monkeypatch.delenv("STREAMS_UNIFIED", raising=False)
    monkeypatch.setattr(chat_docs, "SYNC_ADAPTER_CALL_POOL", sync_pool, raising=False)
    monkeypatch.setattr(
        bounded_daemon_module,
        "STREAM_CLEANUP_DAEMON_POOL",
        cleanup_pool,
    )
    response = await chat_docs.generate_document(
        request=GenerateDocumentRequest(
            **_make_payload(provider="ollama", model="model-a", api_key=None, stream=True)
        ),
        http_request=_direct_request(),
        db=object(),
        service_cls=StreamingService,
        principal=AuthPrincipal(kind="user", user_id=1),
    )
    factory_release_count = sync_pool.release_count

    loop.set_default_executor(saturated_executor)
    default_blocker = loop.run_in_executor(None, block_default_executor)
    try:
        await _wait_for_thread_event(default_entered)
        consume = asyncio.create_task(response.body_iterator.__anext__())
        await _wait_for_thread_event(next_entered)
        assert not default_release.is_set()
        assert sync_pool.active_count == 1

        consume.cancel()
        await asyncio.sleep(0.03)
        assert not consume.done()
        next_release.set()

        await _wait_for_thread_event(close_entered)
        assert not default_release.is_set()
        assert not consume.done()
        assert "runtime_close" not in lifecycle
        close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consume, timeout=1.0)
    finally:
        next_release.set()
        close_release.set()
        default_release.set()
        await asyncio.gather(default_blocker, return_exceptions=True)
        if consume is not None and not consume.done():
            consume.cancel()
        if consume is not None:
            await asyncio.gather(consume, return_exceptions=True)
        replacement_executor = previous_executor or ThreadPoolExecutor()
        loop.set_default_executor(replacement_executor)
        saturated_executor.shutdown(wait=True, cancel_futures=True)

    next_release_index = max(
        index for index, value in enumerate(lifecycle) if value == "sync_capacity_release"
    )
    assert lifecycle.index("next_exit") < next_release_index
    assert next_release_index < lifecycle.index("mark_used")
    assert lifecycle.index("mark_used") < lifecycle.index("close_start")
    assert lifecycle.index("close_exit") < lifecycle.index("cleanup_capacity_release")
    assert lifecycle.index("cleanup_capacity_release") < lifecycle.index("runtime_close")
    assert "persist" not in lifecycle
    assert sync_pool.release_count - factory_release_count == 2
    assert cleanup_pool.release_count == 1
    assert sync_pool.active_count == 0
    assert cleanup_pool.active_count == 0
    assert lifecycle.count("mark_used") == 1
    assert lifecycle.count("close_start") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream_mode", "iterator_kind"),
    [("classic", "sync"), ("unified", "async")],
)
async def test_document_terminal_stream_error_is_canonical_and_never_persisted(
    monkeypatch,
    stream_mode,
    iterator_kind,
):
    """Both stream implementations stop at bounded terminal provider errors."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    sentinel = f"document-{stream_mode}-{iterator_kind}-secret-/srv/provider"
    lifecycle = []
    persisted = []

    class SyncErrorStream:
        def __init__(self):
            self.calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.calls += 1
            if self.calls == 1:
                return "partial valid document"
            if self.calls > 2:
                raise StopIteration
            return {"error": {"message": sentinel}}

        def close(self):
            lifecycle.append("upstream_close")

    class AsyncErrorStream:
        def __init__(self):
            self.calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.calls += 1
            if self.calls == 1:
                return "partial valid document"
            if self.calls > 2:
                raise StopAsyncIteration
            return {"error": {"message": sentinel}}

        async def aclose(self):
            lifecycle.append("upstream_close")

    class StreamingService:
        def __init__(self, _db):
            return None

        def generate_document(self, **_kwargs):
            return SyncErrorStream() if iterator_kind == "sync" else AsyncErrorStream()

        def record_streamed_document(self, **kwargs):
            persisted.append(kwargs["content"])

    _install_direct_document_runtime(monkeypatch, lifecycle)
    if stream_mode == "unified":
        monkeypatch.setenv("STREAMS_UNIFIED", "1")
    else:
        monkeypatch.delenv("STREAMS_UNIFIED", raising=False)

    response = await chat_docs.generate_document(
        request=GenerateDocumentRequest(
            **_make_payload(provider="ollama", model="model-a", api_key=None, stream=True)
        ),
        http_request=_direct_request(),
        db=object(),
        service_cls=StreamingService,
        principal=AuthPrincipal(kind="user", user_id=1),
    )
    wire_chunks = [
        chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        async for chunk in response.body_iterator
    ]
    wire = "".join(wire_chunks)

    assert "provider_unavailable" in wire
    assert sentinel not in wire
    assert persisted == []
    assert "mark_used" not in lifecycle
    assert lifecycle.count("upstream_close") == 1
    assert lifecycle[-1] == "runtime_close"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream_mode", "iterator_kind"),
    [("classic", "sync"), ("unified", "async")],
)
@pytest.mark.parametrize("error_kind", ["chat_api", "unexpected"])
async def test_document_raised_stream_error_never_persists_or_marks_partial_output(
    monkeypatch,
    stream_mode,
    iterator_kind,
    error_kind,
):
    """Raised terminal failures invalidate partial output in both stream modes."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    sentinel = f"raised-{stream_mode}-{iterator_kind}-{error_kind}-secret-/srv/provider"
    lifecycle = []
    persisted = []

    def terminal_error():
        if error_kind == "chat_api":
            return chat_docs.ChatAPIError(sentinel)
        return RuntimeError(sentinel)

    class SyncRaisingStream:
        def __init__(self):
            self.calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.calls += 1
            if self.calls == 1:
                return "partial valid document"
            raise terminal_error()

        def close(self):
            lifecycle.append("upstream_close")

    class AsyncRaisingStream:
        def __init__(self):
            self.calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.calls += 1
            if self.calls == 1:
                return "partial valid document"
            raise terminal_error()

        async def aclose(self):
            lifecycle.append("upstream_close")

    class StreamingService:
        def __init__(self, _db):
            return None

        def generate_document(self, **_kwargs):
            return SyncRaisingStream() if iterator_kind == "sync" else AsyncRaisingStream()

        def record_streamed_document(self, **kwargs):
            persisted.append(kwargs["content"])

    _install_direct_document_runtime(monkeypatch, lifecycle)
    if stream_mode == "unified":
        monkeypatch.setenv("STREAMS_UNIFIED", "1")
    else:
        monkeypatch.delenv("STREAMS_UNIFIED", raising=False)

    response = await chat_docs.generate_document(
        request=GenerateDocumentRequest(
            **_make_payload(provider="ollama", model="model-a", api_key=None, stream=True)
        ),
        http_request=_direct_request(),
        db=object(),
        service_cls=StreamingService,
        principal=AuthPrincipal(kind="user", user_id=1),
    )
    wire_chunks = [
        chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        async for chunk in response.body_iterator
    ]
    wire = "".join(wire_chunks)

    expected_public_error = (
        "Chat provider error" if error_kind == "chat_api" else "internal error"
    )
    assert expected_public_error.lower() in wire.lower()
    assert sentinel not in wire
    assert persisted == []
    assert "mark_used" not in lifecycle
    assert lifecycle.count("upstream_close") == 1
    assert lifecycle[-1] == "runtime_close"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["async", "sync"])
@pytest.mark.parametrize(
    ("late_kind", "expected_success"),
    [
        ("valid", True),
        ("empty", False),
        ("terminal_error", False),
        ("clean_exhaustion", False),
        ("done_control", False),
    ],
)
async def test_document_cancelled_next_classifies_late_result_semantically(
    monkeypatch,
    stream_kind,
    late_kind,
    expected_success,
):
    """Async and sync late next results share one fail-closed success contract."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    entered = threading.Event()
    release = threading.Event()
    success_state = {"successful": False}
    resource_holder = {}
    pool = BoundedDaemonPool(2)
    monkeypatch.setattr(chat_docs, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)

    def late_value():
        if late_kind == "clean_exhaustion":
            raise StopIteration
        if late_kind == "terminal_error":
            return {"error": {"message": "provider failed"}}
        if late_kind == "empty":
            return ""
        if late_kind == "done_control":
            return "[DONE]" if stream_kind == "async" else "data: [DONE]"
        return "late valid document chunk"

    class SyncStream:
        def __iter__(self):
            return self

        def __next__(self):
            entered.set()
            assert release.wait(timeout=3.0)
            return late_value()

    class AsyncStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            if late_kind == "clean_exhaustion":
                raise StopAsyncIteration
            return late_value()

    source = AsyncStream() if stream_kind == "async" else SyncStream()
    iterator = chat_docs._iterate_document_stream(
        source,
        resource_holder=resource_holder,
        success_state=success_state,
    )
    task = asyncio.create_task(iterator.__anext__())
    try:
        await _wait_for_thread_event(entered)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await iterator.aclose()

    assert success_state["successful"] is expected_success
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_auth_source", "expected_status"),
    [("aws_default_chain", 200), (None, 503)],
    ids=["bedrock-default-chain", "bedrock-explicit-absent"],
)
async def test_document_bedrock_auth_contract_distinguishes_default_chain_from_absent(
    monkeypatch,
    runtime_auth_source,
    expected_status,
):
    """Bedrock default-chain auth is valid; an explicit absent snapshot fails closed."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    lifecycle = []

    class BedrockRuntime:
        def __init__(self, **_kwargs):
            return None

        async def resolve(self, provider, *, model=None):
            lifecycle.append(("resolve", provider, model))
            config = {}
            if runtime_auth_source is not None:
                config["_runtime_auth_source"] = runtime_auth_source
            return SimpleNamespace(
                provider=provider,
                api_key=None,
                app_config={"bedrock_api": config},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle):
            lifecycle.append("mark_used")

        async def close(self):
            lifecycle.append("runtime_close")

    class BedrockService:
        def __init__(self, _db):
            return None

        def generate_document(self, **_kwargs):
            lifecycle.append("adapter_call")
            return "generated"

        def get_generated_documents(self, conversation_id=None, document_type=None, **_kwargs):
            return [
                {
                    "id": 1,
                    "conversation_id": conversation_id,
                    "document_type": document_type.value,
                    "title": "Doc",
                    "content": "generated",
                    "provider": "bedrock",
                    "model": "model-a",
                    "generation_time_ms": 1,
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                }
            ]

    monkeypatch.setattr(chat_docs, "ProviderCredentialRuntime", BedrockRuntime)
    monkeypatch.setattr(
        chat_docs,
        "derive_trusted_credential_scope",
        lambda _request, _principal: (1, [], [], False),
    )

    request = GenerateDocumentRequest(
        **_make_payload(provider="bedrock", model="model-a", api_key=None, stream=False)
    )
    if expected_status == 200:
        response = await chat_docs.generate_document(
            request=request,
            http_request=_direct_request(),
            db=object(),
            service_cls=BedrockService,
            principal=AuthPrincipal(kind="user", user_id=1),
        )
        assert response.content == "generated"
        assert lifecycle[-2:] == ["mark_used", "runtime_close"]
    else:
        with pytest.raises(HTTPException) as exc_info:
            await chat_docs.generate_document(
                request=request,
                http_request=_direct_request(),
                db=object(),
                service_cls=BedrockService,
                principal=AuthPrincipal(kind="user", user_id=1),
            )
        assert getattr(exc_info.value, "status_code", None) == expected_status
        assert "adapter_call" not in lifecycle
        assert lifecycle[-1] == "runtime_close"


def test_document_generate_streams_as_sse(authenticated_client, auth_token):


    calls = {}

    class StreamingStubService:
        stored_docs: list[dict] = []
        next_id: int = 1

        def __init__(self, db):

            self._db = db

        def generate_document(self, *, stream, **kwargs):

            calls["stream"] = stream

            async def _generator():
                yield "first chunk"
                yield b"second chunk"

            return _generator()

        def record_streamed_document(
            self,
            *,
            conversation_id,
            document_type,
            content,
            provider,
            model,
            generation_time_ms,
            token_count=None,
        ):

            doc_id = StreamingStubService.next_id
            StreamingStubService.next_id += 1
            StreamingStubService.stored_docs.append(
                {
                    "id": doc_id,
                    "conversation_id": conversation_id,
                    "document_type": document_type.value if hasattr(document_type, "value") else document_type,
                    "title": "Streamed Document",
                    "content": content,
                    "provider": provider,
                    "model": model,
                    "generation_time_ms": generation_time_ms,
                    "token_count": token_count,
                    "created_at": datetime.datetime.utcnow(),
                    "metadata": {},
                }
            )
            return doc_id

        def get_generated_documents(self, conversation_id=None, document_type=None, limit=50, offset=0):

            docs = list(StreamingStubService.stored_docs)
            if conversation_id is not None:
                docs = [doc for doc in docs if doc["conversation_id"] == conversation_id]
            if document_type is not None:
                dtype = document_type.value if hasattr(document_type, "value") else document_type
                docs = [doc for doc in docs if doc["document_type"] == dtype]
            docs.sort(key=lambda item: item["id"], reverse=True)
            return docs[offset:offset + limit]

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: StreamingStubService
    StreamingStubService.stored_docs = []
    StreamingStubService.next_id = 1

    response = authenticated_client.post(
        "/api/v1/chat/documents/generate",
        json=_make_payload(stream=True),
    )

    assert response.status_code == 200
    assert calls["stream"] is True
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    assert "data: first chunk\n\n" in body
    assert "data: second chunk\n\n" in body
    assert body.strip().endswith("data: [DONE]")
    response.close()

    headers = get_auth_headers(auth_token, getattr(authenticated_client, "csrf_token", ""))
    list_response = authenticated_client.get(
        "/api/v1/chat/documents",
        params={"conversation_id": "chat-42"},
        headers=headers,
    )
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["total"] == 1
    assert payload["documents"][0]["content"] == "first chunksecond chunk"
    assert StreamingStubService.stored_docs, "Streamed document was not persisted"


def test_document_generate_bubbles_service_error(authenticated_client):


    class FailingStubService:
        record_calls = 0

        def __init__(self, db):

            self._db = db

        def generate_document(self, *args, **kwargs):

            return {"success": False, "error": "No messages found for conversation chat-42"}

        def get_generated_documents(self, *args, **kwargs):

            return []

        def record_streamed_document(self, *args, **kwargs):

            FailingStubService.record_calls += 1
            return None

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: FailingStubService
    FailingStubService.record_calls = 0

    response = authenticated_client.post(
        "/api/v1/chat/documents/generate",
        json=_make_payload(),
    )

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "No messages found for conversation chat-42"}
    response.close()
    assert FailingStubService.record_calls == 0


def test_document_generate_maps_input_error_from_service(authenticated_client):


    class InputErrorStubService:
        def __init__(self, db):

            self._db = db

        def generate_document(self, *args, **kwargs):

            raise InputError("No messages found for conversation chat-42")

        def get_generated_documents(self, *args, **kwargs):

            return []

        def record_streamed_document(self, *args, **kwargs):

            return None

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: InputErrorStubService

    response = authenticated_client.post(
        "/api/v1/chat/documents/generate",
        json=_make_payload(),
    )

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "No messages found for conversation chat-42"}
    response.close()


def test_document_list_maps_database_error_from_service(authenticated_client):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def get_generated_documents(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite list exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.get("/api/v1/chat/documents")

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to list generated documents"}
    response.close()


def test_document_list_includes_canonical_pagination(authenticated_client) -> None:
    """Generated document listing preserves total while exposing canonical offset pagination."""
    calls = {"list": [], "count": []}

    class PaginatedStubService:
        def __init__(self, db):
            self._db = db

        def get_generated_documents(self, conversation_id=None, document_type=None, limit=50, offset=0):
            calls["list"].append(
                {
                    "conversation_id": conversation_id,
                    "document_type": document_type,
                    "limit": limit,
                    "offset": offset,
                }
            )
            docs = [
                {
                    "id": 12,
                    "conversation_id": conversation_id,
                    "document_type": "summary",
                    "title": "Newest Doc",
                    "content": "newer",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "generation_time_ms": 10,
                    "token_count": 3,
                    "created_at": datetime.datetime.utcnow(),
                    "metadata": {},
                },
                {
                    "id": 11,
                    "conversation_id": conversation_id,
                    "document_type": "summary",
                    "title": "Older Doc",
                    "content": "older",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "generation_time_ms": 12,
                    "token_count": 4,
                    "created_at": datetime.datetime.utcnow(),
                    "metadata": {},
                },
            ]
            return docs[offset:offset + limit]

        def count_generated_documents(self, conversation_id=None, document_type=None):
            calls["count"].append(
                {
                    "conversation_id": conversation_id,
                    "document_type": document_type,
                }
            )
            return 4

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: PaginatedStubService

    response = authenticated_client.get(
        "/api/v1/chat/documents",
        params={"conversation_id": "chat-42", "document_type": "summary", "limit": 1, "offset": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [doc["id"] for doc in payload["documents"]] == [11]
    assert payload["total"] == 4
    assert payload["conversation_id"] == "chat-42"
    assert payload["document_type"] == "summary"
    assert payload["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": 1,
        "total": 4,
        "has_more": True,
        "next_offset": 2,
    }
    assert calls["list"] == [
        {
            "conversation_id": "chat-42",
            "document_type": DocumentType.SUMMARY,
            "limit": 1,
            "offset": 1,
        }
    ]
    assert calls["count"] == [
        {
            "conversation_id": "chat-42",
            "document_type": DocumentType.SUMMARY,
        }
    ]
    response.close()


def test_document_get_maps_database_error_from_service(authenticated_client):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def get_generated_document_by_id(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite get exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.get("/api/v1/chat/documents/123")

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to get generated document"}
    response.close()


def test_document_job_status_maps_database_error_from_service(authenticated_client):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def get_job_status(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite job exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.get("/api/v1/chat/documents/jobs/job-1")

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to get generation job status"}
    response.close()


def test_document_cancel_maps_database_error_from_service(authenticated_client, auth_token):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def get_job_status(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite cancel exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.delete(
        "/api/v1/chat/documents/jobs/job-1",
        headers=get_auth_headers(auth_token, getattr(authenticated_client, "csrf_token", "")),
    )

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to cancel generation job"}
    response.close()


def test_document_delete_maps_database_error_from_service(authenticated_client, auth_token):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def delete_generated_document(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite delete exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.delete(
        "/api/v1/chat/documents/123",
        headers=get_auth_headers(auth_token, getattr(authenticated_client, "csrf_token", "")),
    )

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to delete generated document"}
    response.close()


def test_document_save_prompt_maps_database_error_from_service(authenticated_client):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def save_user_prompt_config(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite prompt exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.post(
        "/api/v1/chat/documents/prompts",
        json={
            "document_type": "summary",
            "system_prompt": "Summarize.",
            "user_prompt": "Content: {content}",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
    )

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to save prompt configuration"}
    response.close()


def test_document_bulk_maps_database_error_from_service(authenticated_client):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def create_generation_job(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite bulk exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.post(
        "/api/v1/chat/documents/bulk",
        json={
            "conversation_ids": ["chat-42"],
            "document_types": ["summary"],
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-test",
            "async_generation": True,
        },
    )

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to create bulk generation jobs"}
    response.close()


@pytest.mark.parametrize(
    ("case_name", "service_method", "request_factory", "expected_detail"),
    [
        (
            "job_status",
            "get_job_status",
            lambda client, token: client.get("/api/v1/chat/documents/jobs/job-1"),
            "Failed to get generation job status",
        ),
        (
            "cancel_job",
            "get_job_status",
            lambda client, token: client.delete(
                "/api/v1/chat/documents/jobs/job-1",
                headers=get_auth_headers(token, getattr(client, "csrf_token", "")),
            ),
            "Failed to cancel generation job",
        ),
        (
            "list_documents",
            "get_generated_documents",
            lambda client, token: client.get("/api/v1/chat/documents"),
            "Failed to list generated documents",
        ),
        (
            "get_document",
            "get_generated_document_by_id",
            lambda client, token: client.get("/api/v1/chat/documents/123"),
            "Failed to get generated document",
        ),
        (
            "delete_document",
            "delete_generated_document",
            lambda client, token: client.delete(
                "/api/v1/chat/documents/123",
                headers=get_auth_headers(token, getattr(client, "csrf_token", "")),
            ),
            "Failed to delete generated document",
        ),
        (
            "save_prompt",
            "save_user_prompt_config",
            lambda client, token: client.post(
                "/api/v1/chat/documents/prompts",
                json={
                    "document_type": "summary",
                    "system_prompt": "Summarize.",
                    "user_prompt": "Content: {content}",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
            ),
            "Failed to save prompt configuration",
        ),
        (
            "get_prompt",
            "get_user_prompt_config",
            lambda client, token: client.get("/api/v1/chat/documents/prompts/summary"),
            "Failed to get prompt configuration",
        ),
        (
            "bulk_generate",
            "create_generation_job",
            lambda client, token: client.post(
                "/api/v1/chat/documents/bulk",
                json={
                    "conversation_ids": ["chat-42"],
                    "document_types": ["summary"],
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-test",
                    "async_generation": True,
                },
            ),
            "Failed to create bulk generation jobs",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_document_handlers_sanitize_unexpected_service_errors(
    authenticated_client,
    auth_token,
    case_name,
    service_method,
    request_factory,
    expected_detail,
):
    def _raise_unexpected_error(self, *args, **kwargs):
        _ = (self, args, kwargs)
        raise RuntimeError(f"{case_name} backend unavailable")

    RuntimeErrorStubService = type(
        "RuntimeErrorStubService",
        (),
        {
            "__init__": lambda self, db: setattr(self, "_db", db),
            service_method: _raise_unexpected_error,
        },
    )

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: RuntimeErrorStubService

    response = request_factory(authenticated_client, auth_token)

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": expected_detail}
    response.close()


def test_document_generate_uses_configured_api_key(monkeypatch, authenticated_client):


    captured = {}

    class KeyCaptureService:
        def __init__(self, db):
            self._db = db

        def generate_document(self, *, stream, **kwargs):

            captured["api_key"] = kwargs.get("api_key")
            captured["provider"] = kwargs.get("provider")
            return "Generated content"

        def get_generated_documents(self, conversation_id=None, document_type=None, limit=50, offset=0):

            return [
                {
                    "id": 101,
                    "conversation_id": conversation_id,
                    "document_type": document_type.value if hasattr(document_type, "value") else document_type,
                    "title": "Doc",
                    "content": "Generated content",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "generation_time_ms": 123,
                    "created_at": datetime.datetime.utcnow(),
                }
            ]

    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: KeyCaptureService
    monkeypatch.setattr(
        provider_credential_runtime,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"api_key": "sk-configured"}},
    )

    payload = _make_payload()
    payload.pop("api_key", None)

    response = authenticated_client.post(
        "/api/v1/chat/documents/generate",
        json=payload,
    )

    assert response.status_code == 200, response.text
    assert captured["api_key"] == "sk-configured"
    assert captured["provider"] == "openai"


@pytest.mark.parametrize("captured_key", ["document-key-a", None], ids=["a-to-b", "absent-to-b"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_document_generate_keeps_static_snapshot_at_service_boundary(
    monkeypatch,
    authenticated_client,
    captured_key,
    stream,
):
    """The endpoint must send one structured fallback snapshot to its service."""
    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    config_a = {"ollama": {"model": "model-a", "base_url": "http://a.invalid"}}
    captured = {}
    lifecycle = []
    handles = []

    class SnapshotService:
        def __init__(self, _db):
            return None

        def generate_document(self, *, stream, **kwargs):
            captured.update(kwargs)
            return "Generated content"

        def get_generated_documents(self, conversation_id=None, document_type=None, **_kwargs):
            return [
                {
                    "id": 101,
                    "conversation_id": conversation_id,
                    "document_type": document_type.value,
                    "title": "Doc",
                    "content": "Generated content",
                    "provider": "ollama",
                    "model": "model-a",
                    "generation_time_ms": 1,
                    "created_at": datetime.datetime.utcnow(),
                }
            ]

        def record_streamed_document(self, **_kwargs):
            return 101

    class FakeRuntime:
        def __init__(self, **kwargs):
            lifecycle.append(("init", kwargs))

        async def resolve(self, provider, *, model=None):
            lifecycle.append(("resolve", provider, model))
            handle = type(
                "Handle",
                (),
                {
                    "api_key": captured_key,
                    "app_config": config_a,
                    "credentials_resolved": True,
                },
            )()
            handles.append(handle)
            return handle

        async def mark_used(self, _handle):
            lifecycle.append("mark_used")

        async def close(self):
            lifecycle.append("close")

    async def forbidden_low_level_resolver(*_args, **_kwargs):
        raise AssertionError("document generation bypassed ProviderCredentialRuntime")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: SnapshotService
    monkeypatch.setattr(chat_docs, "ProviderCredentialRuntime", FakeRuntime)
    monkeypatch.setattr(
        chat_docs,
        "derive_trusted_credential_scope",
        lambda _request, _principal: (1, [2], [3], True),
    )
    monkeypatch.setattr(
        chat_docs,
        "resolve_byok_credentials",
        forbidden_low_level_resolver,
        raising=False,
    )
    monkeypatch.setattr(
        chat_docs,
        "resolve_provider_api_key",
        lambda *_args, **_kwargs: ("document-key-b", {}),
        raising=False,
    )

    response = authenticated_client.post(
        "/api/v1/chat/documents/generate",
        json=_make_payload(provider="ollama", model="model-a", api_key=None, stream=stream),
    )

    assert response.status_code == 200, response.text
    assert captured["api_key"] == (captured_key or "")
    assert captured["app_config"] == config_a
    assert captured["credentials_resolved"] is True
    assert captured["provider_credentials"] is handles[0]
    init_kwargs = lifecycle[0][1]
    assert init_kwargs["user_id"] == 1
    assert init_kwargs["team_ids"] == [2]
    assert init_kwargs["org_ids"] == [3]
    assert init_kwargs["trusted_base_url_override"] is True
    assert lifecycle[1:] == [
        ("resolve", "ollama", "model-a"),
        "mark_used",
        "close",
    ]
    response.close()


def test_document_generate_missing_provider_credentials_returns_503(monkeypatch, authenticated_client):


    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs
    class MissingRuntime:
        def __init__(self, **_kwargs):
            return None

        async def resolve(self, _provider, *, model=None):
            return type(
                "Handle",
                (),
                {"api_key": None, "app_config": None, "credentials_resolved": True},
            )()

        async def close(self):
            return None

    monkeypatch.setattr(chat_docs, "ProviderCredentialRuntime", MissingRuntime)

    payload = _make_payload()
    payload.pop("api_key", None)

    response = authenticated_client.post(
        "/api/v1/chat/documents/generate",
        json=payload,
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    detail = response.json().get("detail", {})
    assert detail.get("error_code") == "missing_provider_credentials"
