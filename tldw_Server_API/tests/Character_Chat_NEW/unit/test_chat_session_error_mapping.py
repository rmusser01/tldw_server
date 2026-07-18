"""Direct route tests for chat session DB error mapping."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import (
    CharacterChatCompletionV2Request,
    CharacterChatStreamPersistRequest,
    ChatSessionUpdate,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError
from tldw_Server_API.app.core.Chat.prompt_cost_guardrails import PromptCostGuardrailConfig
from tldw_Server_API.app.core.LLM_Calls.routing.models import RoutingDecision


pytestmark = pytest.mark.unit


def _test_user() -> User:
    return User(id=1, username="tester", email="tester@example.com", is_active=True)


def _conversation(*, deleted: bool = False) -> dict[str, Any]:
    return {
        "id": "chat-1",
        "client_id": "1",
        "character_id": 7,
        "title": "Test Chat",
        "version": 1,
        "deleted": deleted,
        "scope_type": "global",
        "workspace_id": None,
        "created_at": "2026-01-01T00:00:00Z",
        "last_modified": "2026-01-01T00:00:00Z",
    }


def test_complete_v2_request_accepts_inference_prefix_cache_intent() -> None:
    body = CharacterChatCompletionV2Request(
        inference_prefix_cache_intent={
            "enabled": True,
            "scope": ["world_books"],
        },
    )

    assert body.inference_prefix_cache_intent == {
        "enabled": True,
        "scope": ["world_books"],
    }


class _BrokenChatSessionDb:
    def __init__(self, exc: Exception, *, deleted: bool = False, raise_on_get: bool = False) -> None:
        self.exc = exc
        self.deleted = deleted
        self.raise_on_get = raise_on_get

    def get_conversation_by_id(self, chat_id: str, include_deleted: bool = False) -> dict[str, Any]:
        if self.raise_on_get:
            raise self.exc
        return _conversation(deleted=self.deleted)

    def update_conversation(self, *args: Any, **kwargs: Any) -> None:
        raise self.exc

    def get_messages_for_conversation(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def soft_delete_conversation(self, *args: Any, **kwargs: Any) -> None:
        raise self.exc

    def restore_conversation(self, *args: Any, **kwargs: Any) -> None:
        raise self.exc

    def count_messages_for_conversation(self, *args: Any, **kwargs: Any) -> int:
        return 0


class _CompletionReadyChatSessionDb:
    def get_conversation_by_id(self, chat_id: str, include_deleted: bool = False) -> dict[str, Any]:
        return _conversation()

    def get_conversation_settings(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def get_messages_for_conversation(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"sender": "user", "content": "hello", "deleted": False}]

    def get_character_card_by_id(self, character_id: int) -> dict[str, Any]:
        return {"id": character_id, "name": "Assistant", "content": ""}

    def count_messages_for_conversation(self, *args: Any, **kwargs: Any) -> int:
        return 1

    def list_persona_memory_entries(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []


class _NoopCharacterRateLimiter:
    async def check_soft_message_limit(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def check_message_limit(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def check_chat_completion_rate(self, *args: Any, **kwargs: Any) -> None:
        return None


class _ByokResolution:
    api_key = None
    app_config: dict[str, Any] = {}

    async def touch_last_used(self) -> None:
        return None


def _install_simple_credential_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a no-I/O runtime for legacy completion-route unit tests."""

    class SimpleCredentialRuntime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            del model
            return SimpleNamespace(
                provider=provider,
                api_key=None,
                app_config={},
                credentials_resolved=True,
            )

        async def mark_used(self, _credentials: Any) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        character_chat_sessions,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [], [], False),
    )
    monkeypatch.setattr(
        character_chat_sessions,
        "ProviderCredentialRuntime",
        SimpleCredentialRuntime,
    )


def _install_character_completion_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider_response: Any,
    lifecycle: list[Any],
) -> None:
    """Install a deterministic credential runtime and provider adapter."""

    class RecordingCredentialRuntime:
        def __init__(self, **kwargs: Any) -> None:
            lifecycle.append(
                (
                    "init",
                    kwargs["user_id"],
                    kwargs["team_ids"],
                    kwargs["org_ids"],
                    kwargs["trusted_base_url_override"],
                )
            )

        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            lifecycle.append(("resolve", provider, model))
            return SimpleNamespace(
                provider=provider,
                api_key="key-a",
                app_config={"local_llm_api": {"base_url": "http://generation-a.invalid"}},
                credentials_resolved=True,
            )

        async def mark_used(self, _credentials: Any) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")

    monkeypatch.setenv("ENABLE_LOCAL_LLM_PROVIDER", "true")
    monkeypatch.setenv("STREAMS_UNIFIED", "0")
    monkeypatch.setattr(
        character_chat_sessions,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [11], [22], True),
    )
    monkeypatch.setattr(
        character_chat_sessions,
        "ProviderCredentialRuntime",
        RecordingCredentialRuntime,
    )
    monkeypatch.setattr(
        character_chat_sessions,
        "provider_requires_api_key",
        lambda _provider: False,
    )
    monkeypatch.setattr(
        character_chat_sessions,
        "get_character_rate_limiter",
        lambda: _NoopCharacterRateLimiter(),
    )
    monkeypatch.setattr(
        character_chat_sessions,
        "_should_enforce_char_chat_strict_model_selection",
        lambda: False,
    )
    monkeypatch.setattr(
        character_chat_sessions,
        "perform_chat_api_call",
        lambda **_kwargs: provider_response,
    )


@pytest.mark.asyncio
async def test_complete_v2_sync_stream_keeps_credentials_until_blocked_next_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation must not release credentials while a sync adapter still runs."""
    lifecycle: list[Any] = []
    entered = threading.Event()
    release = threading.Event()

    class BlockingStream:
        def __iter__(self) -> BlockingStream:
            return self

        def __next__(self) -> str:
            entered.set()
            release.wait(timeout=5)
            lifecycle.append("next_exit")
            return "data: first"

        def close(self) -> None:
            lifecycle.append("upstream_close")

    _install_character_completion_runtime(
        monkeypatch,
        provider_response=BlockingStream(),
        lifecycle=lifecycle,
    )

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            stream=True,
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )

    consume = asyncio.create_task(response.body_iterator.__anext__())
    assert await asyncio.to_thread(entered.wait, 2)
    consume.cancel()
    await asyncio.sleep(0)
    assert "runtime_close" not in lifecycle

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await consume

    for _ in range(100):
        if "runtime_close" in lifecycle:
            break
        await asyncio.sleep(0.01)

    assert lifecycle.index("next_exit") < lifecycle.index("upstream_close")
    assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
@pytest.mark.parametrize("result_kind", ["nonstream", "lazy_stream"])
async def test_complete_v2_cancelled_sync_factory_hands_off_completed_result_before_close(
    monkeypatch: pytest.MonkeyPatch,
    result_kind: str,
) -> None:
    """A cancelled caller must not abandon a later successful sync factory result."""
    lifecycle: list[Any] = []
    entered = threading.Event()
    release = threading.Event()

    class UnconsumedStream:
        def __iter__(self) -> UnconsumedStream:
            return self

        def __next__(self) -> str:
            raise AssertionError("cancelled factory stream must not be consumed")

        def close(self) -> None:
            lifecycle.append("upstream_close")

    result: Any = (
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        if result_kind == "nonstream"
        else UnconsumedStream()
    )
    _install_character_completion_runtime(
        monkeypatch,
        provider_response=result,
        lifecycle=lifecycle,
    )

    def blocking_provider_call(**_kwargs: Any) -> Any:
        entered.set()
        release.wait(timeout=5)
        lifecycle.append("factory_exit")
        return result

    monkeypatch.setattr(
        character_chat_sessions,
        "perform_chat_api_call",
        blocking_provider_call,
    )

    request_task = asyncio.create_task(
        character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(
                provider="local-llm",
                model="local-test",
                stream=result_kind == "lazy_stream",
                save_to_db=False,
                include_character_context=False,
            ),
            db=_CompletionReadyChatSessionDb(),
            current_user=_test_user(),
            http_request=SimpleNamespace(state=SimpleNamespace()),
        )
    )
    assert await asyncio.to_thread(entered.wait, 2)
    request_task.cancel()
    await asyncio.sleep(0)
    assert "runtime_close" not in lifecycle

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await request_task

    for _ in range(100):
        if "runtime_close" in lifecycle:
            break
        await asyncio.sleep(0.01)

    assert lifecycle.index("factory_exit") < lifecycle.index("runtime_close")
    if result_kind == "nonstream":
        assert lifecycle.index("mark_used") < lifecycle.index("runtime_close")
        assert "upstream_close" not in lifecycle
    else:
        assert "mark_used" not in lifecycle
        assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
async def test_complete_v2_sync_provider_factory_timeout_is_bounded_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck sync adapter factory cannot retain the request indefinitely."""
    lifecycle: list[Any] = []
    bounded_calls: list[dict[str, Any]] = []
    _install_character_completion_runtime(
        monkeypatch,
        provider_response={
            "choices": [{"message": {"role": "assistant", "content": "late"}}]
        },
        lifecycle=lifecycle,
    )

    async def timeout_boundary(
        _call,
        *,
        pool,
        name: str,
        timeout_seconds: float,
        timeout_message: str,
        released_event: threading.Event | None = None,
        retain_result_after_timeout: bool = False,
    ) -> Any:
        assert retain_result_after_timeout is True
        bounded_calls.append(
            {
                "pool": pool,
                "name": name,
                "timeout_seconds": timeout_seconds,
                "timeout_message": timeout_message,
            }
        )
        if released_event is not None:
            released_event.set()
        raise TimeoutError("private provider timeout sentinel")

    monkeypatch.setattr(
        character_chat_sessions,
        "await_bounded_daemon_with_timeout",
        timeout_boundary,
    )

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(
                provider="local-llm",
                model="local-test",
                save_to_db=False,
                include_character_context=False,
            ),
            db=_CompletionReadyChatSessionDb(),
            current_user=_test_user(),
            http_request=SimpleNamespace(state=SimpleNamespace()),
        )

    assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
    assert exc_info.value.detail == "Chat provider error"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert [call["name"] for call in bounded_calls] == ["character-provider-call"]
    assert bounded_calls[0]["timeout_seconds"] > 0
    assert "mark_used" not in lifecycle
    assert lifecycle[-1] == "runtime_close"


@pytest.mark.asyncio
async def test_complete_v2_sync_stream_uses_bounded_factory_iteration_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every blocking sync stream operation crosses a capacity/deadline boundary."""
    lifecycle: list[Any] = []
    bounded_names: list[str] = []

    class SyncStream:
        def __init__(self) -> None:
            self._chunks = iter(("data: ok", "data: [DONE]"))

        def __iter__(self) -> SyncStream:
            return self

        def __next__(self) -> str:
            return next(self._chunks)

        def close(self) -> None:
            lifecycle.append("upstream_close")

    _install_character_completion_runtime(
        monkeypatch,
        provider_response=SyncStream(),
        lifecycle=lifecycle,
    )

    async def run_inline(
        call,
        *,
        pool,
        name: str,
        timeout_seconds: float,
        timeout_message: str,
        released_event: threading.Event | None = None,
        retain_result_after_timeout: bool = False,
    ) -> Any:
        del pool, timeout_message
        assert timeout_seconds > 0
        assert retain_result_after_timeout is True
        bounded_names.append(name)
        try:
            return call()
        finally:
            if released_event is not None:
                released_event.set()

    monkeypatch.setattr(
        character_chat_sessions,
        "await_bounded_daemon_with_timeout",
        run_inline,
    )

    async def close_inline(close, *, timeout: float) -> None:
        assert timeout > 0
        bounded_names.append("character-stream-close")
        close()

    monkeypatch.setattr(
        character_chat_sessions,
        "invoke_owned_stream_close",
        close_inline,
    )

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            stream=True,
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )
    chunks = [chunk async for chunk in response.body_iterator]

    assert any("data: ok" in str(chunk) for chunk in chunks)
    assert bounded_names[0] == "character-provider-call"
    assert "character-stream-iterator" in bounded_names
    assert bounded_names.count("character-stream-next") == 2
    assert bounded_names[-1] == "character-stream-close"
    assert lifecycle.index("mark_used") < lifecycle.index("upstream_close")
    assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
async def test_character_provider_timeout_retains_capacity_until_worker_really_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timed-out adapter work stays counted and rejects excess concurrency."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    lifecycle: list[Any] = []
    started = threading.Event()
    release = threading.Event()
    call_count = 0
    pool = BoundedDaemonPool(1)
    _install_character_completion_runtime(
        monkeypatch,
        provider_response={
            "choices": [{"message": {"role": "assistant", "content": "late"}}]
        },
        lifecycle=lifecycle,
    )

    def blocking_provider(**_kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        started.set()
        release.wait(timeout=2)
        return {"choices": [{"message": {"role": "assistant", "content": "late"}}]}

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", blocking_provider)
    monkeypatch.setattr(character_chat_sessions, "STREAM_DAEMON_POOL", pool)
    monkeypatch.setattr(
        character_chat_sessions,
        "CHARACTER_PROVIDER_CALL_TIMEOUT_SECONDS",
        0.05,
    )

    async def request_once() -> Any:
        return await character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(
                provider="local-llm",
                model="local-test",
                save_to_db=False,
                include_character_context=False,
            ),
            db=_CompletionReadyChatSessionDb(),
            current_user=_test_user(),
            http_request=SimpleNamespace(state=SimpleNamespace()),
        )

    first = asyncio.create_task(request_once())
    assert await asyncio.to_thread(started.wait, 1)
    with pytest.raises(HTTPException) as first_error:
        await asyncio.wait_for(first, timeout=0.5)
    assert first_error.value.status_code == status.HTTP_502_BAD_GATEWAY
    assert first_error.value.__cause__ is None
    assert first_error.value.__context__ is None
    assert pool.active_count == 1
    assert "runtime_close" not in lifecycle

    with pytest.raises(HTTPException) as capacity_error:
        await asyncio.wait_for(request_once(), timeout=0.5)
    assert capacity_error.value.status_code == status.HTTP_502_BAD_GATEWAY
    assert capacity_error.value.__cause__ is None
    assert capacity_error.value.__context__ is None
    assert call_count == 1
    assert lifecycle.count("runtime_close") == 1

    release.set()
    for _ in range(100):
        if pool.active_count == 0:
            break
        await asyncio.sleep(0.01)
    assert pool.active_count == 0
    for _ in range(100):
        if lifecycle.count("runtime_close") >= 2:
            break
        await asyncio.sleep(0.01)
    assert lifecycle.count("runtime_close") == 2

    recovered = await asyncio.wait_for(request_once(), timeout=0.5)
    assert recovered.assistant_content == "late"
    assert call_count == 2


@pytest.mark.asyncio
async def test_timed_out_character_factory_closes_late_stream_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lazy stream returned after the request deadline cannot leak resources."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    lifecycle: list[Any] = []
    started = threading.Event()
    release = threading.Event()
    late_closed = threading.Event()
    pool = BoundedDaemonPool(1)

    class LateStream:
        def __iter__(self) -> LateStream:
            return self

        def __next__(self) -> str:
            raise AssertionError("late stream must never be consumed")

        def close(self) -> None:
            lifecycle.append("late_stream_close")
            late_closed.set()

    _install_character_completion_runtime(
        monkeypatch,
        provider_response=LateStream(),
        lifecycle=lifecycle,
    )

    def blocking_provider(**_kwargs: Any) -> LateStream:
        started.set()
        release.wait(timeout=2)
        return LateStream()

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", blocking_provider)
    monkeypatch.setattr(character_chat_sessions, "STREAM_DAEMON_POOL", pool)
    monkeypatch.setattr(
        character_chat_sessions,
        "CHARACTER_PROVIDER_CALL_TIMEOUT_SECONDS",
        0.05,
    )

    request = asyncio.create_task(
        character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(
                provider="local-llm",
                model="local-test",
                stream=True,
                save_to_db=False,
                include_character_context=False,
            ),
            db=_CompletionReadyChatSessionDb(),
            current_user=_test_user(),
            http_request=SimpleNamespace(state=SimpleNamespace()),
        )
    )
    assert await asyncio.to_thread(started.wait, 1)
    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(request, timeout=0.5)
    assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
    assert "runtime_close" not in lifecycle

    release.set()
    assert await asyncio.to_thread(late_closed.wait, 1)
    for _ in range(100):
        if "runtime_close" in lifecycle:
            break
        await asyncio.sleep(0.01)
    assert lifecycle.count("late_stream_close") == 1
    assert lifecycle.index("late_stream_close") < lifecycle.index("runtime_close")
    assert lifecycle.count("runtime_close") == 1


@pytest.mark.asyncio
async def test_character_async_next_timeout_defers_stream_runtime_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async provider advancement cannot outlive stream credential ownership."""
    from tldw_Server_API.app.core.Chat import streaming_utils

    release = asyncio.Event()
    started = asyncio.Event()
    close_started = threading.Event()
    close_release = threading.Event()
    cleanup_done = asyncio.Event()
    cleanup_claimed = threading.Event()
    lifecycle: list[str] = []

    class Stream:
        def __aiter__(self) -> Stream:
            return self

        async def __anext__(self) -> str:
            started.set()
            await release.wait()
            lifecycle.append("next_exit")
            return "late"

        def close(self) -> None:
            close_started.set()
            close_release.wait(timeout=2.0)
            lifecycle.append("stream_close")

    stream = Stream()
    holder: dict[str, Any] = {}
    success = {"successful": False}

    async def cleanup(*, after_release: bool = False) -> None:
        assert after_release is True
        await character_chat_sessions._close_character_provider_stream(
            stream,
            holder,
            owned_cleanup=True,
        )
        lifecycle.append("runtime_close")
        cleanup_done.set()

    monkeypatch.setattr(
        character_chat_sessions,
        "CHARACTER_STREAM_NEXT_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(streaming_utils, "STREAM_CLEANUP_TASK_MAX_ACTIVE", 1)
    iterator = character_chat_sessions._iterate_character_provider_stream(
        stream,
        resource_holder=holder,
        success_state=success,
        on_abandoned=lambda: cleanup(after_release=True),
        cleanup_claimed=cleanup_claimed,
    )

    with pytest.raises(TimeoutError, match="character-stream-next timed out"):
        await iterator.__anext__()
    assert started.is_set()
    assert cleanup_claimed.is_set()
    assert lifecycle == []

    release.set()
    assert await asyncio.to_thread(close_started.wait, 1.0)
    assert cleanup_done.is_set() is False
    assert lifecycle == ["next_exit"]
    close_release.set()
    await asyncio.wait_for(cleanup_done.wait(), timeout=1.0)
    assert lifecycle == ["next_exit", "stream_close", "runtime_close"]


@pytest.mark.asyncio
async def test_character_cleanup_bounds_returned_awaitable_and_retains_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out close awaitable keeps the runtime until it really exits."""
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    close_finished = asyncio.Event()
    runtime_closed = asyncio.Event()
    lifecycle: list[str] = []

    class Runtime:
        async def mark_used(self, _credentials: Any) -> None:
            raise AssertionError("an unused stream must not be marked")

        async def close(self) -> None:
            lifecycle.append("runtime_close")
            runtime_closed.set()

    class Stream:
        def close(self) -> Any:
            lifecycle.append("close_invoked")

            async def finish_close() -> None:
                close_started.set()
                await close_release.wait()
                lifecycle.append("close_finished")
                close_finished.set()

            return finish_close()

    monkeypatch.setattr(
        character_chat_sessions,
        "CHARACTER_STREAM_CLOSE_TIMEOUT_SECONDS",
        0.01,
    )
    cleanup = character_chat_sessions._build_character_stream_cleanup(
        runtime=Runtime(),
        credentials=SimpleNamespace(),
        source=Stream(),
        resource_holder={},
        success_state={"successful": False},
    )
    cleanup_task = asyncio.create_task(cleanup())
    try:
        await asyncio.wait_for(close_started.wait(), timeout=1.0)
        done, _pending = await asyncio.wait({cleanup_task}, timeout=0.2)
        assert cleanup_task in done
        await cleanup_task
        assert runtime_closed.is_set() is False
    finally:
        close_release.set()
        await asyncio.gather(cleanup_task, return_exceptions=True)

    await asyncio.wait_for(close_finished.wait(), timeout=1.0)
    await asyncio.wait_for(runtime_closed.wait(), timeout=1.0)
    assert lifecycle == ["close_invoked", "close_finished", "runtime_close"]


@pytest.mark.asyncio
async def test_character_child_close_cancellation_does_not_skip_remaining_cleanup() -> None:
    """A self-cancelled child close is not cancellation of the cleanup owner."""
    child_started = asyncio.Event()
    child_release = asyncio.Event()
    source_closed = asyncio.Event()
    runtime_closed = asyncio.Event()
    lifecycle: list[str] = []

    class Runtime:
        async def mark_used(self, _credentials: Any) -> None:
            return None

        async def close(self) -> None:
            lifecycle.append("runtime_close")
            runtime_closed.set()

    class Iterator:
        async def aclose(self) -> None:
            child_started.set()
            await child_release.wait()
            lifecycle.append("iterator_cancel")
            raise asyncio.CancelledError

    class Source:
        async def aclose(self) -> None:
            lifecycle.append("source_close")
            source_closed.set()

    cleanup = character_chat_sessions._build_character_stream_cleanup(
        runtime=Runtime(),
        credentials=SimpleNamespace(),
        source=Source(),
        resource_holder={"iterator": Iterator()},
        success_state={"successful": False},
    )
    cleanup_task = asyncio.create_task(cleanup())
    await asyncio.wait_for(child_started.wait(), timeout=1.0)
    child_release.set()
    await asyncio.wait_for(cleanup_task, timeout=1.0)

    assert source_closed.is_set()
    assert runtime_closed.is_set()
    assert lifecycle == ["iterator_cancel", "source_close", "runtime_close"]


@pytest.mark.asyncio
async def test_character_cleanup_capacity_rejection_still_drains_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup admission failure must not strand the stream credential runtime."""
    from tldw_Server_API.app.core.Chat import streaming_utils

    lifecycle: list[str] = []

    class Runtime:
        async def mark_used(self, _credentials: Any) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")

    class Source:
        def close(self) -> None:
            lifecycle.append("source_close")

    monkeypatch.setattr(streaming_utils, "STREAM_CLEANUP_TASK_MAX_ACTIVE", 0)
    cleanup = character_chat_sessions._build_character_stream_cleanup(
        runtime=Runtime(),
        credentials=SimpleNamespace(),
        source=Source(),
        resource_holder={},
        success_state={"successful": True},
    )

    await cleanup()

    assert lifecycle == ["mark_used", "source_close", "runtime_close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("abandonment", ["timeout", "cancel"])
@pytest.mark.parametrize(
    ("result", "expected_marks"),
    [
        (
            {"choices": [{"message": {"role": "assistant", "content": "late"}}]},
            1,
        ),
        ({"choices": []}, 0),
        ({"error": {"message": "private late provider error"}}, 0),
    ],
    ids=["content", "empty", "error"],
)
async def test_character_late_usage_requires_valid_nonempty_content(
    monkeypatch: pytest.MonkeyPatch,
    abandonment: str,
    result: dict[str, Any],
    expected_marks: int,
) -> None:
    """Timeout and cancellation mark only validated late provider content."""
    loop = asyncio.get_running_loop()
    provider_started = asyncio.Event()
    provider_release = threading.Event()
    runtime_closed = asyncio.Event()
    lifecycle: list[str] = []

    _install_character_completion_runtime(
        monkeypatch,
        provider_response=result,
        lifecycle=[],
    )

    class Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            return SimpleNamespace(
                provider=provider,
                api_key="key-a",
                app_config={"local_llm_api": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _credentials: Any) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")
            runtime_closed.set()

    def provider_call(**_kwargs: Any) -> dict[str, Any]:
        loop.call_soon_threadsafe(provider_started.set)
        provider_release.wait(timeout=2.0)
        lifecycle.append("provider_exit")
        return result

    monkeypatch.setattr(character_chat_sessions, "ProviderCredentialRuntime", Runtime)
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", provider_call)
    monkeypatch.setattr(
        character_chat_sessions,
        "CHARACTER_PROVIDER_CALL_TIMEOUT_SECONDS",
        0.01 if abandonment == "timeout" else 30.0,
    )

    request_task = asyncio.create_task(
        character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(
                provider="local-llm",
                model="local-test",
                save_to_db=False,
                include_character_context=False,
            ),
            db=_CompletionReadyChatSessionDb(),
            current_user=_test_user(),
            http_request=SimpleNamespace(state=SimpleNamespace()),
        )
    )
    try:
        await asyncio.wait_for(provider_started.wait(), timeout=1.0)
        if abandonment == "cancel":
            request_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request_task
        else:
            with pytest.raises(HTTPException) as exc_info:
                await request_task
            assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
        assert runtime_closed.is_set() is False
    finally:
        provider_release.set()
        await asyncio.gather(request_task, return_exceptions=True)

    await asyncio.wait_for(runtime_closed.wait(), timeout=1.0)
    assert lifecycle.count("mark_used") == expected_marks
    assert lifecycle[-1] == "runtime_close"


@pytest.mark.asyncio
@pytest.mark.parametrize("abandonment", ["timeout", "cancel"])
async def test_character_sync_boundary_never_queues_a_late_default_executor_start(
    monkeypatch: pytest.MonkeyPatch,
    abandonment: str,
) -> None:
    """Daemon admission occurs before timeout/cancellation, outside the default pool."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    loop = asyncio.get_running_loop()
    default_started = asyncio.Event()
    provider_started = asyncio.Event()
    provider_finished = asyncio.Event()
    cleanup_finished = asyncio.Event()
    default_release = threading.Event()
    provider_release = threading.Event()
    call_count = 0

    def occupy_default_executor() -> None:
        loop.call_soon_threadsafe(default_started.set)
        default_release.wait(timeout=2.0)

    def provider_call() -> str:
        nonlocal call_count
        call_count += 1
        loop.call_soon_threadsafe(provider_started.set)
        provider_release.wait(timeout=2.0)
        loop.call_soon_threadsafe(provider_finished.set)
        return "late"

    async def cleanup() -> None:
        cleanup_finished.set()

    previous_executor = getattr(loop, "_default_executor", None)
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    default_future = loop.run_in_executor(None, occupy_default_executor)
    monkeypatch.setattr(character_chat_sessions, "STREAM_DAEMON_POOL", BoundedDaemonPool(1))
    operation = asyncio.create_task(
        character_chat_sessions._run_bounded_character_sync_call(
            provider_call,
            name="character-default-executor-regression",
            timeout_seconds=0.01 if abandonment == "timeout" else 30.0,
            on_abandoned=cleanup,
            cleanup_claimed=threading.Event(),
        )
    )
    try:
        await asyncio.wait_for(default_started.wait(), timeout=1.0)
        if abandonment == "cancel":
            checkpoint = asyncio.Event()
            loop.call_soon(checkpoint.set)
            await checkpoint.wait()
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
        else:
            with pytest.raises(TimeoutError):
                await operation
        assert provider_started.is_set()
        assert call_count == 1
    finally:
        provider_release.set()
        default_release.set()
        await asyncio.gather(default_future, return_exceptions=True)
        await asyncio.gather(operation, return_exceptions=True)
        executor.shutdown(wait=True)
        loop.set_default_executor(previous_executor or ThreadPoolExecutor())

    await asyncio.wait_for(provider_finished.wait(), timeout=1.0)
    await asyncio.wait_for(cleanup_finished.wait(), timeout=1.0)
    assert call_count == 1


@pytest.mark.asyncio
async def test_complete_v2_auto_router_and_final_dispatch_share_one_scoped_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto selection and final dispatch must share one execution credential scope."""
    lifecycle: list[Any] = []
    _install_character_completion_runtime(
        monkeypatch,
        provider_response={
            "choices": [{"message": {"role": "assistant", "content": "routed"}}]
        },
        lifecycle=lifecycle,
    )

    async def fake_auto_route(**kwargs: Any) -> tuple[RoutingDecision, dict[str, Any]]:
        runtime = kwargs["credential_runtime"]
        router_credentials = await runtime.resolve("local-llm", model="router-model")
        await runtime.mark_used(router_credentials)
        return (
            RoutingDecision(
                provider="local-llm",
                model="routed-model",
                canonical=True,
                decision_source="llm_router",
            ),
            {"candidate_count": 1},
        )

    monkeypatch.setattr(
        character_chat_sessions,
        "_resolve_auto_character_chat_routing_decision",
        fake_auto_route,
    )

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="auto",
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )

    assert response.assistant_content == "routed"
    assert [entry for entry in lifecycle if isinstance(entry, tuple) and entry[0] == "init"] == [
        ("init", 1, [11], [22], True)
    ]
    assert [entry for entry in lifecycle if isinstance(entry, tuple) and entry[0] == "resolve"] == [
        ("resolve", "local-llm", "router-model"),
        ("resolve", "local-llm", "routed-model"),
    ]
    assert lifecycle.count("mark_used") == 2
    assert lifecycle[-1] == "runtime_close"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_auth_source", "expected_status"),
    [("aws_default_chain", 200), (None, 503)],
    ids=["bedrock-default-chain", "bedrock-explicit-absent"],
)
async def test_complete_v2_bedrock_auth_contract_distinguishes_default_chain_from_absent(
    monkeypatch: pytest.MonkeyPatch,
    runtime_auth_source: str | None,
    expected_status: int,
) -> None:
    """Bedrock default-chain auth is valid while an absent snapshot fails closed."""
    lifecycle: list[Any] = []

    class BedrockRuntime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            lifecycle.append(("resolve", provider, model))
            provider_config: dict[str, Any] = {}
            if runtime_auth_source is not None:
                provider_config["_runtime_auth_source"] = runtime_auth_source
            return SimpleNamespace(
                provider=provider,
                api_key=None,
                app_config={"bedrock_api": provider_config},
                credentials_resolved=True,
            )

        async def mark_used(self, _credentials: Any) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")

    def provider_call(**_kwargs: Any) -> dict[str, Any]:
        lifecycle.append("adapter_call")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setenv("ENABLE_LOCAL_LLM_PROVIDER", "true")
    monkeypatch.setattr(
        character_chat_sessions,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [], [], False),
    )
    monkeypatch.setattr(character_chat_sessions, "ProviderCredentialRuntime", BedrockRuntime)
    monkeypatch.setattr(character_chat_sessions, "get_character_rate_limiter", lambda: _NoopCharacterRateLimiter())
    monkeypatch.setattr(character_chat_sessions, "_should_enforce_char_chat_strict_model_selection", lambda: False)
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", provider_call)

    kwargs = {
        "chat_id": "chat-1",
        "body": CharacterChatCompletionV2Request(
            provider="bedrock",
            model="model-a",
            save_to_db=False,
            include_character_context=False,
        ),
        "db": _CompletionReadyChatSessionDb(),
        "current_user": _test_user(),
        "http_request": SimpleNamespace(state=SimpleNamespace()),
    }
    if expected_status == 200:
        response = await character_chat_sessions.character_chat_completion(**kwargs)
        assert response.assistant_content == "ok"
        assert lifecycle[-2:] == ["mark_used", "runtime_close"]
    else:
        with pytest.raises(HTTPException) as exc_info:
            await character_chat_sessions.character_chat_completion(**kwargs)
        assert exc_info.value.status_code == expected_status
        assert "adapter_call" not in lifecycle
        assert lifecycle[-1] == "runtime_close"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["sync", "async"])
async def test_complete_v2_stream_marks_success_then_closes_upstream_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    stream_kind: str,
) -> None:
    """Successful lazy streams retain one scoped runtime through adapter cleanup."""
    lifecycle: list[Any] = []

    class SyncStream:
        def __init__(self) -> None:
            self._chunks = iter(("data: ok", "data: [DONE]"))

        def __iter__(self) -> SyncStream:
            return self

        def __next__(self) -> str:
            return next(self._chunks)

        def close(self) -> None:
            lifecycle.append("upstream_close")

    class AsyncStream:
        def __init__(self) -> None:
            self._chunks = iter(("data: ok", "data: [DONE]"))

        def __aiter__(self) -> AsyncStream:
            return self

        async def __anext__(self) -> str:
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration from None

        async def aclose(self) -> None:
            lifecycle.append("upstream_close")

    stream = SyncStream() if stream_kind == "sync" else AsyncStream()
    _install_character_completion_runtime(
        monkeypatch,
        provider_response=stream,
        lifecycle=lifecycle,
    )

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            stream=True,
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )
    chunks = [chunk async for chunk in response.body_iterator]

    assert any("data: ok" in str(chunk) for chunk in chunks)
    assert lifecycle[:2] == [
        ("init", 1, [11], [22], True),
        ("resolve", "local-llm", "local-test"),
    ]
    assert lifecycle.index("mark_used") < lifecycle.index("upstream_close")
    assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
@pytest.mark.parametrize("streams_unified", ["0", "1"], ids=["legacy", "unified"])
@pytest.mark.parametrize("stream_kind", ["sync", "async"])
@pytest.mark.parametrize(
    "error_framing",
    [
        "data",
        "event-data",
        "raw-prefix",
        "data-raw-prefix",
        "canonical-code",
        "serialized",
    ],
)
async def test_complete_v2_stream_sanitizes_terminal_provider_error_frames(
    monkeypatch: pytest.MonkeyPatch,
    streams_unified: str,
    stream_kind: str,
    error_framing: str,
) -> None:
    """Untrusted provider error frames are bounded, terminal, and never successful."""
    lifecycle: list[Any] = []
    sentinel = "sk-provider-stream-/private/provider-cache.json"
    if error_framing == "data":
        error_chunks = [f'data: {{"error": {{"message": "{sentinel}"}}}}']
    elif error_framing == "event-data":
        error_chunks = ["event: error", f'data: {{"message": "{sentinel}"}}']
    elif error_framing == "raw-prefix":
        error_chunks = [f"Error: {sentinel}"]
    elif error_framing == "data-raw-prefix":
        error_chunks = [f"data: Error: {sentinel}"]
    elif error_framing == "canonical-code":
        error_chunks = ["provider_unavailable"]
    else:
        error_chunks = [
            json.dumps(
                {
                    "error": {
                        "code": "provider_unavailable",
                        "message": sentinel,
                    }
                }
            )
        ]
    chunks = [
        'data: {"choices": [{"delta": {"content": "partial-safe-output"}}]}',
        *error_chunks,
        'data: {"choices": [{"delta": {"content": "must-not-appear"}}]}',
        "data: [DONE]",
    ]

    class SyncStream:
        def __init__(self) -> None:
            self._chunks = iter(chunks)

        def __iter__(self) -> SyncStream:
            return self

        def __next__(self) -> str:
            return next(self._chunks)

        def close(self) -> None:
            lifecycle.append("upstream_close")

    class AsyncStream:
        def __init__(self) -> None:
            self._chunks = iter(chunks)

        def __aiter__(self) -> AsyncStream:
            return self

        async def __anext__(self) -> str:
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration from None

        async def aclose(self) -> None:
            lifecycle.append("upstream_close")

    stream = SyncStream() if stream_kind == "sync" else AsyncStream()
    _install_character_completion_runtime(
        monkeypatch,
        provider_response=stream,
        lifecycle=lifecycle,
    )
    monkeypatch.setenv("STREAMS_UNIFIED", streams_unified)

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            stream=True,
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )
    rendered = "".join([
        chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        async for chunk in response.body_iterator
    ])

    assert sentinel not in rendered
    assert "partial-safe-output" in rendered
    assert "must-not-appear" not in rendered
    assert '"code": "provider_unavailable"' in rendered
    assert rendered.lower().count("data: [done]") == 1
    assert "mark_used" not in lifecycle
    assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
@pytest.mark.parametrize("streams_unified", ["0", "1"], ids=["legacy", "unified"])
@pytest.mark.parametrize("stream_kind", ["sync", "async"])
@pytest.mark.parametrize("failure_kind", ["typed", "unexpected"])
async def test_complete_v2_partial_stream_then_lazy_failure_is_not_marked_used(
    monkeypatch: pytest.MonkeyPatch,
    streams_unified: str,
    stream_kind: str,
    failure_kind: str,
) -> None:
    """A terminal lazy-iteration failure must override earlier partial output."""
    lifecycle: list[Any] = []
    content_frame = 'data: {"choices": [{"delta": {"content": "partial"}}]}'
    failure = (
        ChatAPIError("private provider failure")
        if failure_kind == "typed"
        else RuntimeError("private iterator failure")
    )

    class SyncStream:
        def __init__(self) -> None:
            self._returned_content = False

        def __iter__(self) -> SyncStream:
            return self

        def __next__(self) -> str:
            if not self._returned_content:
                self._returned_content = True
                return content_frame
            raise failure

        def close(self) -> None:
            lifecycle.append("upstream_close")

    class AsyncStream:
        def __init__(self) -> None:
            self._returned_content = False

        def __aiter__(self) -> AsyncStream:
            return self

        async def __anext__(self) -> str:
            if not self._returned_content:
                self._returned_content = True
                return content_frame
            raise failure

        async def aclose(self) -> None:
            lifecycle.append("upstream_close")

    stream = SyncStream() if stream_kind == "sync" else AsyncStream()
    _install_character_completion_runtime(
        monkeypatch,
        provider_response=stream,
        lifecycle=lifecycle,
    )
    monkeypatch.setenv("STREAMS_UNIFIED", streams_unified)

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            stream=True,
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )
    rendered = "".join([
        chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        async for chunk in response.body_iterator
    ])

    assert "partial" in rendered
    assert "private provider failure" not in rendered
    assert "private iterator failure" not in rendered
    assert "mark_used" not in lifecycle
    assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
@pytest.mark.parametrize("streams_unified", ["0", "1"], ids=["legacy", "unified"])
async def test_complete_v2_control_only_stream_is_not_marked_used(
    monkeypatch: pytest.MonkeyPatch,
    streams_unified: str,
) -> None:
    """SSE metadata, role, finish, and DONE frames are not assistant output."""
    lifecycle: list[Any] = []
    chunks = (
        'data: {"choices": [{"delta": {"role": "assistant"}}]}',
        "event: ping",
        "id: control-only",
        "retry: 500",
        ": heartbeat",
        'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
        "data: [DONE]",
    )

    class ControlOnlyStream:
        def __init__(self) -> None:
            self._chunks = iter(chunks)

        def __iter__(self) -> ControlOnlyStream:
            return self

        def __next__(self) -> str:
            return next(self._chunks)

        def close(self) -> None:
            lifecycle.append("upstream_close")

    _install_character_completion_runtime(
        monkeypatch,
        provider_response=ControlOnlyStream(),
        lifecycle=lifecycle,
    )
    monkeypatch.setenv("STREAMS_UNIFIED", streams_unified)

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            stream=True,
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )
    rendered = "".join([
        chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        async for chunk in response.body_iterator
    ])

    assert rendered.lower().count("data: [done]") == 1
    assert "mark_used" not in lifecycle
    assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
@pytest.mark.parametrize("streams_unified", ["0", "1"], ids=["legacy", "unified"])
@pytest.mark.parametrize("limit_kind", ["chunks", "bytes"])
async def test_complete_v2_stream_limit_failure_overrides_partial_success(
    monkeypatch: pytest.MonkeyPatch,
    streams_unified: str,
    limit_kind: str,
) -> None:
    """Server-enforced terminal limits cannot leave partial output successful."""
    lifecycle: list[Any] = []
    content_frame = 'data: {"choices": [{"delta": {"content": "partial"}}]}'

    class LimitedStream:
        def __init__(self) -> None:
            self._chunks = iter((content_frame, content_frame, "data: [DONE]"))

        def __iter__(self) -> LimitedStream:
            return self

        def __next__(self) -> str:
            return next(self._chunks)

        def close(self) -> None:
            lifecycle.append("upstream_close")

    _install_character_completion_runtime(
        monkeypatch,
        provider_response=LimitedStream(),
        lifecycle=lifecycle,
    )
    monkeypatch.setenv("STREAMS_UNIFIED", streams_unified)
    if limit_kind == "chunks":
        monkeypatch.setattr(character_chat_sessions, "MAX_STREAMING_CHUNKS", 1)
    else:
        monkeypatch.setattr(
            character_chat_sessions,
            "MAX_STREAMING_BYTES",
            len(character_chat_sessions.ensure_sse_line(content_frame).encode("utf-8"))
            + 1,
        )

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            stream=True,
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )
    rendered = "".join([
        chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        async for chunk in response.body_iterator
    ])

    assert "Streaming" in rendered
    assert "mark_used" not in lifecycle
    assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize("streams_unified", ["0", "1"], ids=["legacy", "unified"])
@pytest.mark.parametrize("stream_kind", ["sync", "async"])
@pytest.mark.parametrize(
    "late_outcome",
    ["terminal_error", "clean_exhaustion", "done_control", "event_control"],
)
async def test_cancelled_character_stream_late_non_output_is_not_marked_used(
    monkeypatch: pytest.MonkeyPatch,
    streams_unified: str,
    stream_kind: str,
    late_outcome: str,
) -> None:
    """A cancelled next must produce valid output before it can count as success."""
    lifecycle: list[Any] = []
    sync_entered = threading.Event()
    sync_release = threading.Event()
    async_entered = asyncio.Event()
    async_release = asyncio.Event()
    terminal_error = 'data: {"error": {"message": "private late provider error"}}'

    class SyncStream:
        def __iter__(self) -> SyncStream:
            return self

        def __next__(self) -> str:
            sync_entered.set()
            sync_release.wait(timeout=5)
            lifecycle.append("next_exit")
            if late_outcome == "clean_exhaustion":
                raise StopIteration
            if late_outcome == "done_control":
                return "data: [DONE]"
            if late_outcome == "event_control":
                return "event: ping"
            return terminal_error

        def close(self) -> None:
            lifecycle.append("upstream_close")

    class AsyncStream:
        def __aiter__(self) -> AsyncStream:
            return self

        async def __anext__(self) -> str:
            async_entered.set()
            await async_release.wait()
            lifecycle.append("next_exit")
            if late_outcome == "clean_exhaustion":
                raise StopAsyncIteration
            if late_outcome == "done_control":
                return "data: [DONE]"
            if late_outcome == "event_control":
                return "event: ping"
            return terminal_error

        async def aclose(self) -> None:
            lifecycle.append("upstream_close")

    stream = SyncStream() if stream_kind == "sync" else AsyncStream()
    _install_character_completion_runtime(
        monkeypatch,
        provider_response=stream,
        lifecycle=lifecycle,
    )
    monkeypatch.setenv("STREAMS_UNIFIED", streams_unified)

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            stream=True,
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )
    consume = asyncio.create_task(response.body_iterator.__anext__())
    if stream_kind == "sync":
        assert await asyncio.to_thread(sync_entered.wait, 2)
    else:
        await asyncio.wait_for(async_entered.wait(), timeout=2)

    consume.cancel()
    await asyncio.sleep(0)
    assert "runtime_close" not in lifecycle

    if stream_kind == "sync":
        sync_release.set()
    else:
        async_release.set()
    with pytest.raises(asyncio.CancelledError):
        await consume

    for _ in range(100):
        if "runtime_close" in lifecycle:
            break
        await asyncio.sleep(0.01)

    assert "mark_used" not in lifecycle
    assert lifecycle.count("upstream_close") == 1
    assert lifecycle.count("runtime_close") == 1
    assert lifecycle.index("next_exit") < lifecycle.index("upstream_close")
    assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize("streams_unified", ["0", "1"], ids=["legacy", "unified"])
@pytest.mark.parametrize("stream_kind", ["sync", "async"])
async def test_cancelled_character_stream_late_terminal_error_overrides_partial_output(
    monkeypatch: pytest.MonkeyPatch,
    streams_unified: str,
    stream_kind: str,
) -> None:
    """A late terminal frame remains authoritative after earlier delivered text."""
    lifecycle: list[Any] = []
    sync_entered = threading.Event()
    sync_release = threading.Event()
    async_entered = asyncio.Event()
    async_release = asyncio.Event()
    content_frame = 'data: {"choices": [{"delta": {"content": "partial"}}]}'
    terminal_error = 'data: {"error": {"message": "private late provider error"}}'

    class SyncStream:
        def __init__(self) -> None:
            self._returned_content = False

        def __iter__(self) -> SyncStream:
            return self

        def __next__(self) -> str:
            if not self._returned_content:
                self._returned_content = True
                return content_frame
            sync_entered.set()
            sync_release.wait(timeout=5)
            lifecycle.append("next_exit")
            return terminal_error

        def close(self) -> None:
            lifecycle.append("upstream_close")

    class AsyncStream:
        def __init__(self) -> None:
            self._returned_content = False

        def __aiter__(self) -> AsyncStream:
            return self

        async def __anext__(self) -> str:
            if not self._returned_content:
                self._returned_content = True
                return content_frame
            async_entered.set()
            await async_release.wait()
            lifecycle.append("next_exit")
            return terminal_error

        async def aclose(self) -> None:
            lifecycle.append("upstream_close")

    stream = SyncStream() if stream_kind == "sync" else AsyncStream()
    _install_character_completion_runtime(
        monkeypatch,
        provider_response=stream,
        lifecycle=lifecycle,
    )
    monkeypatch.setenv("STREAMS_UNIFIED", streams_unified)

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            stream=True,
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )
    first = await asyncio.wait_for(response.body_iterator.__anext__(), timeout=2)
    assert "partial" in str(first)

    consume = asyncio.create_task(response.body_iterator.__anext__())
    if stream_kind == "sync":
        assert await asyncio.to_thread(sync_entered.wait, 2)
    else:
        await asyncio.wait_for(async_entered.wait(), timeout=2)
    consume.cancel()
    await asyncio.sleep(0)
    assert "runtime_close" not in lifecycle

    if stream_kind == "sync":
        sync_release.set()
    else:
        async_release.set()
    with pytest.raises(asyncio.CancelledError):
        await consume

    for _ in range(100):
        if "runtime_close" in lifecycle:
            break
        await asyncio.sleep(0.01)

    assert "mark_used" not in lifecycle
    assert lifecycle.count("upstream_close") == 1
    assert lifecycle.count("runtime_close") == 1
    assert lifecycle.index("next_exit") < lifecycle.index("upstream_close")
    assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
@pytest.mark.parametrize("streams_unified", ["0", "1"], ids=["legacy", "unified"])
async def test_complete_v2_stream_preserves_valid_content_and_tool_frames(
    monkeypatch: pytest.MonkeyPatch,
    streams_unified: str,
) -> None:
    """Sanitization must not rewrite valid adapter content or tool-call frames."""
    lifecycle: list[Any] = []
    content_frame = 'data: {"choices": [{"delta": {"content": "hello"}}]}'
    tool_frame = (
        'data: {"choices": [{"delta": {"tool_calls": '
        '[{"index": 0, "function": {"name": "lookup"}}]}}]}'
    )

    class ValidStream:
        def __init__(self) -> None:
            self._chunks = iter((content_frame, tool_frame, "data: [DONE]"))

        def __iter__(self) -> ValidStream:
            return self

        def __next__(self) -> str:
            return next(self._chunks)

        def close(self) -> None:
            lifecycle.append("upstream_close")

    _install_character_completion_runtime(
        monkeypatch,
        provider_response=ValidStream(),
        lifecycle=lifecycle,
    )
    monkeypatch.setenv("STREAMS_UNIFIED", streams_unified)

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            stream=True,
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )
    rendered = "".join([
        chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        async for chunk in response.body_iterator
    ])

    assert content_frame in rendered
    assert tool_frame in rendered
    assert rendered.lower().count("data: [done]") == 1
    assert lifecycle.index("mark_used") < lifecycle.index("upstream_close")
    assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
@pytest.mark.parametrize("streams_unified", ["0", "1"], ids=["legacy", "unified"])
async def test_complete_v2_tool_only_stream_is_marked_used(
    monkeypatch: pytest.MonkeyPatch,
    streams_unified: str,
) -> None:
    """A valid tool-call delta remains semantic provider output."""
    lifecycle: list[Any] = []
    tool_frame = (
        'data: {"choices": [{"delta": {"tool_calls": '
        '[{"index": 0, "function": {"name": "lookup"}}]}}]}'
    )

    class ToolOnlyStream:
        def __init__(self) -> None:
            self._chunks = iter((tool_frame, "data: [DONE]"))

        def __iter__(self) -> ToolOnlyStream:
            return self

        def __next__(self) -> str:
            return next(self._chunks)

        def close(self) -> None:
            lifecycle.append("upstream_close")

    _install_character_completion_runtime(
        monkeypatch,
        provider_response=ToolOnlyStream(),
        lifecycle=lifecycle,
    )
    monkeypatch.setenv("STREAMS_UNIFIED", streams_unified)

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            stream=True,
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )
    rendered = "".join([
        chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        async for chunk in response.body_iterator
    ])

    assert tool_frame in rendered
    assert lifecycle.index("mark_used") < lifecycle.index("upstream_close")
    assert lifecycle.index("upstream_close") < lifecycle.index("runtime_close")


@pytest.mark.asyncio
@pytest.mark.parametrize("close_kind", ["sync", "async"])
@pytest.mark.parametrize("iterator_raises", [False, True])
async def test_character_stream_cleanup_closes_distinct_iterator_and_source_before_runtime(
    close_kind: str,
    iterator_raises: bool,
) -> None:
    """Iterator cleanup failure cannot skip the owning source or runtime release."""
    lifecycle: list[str] = []

    class Runtime:
        async def mark_used(self, _credentials: Any) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")

    class Resource:
        def __init__(self, name: str, *, raises: bool = False) -> None:
            self.name = name
            self.raises = raises

        def close(self) -> None:
            lifecycle.append(f"{self.name}_close")
            if self.raises:
                raise RuntimeError("cleanup sentinel")

        async def aclose(self) -> None:
            lifecycle.append(f"{self.name}_aclose")
            if self.raises:
                raise RuntimeError("cleanup sentinel")

    iterator = Resource("iterator", raises=iterator_raises)
    source = Resource("source")
    if close_kind == "sync":
        iterator.aclose = None  # type: ignore[method-assign]
        source.aclose = None  # type: ignore[method-assign]
        iterator_event = "iterator_close"
        source_event = "source_close"
    else:
        iterator_event = "iterator_aclose"
        source_event = "source_aclose"

    cleanup = character_chat_sessions._build_character_stream_cleanup(
        runtime=Runtime(),
        credentials=SimpleNamespace(),
        source=source,
        resource_holder={"iterator": iterator},
        success_state={"successful": True},
    )
    await cleanup()

    assert lifecycle.count("mark_used") == 1
    assert lifecycle.count(iterator_event) == 1
    assert lifecycle.count(source_event) == 1
    assert lifecycle.index("mark_used") < lifecycle.index(iterator_event)
    assert lifecycle.index(iterator_event) < lifecycle.index(source_event)
    assert lifecycle.index(source_event) < lifecycle.index("runtime_close")


@pytest.mark.asyncio
async def test_complete_v2_never_started_stream_background_closes_without_marking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ASGI background cleanup releases a returned stream that never starts."""
    lifecycle: list[Any] = []

    class NeverStartedStream:
        def __iter__(self) -> NeverStartedStream:
            return self

        def __next__(self) -> str:
            raise AssertionError("stream must not start")

        def close(self) -> None:
            lifecycle.append("upstream_close")

    _install_character_completion_runtime(
        monkeypatch,
        provider_response=NeverStartedStream(),
        lifecycle=lifecycle,
    )
    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            stream=True,
            save_to_db=False,
            include_character_context=False,
        ),
        db=_CompletionReadyChatSessionDb(),
        current_user=_test_user(),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )

    assert response.background is not None
    await response.background()

    assert "mark_used" not in lifecycle
    assert lifecycle[-2:] == ["upstream_close", "runtime_close"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (ByokResolutionError("credential_scope_revoked", "credential_scope"), 403, "credential_scope_revoked"),
        (ByokResolutionError("credential_store_unavailable", "local-llm"), 503, "credential_store_unavailable"),
    ],
)
async def test_complete_v2_maps_typed_credential_failures_to_bounded_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: ByokResolutionError,
    expected_status: int,
    expected_code: str,
) -> None:
    """Credential policy/storage failures remain bounded at the route boundary."""

    class FailingRuntime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, _provider: str, *, model: str | None = None) -> Any:
            del model
            raise error

        async def close(self) -> None:
            return None

    monkeypatch.setenv("ENABLE_LOCAL_LLM_PROVIDER", "true")
    monkeypatch.setattr(
        character_chat_sessions,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [], [], False),
    )
    monkeypatch.setattr(character_chat_sessions, "ProviderCredentialRuntime", FailingRuntime)
    monkeypatch.setattr(character_chat_sessions, "provider_requires_api_key", lambda _provider: False)
    monkeypatch.setattr(character_chat_sessions, "get_character_rate_limiter", lambda: _NoopCharacterRateLimiter())
    monkeypatch.setattr(character_chat_sessions, "_should_enforce_char_chat_strict_model_selection", lambda: False)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(
                provider="local-llm",
                model="local-test",
                save_to_db=False,
                include_character_context=False,
            ),
            db=_CompletionReadyChatSessionDb(),
            current_user=_test_user(),
            http_request=SimpleNamespace(state=SimpleNamespace()),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail["error_code"] == expected_code
    assert "local-llm" not in str(exc_info.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("captured_key", ["completion-key-a", None], ids=["a-to-b", "absent-to-b"])
async def test_complete_v2_dispatch_keeps_one_static_credential_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    captured_key: str | None,
) -> None:
    """Completion dispatch must carry the key and adapter config from one capture."""
    db = _CompletionReadyChatSessionDb()
    config_a = {"local_llm_api": {"base_url": "http://generation-a.invalid"}}
    current_key = {"value": captured_key}
    captured: dict[str, Any] = {}
    lifecycle: list[Any] = []
    handles: list[Any] = []

    class RecordingCredentialRuntime:
        def __init__(self, **kwargs: Any) -> None:
            lifecycle.append(("init", kwargs["user_id"]))
            assert "fallback_resolver" not in kwargs
            self._api_key = captured_key
            self._app_config = config_a

        async def resolve(self, provider: str, *, model: str | None = None):
            lifecycle.append(("resolve", provider, model))
            current_key["value"] = "completion-key-b"
            handle = SimpleNamespace(
                provider=provider,
                api_key=self._api_key,
                app_config=self._app_config,
                credentials_resolved=True,
            )
            handles.append(handle)
            return handle

        async def mark_used(self, _credentials: Any) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("close")

    def provider_call(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    from tldw_Server_API.app.api.v1.schemas import chat_request_schemas

    monkeypatch.setenv("ENABLE_LOCAL_LLM_PROVIDER", "true")
    monkeypatch.setattr(
        chat_request_schemas,
        "get_api_keys",
        lambda: {"local-llm": current_key["value"]},
    )
    monkeypatch.setattr(
        character_chat_sessions,
        "ProviderCredentialRuntime",
        RecordingCredentialRuntime,
        raising=False,
    )
    monkeypatch.setattr(character_chat_sessions, "provider_requires_api_key", lambda _provider: False)
    monkeypatch.setattr(
        character_chat_sessions,
        "get_character_rate_limiter",
        lambda: _NoopCharacterRateLimiter(),
    )
    monkeypatch.setattr(
        character_chat_sessions,
        "_should_enforce_char_chat_strict_model_selection",
        lambda: False,
    )
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", provider_call)

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            save_to_db=False,
            include_character_context=False,
        ),
        db=db,
        current_user=_test_user(),
    )

    assert response.assistant_content == "ok"
    assert captured["api_key"] == captured_key
    assert captured["app_config"] == config_a
    assert captured["credentials_resolved"] is True
    assert captured[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handles[0]
    assert captured["timeout"] == character_chat_sessions.CHARACTER_PROVIDER_CALL_TIMEOUT_SECONDS
    assert lifecycle == [
        ("init", 1),
        ("resolve", "local-llm", "local-test"),
        "mark_used",
        "close",
    ]


@pytest.mark.asyncio
async def test_update_chat_session_maps_db_error_to_sanitized_500() -> None:
    db = _BrokenChatSessionDb(CharactersRAGDBError("sqlite update exploded"))

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.update_chat_session(
            chat_id="chat-1",
            update_data=ChatSessionUpdate(title="Updated"),
            expected_version=1,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to update chat session"


@pytest.mark.asyncio
async def test_update_chat_session_maps_conflict_error_to_409() -> None:
    db = _BrokenChatSessionDb(ConflictError("chat update conflict"))

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.update_chat_session(
            chat_id="chat-1",
            update_data=ChatSessionUpdate(title="Updated"),
            expected_version=1,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "chat update conflict"


@pytest.mark.asyncio
async def test_delete_chat_session_maps_db_error_to_sanitized_500() -> None:
    db = _BrokenChatSessionDb(CharactersRAGDBError("sqlite delete exploded"))

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.delete_chat_session(
            chat_id="chat-1",
            expected_version=1,
            hard_delete=False,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to delete chat session"


@pytest.mark.asyncio
async def test_delete_chat_session_maps_conflict_error_to_409() -> None:
    db = _BrokenChatSessionDb(ConflictError("chat delete conflict"))

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.delete_chat_session(
            chat_id="chat-1",
            expected_version=1,
            hard_delete=False,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "chat delete conflict"


@pytest.mark.asyncio
async def test_restore_chat_session_maps_db_error_to_sanitized_500() -> None:
    db = _BrokenChatSessionDb(CharactersRAGDBError("sqlite restore exploded"), deleted=True)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.restore_chat_session(
            chat_id="chat-1",
            expected_version=1,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to restore chat session"


@pytest.mark.asyncio
async def test_restore_chat_session_maps_conflict_error_to_409() -> None:
    db = _BrokenChatSessionDb(ConflictError("chat restore conflict"), deleted=True)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.restore_chat_session(
            chat_id="chat-1",
            expected_version=1,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "chat restore conflict"


@pytest.mark.asyncio
async def test_complete_v2_maps_db_error_to_sanitized_500() -> None:
    db = _BrokenChatSessionDb(CharactersRAGDBError("sqlite completion exploded"), raise_on_get=True)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(),
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to complete character chat"


@pytest.mark.asyncio
async def test_complete_v2_maps_conflict_error_to_409() -> None:
    db = _BrokenChatSessionDb(ConflictError("chat completion conflict"), raise_on_get=True)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(),
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "chat completion conflict"


@pytest.mark.asyncio
async def test_complete_v2_maps_chat_api_server_error_to_sanitized_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _CompletionReadyChatSessionDb()

    def fake_provider_call(**_kwargs: Any) -> dict[str, Any]:
        raise ChatAPIError(
            "provider leaked token and /private/provider/cache/path",
            status_code=status.HTTP_502_BAD_GATEWAY,
            provider="local-llm",
        )

    monkeypatch.setenv("ENABLE_LOCAL_LLM_PROVIDER", "true")
    _install_simple_credential_runtime(monkeypatch)
    monkeypatch.setattr(character_chat_sessions, "provider_requires_api_key", lambda provider: False)
    monkeypatch.setattr(character_chat_sessions, "get_character_rate_limiter", lambda: _NoopCharacterRateLimiter())
    monkeypatch.setattr(character_chat_sessions, "_should_enforce_char_chat_strict_model_selection", lambda: False)
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", fake_provider_call)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(
                provider="local-llm",
                model="local-test",
                save_to_db=False,
                include_character_context=False,
            ),
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
    assert exc_info.value.detail == "Chat provider error"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_status", "expected_status"),
    [
        ("malformed", status.HTTP_502_BAD_GATEWAY),
        (status.HTTP_200_OK, status.HTTP_502_BAD_GATEWAY),
        (status.HTTP_302_FOUND, status.HTTP_502_BAD_GATEWAY),
        (799, status.HTTP_502_BAD_GATEWAY),
        (status.HTTP_401_UNAUTHORIZED, status.HTTP_401_UNAUTHORIZED),
        (status.HTTP_429_TOO_MANY_REQUESTS, status.HTTP_429_TOO_MANY_REQUESTS),
        (status.HTTP_503_SERVICE_UNAVAILABLE, status.HTTP_503_SERVICE_UNAVAILABLE),
    ],
)
async def test_complete_v2_clamps_untrusted_provider_http_status(
    monkeypatch: pytest.MonkeyPatch,
    provider_status: Any,
    expected_status: int,
) -> None:
    """Provider exceptions may select only valid HTTP error status codes."""

    def fake_provider_call(**_kwargs: Any) -> dict[str, Any]:
        raise ChatAPIError(
            "provider status sentinel",
            status_code=provider_status,
            provider="local-llm",
        )

    monkeypatch.setenv("ENABLE_LOCAL_LLM_PROVIDER", "true")
    _install_simple_credential_runtime(monkeypatch)
    monkeypatch.setattr(character_chat_sessions, "provider_requires_api_key", lambda _provider: False)
    monkeypatch.setattr(character_chat_sessions, "get_character_rate_limiter", lambda: _NoopCharacterRateLimiter())
    monkeypatch.setattr(character_chat_sessions, "_should_enforce_char_chat_strict_model_selection", lambda: False)
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", fake_provider_call)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(
                provider="local-llm",
                model="local-test",
                save_to_db=False,
                include_character_context=False,
            ),
            db=_CompletionReadyChatSessionDb(),
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == "Chat provider error"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_complete_v2_detaches_unexpected_provider_error_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected adapter errors cannot remain reachable from the public error."""

    db = _CompletionReadyChatSessionDb()
    sentinel = "sk-character-provider-/private/provider/cache/path"

    def fake_provider_call(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(sentinel)

    monkeypatch.setenv("ENABLE_LOCAL_LLM_PROVIDER", "true")
    _install_simple_credential_runtime(monkeypatch)
    monkeypatch.setattr(character_chat_sessions, "provider_requires_api_key", lambda provider: False)
    monkeypatch.setattr(character_chat_sessions, "get_character_rate_limiter", lambda: _NoopCharacterRateLimiter())
    monkeypatch.setattr(character_chat_sessions, "_should_enforce_char_chat_strict_model_selection", lambda: False)
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", fake_provider_call)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(
                provider="local-llm",
                model="local-test",
                save_to_db=False,
                include_character_context=False,
            ),
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
    assert exc_info.value.detail == "Chat provider error"
    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_complete_v2_forwards_inference_prefix_cache_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _CompletionReadyChatSessionDb()
    captured_kwargs: dict[str, Any] = {}

    def fake_provider_call(**kwargs: Any) -> dict[str, Any]:
        captured_kwargs.update(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setenv("ENABLE_LOCAL_LLM_PROVIDER", "true")
    _install_simple_credential_runtime(monkeypatch)
    monkeypatch.setattr(character_chat_sessions, "provider_requires_api_key", lambda provider: False)
    monkeypatch.setattr(character_chat_sessions, "get_character_rate_limiter", lambda: _NoopCharacterRateLimiter())
    monkeypatch.setattr(character_chat_sessions, "_should_enforce_char_chat_strict_model_selection", lambda: False)
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", fake_provider_call)

    response = await character_chat_sessions.character_chat_completion(
        chat_id="chat-1",
        body=CharacterChatCompletionV2Request(
            provider="local-llm",
            model="local-test",
            save_to_db=False,
            include_character_context=False,
            inference_prefix_cache_intent={"enabled": True, "scope": ["world_books"]},
        ),
        db=db,
        current_user=_test_user(),
    )

    assert response.assistant_content == "ok"
    assert captured_kwargs["inference_prefix_cache_intent"] == {
        "enabled": True,
        "scope": ["world_books"],
    }


@pytest.mark.asyncio
async def test_complete_v2_prompt_guardrail_errors_fail_closed_before_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _CompletionReadyChatSessionDb()
    called_provider = False

    def fake_provider_call(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called_provider
        called_provider = True
        return {"choices": [{"message": {"role": "assistant", "content": "late"}}]}

    def raise_guardrail_error(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("guardrail evaluator failed")

    monkeypatch.setenv("ENABLE_LOCAL_LLM_PROVIDER", "true")
    _install_simple_credential_runtime(monkeypatch)
    monkeypatch.setattr(character_chat_sessions, "provider_requires_api_key", lambda provider: False)
    monkeypatch.setattr(character_chat_sessions, "get_character_rate_limiter", lambda: _NoopCharacterRateLimiter())
    monkeypatch.setattr(character_chat_sessions, "_should_enforce_char_chat_strict_model_selection", lambda: False)
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", fake_provider_call)
    monkeypatch.setattr(
        character_chat_sessions,
        "load_prompt_cost_guardrail_config",
        lambda: PromptCostGuardrailConfig(enabled=True),
    )
    monkeypatch.setattr(character_chat_sessions, "evaluate_prompt_cost_guardrails", raise_guardrail_error)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.character_chat_completion(
            chat_id="chat-1",
            body=CharacterChatCompletionV2Request(
                provider="local-llm",
                model="local-test",
                save_to_db=False,
                include_character_context=False,
            ),
            db=db,
            current_user=_test_user(),
        )

    assert called_provider is False
    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail["type"] == "prompt_cost_guardrail_error"


@pytest.mark.asyncio
async def test_persist_streamed_assistant_message_maps_db_error_to_sanitized_500() -> None:
    db = _BrokenChatSessionDb(CharactersRAGDBError("sqlite persist exploded"), raise_on_get=True)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.persist_streamed_assistant_message(
            chat_id="chat-1",
            body=CharacterChatStreamPersistRequest(assistant_content="hello"),
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to persist assistant message"


@pytest.mark.asyncio
async def test_persist_streamed_assistant_message_maps_conflict_error_to_409() -> None:
    db = _BrokenChatSessionDb(ConflictError("chat persist conflict"), raise_on_get=True)

    with pytest.raises(HTTPException) as exc_info:
        await character_chat_sessions.persist_streamed_assistant_message(
            chat_id="chat-1",
            body=CharacterChatStreamPersistRequest(assistant_content="hello"),
            db=db,
            current_user=_test_user(),
        )

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "chat persist conflict"
