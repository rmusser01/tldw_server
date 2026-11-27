import asyncio

import pytest

from tldw_Server_API.app.core.Chat import chat_service


class DummyQueue:
    def __init__(self):
        self.is_running = True
        self.estimated_tokens = None

    async def enqueue(
        self,
        *,
        request_id,
        request_data,
        client_id,
        priority,
        estimated_tokens,
        processor,
        processor_args,
        processor_kwargs,
        streaming,
        stream_channel,
    ):
        self.estimated_tokens = estimated_tokens
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(processor())
        return fut


class DummyMetrics:
    def track_llm_call(self, *args, **kwargs):
        pass

    def track_provider_fallback_success(self, *args, **kwargs):
        pass

    def track_rate_limit(self, *args, **kwargs):
        pass


class DummyProviderManager:
    def record_success(self, *args, **kwargs):
        pass

    def record_failure(self, *args, **kwargs):
        pass


class DummyPolicy:
    enabled = False
    output_enabled = False


class DummyModeration:
    def get_effective_policy(self, _user_id, surface=None):
        return DummyPolicy()


@pytest.mark.asyncio
async def test_queue_estimate_uses_sanitized_request(monkeypatch):
    request_json = (
        "{\"messages\":[{\"content\":\"data:image/png;base64," + "A" * 10000 + "\"}]}"
    )
    expected = chat_service.estimate_tokens_from_json(request_json)

    queue = DummyQueue()
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "log_llm_usage", lambda *args, **kwargs: asyncio.sleep(0))

    async def _noop_save(*args, **kwargs):
        return None

    response = await chat_service.execute_non_stream_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={"messages_payload": []},
        selected_provider="openai",
        provider="openai",
        model="gpt-test",
        request_json=request_json,
        request=None,
        metrics=DummyMetrics(),
        provider_manager=DummyProviderManager(),
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv",
        character_card_for_context=None,
        chat_db=None,
        save_message_fn=_noop_save,
        system_message_id=None,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=True,
        enable_provider_fallback=False,
        llm_call_func=lambda: {"choices": [{"message": {"content": "ok"}}]},
        refresh_provider_params=lambda provider: ({"messages_payload": []}, "gpt-test"),
        moderation_getter=lambda: DummyModeration(),
        on_success=None,
    )

    assert response
    assert queue.estimated_tokens == expected
