import pytest
from unittest.mock import patch


@pytest.mark.integration
@pytest.mark.streaming
@pytest.mark.asyncio
async def test_chat_stream_disconnect_no_generatorexit(async_client, auth_headers, monkeypatch):
    """Open a streaming chat, read a bit, then disconnect early.
    Ensures no RuntimeError('generator ignored GeneratorExit') surfaces.
    Consolidated from Chat_NEW.
    """

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Mock provider to return a simple sync generator of text chunks
    def provider_stream_generator():
        yield "This is"
        yield " a streaming"
        yield " response."

    with patch(
        "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
        return_value=provider_stream_generator(),
    ):
        async with async_client.stream(
            "POST",
            "/api/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
            headers=auth_headers,
        ) as resp:
            assert resp.status_code == 200
            # Read only a couple of chunks then disconnect
            received_some_data = False
            async for chunk in resp.aiter_text():
                if not chunk:
                    continue
                if "data: " in chunk and "\"delta\"" in chunk:
                    received_some_data = True
                    break  # Simulate client disconnect early
            assert received_some_data is True
        # Exiting the context closes the connection. The absence of exceptions here
        # indicates generators handled GeneratorExit correctly.
