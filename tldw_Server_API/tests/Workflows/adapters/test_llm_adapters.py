"""Comprehensive tests for LLM workflow adapters.

This module tests all 7 LLM adapters:
1. run_llm_adapter - Basic LLM chat completion
2. run_llm_with_tools_adapter - LLM with tool calling
3. run_llm_compare_adapter - Compare multiple LLM responses
4. run_llm_critique_adapter - LLM critique of text
5. run_moderation_adapter - Content moderation
6. run_policy_check_adapter - Policy compliance check
7. run_translate_adapter - Translation via LLM
"""

import pytest

import tldw_Server_API.app.core.Workflows.adapters as wf_adapters
from tldw_Server_API.app.core.Workflows.adapters import (
    run_llm_adapter,
    run_llm_with_tools_adapter,
    run_llm_compare_adapter,
    run_llm_critique_adapter,
    run_moderation_adapter,
    run_policy_check_adapter,
    run_translate_adapter,
)

pytestmark = pytest.mark.unit


# =============================================================================
# run_llm_adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_llm_adapter_with_prompt(monkeypatch):
    """Test LLM adapter with prompt config."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def fake_call(**kwargs):
        return {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {"provider": "openai", "model": "gpt-4", "prompt": "Hello"}
    context = {"user_id": "1"}

    result = await run_llm_adapter(config, context)
    assert result["text"] == "Test response"


@pytest.mark.asyncio
async def test_llm_adapter_with_template_rendering(monkeypatch):
    """Test LLM adapter renders templates in prompts."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    calls = {}

    async def fake_call(**kwargs):
        calls["kwargs"] = kwargs
        return {
            "choices": [{"message": {"content": "Hello Alice"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {"provider": "openai", "model": "gpt-4", "prompt": "Greet {{ inputs.name }}"}
    context = {"inputs": {"name": "Alice"}, "user_id": "1"}

    result = await run_llm_adapter(config, context)
    assert result["text"] == "Hello Alice"
    assert calls["kwargs"]["messages_payload"][0]["content"] == "Greet Alice"


@pytest.mark.asyncio
async def test_llm_adapter_template_fallback_log_sanitizes_backend_error(monkeypatch):
    """Test template fallback logs hide raw backend exception details."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service
    import tldw_Server_API.app.core.Chat.prompt_template_manager as prompt_template_manager
    from tldw_Server_API.app.core.Workflows.adapters.llm import llm

    async def fake_call(**kwargs):
        return {"choices": [{"message": {"content": "Fallback response"}}]}

    def broken_template(_value, _context):
        raise RuntimeError("template backend exploded at /private/templates.db")

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)
    monkeypatch.setattr(prompt_template_manager, "apply_template_to_string", broken_template)

    messages = []
    sink_id = llm.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        result = await run_llm_adapter(
            {"provider": "openai", "model": "gpt-4", "prompt": "Greet {{ inputs.name }}"},
            {"inputs": {"name": "Alice"}, "user_id": "1"},
        )
    finally:
        llm.logger.remove(sink_id)

    assert result["text"] == "Fallback response"
    joined = "\n".join(messages)
    assert "LLM adapter: template rendering failed" in joined
    assert "templates.db" not in joined


@pytest.mark.asyncio
async def test_llm_adapter_with_messages(monkeypatch):
    """Test LLM adapter with messages array instead of prompt."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    calls = {}

    async def fake_call(**kwargs):
        calls["kwargs"] = kwargs
        return {"choices": [{"message": {"content": "Response"}}]}

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {
        "provider": "openai",
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
    }
    context = {"user_id": "1"}

    result = await run_llm_adapter(config, context)
    assert result["text"] == "Response"
    assert len(calls["kwargs"]["messages_payload"]) == 3


@pytest.mark.asyncio
async def test_llm_adapter_with_system_message(monkeypatch):
    """Test LLM adapter passes system message."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    calls = {}

    async def fake_call(**kwargs):
        calls["kwargs"] = kwargs
        return {"choices": [{"message": {"content": "I am a pirate!"}}]}

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {
        "provider": "openai",
        "model": "gpt-4",
        "prompt": "Introduce yourself",
        "system_message": "You are a pirate.",
    }
    context = {"user_id": "1"}

    result = await run_llm_adapter(config, context)
    assert result["text"] == "I am a pirate!"
    assert calls["kwargs"]["system_message"] == "You are a pirate."


@pytest.mark.asyncio
async def test_llm_adapter_includes_metadata(monkeypatch):
    """Test LLM adapter returns token usage metadata."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def fake_call(**kwargs):
        return {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "cost_usd": 0.001,
        }

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {"provider": "openai", "model": "gpt-4", "prompt": "Test"}
    context = {"user_id": "1"}

    result = await run_llm_adapter(config, context)
    assert "metadata" in result
    assert result["metadata"]["token_usage"]["total_tokens"] == 30
    assert result["metadata"]["cost_usd"] == 0.001


@pytest.mark.asyncio
async def test_llm_adapter_include_response(monkeypatch):
    """Test LLM adapter includes raw response when requested."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def fake_call(**kwargs):
        return {
            "choices": [{"message": {"content": "Raw response"}}],
            "model": "gpt-4",
        }

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {"provider": "openai", "model": "gpt-4", "prompt": "Test", "include_response": True}
    context = {"user_id": "1"}

    result = await run_llm_adapter(config, context)
    assert "response" in result
    assert result["response"]["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_llm_adapter_missing_prompt_error(monkeypatch):
    """Test LLM adapter raises error when prompt is missing."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    from tldw_Server_API.app.core.exceptions import AdapterError

    config = {"provider": "openai", "model": "gpt-4"}
    context = {"user_id": "1"}

    with pytest.raises(AdapterError) as exc_info:
        await run_llm_adapter(config, context)
    assert "missing_prompt" in str(exc_info.value)


@pytest.mark.asyncio
async def test_llm_adapter_test_mode_simulation(monkeypatch):
    """Test LLM adapter returns simulated response in TEST_MODE."""
    monkeypatch.setenv("TEST_MODE", "1")

    config = {"provider": "openai", "model": "gpt-4", "prompt": "Hello test"}
    context = {"user_id": "1"}

    result = await run_llm_adapter(config, context)
    assert result["simulated"] is True
    assert result["text"] == "Hello test"
    assert result["provider"] == "openai"


@pytest.mark.asyncio
async def test_llm_adapter_streaming(monkeypatch):
    """Test LLM adapter handles streaming response."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def make_stream():
        """Create an async generator for streaming."""
        chunks = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " World"}}]}',
            b"data: [DONE]",
        ]
        for chunk in chunks:
            yield chunk

    async def fake_stream_caller(**kwargs):
        """Return the async generator (not awaited by caller in stream mode)."""
        return make_stream()

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_stream_caller)

    config = {"provider": "openai", "model": "gpt-4", "prompt": "Test", "stream": True}
    context = {"user_id": "1"}

    result = await run_llm_adapter(config, context)
    assert result["streamed"] is True
    assert result["text"] == "Hello World"


@pytest.mark.asyncio
async def test_llm_adapter_streaming_sanitizes_event_dispatch_errors(monkeypatch):
    """Test streaming event dispatch logs hide raw exception details."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service
    from tldw_Server_API.app.core.Workflows.adapters.llm import llm

    async def make_stream():
        yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}'
        yield b"data: [DONE]"

    async def fake_stream_caller(**kwargs):
        return make_stream()

    def broken_append_event(_name, _payload):
        raise RuntimeError("event bus exploded at /private/stream-events")

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_stream_caller)

    messages = []
    sink_id = llm.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        result = await run_llm_adapter(
            {"provider": "openai", "model": "gpt-4", "prompt": "Test", "stream": True},
            {"user_id": "1", "append_event": broken_append_event},
        )
    finally:
        llm.logger.remove(sink_id)

    assert result == {"text": "Hello", "streamed": True}
    joined = "\n".join(messages)
    assert "LLM stream event dispatch failed" in joined
    assert "stream-events" not in joined


# =============================================================================
# run_llm_with_tools_adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_llm_with_tools_adapter_basic(monkeypatch):
    """Test LLM with tools adapter basic functionality."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def fake_call(**kwargs):
        return {"choices": [{"message": {"content": "Tool response complete"}}]}

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {
        "provider": "openai",
        "model": "gpt-4",
        "prompt": "Search for information",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for info",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    context = {"user_id": "1"}

    result = await run_llm_with_tools_adapter(config, context)
    assert result["text"] == "Tool response complete"
    assert isinstance(result["tool_results"], list)


@pytest.mark.asyncio
async def test_llm_with_tools_adapter_tool_execution(monkeypatch):
    """Test LLM with tools adapter executes tools.

    Note: The MCP manager module may not exist, so this test verifies
    that tool calls are attempted and errors are handled gracefully.
    """
    monkeypatch.delenv("TEST_MODE", raising=False)

    import sys
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    call_count = 0

    async def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": "Weather is sunny"}}]}

    class FakeMCPManager:
        async def execute_tool(self, name, args, context=None):
            return {"result": "sunny", "temp": 72}

    # Create a mock module for MCP manager
    class FakeManagerModule:
        @staticmethod
        def get_mcp_manager():
            return FakeMCPManager()

    # Insert mock module before importing
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.MCP_unified.manager", FakeManagerModule)
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {
        "provider": "openai",
        "model": "gpt-4",
        "prompt": "What is the weather?",
        "tools": [],
        "auto_execute": True,
    }
    context = {"user_id": "1"}

    result = await run_llm_with_tools_adapter(config, context)
    assert result["text"] == "Weather is sunny"
    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["tool"] == "get_weather"


@pytest.mark.asyncio
async def test_llm_with_tools_adapter_sanitizes_tool_execution_errors(monkeypatch):
    """Test tool execution failures hide raw backend exception details."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import sys
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    call_count = 0

    async def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {"name": "lookup_secret", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": "Handled sanitized tool error"}}]}

    class FakeMCPManager:
        async def execute_tool(self, name, args, context=None):
            raise RuntimeError("tool backend exploded at /private/tool-cache")

    class FakeManagerModule:
        @staticmethod
        def get_mcp_manager():
            return FakeMCPManager()

    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.MCP_unified.manager", FakeManagerModule)
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {"prompt": "Use a tool", "tools": [], "auto_execute": True}
    result = await run_llm_with_tools_adapter(config, {"user_id": "1"})

    assert result["text"] == "Handled sanitized tool error"
    assert result["tool_results"] == [{"tool": "lookup_secret", "error": "tool_execution_error"}]
    assert "tool-cache" not in str(result)


@pytest.mark.asyncio
async def test_llm_with_tools_adapter_cancellation(monkeypatch):
    """Test LLM with tools adapter respects cancellation."""
    config = {"prompt": "Test", "tools": []}
    context = {"user_id": "1", "is_cancelled": lambda: True}

    result = await run_llm_with_tools_adapter(config, context)
    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_llm_with_tools_adapter_max_iterations(monkeypatch):
    """Test LLM with tools adapter respects max_tool_calls."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import sys
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def fake_call(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "call_1", "function": {"name": "loop", "arguments": "{}"}}
                        ],
                    }
                }
            ]
        }

    class FakeMCPManager:
        async def execute_tool(self, name, args, context=None):
            return {"status": "ok"}

    # Create a mock module for MCP manager
    class FakeManagerModule:
        @staticmethod
        def get_mcp_manager():
            return FakeMCPManager()

    # Insert mock module before importing
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.MCP_unified.manager", FakeManagerModule)
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {"prompt": "Loop forever", "tools": [], "max_tool_calls": 2, "auto_execute": True}
    context = {"user_id": "1"}

    result = await run_llm_with_tools_adapter(config, context)
    # Should stop after max iterations
    assert result["iterations"] <= 3  # max_tool_calls + 1


@pytest.mark.asyncio
async def test_llm_with_tools_adapter_error_handling(monkeypatch):
    """Test LLM with tools adapter handles errors gracefully."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def fake_call(**kwargs):
        raise RuntimeError("LLM tools backend exploded at /private/llm-cache")

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {"prompt": "Test", "tools": []}
    context = {"user_id": "1"}

    result = await run_llm_with_tools_adapter(config, context)
    assert result == {"error": "llm_with_tools_error", "text": "", "tool_results": []}
    assert "llm-cache" not in str(result)


# =============================================================================
# run_llm_compare_adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_llm_compare_adapter_multiple_providers(monkeypatch):
    """Test LLM compare adapter with multiple providers."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    responses = {
        "openai": "OpenAI response",
        "anthropic": "Anthropic response",
    }

    async def fake_call(**kwargs):
        provider = kwargs.get("api_provider", "openai")
        return {"choices": [{"message": {"content": responses.get(provider, "default")}}]}

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {
        "prompt": "What is AI?",
        "providers": [
            {"provider": "openai", "model": "gpt-4"},
            {"provider": "anthropic", "model": "claude-3"},
        ],
    }
    context = {"user_id": "1"}

    result = await run_llm_compare_adapter(config, context)
    assert len(result["responses"]) == 2
    assert result["comparison"]["provider_count"] == 2
    assert result["comparison"]["successful"] == 2


@pytest.mark.asyncio
async def test_llm_compare_adapter_with_failures(monkeypatch):
    """Test LLM compare adapter handles provider failures."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    call_count = 0

    async def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Provider exploded at /private/llm-cache")
        return {"choices": [{"message": {"content": "Success"}}]}

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {
        "prompt": "Test",
        "providers": [
            {"provider": "openai", "model": "gpt-4"},
            {"provider": "anthropic", "model": "claude-3"},
        ],
    }
    context = {"user_id": "1"}

    result = await run_llm_compare_adapter(config, context)
    assert result["comparison"]["successful"] == 1
    assert result["comparison"]["failed"] == 1
    assert result["responses"][1]["error"] == "llm_provider_error"
    assert "llm-cache" not in str(result)


@pytest.mark.asyncio
async def test_llm_compare_adapter_missing_prompt():
    """Test LLM compare adapter returns error for missing prompt."""
    config = {"providers": [{"provider": "openai"}]}
    context = {"user_id": "1"}

    result = await run_llm_compare_adapter(config, context)
    assert result["error"] == "missing_prompt"


@pytest.mark.asyncio
async def test_llm_compare_adapter_missing_providers():
    """Test LLM compare adapter returns error for missing providers."""
    config = {"prompt": "Test"}
    context = {"user_id": "1"}

    result = await run_llm_compare_adapter(config, context)
    assert result["error"] == "missing_providers"


@pytest.mark.asyncio
async def test_llm_compare_adapter_cancellation():
    """Test LLM compare adapter respects cancellation."""
    config = {"prompt": "Test", "providers": [{"provider": "openai"}]}
    context = {"user_id": "1", "is_cancelled": lambda: True}

    result = await run_llm_compare_adapter(config, context)
    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_llm_compare_adapter_timing_metadata(monkeypatch):
    """Test LLM compare adapter includes timing metadata."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def fake_call(**kwargs):
        return {"choices": [{"message": {"content": "Response"}}]}

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {
        "prompt": "Test",
        "providers": [{"provider": "openai", "model": "gpt-4"}],
    }
    context = {"user_id": "1"}

    result = await run_llm_compare_adapter(config, context)
    assert "elapsed_ms" in result["responses"][0]
    assert "char_count" in result["responses"][0]


# =============================================================================
# run_llm_critique_adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_llm_critique_adapter_basic(monkeypatch):
    """Test LLM critique adapter basic functionality."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    call_count = 0

    async def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"choices": [{"message": {"content": "The text lacks clarity."}}]}
        return {"choices": [{"message": {"content": "Improved version here."}}]}

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {
        "content": "This is some test content.",
        "criteria": ["clarity", "accuracy"],
        "revise": True,
    }
    context = {"user_id": "1"}

    result = await run_llm_critique_adapter(config, context)
    assert result["critique"] == "The text lacks clarity."
    assert result["revised"] == "Improved version here."
    assert result["criteria"] == ["clarity", "accuracy"]


@pytest.mark.asyncio
async def test_llm_critique_adapter_no_revise(monkeypatch):
    """Test LLM critique adapter without revision."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def fake_call(**kwargs):
        return {"choices": [{"message": {"content": "Critique only"}}]}

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {"content": "Test content", "revise": False}
    context = {"user_id": "1"}

    result = await run_llm_critique_adapter(config, context)
    assert result["critique"] == "Critique only"
    assert result["revised"] == ""


@pytest.mark.asyncio
async def test_llm_critique_adapter_missing_content():
    """Test LLM critique adapter returns error for missing content."""
    config = {"criteria": ["clarity"]}
    context = {"user_id": "1"}

    result = await run_llm_critique_adapter(config, context)
    assert result["error"] == "missing_content"


@pytest.mark.asyncio
async def test_llm_critique_adapter_uses_prev_context(monkeypatch):
    """Test LLM critique adapter uses prev context when content not specified."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def fake_call(**kwargs):
        return {"choices": [{"message": {"content": "Critique of prev content"}}]}

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {"revise": False}
    context = {"user_id": "1", "prev": {"text": "Previous step content"}}

    result = await run_llm_critique_adapter(config, context)
    assert result["critique"] == "Critique of prev content"


@pytest.mark.asyncio
async def test_llm_critique_adapter_cancellation():
    """Test LLM critique adapter respects cancellation."""
    config = {"content": "Test"}
    context = {"user_id": "1", "is_cancelled": lambda: True}

    result = await run_llm_critique_adapter(config, context)
    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_llm_critique_adapter_error_handling(monkeypatch):
    """Test LLM critique adapter handles errors."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def fake_call(**kwargs):
        raise RuntimeError("Critique backend exploded at /private/llm-cache")

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {"content": "Test content"}
    context = {"user_id": "1"}

    result = await run_llm_critique_adapter(config, context)
    assert result == {"error": "llm_critique_error", "critique": "", "revised": ""}
    assert "llm-cache" not in str(result)


# =============================================================================
# run_moderation_adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_moderation_adapter_check_allowed(monkeypatch):
    """Test moderation adapter check action with allowed content."""
    monkeypatch.setenv("TEST_MODE", "1")

    config = {"action": "check", "text": "This is safe content."}
    context = {"user_id": "1"}

    result = await run_moderation_adapter(config, context)
    assert result["allowed"] is True
    assert result["reason"] == "passed"
    assert result["simulated"] is True


@pytest.mark.asyncio
async def test_moderation_adapter_check_blocked(monkeypatch):
    """Test moderation adapter check action with blocked content."""
    monkeypatch.setenv("TEST_MODE", "1")

    config = {"action": "check", "text": "This content is blocked."}
    context = {"user_id": "1"}

    result = await run_moderation_adapter(config, context)
    assert result["allowed"] is False
    assert result["reason"] == "contains_blocked_term"
    assert "test_blocked_term" in result["matched_rules"]


@pytest.mark.asyncio
async def test_moderation_adapter_redact(monkeypatch):
    """Test moderation adapter redact action."""
    monkeypatch.setenv("TEST_MODE", "1")

    config = {"action": "redact", "text": "My password is secret123"}
    context = {"user_id": "1"}

    result = await run_moderation_adapter(config, context)
    assert "[REDACTED]" in result["redacted_text"]
    assert result["redaction_count"] >= 1


@pytest.mark.asyncio
async def test_moderation_adapter_redact_custom_patterns(monkeypatch):
    """Test moderation adapter redact with custom patterns."""
    monkeypatch.setenv("TEST_MODE", "1")

    config = {
        "action": "redact",
        "text": "Contact me at john@example.com",
        "patterns": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
    }
    context = {"user_id": "1"}

    result = await run_moderation_adapter(config, context)
    assert "[REDACTED]" in result["redacted_text"]


@pytest.mark.asyncio
async def test_moderation_adapter_sanitizes_invalid_custom_pattern_logs(monkeypatch):
    """Test invalid custom redaction regex logs hide raw pattern/parser details."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Moderation.moderation_service as moderation_service
    from tldw_Server_API.app.core.Workflows.adapters.llm import moderation as moderation_module

    class _Policy:
        redact_replacement = "[REDACTED]"

    class _Service:
        def get_effective_policy(self, _user_id):
            return _Policy()

        def redact_text_with_count(self, text, _policy):
            return text, 0

    monkeypatch.setattr(moderation_service, "get_moderation_service", lambda: _Service())

    messages: list[str] = []
    sink_id = moderation_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        result = await run_moderation_adapter(
            {"action": "redact", "text": "Contact private-token", "patterns": ["private-token("]},
            {"user_id": "1"},
        )
    finally:
        moderation_module.logger.remove(sink_id)

    assert result["text"] == "Contact private-token"
    joined = "\n".join(messages)
    assert "Invalid custom redaction pattern skipped" in joined
    assert "private-token" not in joined
    assert "missing ), unterminated subpattern" not in joined


@pytest.mark.asyncio
async def test_moderation_adapter_missing_text():
    """Test moderation adapter returns error for missing text."""
    config = {"action": "check"}
    context = {"user_id": "1"}

    result = await run_moderation_adapter(config, context)
    assert result["error"] == "missing_text"


@pytest.mark.asyncio
async def test_moderation_adapter_cancellation():
    """Test moderation adapter respects cancellation."""
    config = {"action": "check", "text": "Test"}
    context = {"user_id": "1", "is_cancelled": lambda: True}

    result = await run_moderation_adapter(config, context)
    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_moderation_adapter_uses_prev_context(monkeypatch):
    """Test moderation adapter uses prev context when text not specified."""
    monkeypatch.setenv("TEST_MODE", "1")

    config = {"action": "check"}
    context = {"user_id": "1", "prev": {"text": "Safe content from previous step"}}

    result = await run_moderation_adapter(config, context)
    assert result["allowed"] is True


@pytest.mark.asyncio
async def test_moderation_adapter_unknown_action(monkeypatch):
    """Test moderation adapter returns error for unknown action."""
    monkeypatch.setenv("TEST_MODE", "1")

    config = {"action": "unknown", "text": "Test"}
    context = {"user_id": "1"}

    result = await run_moderation_adapter(config, context)
    assert "unknown_action" in result["error"]


@pytest.mark.asyncio
async def test_moderation_adapter_sanitizes_backend_errors(monkeypatch):
    """Test moderation backend failures hide raw exception details."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Moderation.moderation_service as moderation_service

    def fake_get_service():
        raise RuntimeError("moderation backend exploded at /private/policy.db")

    monkeypatch.setattr(moderation_service, "get_moderation_service", fake_get_service)

    result = await run_moderation_adapter({"action": "check", "text": "safe text"}, {"user_id": "1"})

    assert result == {"error": "moderation_error"}
    assert "policy.db" not in str(result)


# =============================================================================
# run_policy_check_adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_policy_check_adapter_no_violations():
    """Test policy check adapter with no violations."""
    config = {"text_source": "last"}
    context = {"user_id": "1", "prev": {"text": "Clean text without issues"}}

    result = await run_policy_check_adapter(config, context)
    assert result["blocked"] is False
    assert result["reasons"] == []


@pytest.mark.asyncio
async def test_policy_check_adapter_block_words():
    """Test policy check adapter detects blocked words."""
    config = {"text_source": "last", "block_words": ["forbidden", "banned"]}
    context = {"user_id": "1", "prev": {"text": "This contains forbidden content"}}

    result = await run_policy_check_adapter(config, context)
    assert result["blocked"] is True
    assert "blocked_terms" in result["reasons"]
    assert "forbidden" in result["flags"]["block_words"]


@pytest.mark.asyncio
async def test_policy_check_adapter_max_length():
    """Test policy check adapter enforces max length."""
    config = {"text_source": "last", "max_length": 10}
    context = {"user_id": "1", "prev": {"text": "This text is way too long for the limit"}}

    result = await run_policy_check_adapter(config, context)
    assert result["blocked"] is True
    assert "too_long" in result["reasons"]
    assert result["flags"]["too_long"] is True


@pytest.mark.asyncio
async def test_policy_check_adapter_inputs_source():
    """Test policy check adapter with inputs source."""
    config = {"text_source": "inputs", "block_words": ["test"]}
    context = {
        "user_id": "1",
        "inputs": {"text": "This is a test input"},
    }

    result = await run_policy_check_adapter(config, context)
    assert result["blocked"] is True


@pytest.mark.asyncio
async def test_policy_check_adapter_field_source():
    """Test policy check adapter with field source."""
    config = {"text_source": "field", "field": "inputs.summary", "block_words": ["bad"]}
    context = {
        "user_id": "1",
        "inputs": {"summary": "This has bad content"},
    }

    result = await run_policy_check_adapter(config, context)
    assert result["blocked"] is True


@pytest.mark.asyncio
async def test_policy_check_adapter_redact_preview():
    """Test policy check adapter with redact preview."""
    config = {"text_source": "last", "redact_preview": True}
    context = {"user_id": "1", "prev": {"text": "Some content to preview"}}

    result = await run_policy_check_adapter(config, context)
    assert "preview" in result


@pytest.mark.asyncio
async def test_policy_check_adapter_combined_violations():
    """Test policy check adapter with multiple violations."""
    config = {
        "text_source": "last",
        "block_words": ["forbidden"],
        "max_length": 10,
    }
    context = {"user_id": "1", "prev": {"text": "This forbidden text is too long"}}

    result = await run_policy_check_adapter(config, context)
    assert result["blocked"] is True
    assert "blocked_terms" in result["reasons"]
    assert "too_long" in result["reasons"]


# =============================================================================
# run_translate_adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_translate_adapter_test_mode(monkeypatch):
    """Test translate adapter in test mode returns simulated response."""
    monkeypatch.setenv("TEST_MODE", "1")

    config = {"input": "Hello world", "target_lang": "es"}
    context = {"user_id": "1"}

    result = await run_translate_adapter(config, context)
    assert result["text"] == "Hello world"
    assert result["target_lang"] == "es"
    assert result["simulated"] is True


@pytest.mark.asyncio
async def test_translate_adapter_with_provider(monkeypatch):
    """Test translate adapter with OpenAI provider."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    # Import the translate module directly to patch get_registry at its usage site
    import tldw_Server_API.app.core.Workflows.adapters.llm.translate as translate_module

    class FakeAdapter:
        async def achat(self, config):
            return {"choices": [{"message": {"content": "Hola mundo"}}]}

    class FakeRegistry:
        def get_adapter(self, name):
            return FakeAdapter()

    # Patch get_registry where it's used in the translate module
    monkeypatch.setattr(translate_module, "get_registry", lambda: FakeRegistry())

    config = {"input": "Hello world", "target_lang": "es"}
    context = {"user_id": "1"}

    result = await run_translate_adapter(config, context)
    assert result["text"] == "Hola mundo"
    assert result["target_lang"] == "es"
    assert result["provider"] == "openai"


@pytest.mark.asyncio
async def test_translate_adapter_missing_input():
    """Test translate adapter returns error for missing input."""
    config = {"target_lang": "es"}
    context = {"user_id": "1"}

    result = await run_translate_adapter(config, context)
    assert result["error"] == "missing_input_text"


@pytest.mark.asyncio
async def test_translate_adapter_uses_prev_context(monkeypatch):
    """Test translate adapter uses prev context when input not specified."""
    monkeypatch.setenv("TEST_MODE", "1")

    config = {"target_lang": "fr"}
    context = {"user_id": "1", "prev": {"text": "Previous text to translate"}}

    result = await run_translate_adapter(config, context)
    assert result["text"] == "Previous text to translate"
    assert result["target_lang"] == "fr"


@pytest.mark.asyncio
async def test_translate_adapter_fallback_on_error(monkeypatch):
    """Test translate adapter fallback when provider fails."""
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.LLM_Calls.adapter_registry as adapter_registry

    class FakeRegistry:
        def get_adapter(self, name):
            raise Exception("Provider unavailable")

    monkeypatch.setattr(adapter_registry, "get_registry", lambda: FakeRegistry())

    config = {"input": "Hello", "target_lang": "de"}
    context = {"user_id": "1"}

    result = await run_translate_adapter(config, context)
    assert result["text"] == "Hello"
    assert result["fallback"] is True


@pytest.mark.asyncio
async def test_translate_adapter_default_target_lang(monkeypatch):
    """Test translate adapter uses default target language."""
    monkeypatch.setenv("TEST_MODE", "1")

    config = {"input": "Bonjour"}
    context = {"user_id": "1"}

    result = await run_translate_adapter(config, context)
    assert result["target_lang"] == "en"


# =============================================================================
# Import Verification Tests
# =============================================================================


def test_all_llm_adapters_importable():
    """Test that all LLM adapters can be imported."""
    adapters = [
        run_llm_adapter,
        run_llm_with_tools_adapter,
        run_llm_compare_adapter,
        run_llm_critique_adapter,
        run_moderation_adapter,
        run_policy_check_adapter,
        run_translate_adapter,
    ]
    for adapter in adapters:
        assert callable(adapter)


def test_llm_adapters_registered():
    """Test that all LLM adapters are registered in the registry."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    expected = ["llm", "llm_with_tools", "llm_compare", "llm_critique", "moderation", "policy_check", "translate"]
    for name in expected:
        assert registry.get_adapter(name) is not None, f"Adapter '{name}' not registered"


def test_llm_adapters_have_config_models():
    """Test that all LLM adapters have config models."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    expected = ["llm", "llm_with_tools", "llm_compare", "llm_critique", "moderation", "policy_check", "translate"]
    for name in expected:
        spec = registry.get_spec(name)
        assert spec.config_model is not None, f"Adapter '{name}' missing config_model"
