"""Tests for ACP prompt preprocessing utilities (@mention resolution)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_extracts_mentions():
    """@tool_name patterns are extracted from message content."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    messages = [{"role": "user", "content": "Use @search_web to find info about @linear tasks"}]
    msgs, hints = await preprocess_mentions(messages)
    assert "search_web" in hints
    assert "linear" in hints


@pytest.mark.asyncio
async def test_no_mentions():
    """Messages without @mentions return empty hints."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    messages = [{"role": "user", "content": "Hello world"}]
    msgs, hints = await preprocess_mentions(messages)
    assert hints == []


@pytest.mark.asyncio
async def test_registry_validation():
    """Only registered tools appear in hints."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    registry = AsyncMock()
    registry.tool_exists = AsyncMock(side_effect=lambda n: n == "search_web")
    messages = [{"role": "user", "content": "@search_web and @nonexistent"}]
    msgs, hints = await preprocess_mentions(messages, tool_registry=registry)
    assert hints == ["search_web"]


@pytest.mark.asyncio
async def test_cache_prevents_repeated_lookups():
    """Cached results avoid repeated registry calls."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    registry = AsyncMock()
    registry.tool_exists = AsyncMock(return_value=True)
    cache: dict[str, bool] = {}
    messages = [{"role": "user", "content": "@tool1 @tool1 @tool1"}]
    await preprocess_mentions(messages, tool_registry=registry, cache=cache)
    assert registry.tool_exists.call_count == 1  # Only called once due to cache


@pytest.mark.asyncio
async def test_non_string_content_skipped():
    """Non-string content (e.g., list for multimodal) is skipped."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    messages = [{"role": "user", "content": [{"type": "image", "url": "..."}]}]
    msgs, hints = await preprocess_mentions(messages)
    assert hints == []


@pytest.mark.asyncio
async def test_email_not_matched():
    """Email addresses should not be matched as @mentions."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    messages = [{"role": "user", "content": "Contact user@example.com"}]
    msgs, hints = await preprocess_mentions(messages)
    assert "example.com" not in hints


@pytest.mark.asyncio
async def test_messages_not_modified():
    """Original messages are returned unchanged (no text replacement)."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    original_content = "Use @search_web to look things up"
    messages = [{"role": "user", "content": original_content}]
    msgs, hints = await preprocess_mentions(messages)
    assert msgs[0]["content"] == original_content


@pytest.mark.asyncio
async def test_hints_sorted():
    """Returned hints are sorted alphabetically."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    messages = [{"role": "user", "content": "@zebra @alpha @middle"}]
    msgs, hints = await preprocess_mentions(messages)
    assert hints == ["alpha", "middle", "zebra"]


@pytest.mark.asyncio
async def test_cache_persists_across_calls():
    """The same cache dict accumulates results across multiple calls."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    cache: dict[str, bool] = {}
    registry = AsyncMock()
    registry.tool_exists = AsyncMock(return_value=True)

    await preprocess_mentions(
        [{"role": "user", "content": "@tool_a"}],
        tool_registry=registry,
        cache=cache,
    )
    assert "tool_a" in cache

    # Second call with same cache should not call registry for tool_a again
    registry.tool_exists.reset_mock()
    await preprocess_mentions(
        [{"role": "user", "content": "@tool_a @tool_b"}],
        tool_registry=registry,
        cache=cache,
    )
    # Only tool_b should trigger a registry call
    registry.tool_exists.assert_called_once_with("tool_b")


@pytest.mark.asyncio
async def test_registry_exception_treated_as_unresolved():
    """If registry raises, the mention is treated as unresolved."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    registry = AsyncMock()
    registry.tool_exists = AsyncMock(side_effect=RuntimeError("connection failed"))
    messages = [{"role": "user", "content": "@broken_tool"}]
    msgs, hints = await preprocess_mentions(messages, tool_registry=registry)
    assert hints == []


@pytest.mark.asyncio
async def test_multiple_messages_scanned():
    """Mentions across multiple messages are all collected."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    messages = [
        {"role": "system", "content": "You can use @system_tool"},
        {"role": "user", "content": "Please use @user_tool"},
    ]
    msgs, hints = await preprocess_mentions(messages)
    assert "system_tool" in hints
    assert "user_tool" in hints


@pytest.mark.asyncio
async def test_dotted_tool_name():
    """Tool names with dots (e.g., namespace.tool) are matched."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    messages = [{"role": "user", "content": "Use @mcp.search_web"}]
    msgs, hints = await preprocess_mentions(messages)
    assert "mcp.search_web" in hints


@pytest.mark.asyncio
async def test_hyphenated_tool_name():
    """Tool names with hyphens are matched."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

    messages = [{"role": "user", "content": "Use @my-tool"}]
    msgs, hints = await preprocess_mentions(messages)
    assert "my-tool" in hints
