import pytest
from loguru import logger

from tldw_Server_API.app.core.RAG.rag_service import media_search as ms


pytestmark = pytest.mark.unit


@pytest.fixture
def captured_log_messages():
    messages: list[str] = []
    handler_id = logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        yield messages
    finally:
        logger.remove(handler_id)


@pytest.mark.asyncio
async def test_search_images_uses_to_thread_and_returns_normalized_results(monkeypatch):
    import tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs as web_apis

    captured: dict[str, object] = {"to_thread_calls": 0}

    async def _fake_reformulate_query(_query, _system, _provider, _model):  # noqa: ANN001
        return "KFC logo"

    def _fake_perform_websearch(**kwargs):  # noqa: ANN003
        captured["search_kwargs"] = kwargs
        return {
            "results": [
                {
                    "title": "KFC Logo",
                    "url": "https://example.com/kfc-logo",
                    "thumbnail": "https://example.com/kfc-logo-thumb.jpg",
                    "snippet": "Official KFC logo image",
                }
            ]
        }

    async def _fake_to_thread(func, *args, **kwargs):  # noqa: ANN001
        captured["to_thread_calls"] = int(captured["to_thread_calls"]) + 1
        captured["to_thread_func"] = getattr(func, "__name__", str(func))
        return func(*args, **kwargs)

    monkeypatch.setattr(ms, "_reformulate_query", _fake_reformulate_query)
    monkeypatch.setattr(web_apis, "perform_websearch", _fake_perform_websearch)
    monkeypatch.setattr(ms.asyncio, "to_thread", _fake_to_thread)

    images = await ms.search_images(
        query="What does the KFC logo look like?",
        max_results=1,
    )

    assert captured["to_thread_calls"] == 1
    assert captured["to_thread_func"] == "_fake_perform_websearch"
    assert captured["search_kwargs"]["search_query"] == "KFC logo images"
    assert len(images) == 1
    assert images[0]["title"] == "KFC Logo"
    assert images[0]["url"] == "https://example.com/kfc-logo"
    assert images[0]["thumbnail_url"] == "https://example.com/kfc-logo-thumb.jpg"


@pytest.mark.asyncio
async def test_search_videos_uses_to_thread_and_builds_youtube_thumbnail(monkeypatch):
    import tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs as web_apis

    captured: dict[str, object] = {"to_thread_calls": 0}

    async def _fake_reformulate_query(_query, _system, _provider, _model):  # noqa: ANN001
        return "python beginner tutorial"

    def _fake_perform_websearch(**kwargs):  # noqa: ANN003
        captured["search_kwargs"] = kwargs
        return {
            "results": [
                {
                    "title": "Python Beginner Tutorial",
                    "url": "https://www.youtube.com/watch?v=abcdefghijk",
                    "snippet": "Learn Python step by step.",
                }
            ]
        }

    async def _fake_to_thread(func, *args, **kwargs):  # noqa: ANN001
        captured["to_thread_calls"] = int(captured["to_thread_calls"]) + 1
        captured["to_thread_func"] = getattr(func, "__name__", str(func))
        return func(*args, **kwargs)

    monkeypatch.setattr(ms, "_reformulate_query", _fake_reformulate_query)
    monkeypatch.setattr(web_apis, "perform_websearch", _fake_perform_websearch)
    monkeypatch.setattr(ms.asyncio, "to_thread", _fake_to_thread)

    videos = await ms.search_videos(
        query="How do I learn Python?",
        max_results=1,
    )

    assert captured["to_thread_calls"] == 1
    assert captured["to_thread_func"] == "_fake_perform_websearch"
    assert captured["search_kwargs"]["search_query"] == "site:youtube.com python beginner tutorial"
    assert len(videos) == 1
    assert videos[0]["source"] == "youtube"
    assert videos[0]["thumbnail_url"] == "https://img.youtube.com/vi/abcdefghijk/mqdefault.jpg"


@pytest.mark.asyncio
async def test_reformulate_query_fallback_log_sanitizes_exception_repr(
    monkeypatch,
    captured_log_messages,
):
    secret = "sk-test-secret"
    path = "/Users/example/private/config.txt"

    async def _raise_sensitive_exception(**_kwargs):  # noqa: ANN003
        raise RuntimeError(f"token={secret} path={path}")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        _raise_sensitive_exception,
    )

    query = "What does the KFC logo look like?"
    result = await ms._reformulate_query(
        query=query,
        system_prompt="system",
        llm_provider="openai",
        llm_model=None,
    )

    assert result == query
    joined_logs = "\n".join(captured_log_messages)
    assert "Media query reformulation failed" in joined_logs
    assert secret not in joined_logs
    assert path not in joined_logs
    assert "RuntimeError" not in joined_logs


@pytest.mark.asyncio
async def test_search_images_fallback_log_sanitizes_exception_repr(
    monkeypatch,
    captured_log_messages,
):
    import tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs as web_apis

    secret = "ddg-secret-token"
    path = "/tmp/private/images-cache.db"

    async def _fake_reformulate_query(_query, _system, _provider, _model):  # noqa: ANN001
        return "KFC logo"

    def _raise_sensitive_exception(**_kwargs):  # noqa: ANN003
        raise RuntimeError(f"token={secret} path={path}")

    monkeypatch.setattr(ms, "_reformulate_query", _fake_reformulate_query)
    monkeypatch.setattr(web_apis, "perform_websearch", _raise_sensitive_exception)

    images = await ms.search_images("show KFC logo")

    assert images == []
    joined_logs = "\n".join(captured_log_messages)
    assert "Image search failed" in joined_logs
    assert secret not in joined_logs
    assert path not in joined_logs
    assert "RuntimeError" not in joined_logs


@pytest.mark.asyncio
async def test_search_videos_fallback_log_sanitizes_exception_repr(
    monkeypatch,
    captured_log_messages,
):
    import tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs as web_apis

    secret = "youtube-secret-token"
    path = "/var/private/video-cache.sqlite"

    async def _fake_reformulate_query(_query, _system, _provider, _model):  # noqa: ANN001
        return "python tutorial"

    def _raise_sensitive_exception(**_kwargs):  # noqa: ANN003
        raise RuntimeError(f"token={secret} path={path}")

    monkeypatch.setattr(ms, "_reformulate_query", _fake_reformulate_query)
    monkeypatch.setattr(web_apis, "perform_websearch", _raise_sensitive_exception)

    videos = await ms.search_videos("python tutorial")

    assert videos == []
    joined_logs = "\n".join(captured_log_messages)
    assert "Video search failed" in joined_logs
    assert secret not in joined_logs
    assert path not in joined_logs
    assert "RuntimeError" not in joined_logs
