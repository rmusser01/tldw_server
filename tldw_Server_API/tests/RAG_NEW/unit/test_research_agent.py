import pytest

from tldw_Server_API.app.core.RAG.rag_service.query_classifier import QueryClassification
from tldw_Server_API.app.core.RAG.rag_service import research_agent as ra
from tldw_Server_API.app.core.RAG.rag_service.research_agent import create_default_registry
from tldw_Server_API.tests.RAG_NEW.unit.test_generation_executor import (
    _RecordingCredentialRuntime,
    _install_explicit_chat_capture,
)


pytestmark = pytest.mark.unit


def test_parse_research_action_parses_fenced_json_with_think_tags():
    raw = (
        "<think>deliberation</think>\n"
        "```json\n"
        '{"reasoning":"search first","action":"web_search","params":{"query":"rag evals"}}\n'
        "```"
    )

    parsed = ra._parse_research_action(raw)
    assert parsed["action"] == "web_search"
    assert parsed["params"]["query"] == "rag evals"


@pytest.mark.asyncio
async def test_registry_disables_url_scrape_action_when_requested():
    registry = create_default_registry(enable_url_scraping=False)
    assert registry.get("scrape_url") is None
    assert registry.get("discussion_search") is not None
    assert registry.get("done") is not None


@pytest.mark.asyncio
async def test_scrape_url_action_surfaces_shared_policy_block(monkeypatch):
    import tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib as article_lib

    async def _fake_scrape_article(target_url: str):  # noqa: ANN001
        return {
            "extraction_successful": False,
            "url": target_url,
            "error": "Blocked by outbound policy",
            "policy_reason": "robots_unreachable",
        }

    monkeypatch.setattr(article_lib, "scrape_article", _fake_scrape_article)

    registry = create_default_registry(enable_url_scraping=True)
    out = await registry.execute(
        "scrape_url",
        {"url": "https://example.com/blocked"},
    )

    assert out.success is False
    assert out.error == "Blocked by outbound policy"


@pytest.mark.asyncio
async def test_discussion_action_uses_configured_default_platforms(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_search_discussions(query, platforms=None, max_results=10, search_engine="duckduckgo"):  # noqa: ANN001
        captured["query"] = query
        captured["platforms"] = platforms
        captured["max_results"] = max_results
        captured["search_engine"] = search_engine
        return [
            {
                "title": "Thread",
                "url": "https://reddit.com/r/test/comments/1",
                "content": "Community answer",
                "source": "discussion",
                "platform": "reddit",
            }
        ]

    import tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs as web_apis

    monkeypatch.setattr(web_apis, "search_discussions", _fake_search_discussions)

    registry = create_default_registry(
        discussion_platforms=["reddit", "stackoverflow"],
        enable_url_scraping=True,
    )
    classification = QueryClassification(
        skip_search=False,
        search_local_db=False,
        search_web=False,
        search_academic=False,
        search_discussions=True,
        standalone_query="community feedback",
        detected_intent="exploratory",
    )
    available_names = {a.name for a in registry.get_available(classification)}
    assert "discussion_search" in available_names

    out = await registry.execute("discussion_search", {"query": "community feedback"})
    assert out.success is True
    assert out.result_count == 1
    assert captured.get("platforms") == ["reddit", "stackoverflow"]


@pytest.mark.asyncio
async def test_web_search_action_uses_to_thread_and_skips_reprocessing(monkeypatch):
    import tldw_Server_API.app.core.RAG.rag_service.research_agent as ra
    import tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs as web_apis

    captured: dict[str, object] = {}

    def _fake_perform_websearch(**kwargs):  # noqa: ANN003
        captured["search_query"] = kwargs.get("search_query")
        return {
            "results": [
                {
                    "title": "Latest RAG Update",
                    "url": "https://example.com/rag-update",
                    "content": "New retrieval pipeline details.",
                }
            ]
        }

    def _fail_process(_payload, _engine):  # noqa: ANN001
        raise AssertionError("process_web_search_results should not be called for normalized payloads")

    async def _fake_to_thread(func, *args, **kwargs):  # noqa: ANN001
        captured["to_thread_called"] = True
        captured["to_thread_func"] = getattr(func, "__name__", str(func))
        return func(*args, **kwargs)

    monkeypatch.setattr(web_apis, "perform_websearch", _fake_perform_websearch)
    monkeypatch.setattr(web_apis, "process_web_search_results", _fail_process)
    monkeypatch.setattr(ra.asyncio, "to_thread", _fake_to_thread)

    registry = create_default_registry(enable_url_scraping=True)
    out = await registry.execute(
        "web_search",
        {"query": "latest rag update", "engine": "duckduckgo", "result_count": 1},
    )

    assert captured.get("to_thread_called") is True
    assert out.success is True
    assert out.result_count == 1
    assert out.results[0]["url"] == "https://example.com/rag-update"
    assert out.results[0]["source"] == "web"


@pytest.mark.asyncio
async def test_academic_search_action_processes_raw_results_once(monkeypatch):
    import tldw_Server_API.app.core.RAG.rag_service.research_agent as ra
    import tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs as web_apis

    captured = {"process_calls": 0, "to_thread_calls": 0}

    def _fake_perform_websearch(**kwargs):  # noqa: ANN003
        assert "site:arxiv.org" in str(kwargs.get("search_query", ""))
        return {
            "results": [
                {
                    "title": "RAG Paper",
                    "href": "https://arxiv.org/abs/1234.5678",
                    "body": "Paper abstract snippet.",
                }
            ]
        }

    def _fake_process(payload, engine):  # noqa: ANN001
        captured["process_calls"] += 1
        assert engine == "duckduckgo"
        result = payload["results"][0]
        return {
            "results": [
                {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "content": result.get("body", ""),
                }
            ]
        }

    async def _fake_to_thread(func, *args, **kwargs):  # noqa: ANN001
        captured["to_thread_calls"] += 1
        return func(*args, **kwargs)

    monkeypatch.setattr(web_apis, "perform_websearch", _fake_perform_websearch)
    monkeypatch.setattr(web_apis, "process_web_search_results", _fake_process)
    monkeypatch.setattr(ra.asyncio, "to_thread", _fake_to_thread)

    registry = create_default_registry(enable_url_scraping=True)
    out = await registry.execute(
        "academic_search",
        {"query": "rag benchmark", "result_count": 1},
    )

    assert captured["to_thread_calls"] == 1
    assert captured["process_calls"] == 1
    assert out.success is True
    assert out.result_count == 1
    assert out.results[0]["url"] == "https://arxiv.org/abs/1234.5678"
    assert out.results[0]["source"] == "academic"


@pytest.mark.asyncio
async def test_research_loop_skips_duplicate_scrape_url_fetch(monkeypatch):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service
    import tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib as article_lib

    url = "https://example.com/deep-dive"
    llm_responses = iter([
        f'{{"reasoning":"Need full text","action":"scrape_url","params":{{"url":"{url}"}}}}',
        f'{{"reasoning":"Try scraping again","action":"scrape_url","params":{{"url":"{url}"}}}}',
        '{"reasoning":"Enough evidence","action":"done","params":{"reason":"done"}}',
    ])

    async def _fake_chat_call_async(**_kwargs):  # noqa: ANN001
        return next(llm_responses)

    scrape_calls = {"count": 0}

    async def _fake_scrape_article(target_url: str):  # noqa: ANN001
        scrape_calls["count"] += 1
        return {
            "extraction_successful": True,
            "url": target_url,
            "title": "Deep Dive",
            "content": "Detailed article content for the query.",
            "author": "Analyst",
            "date": "2026-01-01",
        }

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", _fake_chat_call_async)
    monkeypatch.setattr(article_lib, "scrape_article", _fake_scrape_article)

    classification = QueryClassification(
        skip_search=False,
        search_local_db=False,
        search_web=False,
        search_academic=False,
        search_discussions=False,
        standalone_query="scrape once and reuse duplicate url",
        detected_intent="navigational",
    )

    output = await ra.research_loop(
        query="scrape once and reuse duplicate url",
        classification=classification,
        mode="speed",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        max_iterations=3,
    )

    assert output.completed is True
    assert scrape_calls["count"] == 1
    assert output.total_results == 1
    assert output.all_results[0]["url"] == url
    assert output.metadata["url_dedup"]["urls_seen"] == 1
    assert output.metadata["url_dedup"]["duplicates_merged"] == 1
    assert output.metadata["url_dedup"]["duplicate_fetches_skipped"] == 1


@pytest.mark.asyncio
async def test_research_loop_skips_duplicate_web_search_signature(monkeypatch):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service
    import tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs as web_apis

    llm_responses = iter([
        '{"reasoning":"first","action":"web_search","params":{"query":"rag updates","result_count":1}}',
        '{"reasoning":"duplicate","action":"web_search","params":{"query":"rag updates","result_count":1}}',
        '{"reasoning":"done","action":"done","params":{"reason":"enough"}}',
    ])

    async def _fake_chat_call_async(**_kwargs):  # noqa: ANN001
        return next(llm_responses)

    calls = {"to_thread": 0}

    def _fake_perform_websearch(**_kwargs):  # noqa: ANN001
        return {
            "results": [
                {
                    "title": "RAG Updates",
                    "url": "https://example.com/rag-updates",
                    "content": "Latest updates",
                }
            ]
        }

    async def _fake_to_thread(func, *args, **kwargs):  # noqa: ANN001
        calls["to_thread"] += 1
        return func(*args, **kwargs)

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", _fake_chat_call_async)
    monkeypatch.setattr(web_apis, "perform_websearch", _fake_perform_websearch)
    monkeypatch.setattr(ra.asyncio, "to_thread", _fake_to_thread)

    classification = QueryClassification(
        skip_search=False,
        search_local_db=False,
        search_web=True,
        standalone_query="rag updates",
    )
    out = await ra.research_loop(
        query="rag updates",
        classification=classification,
        mode="speed",
        max_iterations=3,
    )
    assert out.metadata["action_dedup"]["duplicates_skipped"] >= 1
    assert calls["to_thread"] == 1


@pytest.mark.asyncio
async def test_research_loop_uses_explicit_runtime_credentials(monkeypatch):
    runtime = _RecordingCredentialRuntime()
    captured = _install_explicit_chat_capture(
        monkeypatch,
        '{"reasoning":"enough evidence","action":"done","params":{"reason":"done"}}',
    )
    classification = QueryClassification(
        skip_search=False,
        search_local_db=True,
        standalone_query="credential runtime research",
    )

    output = await ra.research_loop(
        query="credential runtime research",
        classification=classification,
        mode="speed",
        llm_provider="anthropic",
        llm_model="claude-test",
        max_iterations=1,
        credential_runtime=runtime,
    )

    assert output.completed is True
    assert runtime.resolved == ["anthropic"]
    assert runtime.marked == [runtime.handle]
    assert captured["kwargs"]["api_key"] == "runtime-only-key"
    assert captured["kwargs"]["app_config"] == runtime.handle.app_config
    assert captured["kwargs"]["credentials_resolved"] is True


@pytest.mark.asyncio
async def test_research_loop_runtime_failure_records_bounded_unavailability(monkeypatch):
    class FailingRuntime:
        async def resolve(self, _provider):
            raise RuntimeError("secret-key /private/credential-store.db")

    warnings: list[str] = []
    monkeypatch.setattr(ra.logger, "warning", warnings.append)
    classification = QueryClassification(
        skip_search=False,
        search_local_db=True,
        standalone_query="credential runtime research",
    )

    output = await ra.research_loop(
        query="credential runtime research",
        classification=classification,
        mode="speed",
        llm_provider="anthropic",
        llm_model="claude-test",
        max_iterations=1,
        credential_runtime=FailingRuntime(),
    )

    assert output.metadata["provider_stage"] == {
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert warnings == ["Research loop provider unavailable"]
    assert "secret-key" not in str(output)
    assert "/private/" not in str(output)


@pytest.mark.asyncio
async def test_default_registry_passes_runtime_to_media_actions(monkeypatch):
    from tldw_Server_API.app.core.RAG.rag_service import media_search

    runtime = _RecordingCredentialRuntime()
    captured: dict[str, object] = {}

    async def fake_search_images(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(media_search, "search_images", fake_search_images)
    registry = create_default_registry(
        enable_image_search=True,
        credential_runtime=runtime,
    )

    output = await registry.execute("image_search", {"query": "architecture diagram"})

    assert output.success is True
    assert captured["credential_runtime"] is runtime


@pytest.mark.parametrize(
    ("action_name", "search_name", "media_type"),
    [
        ("image_search", "search_images", "images"),
        ("video_search", "search_videos", "videos"),
    ],
)
@pytest.mark.asyncio
async def test_media_action_preserves_results_and_reformulation_trust(
    monkeypatch,
    action_name,
    search_name,
    media_type,
):
    from tldw_Server_API.app.core.RAG.rag_service import media_search

    runtime = _RecordingCredentialRuntime()

    async def fake_media_search(**kwargs):
        kwargs["stage_metadata"].update(
            failure_code="provider_unavailable",
            verification_available=False,
        )
        return [{"title": "Architecture", "url": "https://example.com/image"}]

    monkeypatch.setattr(media_search, search_name, fake_media_search)
    registry = create_default_registry(
        enable_image_search=True,
        enable_video_search=True,
        credential_runtime=runtime,
    )

    output = await registry.execute(action_name, {"query": "architecture diagram"})

    assert output.success is True
    assert output.result_count == 1
    assert output.metadata == {
        "type": media_type,
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
