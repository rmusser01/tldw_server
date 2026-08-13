import asyncio

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
from tldw_Server_API.app.core.RAG.rag_service import research_agent as ra
from tldw_Server_API.app.core.RAG.rag_service.query_classifier import QueryClassification
from tldw_Server_API.app.core.RAG.rag_service.research_agent import create_default_registry
from tldw_Server_API.tests.RAG_NEW.unit.test_generation_executor import (
    _install_blocking_sync_chat_adapter,
    _install_explicit_chat_capture,
    _RecordingCredentialRuntime,
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
    import tldw_Server_API.app.core.Web_Scraping.orchestration as article_orchestration

    async def _fake_scrape_article(target_url: str):  # noqa: ANN001
        return {
            "extraction_successful": False,
            "url": target_url,
            "error": "Blocked by outbound policy",
            "policy_reason": "robots_unreachable",
        }

    monkeypatch.setattr(article_orchestration, "scrape_article", _fake_scrape_article)

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
    import tldw_Server_API.app.core.Web_Scraping.orchestration as article_orchestration

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
    monkeypatch.setattr(article_orchestration, "scrape_article", _fake_scrape_article)

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
    assert runtime.resolved_models == ["claude-test"]
    assert runtime.marked == [runtime.handle]
    assert captured["kwargs"]["api_key"] == "runtime-only-key"
    assert captured["kwargs"]["app_config"] == runtime.handle.app_config
    assert captured["kwargs"]["credentials_resolved"] is True
    assert captured["kwargs"][PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is runtime.handle


@pytest.mark.asyncio
async def test_research_loop_cancellation_marks_completed_sync_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    entered, release = _install_blocking_sync_chat_adapter(
        monkeypatch,
        '{"reasoning":"done","action":"done","params":{"reason":"done"}}',
    )
    classification = QueryClassification(
        skip_search=False,
        search_local_db=True,
        standalone_query="runtime research",
    )
    task = asyncio.create_task(
        ra.research_loop(
            query="runtime research",
            classification=classification,
            mode="speed",
            llm_provider="anthropic",
            llm_model="claude-test",
            max_iterations=1,
            credential_runtime=runtime,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert runtime.marked == [runtime.handle]


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


@pytest.mark.asyncio
async def test_default_registry_passes_runtime_to_local_database_action(monkeypatch):
    from tldw_Server_API.app.core.RAG.rag_service import database_retrievers

    runtime = object()
    captured: dict[str, object] = {}
    media_adapter = object()
    chacha_adapter = object()
    paths: dict[str, object] = {}

    def recording_retriever(name):
        class RecordingRetriever:
            def __init__(self, db_path=None, *args, **kwargs):
                paths[name] = db_path
                captured[f"{name}_args"] = args
                captured[f"{name}_kwargs"] = kwargs
                self.config = None

            async def retrieve(self, query, **_kwargs):
                captured["query"] = query
                captured["retrieval_config"] = self.config
                return []

            def close(self):
                return None

        return RecordingRetriever

    for class_name, path_name in (
        ("MediaDBRetriever", "media"),
        ("NotesDBRetriever", "notes"),
        ("CharacterCardsRetriever", "characters"),
        ("ChatHistoryRetriever", "chats"),
        ("WorldBooksRetriever", "world_books"),
        ("ChatDictionariesRetriever", "dictionaries"),
        ("KanbanDBRetriever", "kanban"),
    ):
        monkeypatch.setattr(
            database_retrievers,
            class_name,
            recording_retriever(path_name),
        )
    registry = create_default_registry(credential_runtime=runtime)

    output = await registry.execute(
        "local_db_search",
        {
            "query": "runtime local search",
            "sources": ["media_db"],
            "top_k": 7,
            "user_id": "42",
            "media_db_path": "media.sqlite",
            "notes_db_path": "notes.sqlite",
            "character_db_path": "characters.sqlite",
            "kanban_db_path": "kanban.sqlite",
            "media_db": media_adapter,
            "chacha_db": chacha_adapter,
        },
    )

    assert output.success is True
    assert paths["media"] == "media.sqlite"
    assert paths["notes"] == "notes.sqlite"
    assert paths["characters"] == "characters.sqlite"
    assert paths["kanban"] == "kanban.sqlite"
    assert captured["media_kwargs"] == {
        "user_id": "42",
        "media_db": media_adapter,
        "credential_runtime": runtime,
    }
    config = captured["retrieval_config"]
    assert isinstance(config, database_retrievers.RetrievalConfig)
    assert config.max_results == 7
    assert config.use_fts is True
    assert config.use_vector is True


@pytest.mark.asyncio
async def test_research_loop_overwrites_local_action_scope_with_trusted_context(monkeypatch):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    captured: dict[str, object] = {}

    async def fake_chat_call_async(**_kwargs):
        return (
            '{"reasoning":"search local","action":"local_db_search","params":'
            '{"query":"scope test","sources":["media_db"],"top_k":3,'
            '"user_id":"attacker","media_db_path":"attacker.db",'
            '"notes_db_path":"attacker-notes.db","chacha_db":"attacker-adapter",'
            '"credential_runtime":"attacker-runtime"}}'
        )

    async def capture_local_action(params):
        captured.update(params)
        return ra.ActionOutput(action_name="local_db_search", success=True)

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call_async)
    registry = ra.ActionRegistry()
    registry.register(
        ra.ResearchAction(
            name="local_db_search",
            description="local",
            schema={},
            enabled=lambda _classification: True,
            execute=capture_local_action,
        )
    )
    classification = QueryClassification(
        skip_search=False,
        search_local_db=True,
        standalone_query="scope test",
    )

    await ra.research_loop(
        query="scope test",
        classification=classification,
        mode="speed",
        max_iterations=1,
        registry=registry,
        db_context={
            "user_id": "trusted-user",
            "media_db_path": "trusted.db",
        },
    )

    assert captured["user_id"] == "trusted-user"
    assert captured["media_db_path"] == "trusted.db"
    assert captured["sources"] == ["media_db"]
    assert captured["top_k"] == 3
    assert "notes_db_path" not in captured
    assert "chacha_db" not in captured
    assert "credential_runtime" not in captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "secrets", "expected_sources"),
    [
        pytest.param(
            '{"reasoning":"local","action":"local_db_search",'
            '"params":"provider-params-secret"}',
            ("provider-params-secret",),
            ["media_db"],
            id="non-mapping-params",
        ),
        pytest.param(
            '{"reasoning":"local","action":"local_db_search","params":'
            '{"query":{"secret":"provider-query-secret"},'
            '"sources":{"secret":"provider-sources-secret"},'
            '"top_k":"provider-top-k-secret"}}',
            (
                "provider-query-secret",
                "provider-sources-secret",
                "provider-top-k-secret",
            ),
            [],
            id="malformed-local-fields",
        ),
        pytest.param(
            '{"reasoning":"local","action":"local_db_search","params":'
            '{"query":"trusted fallback query",'
            '"sources":["provider-unknown-source"],"top_k":10}}',
            ("provider-unknown-source",),
            [],
            id="unknown-source",
        ),
    ],
)
async def test_research_loop_normalizes_malformed_local_action_params(
    monkeypatch,
    response,
    secrets,
    expected_sources,
):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    captured: list[dict[str, object]] = []

    async def fake_chat_call_async(**_kwargs):
        return response

    async def capture_local_action(params):
        captured.append(dict(params))
        return ra.ActionOutput(action_name="local_db_search", success=True)

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call_async)
    registry = ra.ActionRegistry()
    registry.register(
        ra.ResearchAction(
            name="local_db_search",
            description="local",
            schema={},
            enabled=lambda _classification: True,
            execute=capture_local_action,
        )
    )
    classification = QueryClassification(
        skip_search=False,
        search_local_db=True,
        standalone_query="trusted fallback query",
    )

    output = await ra.research_loop(
        query="trusted fallback query",
        classification=classification,
        mode="speed",
        max_iterations=1,
        registry=registry,
    )

    assert captured == [
        {
            "query": "trusted fallback query",
            "sources": expected_sources,
            "top_k": 10,
        }
    ]
    for secret in secrets:
        assert secret not in str(output)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action_name", "limit_name", "limit_json", "expected_limit", "sentinel"),
    [
        pytest.param(
            "local_db_search",
            "top_k",
            "1e309",
            10,
            "provider-local-infinity-secret",
            id="local-overflow-default",
        ),
        pytest.param(
            "local_db_search",
            "top_k",
            "999999999999999999999999",
            25,
            "provider-local-extreme-secret",
            id="local-extreme-clamp",
        ),
        pytest.param(
            "web_search",
            "result_count",
            '"provider-web-limit-secret"',
            5,
            "provider-web-limit-secret",
            id="web-value-error-default",
        ),
        pytest.param(
            "web_search",
            "result_count",
            "-999999999999999999",
            1,
            "provider-web-negative-secret",
            id="web-negative-clamp",
        ),
        pytest.param(
            "academic_search",
            "result_count",
            "null",
            5,
            "provider-academic-type-secret",
            id="academic-type-error-default",
        ),
        pytest.param(
            "discussion_search",
            "max_results",
            "1e309",
            10,
            "provider-discussion-overflow-secret",
            id="discussion-overflow-default",
        ),
        pytest.param(
            "image_search",
            "max_results",
            "999999999999999999999999",
            25,
            "provider-image-extreme-secret",
            id="image-extreme-clamp",
        ),
        pytest.param(
            "video_search",
            "max_results",
            "-999999999999999999",
            1,
            "provider-video-negative-secret",
            id="video-negative-clamp",
        ),
    ],
)
async def test_research_loop_normalizes_all_action_numeric_limits(
    monkeypatch,
    action_name,
    limit_name,
    limit_json,
    expected_limit,
    sentinel,
):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    captured: list[dict[str, object]] = []

    async def fake_chat_call_async(**_kwargs):
        return (
            '{"reasoning":"bounded numeric input","action":'
            f'"{action_name}","params":{{"query":"numeric normalization",'
            f'"{limit_name}":{limit_json},"undeclared":"{sentinel}",'
            '"llm_provider":"attacker-provider","llm_model":"attacker-model",'
            '"search_engine":"attacker-engine"}}}'
        )

    async def capture_action(params):
        captured.append(dict(params))
        return ra.ActionOutput(action_name=action_name, success=True)

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call_async)
    registry = ra.ActionRegistry()
    registry.register(
        ra.ResearchAction(
            name=action_name,
            description="numeric action",
            schema={},
            enabled=lambda _classification: True,
            execute=capture_action,
        )
    )
    classification = QueryClassification(
        skip_search=False,
        standalone_query="numeric normalization",
    )

    output = await ra.research_loop(
        query="numeric normalization",
        classification=classification,
        mode="speed",
        max_iterations=1,
        registry=registry,
    )

    assert captured[0][limit_name] == expected_limit
    assert "undeclared" not in captured[0]
    assert "llm_provider" not in captured[0]
    assert "llm_model" not in captured[0]
    assert "search_engine" not in captured[0]
    assert sentinel not in str(output)


@pytest.mark.asyncio
@pytest.mark.parametrize("action_name", ["image_search", "video_search"])
async def test_research_loop_media_dedup_uses_normalized_query_and_limit(
    monkeypatch,
    action_name,
):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    responses = iter(
        [
            '{"reasoning":"first","action":'
            f'"{action_name}","params":{{"query":"alpha topic","max_results":2}}}}',
            '{"reasoning":"different query","action":'
            f'"{action_name}","params":{{"query":"beta topic","max_results":2}}}}',
            '{"reasoning":"different limit","action":'
            f'"{action_name}","params":{{"query":"beta topic","max_results":3}}}}',
            '{"reasoning":"normalized duplicate","action":'
            f'"{action_name}","params":{{"query":"  BETA   TOPIC  ","max_results":3}}}}',
        ]
    )
    calls: list[dict[str, object]] = []

    async def fake_chat_call_async(**_kwargs):
        return next(responses)

    async def capture_action(params):
        calls.append(dict(params))
        return ra.ActionOutput(
            action_name=action_name,
            success=True,
            results=[{"id": str(len(calls))}],
            result_count=1,
        )

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call_async)
    registry = ra.ActionRegistry()
    registry.register(
        ra.ResearchAction(
            name=action_name,
            description="media action",
            schema={},
            enabled=lambda _classification: True,
            execute=capture_action,
        )
    )
    classification = QueryClassification(
        skip_search=False,
        standalone_query="media dedup",
    )

    output = await ra.research_loop(
        query="media dedup",
        classification=classification,
        mode="speed",
        max_iterations=4,
        registry=registry,
    )

    assert calls == [
        {"query": "alpha topic", "max_results": 2},
        {"query": "beta topic", "max_results": 2},
        {"query": "beta topic", "max_results": 3},
    ]
    assert output.metadata["action_dedup"]["duplicates_skipped"] == 1


@pytest.mark.asyncio
async def test_research_loop_normalizes_known_action_name_case(monkeypatch):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    captured: list[dict[str, object]] = []

    async def fake_chat_call_async(**_kwargs):
        return (
            '{"reasoning":"local","action":" LOCAL_DB_SEARCH ","params":'
            '{"query":"case normalized","sources":["media_db"],"top_k":2}}'
        )

    async def capture_local_action(params):
        captured.append(dict(params))
        return ra.ActionOutput(action_name="local_db_search", success=True)

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call_async)
    registry = ra.ActionRegistry()
    registry.register(
        ra.ResearchAction(
            name="local_db_search",
            description="local",
            schema={},
            enabled=lambda _classification: True,
            execute=capture_local_action,
        )
    )
    classification = QueryClassification(
        skip_search=False,
        search_local_db=True,
        standalone_query="case normalized",
    )

    output = await ra.research_loop(
        query="case normalized",
        classification=classification,
        mode="speed",
        max_iterations=1,
        registry=registry,
    )

    assert output.steps[0].action_name == "local_db_search"
    assert captured[0]["top_k"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_json",
    ['["provider-action-secret"]', '"provider-action-secret"'],
)
async def test_research_loop_maps_untrusted_action_names_to_safe_completion(
    monkeypatch,
    action_json,
):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    async def fake_chat_call_async(**_kwargs):
        return (
            '{"reasoning":"stop safely","action":'
            f"{action_json}"
            ',"params":{}}'
        )

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call_async)
    classification = QueryClassification(
        skip_search=False,
        standalone_query="safe action normalization",
    )

    output = await ra.research_loop(
        query="safe action normalization",
        classification=classification,
        mode="speed",
        max_iterations=1,
        registry=ra.ActionRegistry(),
    )

    assert output.steps[0].action_name == "done"
    assert "provider-action-secret" not in str(output)


@pytest.mark.asyncio
async def test_research_loop_canonicalizes_local_source_aliases_before_dedup(monkeypatch):
    import tldw_Server_API.app.core.Chat.chat_service as chat_service
    from tldw_Server_API.app.core.RAG.rag_service import database_retrievers
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

    calls = {"character_cards": 0, "chat_history": 0}

    def recording_retriever(name, source):
        class RecordingRetriever:
            def __init__(self, *_args, **_kwargs):
                self.config = None

            async def retrieve(self, _query, **_kwargs):
                calls[name] += 1
                return [
                    Document(
                        id=f"{name}-1",
                        content=name,
                        metadata={},
                        source=source,
                        score=0.9,
                    )
                ]

            def close(self):
                return None

        return RecordingRetriever

    class InertRetriever:
        def __init__(self, *_args, **_kwargs):
            self.config = None

        async def retrieve(self, _query, **_kwargs):
            return []

        def close(self):
            return None

    monkeypatch.setattr(
        database_retrievers,
        "CharacterCardsRetriever",
        recording_retriever("character_cards", DataSource.CHARACTER_CARDS),
    )
    monkeypatch.setattr(
        database_retrievers,
        "ChatHistoryRetriever",
        recording_retriever("chat_history", DataSource.CHAT_HISTORY),
    )
    monkeypatch.setattr(database_retrievers, "WorldBooksRetriever", InertRetriever)
    monkeypatch.setattr(database_retrievers, "ChatDictionariesRetriever", InertRetriever)

    responses = iter(
        [
            '{"reasoning":"aliases","action":"local_db_search","params":'
            '{"query":"local aliases","sources":["characters","chats"],"top_k":4}}',
            '{"reasoning":"canonical duplicate","action":"local_db_search","params":'
            '{"query":"local aliases","sources":["character_cards","chat_history"],"top_k":4}}',
        ]
    )

    async def fake_chat_call_async(**_kwargs):
        return next(responses)

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call_async)
    classification = QueryClassification(
        skip_search=False,
        search_local_db=True,
        standalone_query="local aliases",
    )

    output = await ra.research_loop(
        query="local aliases",
        classification=classification,
        mode="speed",
        max_iterations=2,
        registry=create_default_registry(),
        db_context={"character_db_path": "characters.sqlite"},
    )

    assert calls == {"character_cards": 1, "chat_history": 1}
    assert output.metadata["action_dedup"]["duplicates_skipped"] == 1


@pytest.mark.asyncio
async def test_local_database_action_sanitizes_typed_provider_failure(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
    from tldw_Server_API.app.core.RAG.rag_service import database_retrievers

    secret = "local-retrieval-secret"

    class FailingMediaRetriever:
        def __init__(self, *_args, **_kwargs):
            self.config = None

        async def retrieve(self, _query, **_kwargs):
            raise ByokResolutionError("credential_store_unavailable", "openai")

        def close(self):
            return None

    monkeypatch.setattr(database_retrievers, "MediaDBRetriever", FailingMediaRetriever)
    registry = create_default_registry(credential_runtime=object())

    output = await registry.execute(
        "local_db_search",
        {
            "query": secret,
            "sources": ["media_db"],
            "media_db_path": "media.sqlite",
        },
    )

    assert output.success is False
    assert output.error == "credential_store_unavailable"
    assert output.metadata == {"failure_code": "credential_store_unavailable"}
    assert secret not in str(output)


@pytest.mark.asyncio
async def test_local_database_action_sanitizes_unexpected_failure():
    secret = "local-action-secret"

    class FailingUserId:
        def __str__(self):
            raise RuntimeError(secret)

    registry = create_default_registry(credential_runtime=object())

    output = await registry.execute(
        "local_db_search",
        {
            "query": "unexpected failure",
            "sources": [],
            "user_id": FailingUserId(),
        },
    )

    assert output.success is False
    assert output.error == "action_failed"
    assert output.metadata == {"failure_code": "action_failed"}
    assert secret not in str(output)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_factory", "expected_code"),
    [
        pytest.param(
            lambda secret: ByokResolutionError("credential_store_unavailable", "openai"),
            "credential_store_unavailable",
            id="byok",
        ),
        pytest.param(
            lambda secret: ChatAuthenticationError(secret, provider="openai"),
            "invalid_provider_credentials",
            id="chat-auth",
        ),
        pytest.param(lambda secret: RuntimeError(secret), "action_failed", id="unexpected"),
    ],
)
async def test_action_registry_sanitizes_outer_failures(
    monkeypatch,
    failure_factory,
    expected_code,
):
    secret = "outer-action-secret"
    warnings: list[object] = []

    async def fail_action(_params):
        raise failure_factory(secret)

    def capture_warning(*args, **kwargs):
        warnings.append((args, kwargs))

    monkeypatch.setattr(ra.logger, "warning", capture_warning)
    registry = ra.ActionRegistry()
    registry.register(
        ra.ResearchAction(
            name="failing_action",
            description="failure",
            schema={},
            enabled=lambda _classification: True,
            execute=fail_action,
        )
    )

    output = await registry.execute("failing_action", {})

    assert output.success is False
    assert output.error == expected_code
    assert output.metadata == {"failure_code": expected_code}
    assert secret not in str(output)
    assert secret not in str(warnings)


@pytest.mark.asyncio
async def test_action_registry_propagates_cancellation():
    async def cancel_action(_params):
        raise asyncio.CancelledError

    registry = ra.ActionRegistry()
    registry.register(
        ra.ResearchAction(
            name="cancel_action",
            description="cancel",
            schema={},
            enabled=lambda _classification: True,
            execute=cancel_action,
        )
    )

    with pytest.raises(asyncio.CancelledError):
        await registry.execute("cancel_action", {})


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
