"""Owner web-summary instructions through real HTTP, storage and model assembly."""

import asyncio
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.media import process_web_scraping as endpoint
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summary
from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.Utils import prompt_loader
from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as scraper_mod
from tldw_Server_API.app.services import enhanced_web_scraping_service as enhanced
from tldw_Server_API.app.services import web_scraping_service as service
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage

pytestmark = pytest.mark.integration
PROMPT_ID = "media.web.summarization"
METHODS = ["Individual URLs", "Sitemap", "URL Level", "Recursive Scraping"]


def article(url: str = "https://example.com/a") -> dict[str, Any]:
    """Return an extracted page without contacting its origin."""
    return {
        "url": url,
        "title": "A report",
        "content": "Article facts.",
        "author": "Writer",
        "extraction_successful": True,
    }


@pytest.fixture
def context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client_with_single_user: tuple[TestClient, object]
) -> Iterator[SimpleNamespace]:
    """Keep orchestration and summarization real; substitute external fetch/model operations."""
    client, _ = client_with_single_user
    state = SimpleNamespace(
        client=client,
        owner=1,
        calls=[],
        reads=[],
        fetches=[],
        metadata=[],
        databases={owner: PromptsDatabase(tmp_path / f"{owner}.sqlite", "web-test") for owner in (1, 2)},
    )

    async def user() -> User:
        """Capture the authenticated identity for each request."""
        return User(id=state.owner, username=f"owner-{state.owner}")

    async def database(request: Request, owner: User) -> PromptsDatabase:
        """Provide real per-owner storage and record acquisition."""
        state.reads.append(owner.id)
        return state.databases[owner.id]

    async def queue(job: Any) -> asyncio.Future:
        """Complete the external fetch job with an extracted page."""
        state.fetches.append(job.url)
        state.metadata.append(json.loads(json.dumps(job.metadata)))
        future = asyncio.get_running_loop().create_future()
        future.set_result(article(job.url))
        return future

    async def pages(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Replace remote discovery and extraction for crawl methods."""
        state.fetches.append(args[0] if args else kwargs["base_url"])
        return [article(), article("https://example.com/b")]

    async def page(url: str, **kwargs: Any) -> dict[str, Any]:
        """Replace the legacy external article fetch."""
        state.fetches.append(url)
        return article(url)

    async def progress(*args: Any, **kwargs: Any) -> None:
        """Avoid creating a crawl progress artifact for fake network work."""

    def model(self: OpenAIAdapter, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Return model output through the public adapter contract."""
        state.calls.append(request)
        return {"choices": [{"message": {"role": "assistant", "content": "Summary result."}, "finish_reason": "stop"}]}

    monkeypatch.setitem(client.app.dependency_overrides, get_request_user, user)
    monkeypatch.setattr(endpoint, "get_prompts_db_for_user", database, raising=False)
    monkeypatch.setattr(scraper_mod, "get_database_dir", lambda: str(tmp_path))
    state.scraper = scraper_mod.EnhancedWebScraper({})
    monkeypatch.setattr(state.scraper.job_queue, "add_job", queue)
    monkeypatch.setattr(state.scraper, "scrape_sitemap", pages)
    monkeypatch.setattr(state.scraper, "recursive_scrape", pages)
    monkeypatch.setattr(state.scraper, "save_progress", progress)
    state.service = enhanced.WebScrapingService()
    state.service.scraper = state.scraper
    state.service._initialized = True
    monkeypatch.setattr(service, "get_web_scraping_service", lambda: state.service)
    monkeypatch.setattr(legacy, "scrape_article", page)
    monkeypatch.setattr(
        service, "scrape_from_sitemap", lambda *args, **kwargs: [article(), article("https://example.com/b")]
    )
    monkeypatch.setattr(
        service, "scrape_by_url_level", lambda *args, **kwargs: [article(), article("https://example.com/b")]
    )
    monkeypatch.setattr(service, "recursive_scrape", pages)
    monkeypatch.setattr(summary, "loaded_config_data", {"openai_api": {"model": "test-model"}})
    monkeypatch.setattr(
        prompt_loader, "_prompts_dir", lambda: str(Path(__file__).resolve().parents[2] / "Config_Files" / "Prompts")
    )
    monkeypatch.setattr(prompt_loader, "get_global_context_integrity_resolver", lambda: None)
    monkeypatch.setattr(OpenAIAdapter, "chat", model)
    yield state
    for database in state.databases.values():
        database.close_connection()


def save(context: SimpleNamespace, owner: int = 1) -> Any:
    """Save the atomic pair in real owner storage."""
    return context.databases[owner].save_service_prompt_override(
        PROMPT_ID, {"system": f"Owner {owner} system {{literal}}", "user": f"Owner {owner} summary {{literal}}"}, None
    )


def process(context: SimpleNamespace, method: str = "Individual URLs", **options: Any) -> dict[str, Any]:
    """Exercise the authenticated synchronous JSON endpoint."""
    response = context.client.post(
        "/api/v1/media/process-web-scraping",
        json={
            "scrape_method": method,
            "url_input": "https://example.com/a\nhttps://example.com/b"
            if method == "Individual URLs"
            else "https://example.com",
            "url_level": 1,
            "max_pages": 2,
            "summarize_checkbox": True,
            "api_name": "openai",
            "mode": "ephemeral",
            "perform_chunking": False,
            **options,
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    if "ephemeral_id" in result:
        return ephemeral_storage.get_data(result["ephemeral_id"])["result"]
    return {"articles": result["results"]}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fallback", [False, True])
def test_saved_pair_reaches_all_pages(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, method: str, fallback: bool
) -> None:
    """Every supported engine/method must consume both owner instructions."""
    save(context)
    if fallback:

        def unavailable() -> None:
            """Select the existing compatibility engine."""
            raise RuntimeError("enhanced unavailable")

        monkeypatch.setattr(service, "get_web_scraping_service", unavailable)
        monkeypatch.setenv("TLDW_ENABLE_LEGACY_WEB_SCRAPING_FALLBACK", "1")
    result = process(context, method)
    assert [item["summary"] for item in result["articles"]] == ["Summary result."] * 2
    assert len(context.calls) == 2
    assert {call["system_message"] for call in context.calls} == {"Owner 1 system {literal}"}
    assert {call["messages"][0]["content"] for call in context.calls} == {
        "Article facts.\n\n\n\nOwner 1 summary {literal}"
    }
    assert all("summary_prompt_overrides" not in metadata for metadata in context.metadata)


@pytest.mark.parametrize("method", ["Individual URLs", "Sitemap"])
@pytest.mark.parametrize("part", ["system_prompt", "custom_prompt"])
@pytest.mark.parametrize("value", ["Explicit {literal}", ""])
def test_explicit_parts_win_independently(context: SimpleNamespace, method: str, part: str, value: str) -> None:
    """Explicit empty and literal instructions must not be replaced by saved parts."""
    save(context)
    process(context, method, **{part: value})
    assert {call["system_message"] for call in context.calls} == {
        value if part == "system_prompt" else "Owner 1 system {literal}"
    }
    suffix = value if part == "custom_prompt" else "Owner 1 summary {literal}"
    expected = "Article facts." + ("\n\n\n\n" + suffix if suffix else "")
    assert {call["messages"][0]["content"] for call in context.calls} == {expected}


@pytest.mark.parametrize("options", [{"summarize_checkbox": False}, {"system_prompt": "", "custom_prompt": ""}])
def test_irrelevant_corrupt_storage_is_not_read(context: SimpleNamespace, options: dict[str, Any]) -> None:
    """Disabled analysis or fully explicit parts must bypass saved storage."""
    context.databases[1].save_service_prompt_override(PROMPT_ID, {"bad": "corrupt"}, None)
    process(context, **options)
    assert context.reads == []
    if options.get("summarize_checkbox") is False:
        assert context.calls == []


def test_owner_isolation(context: SimpleNamespace) -> None:
    """The request identity, not a database wrapper attribute, selects saved prompts."""
    save(context, 1)
    save(context, 2)
    context.owner = 2
    process(context)
    assert context.reads == [2]
    assert {call["system_message"] for call in context.calls} == {"Owner 2 system {literal}"}


def test_snapshot_survives_mid_request_edit(context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """All pages use the revision captured before the first fetch."""
    row = save(context)
    original = context.scraper.job_queue.add_job

    async def edit(job: Any) -> asyncio.Future:
        """Edit settings while external scraping is in progress."""
        if not context.fetches:
            db = context.databases[1]
            try:
                db.save_service_prompt_override(PROMPT_ID, {"system": "Future", "user": "Future"}, row.revision)
            finally:
                db.close_connection()
            context.owner = 2
        return await original(job)

    monkeypatch.setattr(context.scraper.job_queue, "add_job", edit)
    process(context)
    assert context.reads == [1]
    assert {call["system_message"] for call in context.calls} == {"Owner 1 system {literal}"}


@pytest.mark.parametrize("corrupt", [False, True])
def test_lookup_connection_closes_on_worker(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    """Prompt storage cleanup runs on its read thread even when validation fails."""
    database = context.databases[1]
    if corrupt:
        database.save_service_prompt_override(PROMPT_ID, {"bad": "corrupt"}, None)
    else:
        save(context)
    reads, closes = [], []
    read, close = database.get_service_prompt_override, database.close_connection

    def track_read(definition_id: str) -> Any:
        """Record the worker performing a real database lookup."""
        reads.append(threading.get_ident())
        return read(definition_id)

    def track_close() -> None:
        """Record cleanup after closing the actual connection."""
        close()
        closes.append(threading.get_ident())

    monkeypatch.setattr(database, "get_service_prompt_override", track_read)
    monkeypatch.setattr(database, "close_connection", track_close)
    if corrupt:
        response = context.client.post(
            "/api/v1/media/process-web-scraping",
            json={
                "scrape_method": "Individual URLs",
                "url_input": "https://example.com",
                "summarize_checkbox": True,
                "api_name": "openai",
            },
        )
        assert response.status_code >= 400
        assert context.fetches == []
    else:
        process(context)
    assert len(reads) == 1
    assert closes == reads


@pytest.mark.parametrize("method", METHODS)
def test_reset_restores_engine_defaults(context: SimpleNamespace, method: str) -> None:
    """Saving and resetting must not unify the engines' different no-override prompts."""
    row = save(context)
    context.databases[1].reset_service_prompt_override(PROMPT_ID, row.revision)
    process(context, method)
    if method == "Individual URLs":
        assert {call["system_message"] for call in context.calls} == {"Summarize this article concisely."}
        assert {call["messages"][0]["content"] for call in context.calls} == {"Article facts."}
    else:
        assert {call["system_message"] for call in context.calls} == {
            "You are a professional summarizer who produces accurate, concise summaries of web content."
        }
        assert {call["messages"][0]["content"] for call in context.calls} == {
            "Article facts.\n\n\n\nSummarize this article concisely. Focus on the main points, facts, and any actionable insights."
        }


def test_saved_pair_does_not_load_unused_deployment_defaults(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid saved pair must not depend on loading fallback prompt files."""
    save(context)

    def unavailable(*args: Any, **kwargs: Any) -> str:
        """Fail only when the unused fallback source is accessed."""
        raise ValueError("deployment prompt unavailable")

    monkeypatch.setattr(enhanced, "load_prompt", unavailable)
    result = process(context, "Sitemap")
    assert [item["summary"] for item in result["articles"]] == ["Summary result."] * 2


@pytest.mark.parametrize("method", METHODS)
def test_legacy_reset_preserves_existing_defaults(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Compatibility-engine defaults remain distinct from deployed crawl prompts."""
    row = save(context)
    context.databases[1].reset_service_prompt_override(PROMPT_ID, row.revision)

    def unavailable() -> None:
        """Force selection of the legacy scraping engine."""
        raise RuntimeError("enhanced unavailable")

    monkeypatch.setattr(service, "get_web_scraping_service", unavailable)
    monkeypatch.setenv("TLDW_ENABLE_LEGACY_WEB_SCRAPING_FALLBACK", "1")
    process(context, method)
    expected = "Act as a professional summarizer and summarize this article." if method == "Individual URLs" else ""
    assert {call["system_message"] for call in context.calls} == {expected}
    assert {call["messages"][0]["content"] for call in context.calls} == {
        "Article facts." + ("\n\n\n\n" + expected if expected else "")
    }


def test_missing_deployment_assets_preserves_builtin_defaults(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Existing literal fallbacks still work without deployment prompt assets."""
    monkeypatch.setattr(prompt_loader, "_prompts_dir", lambda: str(tmp_path))
    process(context, "Sitemap")
    assert {call["system_message"] for call in context.calls} == {"You are a professional summarizer."}
    assert {call["messages"][0]["content"] for call in context.calls} == {
        "Article facts.\n\n\n\nSummarize this article concisely."
    }


def test_explicit_system_reaches_individual_scraper_without_saved_override(
    context: SimpleNamespace,
) -> None:
    """The HTTP system_prompt name reaches the individual scraper's model call."""
    process(context, system_prompt="Explicit system")
    assert {call["system_message"] for call in context.calls} == {"Explicit system"}


@pytest.mark.asyncio
async def test_unscoped_caller_keeps_defaults_without_owner_lookup(context: SimpleNamespace) -> None:
    """Direct service calls stay outside account-specific prompt resolution."""
    save(context)
    await context.service.process_web_scraping_task(
        scrape_method="Sitemap",
        url_input="https://example.com",
        summarize_checkbox=True,
        api_name="openai",
        mode="ephemeral",
        perform_chunking=False,
    )
    assert context.reads == []
    assert {call["system_message"] for call in context.calls} == {
        "You are a professional summarizer who produces accurate, concise summaries of web content."
    }


def test_persistence_uses_authenticated_prompt_owner(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Saved instructions and persisted media must belong to the same authenticated user."""
    save(context, 2)
    context.owner = 2
    owners = []

    def path(owner: int) -> str:
        """Route real persistence into a per-owner temporary media database."""
        owners.append(owner)
        return str(tmp_path / f"media-{owner}.db")

    monkeypatch.setattr(enhanced, "get_user_media_db_path", path)
    response = context.client.post(
        "/api/v1/media/process-web-scraping",
        json={
            "scrape_method": "Individual URLs",
            "url_input": "https://example.com/a",
            "api_name": "openai",
            "summarize_checkbox": True,
            "mode": "persist",
            "perform_chunking": False,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["stored_articles"] == 1
    assert owners == [2]
    assert {call["system_message"] for call in context.calls} == {"Owner 2 system {literal}"}
