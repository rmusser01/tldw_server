import asyncio
import dataclasses
import threading
from types import MethodType, SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Web_Scraping.orchestration import article as canonical_article
from tldw_Server_API.tests.Web_Scraping.test_phase3_article_preflight_facade import (
    URL,
    install_article_defaults,
)
from tldw_Server_API.tests.Web_Scraping.test_phase3_enhanced_preflight_facade import (
    install_enhanced_defaults,
)


def _scrape_plan() -> SimpleNamespace:
    return SimpleNamespace(
        backend="auto",
        handler="",
        ua_profile="test-profile",
        extra_headers={},
        cookies={},
        respect_robots=True,
        impersonate=None,
        proxies=None,
        strategy_order=None,
        schema_rules=None,
        llm_settings=None,
        regex_settings=None,
        cluster_settings=None,
    )


@pytest.mark.unit
async def test_article_router_resolution_runs_off_loop_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_builder = canonical_article._build_default_dependencies
    harness = install_article_defaults(monkeypatch)
    loop_thread = threading.get_ident()
    events: list[tuple[str, int]] = []
    config = harness.dependencies.load_config()
    target = harness.evaluate_target.return_value
    plan = _scrape_plan()

    def load_config() -> dict[str, Any]:
        events.append(("config", threading.get_ident()))
        return config

    class RecordingRouter:
        @staticmethod
        def load_rules_from_yaml(_path: str) -> dict[str, Any]:
            events.append(("load", threading.get_ident()))
            return {}

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            events.append(("construct", threading.get_ident()))

        def resolve(self, _url: str) -> SimpleNamespace:
            events.append(("resolve", threading.get_ident()))
            return plan

    async def evaluate_target(*_args: Any, **_kwargs: Any) -> Any:
        events.append(("preflight", threading.get_ident()))
        return target

    default_dependencies = default_builder(None)
    dependencies = dataclasses.replace(
        harness.dependencies,
        load_config=load_config,
        resolve_plan=default_dependencies.resolve_plan,
        backend_setting=default_dependencies.backend_setting,
    )
    monkeypatch.setattr(canonical_article, "ScraperRouter", RecordingRouter)
    monkeypatch.setattr(canonical_article, "_build_default_dependencies", lambda _cookies: dependencies)
    harness.evaluate_target.side_effect = evaluate_target

    result = await harness.article.scrape_article(URL)

    assert result["extraction_successful"] is True
    assert [name for name, _thread in events] == [
        "config",
        "load",
        "construct",
        "resolve",
        "preflight",
    ]
    routing_threads = {thread for _name, thread in events[:-1]}
    assert len(routing_threads) == 1
    assert routing_threads != {loop_thread}
    assert events[-1] == ("preflight", loop_thread)


@pytest.mark.unit
async def test_enhanced_router_resolution_runs_off_loop_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_enhanced_defaults(monkeypatch)
    loop_thread = threading.get_ident()
    events: list[tuple[str, int]] = []
    target = harness.evaluate_target.return_value
    plan = _scrape_plan()

    class RecordingRouter:
        @staticmethod
        def load_rules_from_yaml(_path: str) -> dict[str, Any]:
            events.append(("load", threading.get_ident()))
            return {}

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            events.append(("construct", threading.get_ident()))

        def resolve(self, _url: str) -> SimpleNamespace:
            events.append(("resolve", threading.get_ident()))
            return plan

    async def evaluate_target(*_args: Any, **_kwargs: Any) -> Any:
        events.append(("preflight", threading.get_ident()))
        return target

    monkeypatch.setattr(harness.enhanced, "ScraperRouter", RecordingRouter)
    monkeypatch.setattr(
        harness.scraper,
        "_resolve_scrape_plan",
        MethodType(harness.enhanced.EnhancedWebScraper._resolve_scrape_plan, harness.scraper),
    )
    harness.evaluate_target.side_effect = evaluate_target

    result = await harness.scraper.scrape_article(URL)

    assert result["extraction_successful"] is True
    assert [name for name, _thread in events] == [
        "load",
        "construct",
        "resolve",
        "preflight",
    ]
    routing_threads = {thread for _name, thread in events[:-1]}
    assert len(routing_threads) == 1
    assert routing_threads != {loop_thread}
    assert events[-1] == ("preflight", loop_thread)


@pytest.mark.unit
async def test_article_cancellation_discards_late_router_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_builder = canonical_article._build_default_dependencies
    harness = install_article_defaults(monkeypatch)
    started = threading.Event()
    finished = threading.Event()
    release = threading.Event()
    plan = _scrape_plan()

    class SlowRouter:
        @staticmethod
        def load_rules_from_yaml(_path: str) -> dict[str, Any]:
            return {}

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def resolve(self, _url: str) -> SimpleNamespace:
            started.set()
            release.wait(1.0)
            finished.set()
            return plan

    default_dependencies = default_builder(None)
    dependencies = dataclasses.replace(
        harness.dependencies,
        resolve_plan=default_dependencies.resolve_plan,
        backend_setting=default_dependencies.backend_setting,
    )
    monkeypatch.setattr(canonical_article, "ScraperRouter", SlowRouter)
    monkeypatch.setattr(canonical_article, "_build_default_dependencies", lambda _cookies: dependencies)
    task = asyncio.create_task(harness.article.scrape_article(URL))
    cancel_handle = asyncio.get_running_loop().call_later(0.02, task.cancel)

    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert started.is_set()
        assert not finished.is_set()
        harness.evaluate_target.assert_not_awaited()
        release.set()
        assert await asyncio.to_thread(finished.wait, 1.0)
        harness.evaluate_target.assert_not_awaited()
    finally:
        cancel_handle.cancel()
        release.set()
        if started.is_set() and not finished.is_set():
            await asyncio.to_thread(finished.wait, 1.0)


@pytest.mark.unit
async def test_enhanced_cancellation_discards_late_router_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_enhanced_defaults(monkeypatch)
    started = threading.Event()
    finished = threading.Event()
    release = threading.Event()
    plan = _scrape_plan()

    def slow_resolve(_url: str) -> tuple[SimpleNamespace, str, str]:
        started.set()
        release.wait(1.0)
        finished.set()
        return plan, "auto", ""

    monkeypatch.setattr(harness.scraper, "_resolve_scrape_plan", slow_resolve)
    task = asyncio.create_task(harness.scraper.scrape_article(URL))
    cancel_handle = asyncio.get_running_loop().call_later(0.02, task.cancel)

    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert started.is_set()
        assert not finished.is_set()
        harness.evaluate_target.assert_not_awaited()
        release.set()
        assert await asyncio.to_thread(finished.wait, 1.0)
        harness.evaluate_target.assert_not_awaited()
    finally:
        cancel_handle.cancel()
        release.set()
        if started.is_set() and not finished.is_set():
            await asyncio.to_thread(finished.wait, 1.0)


@pytest.mark.unit
async def test_article_extraction_cancellation_is_not_recovered_as_a_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_article_defaults(monkeypatch)
    harness.extractor.side_effect = asyncio.CancelledError("caller cancelled")

    with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
        await harness.article.scrape_article(URL)


@pytest.mark.unit
async def test_enhanced_playwright_extraction_cancellation_closes_browser_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = install_enhanced_defaults(monkeypatch)

    class Page:
        closed = False

        async def goto(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def wait_for_load_state(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        async def content(self) -> str:
            return "<article>Body</article>"

        async def close(self) -> None:
            self.closed = True

    class Context:
        closed = False

        def __init__(self) -> None:
            self.page = Page()

        async def new_page(self) -> Page:
            return self.page

        async def close(self) -> None:
            self.closed = True

    class Browser:
        def __init__(self) -> None:
            self.context = Context()

        async def new_context(self, **_kwargs: Any) -> Context:
            return self.context

    async def cancel_extraction(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise asyncio.CancelledError("caller cancelled")

    browser = Browser()
    harness.scraper._browser = browser
    monkeypatch.setattr(harness.enhanced, "run_extraction_in_thread", cancel_extraction)

    with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
        await harness.enhanced.EnhancedWebScraper._scrape_with_playwright(harness.scraper, URL)

    assert browser.context.page.closed is True
    assert browser.context.closed is True
