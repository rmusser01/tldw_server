"""Lifecycle commands authorize verified principals before touching shared state."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.endpoints import web_scraping as endpoints
from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ScrapingJob, ScrapingJobQueue
from tldw_Server_API.app.services import enhanced_web_scraping_service as services

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("command", ["initialize", "shutdown"])
async def test_claimed_admin_in_options_cannot_control_lifecycle(monkeypatch, command):
    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[endpoints.get_request_user] = lambda: SimpleNamespace(
        id=7, role="user", roles=["user"], permissions=[], is_admin=True
    )

    def forbidden_accessor():
        pytest.fail("Unauthorized lifecycle command reached the singleton accessor")

    monkeypatch.setattr(endpoints, "get_web_scraping_service", forbidden_accessor)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            f"/web-scraping/service/{command}",
            json={"role": "admin", "owner_user_id": 1, "options": {"permissions": ["*"]}},
        )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "claims",
    [
        {"role": "owner"},
        {"roles": ["super_admin"]},
        {"permissions": ["system.configure"]},
        {"permissions": ["admin"]},
        {"roles": ["admin"]},
        {"permissions": ["*"]},
    ],
)
@pytest.mark.parametrize("command", ["initialize", "shutdown"])
async def test_verified_platform_admin_can_control_lifecycle(monkeypatch, claims, command):
    svc = services.WebScrapingService()

    async def factory():
        return SimpleNamespace(stop=stop, job_queue=SimpleNamespace(get_status=lambda: {}))

    async def stop():
        return None

    monkeypatch.setattr(services, "create_enhanced_scraper", factory)
    monkeypatch.setattr(endpoints, "get_web_scraping_service", lambda: svc)
    monkeypatch.setattr(svc, "get_service_status", lambda: {"initialized": svc._initialized})
    await svc.initialize()
    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[endpoints.get_request_user] = lambda: SimpleNamespace(id=1, **claims)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(f"/web-scraping/service/{command}")
    assert response.status_code == 200
    assert svc._initialized is (command == "initialize")


async def test_owner_status_and_cancel_and_wrong_owner_not_found_semantics():
    svc = services.WebScrapingService()
    svc._initialized = True
    queue = ScrapingJobQueue()
    svc.scraper = SimpleNamespace(job_queue=queue)
    job = ScrapingJob("owned", "https://example.com", "auto", user_id="7")
    future = await queue.add_job(job)
    try:
        for operation in (svc.get_job_status, svc.cancel_job):
            for user, job_id, code in (
                (None, "owned", 401),
                (SimpleNamespace(id=8), "owned", 403),
                (SimpleNamespace(id=7), "absent", 404),
            ):
                with pytest.raises(HTTPException) as error:
                    await operation(job_id, user)
                assert error.value.status_code == code
        assert (await svc.get_job_status("owned", SimpleNamespace(id=7)))["user_id"] == "7"
        assert (await svc.get_job_status("owned", SimpleNamespace(id=8, roles=["owner"])))["job_id"] == "owned"
        assert (await svc.cancel_job("owned", SimpleNamespace(id=7)))["status"] == "cancelled"
    finally:
        await queue.stop()
        await asyncio.gather(future, return_exceptions=True)


async def test_concurrent_trusted_initialization_is_idempotent(monkeypatch):
    svc = services.WebScrapingService()
    entered, release = asyncio.Event(), asyncio.Event()
    created = []

    async def factory():
        created.append(SimpleNamespace())
        entered.set()
        await release.wait()
        return created[-1]

    monkeypatch.setattr(services, "create_enhanced_scraper", factory)
    first = asyncio.create_task(svc.initialize())
    await entered.wait()
    second = asyncio.create_task(svc.initialize())
    release.set()
    await asyncio.gather(first, second)
    assert len(created) == 1
    assert svc.scraper is created[0]


async def test_cancelled_initializer_keeps_single_owned_startup(monkeypatch):
    svc = services.WebScrapingService()
    entered, release = asyncio.Event(), asyncio.Event()
    created = []

    async def factory():
        created.append(SimpleNamespace())
        entered.set()
        await release.wait()
        return created[-1]

    monkeypatch.setattr(services, "create_enhanced_scraper", factory)
    first = asyncio.create_task(svc.initialize())
    await entered.wait()
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    second = asyncio.create_task(svc.initialize())
    release.set()
    await second
    assert len(created) == 1
    assert svc.scraper is created[0]


async def test_cancelled_shutdown_waiter_cannot_replace_owned_resources(monkeypatch):
    svc = services.WebScrapingService()
    entered, release = asyncio.Event(), asyncio.Event()

    async def stop():
        entered.set()
        await release.wait()

    scraper = SimpleNamespace(stop=stop, job_queue=SimpleNamespace(get_status=lambda: {}))
    svc.scraper, svc._initialized = scraper, True
    waiter = asyncio.create_task(svc.shutdown())
    await entered.wait()
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    try:
        with pytest.raises(HTTPException) as error:
            await svc.initialize()
        assert error.value.status_code == 503
        assert svc.scraper is scraper
    finally:
        release.set()
        await svc.shutdown()


async def test_owned_queue_drain_blocks_replacement_and_global_cleanup(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.jobs.queue import ProcessLocalJobQueue
    from tldw_Server_API.tests.Web_Scraping.phase5_queue_fakes import Executor, job, settle

    executor = Executor()
    queue = ProcessLocalJobQueue(execute=executor)
    await queue.start()
    future = await queue.add_job(job())
    attempt = await executor.started.get()
    svc = services.WebScrapingService()
    scraper = SimpleNamespace(stop=queue.stop, job_queue=queue)
    svc.scraper, svc._initialized = scraper, True
    monkeypatch.setattr(services, "_web_scraping_service", svc)
    try:
        await services.shutdown_web_scraping_service()
        assert services.get_web_scraping_service() is svc
        with pytest.raises(HTTPException) as error:
            await svc.initialize()
        assert error.value.status_code == 503
        assert svc.scraper is scraper
    finally:
        attempt.drained.set_result(None)
        await asyncio.gather(future, return_exceptions=True)
        await settle(queue)
        await svc.shutdown()


@pytest.mark.parametrize("failure", [OSError, asyncio.CancelledError])
async def test_completed_failed_cleanup_can_be_retried_without_replacing_owner(monkeypatch, failure):
    svc = services.WebScrapingService()
    calls = 0
    entered, release = asyncio.Event(), asyncio.Event()
    state = {"state": "stopping"}

    async def stop():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure("transient cleanup failure")
        entered.set()
        await release.wait()

    scraper = SimpleNamespace(stop=stop, job_queue=SimpleNamespace(get_status=lambda: dict(state)))
    svc.scraper, svc._initialized = scraper, True
    monkeypatch.setattr(services, "_web_scraping_service", svc)
    with pytest.raises(failure):
        await services.shutdown_web_scraping_service()
    assert services.get_web_scraping_service() is svc
    assert svc.scraper is scraper and svc._stopping
    with pytest.raises(HTTPException) as error:
        await svc.initialize()
    assert error.value.status_code == 503
    assert calls == 1

    first = asyncio.create_task(svc.shutdown())
    # Also observe a cached-failure retry promptly instead of waiting on an event
    # that only a new cleanup attempt can set.
    started = asyncio.create_task(entered.wait())
    await asyncio.wait((first, started), return_when=asyncio.FIRST_COMPLETED)
    try:
        assert entered.is_set(), "Explicit retry reused the completed failed cleanup"
        second = asyncio.create_task(svc.shutdown())
        release.set()
        await asyncio.gather(first, second)
        assert calls == 2
        assert svc.scraper is scraper and svc._stopping
        with pytest.raises(HTTPException) as error:
            await svc.initialize()
        assert error.value.status_code == 503
        state["state"] = "stopped"
        await services.shutdown_web_scraping_service()
        assert services._web_scraping_service is None
        assert calls == 2

        replacement = SimpleNamespace()

        async def factory():
            return replacement

        monkeypatch.setattr(services, "create_enhanced_scraper", factory)
        await svc.initialize()
        assert svc.scraper is replacement and svc._initialized
    finally:
        release.set()
        started.cancel()
        await asyncio.gather(first, started, return_exceptions=True)


async def test_initialization_and_task_exception_logs_drop_private_text(monkeypatch):
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    secret = "sentinel-auth Cookie sentinel-cookie https://user:sentinel-pass@example.com/?token=sentinel-token"

    async def failed_factory():
        raise RuntimeError(secret)

    monkeypatch.setattr(services, "create_enhanced_scraper", failed_factory)
    svc = services.WebScrapingService()
    try:
        with pytest.raises(RuntimeError):
            await svc.initialize()
        svc._initialized = True
        monkeypatch.setattr(services, "load_and_log_configs", lambda: {})
        with pytest.raises(HTTPException):
            await svc.process_web_scraping_task(scrape_method=secret, url_input=secret)
        assert "sentinel-" not in "".join(messages)
    finally:
        logger.remove(sink)


async def test_concurrent_global_shutdown_is_idempotent(monkeypatch):
    svc = services.WebScrapingService()
    entered, release = asyncio.Event(), asyncio.Event()

    async def stop():
        entered.set()
        await release.wait()

    svc.scraper = SimpleNamespace(stop=stop, job_queue=SimpleNamespace(get_status=lambda: {}))
    svc._initialized = True
    monkeypatch.setattr(services, "_web_scraping_service", svc)
    first = asyncio.create_task(services.shutdown_web_scraping_service())
    await entered.wait()
    second = asyncio.create_task(services.shutdown_web_scraping_service())
    release.set()
    await asyncio.gather(first, second)
    assert services._web_scraping_service is None


async def test_legacy_worker_and_summary_exception_logs_are_safe(monkeypatch):
    secret = "sentinel-auth Cookie sentinel-cookie"
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")

    async def scrape_article(*args, **kwargs):
        raise ValueError(secret)

    def analyze(**kwargs):
        raise RuntimeError(secret)

    queue = ScrapingJobQueue(parent_scraper=SimpleNamespace(scrape_article=scrape_article), max_concurrent=1)
    await queue.start()
    try:
        future = await queue.add_job(ScrapingJob("failed", "https://example.com", "auto"))
        with pytest.raises(ValueError):
            await future
        monkeypatch.setattr(services, "analyze", analyze)
        result = await services.WebScrapingService()._summarize_content("body", "prompt", "openai", None, "system", 0.7)
        assert result == "Summary generation failed"
        assert "sentinel-" not in "".join(messages)
    finally:
        await queue.stop()
        logger.remove(sink)


async def test_legacy_facade_exception_logs_never_echo_execution_data(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as legacy

    scraper = legacy.EnhancedWebScraper(config={})
    secret = "sentinel-auth Cookie sentinel-cookie <html>sentinel-discovery</html>"
    url = "https://user:sentinel-pass@example.com/article?token=sentinel-token"
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")

    def fail(*args, **kwargs):
        raise ValueError(secret)

    async def acquire():
        return None

    monkeypatch.setattr(scraper.rate_limiter, "acquire", acquire)
    monkeypatch.setattr(scraper, "_resolve_scrape_plan", fail)
    monkeypatch.setattr(legacy, "analyze", fail)
    try:
        result = await scraper.scrape_article(url)
        assert result["extraction_successful"] is False
        assert await scraper._summarize_content("body") == "Summary generation failed"
        assert "sentinel-" not in "".join(messages)
    finally:
        logger.remove(sink)
