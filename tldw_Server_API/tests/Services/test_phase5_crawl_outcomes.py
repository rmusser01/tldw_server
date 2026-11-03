"""Terminal crawl facts survive service postprocessing and physical cleanup."""

import asyncio
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import CrawlOutcome, CrawlReport
from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper
from tldw_Server_API.app.services import enhanced_web_scraping_service as module
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
METHODS = ["Individual URLs", "Sitemap", "URL Level", "Recursive Scraping"]
PARTIAL = {"state": "partial", "stop_reason": "budget_exhausted", "has_partial_results": True}
COMPLETED = {"state": "completed", "stop_reason": "frontier_exhausted", "has_partial_results": False}


def article(url="https://example.com/news/a"):
    return {
        "url": url,
        "title": "A",
        "content": "Article facts",
        "extraction_successful": True,
        "metadata": {"depth": 2, "parent_url": "https://example.com", "score": 0.4},
        "custom_field": {"retained": True},
    }


@pytest.fixture
def harness(monkeypatch):
    """Replace only external acquisition/database/model work, not the service."""
    state = SimpleNamespace(
        report=CrawlReport.interrupted([article()], "budget_exhausted"),
        calls=[],
        summaries=[],
        saves=[],
        writes=[],
        exits=[],
        duplicate=False,
        storage_error=False,
    )
    scraper = SimpleNamespace(_progress={})

    async def report(*args, **kwargs):
        state.calls.append((args, kwargs))
        return state.report

    def set_progress(task_id, **updates):
        if task_id:
            scraper._progress.setdefault(task_id, {}).update(updates)

    async def save_progress(task_id, path):
        state.saves.append(deepcopy(scraper._progress.get(task_id, {})))
        if getattr(state, "save_error", False):
            raise OSError("secret progress path")

    def analyze(**kwargs):
        state.summaries.append(kwargs)
        if getattr(state, "summary_error", False):
            raise RuntimeError("secret model failure")
        return "Summary"

    scraper._batch_report = report
    scraper._crawl_report = report

    async def legacy_batch(*args, **kwargs):
        return list((await scraper._batch_report(*args, **kwargs)).articles)

    async def legacy_crawl(*args, **kwargs):
        return list((await scraper._crawl_report(*args, **kwargs)).articles)

    scraper.scrape_multiple = legacy_batch
    scraper.scrape_sitemap = legacy_crawl
    scraper.recursive_scrape = legacy_crawl
    scraper._set_progress = set_progress
    scraper.save_progress = save_progress
    scraper._summarize_content = EnhancedWebScraper._summarize_content.__get__(scraper)
    svc = module.WebScrapingService()
    svc._initialized = True
    svc.scraper = scraper
    state.service = svc
    state.scraper = scraper

    class DB:
        def add_media_with_keywords(self, **kwargs):
            state.writes.append(kwargs)
            if state.storage_error:
                raise RuntimeError("secret storage error")
            if hasattr(state, "write_entered"):
                state.write_entered.set()
                state.write_release.wait(5)
            return 8, "uuid", "Media 'A' already exists. Overwrite not enabled." if state.duplicate else "Stored"

        def update_media_reprocess_state(self, *args, **kwargs):
            state.writes.append((args, kwargs))

    @contextmanager
    def database(**kwargs):
        try:
            yield DB()
        finally:
            state.exits.append(True)

    monkeypatch.setattr(module, "managed_media_database", database)
    monkeypatch.setattr(module, "get_user_media_db_path", lambda owner: "unused.db")
    monkeypatch.setattr(module, "load_and_log_configs", lambda: {})
    monkeypatch.setattr(module, "analyze", analyze)
    monkeypatch.setattr("tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping.analyze", analyze)
    yield state
    for result_id in getattr(state, "ephemeral_ids", []):
        ephemeral_storage.remove_data(result_id)


async def process(state, method="Individual URLs", **kwargs):
    result = await state.service.process_web_scraping_task(
        scrape_method=method,
        url_input="https://example.com/news/a",
        url_level=2,
        perform_chunking=False,
        mode=kwargs.pop("mode", "ephemeral"),
        **kwargs,
    )
    if "ephemeral_id" in result:
        state.ephemeral_ids = [*getattr(state, "ephemeral_ids", []), result["ephemeral_id"]]
    return result


@pytest.mark.parametrize("method", METHODS)
async def test_ephemeral_receipt_preserves_report_before_summary_and_conversion(harness, method):
    result = await process(
        harness,
        method,
        summarize_checkbox=True,
        custom_titles="Chosen",
        summary_prompt_overrides={"system": "Owner system", "user": "Owner prompt"},
        api_name="provider",
        api_key="secret",
        temperature=0.2,
        custom_cookies=[{"name": "session", "value": "secret"}],
        custom_headers={"X-Test": "value"},
        user_agent="Agent",
        user_id=7,
    )
    payload = ephemeral_storage.get_data(result["ephemeral_id"])
    assert result["status"] == "ephemeral-ok"
    assert result["crawl_outcome"] == payload["crawl_outcome"] == payload["result"]["crawl_outcome"] == PARTIAL
    assert payload["result"]["articles"][0]["summary"] == "Summary"
    assert payload["result"]["articles"][0]["custom_field"] == {"retained": True}
    assert payload["result"]["articles"][0]["metadata"]["depth"] == 2
    call = harness.summaries[0]
    assert (call["custom_prompt_arg"], call["system_message"], call["api_name"], call["api_key"], call["temp"]) == (
        "Owner prompt",
        "Owner system",
        "provider",
        "secret",
        0.2,
    )
    options = harness.calls[0][1]
    credentials = options["options"] if method == "Individual URLs" else options
    assert credentials["custom_cookies"] == [{"name": "session", "value": "secret"}]
    assert credentials["custom_headers"] == {"X-Test": "value"}
    if method == "URL Level":
        assert options["profile"] == "enhanced_recursive"
        assert options["url_filter"]("https://example.com/news/a")
        assert not options["url_filter"]("https://example.com/news/a/b")
    if method == "Individual URLs":
        assert payload["result"]["articles"][0]["title"] == "Chosen"


@pytest.mark.parametrize(
    ("reason", "status"),
    [
        ("queue_overloaded", 503),
        ("queue_stopping", 503),
        ("transport_unavailable", 503),
        ("policy_denied", 403),
        ("deadline_exhausted", 504),
        ("budget_exhausted", 429),
        ("job_payload_too_large", 413),
        ("execution_failed", 500),
        ("checkpoint_failed", 500),
    ],
)
@pytest.mark.parametrize("failures", [False, True])
async def test_empty_failed_http_mapping_precedes_all_side_effects(harness, reason, status, failures):
    harness.report = CrawlReport.interrupted(
        [{"url": "https://secret.invalid", "error": "secret", "extraction_successful": False}] if failures else [],
        reason,
    )
    with pytest.raises(HTTPException) as error:
        await process(harness, summarize_checkbox=True, mode="persist")
    assert error.value.status_code == status
    assert "secret" not in str(error.value.detail)
    assert harness.writes == harness.summaries == []


@pytest.mark.parametrize("disposition", ["persist-ok", "duplicate", "storage-error"])
async def test_persistent_duplicate_and_failed_storage_receipts_keep_outcome(harness, disposition):
    harness.duplicate = disposition == "duplicate"
    harness.storage_error = disposition == "storage-error"
    result = await process(harness, mode="persist", user_id=7)
    assert result["crawl_outcome"] == PARTIAL
    assert result["status"] == ("duplicate" if harness.duplicate else "persist-ok")
    if harness.storage_error:
        assert result["stored_articles"] == 0 and result["errors"]
    else:
        assert harness.writes[0]["owner_user_id"] == 7
        assert '"crawl_depth": 2' in harness.writes[0]["safe_metadata"]


async def test_summary_failure_cannot_rewrite_terminal_outcome(harness):
    harness.summary_error = True
    result = await process(harness, summarize_checkbox=True)
    payload = ephemeral_storage.get_data(result["ephemeral_id"])["result"]
    assert result["crawl_outcome"] == PARTIAL
    assert payload["articles"][0]["summary"] == "Summary generation failed"


async def test_completed_batch_keeps_existing_policy_failure_shape(harness):
    harness.report = CrawlReport(
        (
            {
                "url": "https://example.com",
                "error": "blocked",
                "policy_reason": "robots",
                "extraction_successful": False,
                "title": "N/A",
                "author": "N/A",
                "date": "N/A",
                "content": "",
            },
        ),
        CrawlOutcome(**COMPLETED),
    )
    receipt = await process(harness)
    articles = ephemeral_storage.get_data(receipt["ephemeral_id"])["result"]["articles"]
    assert articles == [
        {"url": "https://example.com", "error": "blocked", "policy_reason": "robots", "extraction_successful": False}
    ]


@pytest.mark.parametrize("reason", ["budget_exhausted", "checkpoint_failed"])
async def test_secondary_progress_save_cannot_replace_primary_partial(harness, reason):
    harness.report = CrawlReport.interrupted([article()], reason)
    harness.save_error = True
    result = await process(harness, "Recursive Scraping")
    assert result["crawl_outcome"] == {**PARTIAL, "stop_reason": reason}
    assert len(harness.saves) == 1


async def test_cancelled_report_launches_no_summary_or_storage(harness):
    harness.report = CrawlReport.interrupted([article()], "cancelled")
    with pytest.raises(asyncio.CancelledError):
        await process(harness, summarize_checkbox=True, mode="persist")
    assert harness.writes == harness.summaries == harness.saves == []


@pytest.mark.parametrize("method", METHODS)
async def test_cancel_before_publication_preserves_cancelled_report(harness, method):
    entered = asyncio.Event()

    async def acquire(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            error.report = CrawlReport.interrupted([], "cancelled")
            raise

    harness.scraper._batch_report = acquire
    harness.scraper._crawl_report = acquire
    task = asyncio.create_task(process(harness, method, summarize_checkbox=True, user_id=7))
    await asyncio.wait_for(entered.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    progress = next(iter(harness.scraper._progress.values()))
    assert progress["crawl_outcome"] == {"state": "cancelled", "stop_reason": "cancelled", "has_partial_results": False}
    assert progress["service_status"] == "cancelled"
    assert progress["owner_user_id"] == "7"
    assert harness.writes == harness.summaries == []


async def test_unexpected_secondary_snapshot_error_does_not_replace_partial(harness):
    class SnapshotError(Exception):
        pass

    async def save(*args):
        raise SnapshotError("secret secondary")

    harness.scraper.save_progress = save
    result = await process(harness, "Recursive Scraping")
    assert result["crawl_outcome"] == PARTIAL


async def test_unexpected_primary_error_is_sanitized(harness):
    class AcquisitionError(Exception):
        pass

    async def acquire(*args, **kwargs):
        raise AcquisitionError("secret acquisition")

    harness.scraper._batch_report = acquire
    with pytest.raises(HTTPException) as error:
        await process(harness)
    assert error.value.status_code == 500
    assert "secret" not in str(error.value.detail)
    assert next(iter(harness.scraper._progress.values()))["service_status"] == "failed"


@pytest.mark.parametrize("method", METHODS)
async def test_midcrawl_failure_entry_keeps_partial_and_only_success_is_processed(harness, method):
    harness.report = CrawlReport.interrupted(
        [
            article(),
            {**article("https://example.com/failed"), "error": "browser_transport_unavailable"},
        ],
        "transport_unavailable",
    )
    result = await process(harness, method, summarize_checkbox=True, mode="persist")
    assert result["crawl_outcome"] == {**PARTIAL, "stop_reason": "transport_unavailable"}
    assert len(harness.summaries) == 1
    assert len([item for item in harness.writes if isinstance(item, dict)]) == 1
    assert result["extraction_failures"]


@pytest.mark.parametrize("method", METHODS)
async def test_cancellation_during_first_summary_never_starts_next_or_storage(harness, method):
    harness.report = CrawlReport((article(), article("https://example.com/news/b")), CrawlOutcome(**COMPLETED))
    entered, release = threading.Event(), threading.Event()

    def summary(**kwargs):
        harness.summaries.append(kwargs)
        entered.set()
        release.wait(5)
        return "Summary"

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "analyze", summary)
        patch.setattr("tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping.analyze", summary)
        task = asyncio.create_task(process(harness, method, summarize_checkbox=True, mode="persist"))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert next(iter(harness.scraper._progress.values()))["crawl_outcome"] == COMPLETED
            assert harness.writes == harness.saves == []
            assert len(harness.summaries) == 1
        finally:
            release.set()


async def test_recursive_primary_error_survives_secondary_snapshot_error(harness):
    harness.save_error = True

    async def fail(*args, **kwargs):
        raise RuntimeError("secret primary")

    harness.scraper._crawl_report = fail
    with pytest.raises(HTTPException) as error:
        await process(harness, "Recursive Scraping")
    assert error.value.status_code == 500
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "secret primary"
    assert len(harness.saves) == 1


async def test_real_adapter_progress_attaches_terminal_outcome_without_private_state(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as core

    report = CrawlReport.interrupted([article()], "deadline_exhausted")

    async def acquire(*args, **kwargs):
        return report

    monkeypatch.setattr(core, "execute_crawl", acquire)
    monkeypatch.setattr(core, "get_database_dir", lambda: str(tmp_path))
    scraper = EnhancedWebScraper({})
    try:
        await scraper._crawl_report(
            "https://example.com", profile="enhanced_recursive", options={}, task_id="owned", user_id="7"
        )
        progress = scraper.get_progress("owned", owner_user_id="7")
        assert progress["crawl_outcome"] == {
            "state": "partial",
            "stop_reason": "deadline_exhausted",
            "has_partial_results": True,
        }
        assert scraper.get_progress("owned", owner_user_id="8") == {}
    finally:
        await scraper.job_queue.stop()


async def test_reused_adapter_progress_replaces_previous_outcome_on_cancellation(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as core

    async def acquire(*args, **kwargs):
        error = asyncio.CancelledError()
        error.report = CrawlReport.interrupted([], "cancelled")
        raise error

    monkeypatch.setattr(core, "execute_crawl", acquire)
    monkeypatch.setattr(core, "get_database_dir", lambda: str(tmp_path))
    scraper = EnhancedWebScraper({})
    scraper._progress["owned"] = {"owner_user_id": "7", "crawl_outcome": COMPLETED}
    try:
        with pytest.raises(asyncio.CancelledError):
            await scraper._crawl_report(
                "https://example.com", profile="enhanced_recursive", options={}, task_id="owned", user_id="7"
            )
        assert scraper.get_progress("owned", owner_user_id="7")["crawl_outcome"] == {
            "state": "cancelled",
            "stop_reason": "cancelled",
            "has_partial_results": False,
        }
    finally:
        await scraper.job_queue.stop()


async def test_late_cancel_drains_blocking_write_without_closing_db_or_launching_next(harness):
    harness.report = CrawlReport((article(), article("https://example.com/news/b")), CrawlOutcome(**COMPLETED))
    harness.write_entered, harness.write_release = threading.Event(), threading.Event()
    task = asyncio.create_task(process(harness, mode="persist"))
    try:
        assert await asyncio.to_thread(harness.write_entered.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert harness.exits == [], "database cannot close before its real write exits"
        progress = next(iter(harness.scraper._progress.values()))
        assert progress["crawl_outcome"] == COMPLETED
        assert progress["service_status"] == "cancelled"
    finally:
        harness.write_release.set()
    for _ in range(1000):
        if harness.exits:
            break
        await asyncio.sleep(0.001)
    assert harness.exits == [True]
    assert len(harness.writes) == 1, "no next article or post-write update after cancellation"


async def test_cancel_preserves_already_committed_media_and_drains_only_started_write(harness, monkeypatch, tmp_path):
    from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database

    path = str(tmp_path / "media.db")
    committed, closed = [], []
    entered, release = threading.Event(), threading.Event()
    harness.report = CrawlReport(
        tuple(article(f"https://example.com/{i}") for i in range(3)), CrawlOutcome(**COMPLETED)
    )

    @contextmanager
    def database(**kwargs):
        with managed_media_database(client_id="7", db_path=path, initialize=False) as db:

            def write(**fields):
                if committed:
                    entered.set()
                    release.wait(5)
                result = db.add_media_with_keywords(**fields)
                committed.append(result[0])
                return result

            try:
                yield SimpleNamespace(
                    add_media_with_keywords=write, update_media_reprocess_state=db.update_media_reprocess_state
                )
            finally:
                closed.append(True)

    monkeypatch.setattr(module, "managed_media_database", database)
    task = asyncio.create_task(process(harness, mode="persist", user_id=7))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(committed) == 1 and closed == []
        with managed_media_database(client_id="reader", db_path=path, initialize=False) as db:
            assert db.get_media_by_id(committed[0])["url"] == "https://example.com/0"
    finally:
        release.set()
    for _ in range(1000):
        if closed:
            break
        await asyncio.sleep(0.001)
    assert len(committed) == 2 and closed == [True]
    with managed_media_database(client_id="reader", db_path=path, initialize=False) as db:
        assert db.get_media_by_id(committed[0])["url"] == "https://example.com/0"


async def test_stopping_service_cannot_launch_work_before_shutdown_task_runs(harness):
    harness.service._stopping = True
    with pytest.raises(HTTPException) as error:
        await process(harness)
    assert error.value.status_code == 503
    assert harness.calls == []


async def test_entered_shutdown_survives_cancelled_waiters_and_keeps_actual_drain(monkeypatch):
    service = module.WebScrapingService()
    startup_entered, release_startup, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    counts = {"startup": 0, "stop": 0}
    queue_state = {"state": "stopping"}

    async def stop():
        counts["stop"] += 1
        stopped.set()

    scraper = SimpleNamespace(stop=stop, job_queue=SimpleNamespace(get_status=lambda: dict(queue_state)))

    async def startup():
        counts["startup"] += 1
        startup_entered.set()
        await release_startup.wait()
        return scraper

    monkeypatch.setattr(module, "create_enhanced_scraper", startup)
    waiter = asyncio.create_task(service.initialize())
    await startup_entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    shutdown = asyncio.create_task(service.shutdown())
    for _ in range(100):
        if service._lifecycle_lock.locked():
            break
        await asyncio.sleep(0)
    assert service._lifecycle_lock.locked(), "shutdown entered the serialized boundary"
    shutdown.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    release_startup.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert stopped.is_set(), "entered shutdown must create cleanup after startup even when waiter leaves"
    with pytest.raises(HTTPException) as error:
        await service.initialize()
    assert error.value.status_code == 503
    assert service.scraper is scraper and service._stopping
    assert service.get_service_status()["status"] == "stopping"
    queue_state["state"] = "stopped"
    await service.initialize()
    assert counts == {"startup": 2, "stop": 1}


async def test_entered_shutdown_with_failed_startup_can_be_retried(monkeypatch):
    service = module.WebScrapingService()
    entered, release = asyncio.Event(), asyncio.Event()

    async def startup():
        entered.set()
        await release.wait()
        raise RuntimeError("startup failed")

    monkeypatch.setattr(module, "create_enhanced_scraper", startup)
    initializer = asyncio.create_task(service.initialize())
    await entered.wait()
    initializer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await initializer
    shutdown = asyncio.create_task(service.shutdown())
    await asyncio.sleep(0)
    assert service._lifecycle_lock.locked()
    release.set()
    with pytest.raises(RuntimeError, match="startup failed"):
        await shutdown
    await service.shutdown()
    assert not service._stopping
    assert service.scraper is None and not service._initialized
