"""Bounded safe history never controls live subscriber delivery."""

import asyncio
import json

import pytest

from tldw_Server_API.app.core.Web_Scraping.jobs.models import QueueLimits
from tldw_Server_API.app.core.Web_Scraping.jobs.queue import ProcessLocalJobQueue
from tldw_Server_API.tests.Web_Scraping.phase5_queue_fakes import Executor, job, page, settle, turn

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("retention", [0, 1, 2])
async def test_count_eviction_and_context_credentials_not_retained(retention):
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor, completed_retention=retention)
    await q.start()
    for i in range(4):
        future = await q.add_job(job(str(i), metadata={"custom_cookies": {"session": "private"}}))
        attempt = await executor.started.get()
        attempt.finish(
            page(
                title="title",
                metadata={"headers": {"x": "secret"}, "labels": ["public"]},
                custom_extra={"api_key": "secret", "rank": 7},
                _discovered_links=["private"],
            )
        )
        delivered = await future
        assert delivered["custom_extra"]["rank"] == 7
        await settle(q)
    assert q.get_status()["completed_jobs"] == retention
    assert q.get_job("0") is None
    if retention:
        retained = q.get_job("3")
        assert retained.metadata == {} and retained.context is None
        assert retained.result == {
            "content": "body",
            "extraction_successful": True,
            "title": "title",
            "metadata": {"labels": ["public"]},
        }
    await q.stop()


async def test_huge_result_delivered_but_not_retained():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor, limits=QueueLimits(max_history_bytes=512))
    await q.start()
    future = await q.add_job(job())
    attempt = await executor.started.get()
    huge = page(content="x" * 1000000)
    attempt.finish(huge)
    assert (await future)["content"] is huge.article["content"]
    await settle(q)
    assert q.get_job("job") is None and q.get_status()["history_bytes"] == 0
    await q.stop()


async def test_byte_eviction_stays_bounded_and_polling_cannot_mutate_history():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor, limits=QueueLimits(max_history_bytes=1500))
    await q.start()
    for i in range(8):
        future = await q.add_job(job(str(i)))
        attempt = await executor.started.get()
        attempt.finish(page(content="x" * 200))
        await future
        await settle(q)
        assert 0 < q.get_status()["history_bytes"] <= 1500
    assert q.get_job("0") is None and q.get_job("7") is not None
    polled = q.get_job("7")
    polled.result["content"] = "changed"
    assert q.get_job("7").result["content"] == "x" * 200
    await q.stop()


async def test_repeated_pending_cancellation_does_not_accumulate_tombstones():
    q = ProcessLocalJobQueue(execute=Executor(), completed_retention=0)
    for i in range(400):
        future = await q.add_job(job(str(i)))
        await q.cancel_job(str(i))
        await asyncio.gather(future, return_exceptions=True)
        assert not q._heap and not q._entries
        assert q.get_status()["live_jobs"] == q.get_status()["history_bytes"] == 0
    await q.stop()


async def test_history_string_encoder_only_sees_bounded_chunks(monkeypatch):
    lengths = []
    dumps = json.dumps

    def measured(value, **kwargs):
        if type(value) is str:
            lengths.append(len(value))
            assert len(value) <= 256
        return dumps(value, **kwargs)

    monkeypatch.setattr(json, "dumps", measured)
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor, limits=QueueLimits(max_history_bytes=512))
    await q.start()
    future = await q.add_job(job())
    attempt = await executor.started.get()
    attempt.finish(page(content="x" * 1000000))
    assert len((await future)["content"]) == 1000000
    await settle(q)
    assert q.get_status()["history_bytes"] == 0 and max(lengths) <= 256
    await q.stop()


async def test_history_exact_byte_boundary_and_one_less():
    from tldw_Server_API.app.core.Web_Scraping.jobs.models import JobStatus
    from tldw_Server_API.app.core.Web_Scraping.jobs.projections import History

    candidate = job()
    candidate.status = JobStatus.COMPLETED
    candidate.result = page().article
    size = len(json.dumps(candidate.to_dict(), separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    exact = History(1, size)
    exact.record(candidate, candidate.result)
    assert exact.get("job").result == page().article and exact.bytes == size
    smaller = History(1, size - 1)
    smaller.record(candidate, candidate.result)
    assert smaller.get("job") is None and smaller.bytes == 0


@pytest.mark.parametrize("retention", [0, 1])
@pytest.mark.parametrize("failure", ["raised", "extraction"])
async def test_public_diagnostics_never_copy_private_live_failures_during_or_after_drain(retention, failure):
    private = "Authorization: Bearer bearer-sentinel; Cookie: session=cookie-sentinel; provider-private-detail"
    original = RuntimeError(private)
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor, completed_retention=retention)
    await q.start()
    future = await q.add_job(job(max_retries=0))
    attempt = await executor.started.get()
    raw = page(error=private, metadata={"nested": {"error": private, "message": private}})
    if failure == "raised":
        attempt.result.set_exception(original)
    else:
        attempt.result.set_result(raw)
    delivered = (await asyncio.gather(future, return_exceptions=True))[0]
    try:
        if failure == "raised":
            assert delivered is original and type(delivered) is RuntimeError
        else:
            assert type(delivered) is Exception
        assert str(delivered) == private
        assert not attempt.drained.done() and q.get_status()["live_jobs"] == 1
        polled = q.get_job("job")
        assert polled.error == ("execution_failed" if failure == "raised" else "source_extraction_failed")
        assert (await q.cancel_job("job")).error == polled.error
        if polled.result is not None:
            assert polled.result["error"] == "source_extraction_failed"
            assert polled.result["metadata"]["nested"] == {"error": "execution_failed", "message": "execution_failed"}
        serialized = json.dumps({"job": polled.to_dict(), "queue": q.get_status()})
        assert all(
            value not in serialized for value in ("bearer-sentinel", "cookie-sentinel", "provider-private-detail")
        )
    finally:
        attempt.drained.set_result(None)
        await settle(q)
        await q.stop()
    assert raw.article["error"] == private
    retained = q.get_job("job")
    if retention:
        assert retained.to_dict() == polled.to_dict()
    else:
        assert retained is None


async def test_public_diagnostics_safe_while_failed_attempt_waits_for_retry_drain():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor)
    await q.start()
    future = await q.add_job(job())
    attempt = await executor.started.get()
    attempt.result.set_result(page(error="Cookie: private-retry-cookie"))
    # Await the queue-owned execution reaching its physical-drain wait.
    while q._entries["job"].job.error is None:
        await turn()
    try:
        assert q.get_job("job").error == "source_extraction_failed"
        assert "private-retry-cookie" not in json.dumps(q.get_job("job").to_dict())
    finally:
        await q.stop()
        attempt.drained.set_result(None)
        await asyncio.gather(future, return_exceptions=True)
        await settle(q)


async def test_public_diagnostics_sanitization_does_not_mutate_successful_live_article():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor)
    await q.start()
    future = await q.add_job(job())
    attempt = await executor.started.get()
    raw = page(metadata={"nested": {"error": "Authorization: Bearer private-success"}})
    attempt.finish(raw)
    delivered = await future
    await settle(q)
    await q.stop()
    assert delivered is raw.article
    assert delivered["metadata"]["nested"]["error"] == "Authorization: Bearer private-success"
    assert q.get_job("job").result["metadata"]["nested"]["error"] == "execution_failed"


async def test_public_diagnostics_history_does_not_trust_supplied_job_error():
    from tldw_Server_API.app.core.Web_Scraping.jobs.models import JobStatus
    from tldw_Server_API.app.core.Web_Scraping.jobs.projections import History

    candidate = job(status=JobStatus.FAILED, error="provider diagnostic: Authorization private-provider")
    history = History(1, 4096)
    history.record(candidate, {"error": "Cookie private-cookie"})
    retained = history.get("job")
    assert retained.error == "execution_failed"
    assert retained.result["error"] == "source_extraction_failed"
    assert candidate.error == "provider diagnostic: Authorization private-provider"
