"""Immediate bounded admission, private snapshots, and legacy model parity."""

import asyncio
import json
from dataclasses import fields, replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as legacy
from tldw_Server_API.app.core.Web_Scraping.jobs.admission import JobPayload, QueueRejected
from tldw_Server_API.app.core.Web_Scraping.jobs.models import JobPriority, JobStatus, QueueLimits, ScrapingJob
from tldw_Server_API.app.core.Web_Scraping.jobs.queue import ProcessLocalJobQueue
from tldw_Server_API.tests.Web_Scraping.phase5_queue_fakes import Executor, job, settle

pytestmark = pytest.mark.unit


def test_public_model_parity():
    assert [(x.name, x.value) for x in JobPriority] == [(x.name, x.value) for x in legacy.JobPriority]
    assert [(x.name, x.value) for x in JobStatus] == [(x.name, x.value) for x in legacy.JobStatus]
    assert [f.name for f in fields(ScrapingJob)] == [f.name for f in fields(legacy.ScrapingJob)]
    old = legacy.ScrapingJob("id", "url", "auto")
    new = ScrapingJob("id", "url", "auto", created_at=old.created_at)
    assert new.to_dict() == old.to_dict()
    assert new.max_retries == old.max_retries == 3


def test_payload_size_matches_utf8_compact_json_without_retained_encoding():
    candidate = job(metadata={"value": ['\u00e9\n\\"', 4, False, None, 1.25]})
    payload = JobPayload.from_job(candidate)
    expected = {
        "job_id": "job",
        "url": "https://example.com/job",
        "operation": "auto",
        "owner": None,
        "options": candidate.metadata,
    }
    assert payload.size_bytes == len(json.dumps(expected, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    assert "value" not in repr(payload) and "options" not in repr(payload)


@pytest.mark.parametrize("field", [f.name for f in fields(QueueLimits)])
@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, float("inf"), float("nan"), "2"])
def test_invalid_limits(field, bad):
    with pytest.raises(ValueError):
        replace(QueueLimits(), **{field: bad})


def test_default_limits():
    limits = QueueLimits()
    assert (limits.max_pending, limits.max_live, limits.max_payload_bytes) == (128, 256, 262144)
    assert (limits.max_total_payload_bytes, limits.max_history_bytes) == (16777216, 67108864)


@pytest.mark.parametrize(
    "options",
    [
        {"max_concurrent": True},
        {"max_concurrent": 0},
        {"completed_retention": True},
        {"completed_retention": -1},
        {"completed_retention": 1.5},
        {"max_concurrent": 3, "limits": QueueLimits(max_live=2)},
        {"max_concurrent": 3, "limits": QueueLimits(max_pending=2)},
    ],
)
def test_incoherent_queue_configuration(options):
    with pytest.raises(ValueError):
        ProcessLocalJobQueue(execute=Executor(), **options)


class Unserializable:
    def __str__(self):
        raise AssertionError("must not stringify private arbitrary objects")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        {"bad": Unserializable()},
        {"bad": lambda: None},
        {1: "bad-key"},
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"bad": {1, 2}},
        {"controls": Unserializable()},
    ],
)
async def test_malformed_payload_has_no_retained_state(metadata):
    q = ProcessLocalJobQueue(execute=Executor())
    with pytest.raises(ValueError):
        await q.add_job(job(metadata=metadata))
    assert q.get_status()["live_jobs"] == q.get_status()["payload_bytes"] == 0
    assert not q._entries and not q._heap and not q._tasks


@pytest.mark.asyncio
async def test_cyclic_payload_rejected():
    metadata = {}
    metadata["cycle"] = metadata
    q = ProcessLocalJobQueue(execute=Executor())
    with pytest.raises(ValueError):
        await q.add_job(job(metadata=metadata))
    assert q.get_status()["live_jobs"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"job_id": None},
        {"url": ""},
        {"method": 7},
        {"user_id": Unserializable()},
        {"priority": Unserializable()},
        {"status": JobStatus.COMPLETED},
        {"retries": True},
        {"max_retries": -1},
        {"metadata": []},
        {"created_at": Unserializable()},
    ],
)
async def test_invalid_job_rejected_before_admission(options):
    candidate = job()
    for name, value in options.items():
        setattr(candidate, name, value)
    q = ProcessLocalJobQueue(execute=Executor())
    with pytest.raises(ValueError):
        await q.add_job(candidate)
    assert not q._entries and q.get_status()["live_jobs"] == 0


@pytest.mark.asyncio
async def test_payload_snapshot_utf8_size_and_individual_exact_boundary():
    candidate = job(metadata={"custom_cookies": {"session": "secret"}, "text": "\u00e9"})
    payload = JobPayload.from_job(candidate)
    q = ProcessLocalJobQueue(
        execute=Executor(),
        max_concurrent=1,
        limits=QueueLimits(max_payload_bytes=payload.size_bytes, max_total_payload_bytes=payload.size_bytes),
    )
    future = await q.add_job(candidate)
    candidate.metadata["custom_cookies"]["session"] = "changed"
    assert q._entries["job"].payload.options["custom_cookies"]["session"] == "secret"
    assert q.get_status()["payload_bytes"] == payload.size_bytes
    await q.cancel_job("job")
    with pytest.raises(Exception, match="Job cancelled"):
        await future
    smaller = ProcessLocalJobQueue(
        execute=Executor(), max_concurrent=1, limits=QueueLimits(max_payload_bytes=payload.size_bytes - 1)
    )
    with pytest.raises(QueueRejected) as caught:
        await smaller.add_job(job(metadata={"custom_cookies": {"session": "secret"}, "text": "\u00e9"}))
    assert caught.value.code == "job_payload_too_large"
    assert smaller.get_status()["live_jobs"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("ceiling", ["pending", "live", "bytes"])
async def test_shared_aggregate_limits_reject_immediately_without_residue(ceiling):
    size = JobPayload.from_job(job("a")).size_bytes
    limits = QueueLimits(
        max_pending=2 if ceiling == "pending" else 8,
        max_live=2 if ceiling == "live" else 8,
        max_total_payload_bytes=2 * size if ceiling == "bytes" else 100000,
    )
    q = ProcessLocalJobQueue(execute=Executor(), max_concurrent=1, limits=limits)
    accepted = [await q.add_job(job(name, user_id=None)) for name in ("a", "b")]
    before = q.get_status()
    rejected = job("c", priority=JobPriority.CRITICAL)
    with pytest.raises(QueueRejected) as caught:
        await q.add_job(rejected)
    assert caught.value.code == "queue_overloaded"
    assert q.get_status() == before and q.get_job("c") is None
    assert rejected.context is None
    await q.stop()
    await asyncio.gather(*accepted, return_exceptions=True)


@pytest.mark.asyncio
async def test_duplicate_id_rejected_without_replacing_subscription():
    q = ProcessLocalJobQueue(execute=Executor())
    first = await q.add_job(job())
    with pytest.raises(ValueError):
        await q.add_job(job())
    await q.stop()
    with pytest.raises(Exception, match="shutdown"):
        await first


@given(st.lists(st.booleans(), min_size=1, max_size=80))
def test_admission_release_state_machine(actions):
    async def scenario():
        q = ProcessLocalJobQueue(
            execute=Executor(), max_concurrent=1, completed_retention=0, limits=QueueLimits(max_pending=4, max_live=4)
        )
        accepted = []
        for index, admit in enumerate(actions):
            if admit:
                try:
                    future = await q.add_job(job(str(index)))
                except QueueRejected:
                    assert len(accepted) == 4
                else:
                    accepted.append((str(index), future))
            elif accepted:
                name, future = accepted.pop(0)
                await q.cancel_job(name)
                await asyncio.gather(future, return_exceptions=True)
            assert q.get_status()["live_jobs"] == len(accepted)
            assert q.get_status()["pending_reservations"] == len(accepted)
            assert len(q._heap) == len(accepted)
        await q.stop()
        await asyncio.gather(*(f for _, f in accepted), return_exceptions=True)
        await settle(q)
        assert q.get_status()["payload_bytes"] == 0

    asyncio.run(scenario())
