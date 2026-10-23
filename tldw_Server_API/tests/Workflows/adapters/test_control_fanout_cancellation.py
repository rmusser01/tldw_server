"""map and parallel cancel their other sub-steps when one raises, and a cancelled sub-step stops its work (TASK-13863)."""

import asyncio
import contextlib
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.exceptions import AdapterError, LLMQuotaExceededError
from tldw_Server_API.app.core.Workflows.adapters.control import flow
from tldw_Server_API.app.core.Workflows.adapters.media import ingest
from tldw_Server_API.app.core.Workflows.subprocess_utils import SubprocessTask

pytestmark = pytest.mark.unit


def _refusal() -> LLMQuotaExceededError:
    return LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )


def _upstream_failure() -> HTTPException:
    return HTTPException(status_code=500, detail="upstream failed")


def _recording_substep(log: list[tuple[str, int]], error: Exception, release: asyncio.Event):
    """Sub-step 0 raises ``error`` at its first await; every other one waits for ``release``. ``log`` says what each did."""

    async def substep(config, context):
        n = context.get("item", config.get("n"))
        log.append(("started", n))
        if n == 0:
            await asyncio.sleep(0)
            raise error
        try:
            await release.wait()
        except asyncio.CancelledError:
            log.append(("cancelled", n))
            raise
        log.append(("finished", n))
        return {"text": str(n)}

    return substep


async def _run_then_release(log: list[tuple[str, int]], error: Exception, release: asyncio.Event, call_step) -> Exception:
    """Run the step, mark when it raised, then release any sibling still waiting and let it run.

    The 5 s bound only matters on dev, where a parallel step without fail_fast waits for its siblings,
    and they wait for this release.
    """
    with pytest.raises(type(error)) as raised:
        await asyncio.wait_for(call_step(), timeout=5)
    log.append(("step_raised", -1))
    release.set()
    for _ in range(20):
        await asyncio.sleep(0)
    return raised.value


def _ran_after_the_step_raised(log: list[tuple[str, int]]) -> list[int]:
    """Siblings that finished, or were still being cancelled, after the step had raised."""
    raised_at = log.index(("step_raised", -1))
    return [n for kind, n in log[raised_at:] if kind in {"finished", "cancelled"}]


def _never_started(log: list[tuple[str, int]], total: int) -> list[int]:
    started = {n for kind, n in log if kind == "started"}
    return [n for n in range(total) if n not in started]


async def test_map_cancels_the_other_items_when_one_raises() -> None:
    """On dev, items 1-5 finished after the step had raised."""
    log: list[tuple[str, int]] = []
    error, release = _refusal(), asyncio.Event()
    config = {"items": [0, 1, 2, 3, 4, 5], "step": {"type": "summarize", "config": {}}, "concurrency": 2}

    with patch.object(flow, "get_adapter", return_value=_recording_substep(log, error, release)):
        raised = await _run_then_release(log, error, release, lambda: flow.run_map_adapter(config, {}))

    assert raised is error
    assert _ran_after_the_step_raised(log) == []
    # Item 2 takes the slot item 0 frees and is cancelled at its first await; 3-5 never start.
    assert _never_started(log, 6) == [3, 4, 5]


@pytest.mark.parametrize("error", [_refusal(), _upstream_failure()], ids=["llm_quota_refusal", "http_500"])
async def test_parallel_with_fail_fast_cancels_the_other_steps_when_one_raises(error: Exception) -> None:
    """On dev, steps 1-3 finished after the step had raised."""
    log: list[tuple[str, int]] = []
    release = asyncio.Event()
    config = {
        "steps": [{"type": "summarize", "config": {"n": n}} for n in range(4)],
        "max_concurrency": 2,
        "fail_fast": True,
    }

    with patch.object(flow, "get_adapter", return_value=_recording_substep(log, error, release)):
        raised = await _run_then_release(log, error, release, lambda: flow.run_parallel_adapter(config, {}))

    assert raised is error
    assert _ran_after_the_step_raised(log) == []
    assert _never_started(log, 4) == [3]


async def test_parallel_without_fail_fast_stops_the_other_steps_on_a_quota_refusal() -> None:
    """The refusal fails the step (TASK-13638); on dev the step waited for its siblings, so wait_for timed out."""
    log: list[tuple[str, int]] = []
    error, release = _refusal(), asyncio.Event()
    config = {"steps": [{"type": "summarize", "config": {"n": n}} for n in range(4)], "max_concurrency": 2}

    with patch.object(flow, "get_adapter", return_value=_recording_substep(log, error, release)):
        raised = await _run_then_release(log, error, release, lambda: flow.run_parallel_adapter(config, {}))

    assert raised is error
    assert _ran_after_the_step_raised(log) == []
    assert _never_started(log, 4) == [3]


async def test_parallel_without_fail_fast_collects_every_other_error() -> None:
    """A sub-step error outside the noncritical tuple is collected too; on dev it was dropped as None."""

    async def substep(config, _context):
        if config["n"] == 0:
            raise AdapterError("boom")
        if config["n"] == 1:
            raise _upstream_failure()
        return {"text": "ok"}

    config = {"steps": [{"type": "summarize", "config": {"n": n}} for n in range(3)]}
    with patch.object(flow, "get_adapter", return_value=substep):
        result = await flow.run_parallel_adapter(config, {})

    assert result["results"] == [{"error": "parallel_step_error"}, {"error": "parallel_step_error"}, {"text": "ok"}]
    assert result["errors"] is None


async def test_parallel_without_fail_fast_collects_a_sub_step_that_raises_cancelled_error() -> None:
    """A sub-step's own CancelledError is an error to collect; only the step's own cancellation stops the step."""

    async def substep(config, _context):
        if config["n"] == 0:
            raise asyncio.CancelledError()
        return {"text": "ok"}

    config = {"steps": [{"type": "summarize", "config": {"n": n}} for n in range(3)]}
    with patch.object(flow, "get_adapter", return_value=substep):
        result = await flow.run_parallel_adapter(config, {})

    assert result["results"] == [{"error": "parallel_step_error"}, {"text": "ok"}, {"text": "ok"}]


async def test_a_map_step_that_times_out_leaves_no_item_running() -> None:
    """The engine's step timeout cancels the step, and every item stops before it raises (dev's gather did this too)."""
    log: list[tuple[str, int]] = []
    release = asyncio.Event()  # never set: the items run until they are cancelled
    config = {"items": [1, 2, 3, 4], "step": {"type": "summarize", "config": {}}, "concurrency": 2}

    with patch.object(flow, "get_adapter", return_value=_recording_substep(log, _refusal(), release)):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(flow.run_map_adapter(config, {}), timeout=0.05)
        log.append(("step_raised", -1))
        for _ in range(20):
            await asyncio.sleep(0)

    assert _ran_after_the_step_raised(log) == []
    assert sorted(n for kind, n in log if kind == "cancelled") == [1, 2]


async def test_a_cancelled_media_ingest_terminates_its_download(tmp_path, monkeypatch) -> None:
    """Cancelled mid-poll, because a sibling failed or the step timed out, the step stops yt-dlp; on dev nothing did."""
    monkeypatch.setenv("WORKFLOWS_TMP", str(tmp_path))
    started = asyncio.Event()
    download = SubprocessTask(
        cmd=["yt-dlp"], workdir=tmp_path, stdout_path=tmp_path / "stdout.log", stderr_path=tmp_path / "stderr.log",
        pid=4242, pgid=4242,
    )

    def fake_start_process(_cmd, workdir, log_dir):
        started.set()
        return download

    terminate = MagicMock(return_value=(True, False))
    config = {"sources": [{"uri": "https://media.example.com/clip"}], "timeout_seconds": 60}
    with patch.object(ingest, "is_test_mode", return_value=False), patch.object(
        ingest, "is_url_allowed_for_tenant", return_value=True
    ), patch.object(ingest, "start_process", side_effect=fake_start_process), patch.object(
        ingest, "terminate_process", terminate
    ):
        step = asyncio.create_task(ingest.run_media_ingest_adapter(config, {"tenant_id": "default"}))
        await started.wait()  # the adapter is now in its poll's sleep
        step.cancel()
        with pytest.raises(asyncio.CancelledError):
            await step

    assert terminate.call_args_list == [call(download)]


async def test_a_map_item_that_raises_runs_once_and_fails_the_step() -> None:
    """On dev the tracing fallback ran the item again before the step failed: calls == [0, 0] (TASK-13886)."""
    calls: list[int] = []
    error = RuntimeError("webhook failed")

    async def substep(_config, context):
        calls.append(context["item"])
        raise error

    with patch.object(flow, "get_adapter", return_value=substep):
        with pytest.raises(RuntimeError) as raised:
            await flow.run_map_adapter({"items": [0], "step": {"type": "summarize", "config": {}}}, {})

    assert raised.value is error
    assert calls == [0]


async def test_a_map_item_still_runs_once_when_its_tracing_span_fails() -> None:
    """A span that fails to open costs the item its trace, not its run (TASK-13886)."""
    calls: list[int] = []

    async def substep(_config, context):
        calls.append(context["item"])
        return {"text": str(context["item"])}

    @contextlib.asynccontextmanager
    async def broken_span(*_args, **_kwargs):
        raise RuntimeError("tracer unavailable")
        yield  # pragma: no cover - never reached

    with patch.object(flow, "get_adapter", return_value=substep), patch.object(flow, "_start_span", broken_span):
        result = await flow.run_map_adapter({"items": [0], "step": {"type": "summarize", "config": {}}}, {})

    assert result == {"results": [{"text": "0"}], "count": 1}
    assert calls == [0]
