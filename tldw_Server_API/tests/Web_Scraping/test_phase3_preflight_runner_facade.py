"""Deterministic runner and typed facade tests for Phase 3 Task 11."""

from __future__ import annotations

import asyncio
import gc
import importlib
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, cast

import pytest
from loguru import logger

from tldw_Server_API.app.core.Web_Scraping.contracts import (
    PreflightAdvice,
    PreflightResult,
    RuntimeFailure,
    WebScrapingStatus,
)
from tldw_Server_API.app.core.Web_Scraping.preflight import facade
from tldw_Server_API.app.core.Web_Scraping.preflight.context import (
    PreflightDeadlineExceeded,
    PreflightExecutionContext,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.options import PreflightOptions
from tldw_Server_API.app.core.Web_Scraping.preflight.probes import ProbeTimeout
from tldw_Server_API.app.core.Web_Scraping.preflight.target import PreflightTarget
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    PolicyDecision,
    RuntimeRequestContext,
)

pytestmark = pytest.mark.unit

ANALYZER_KEYS = [
    "robots",
    "tls",
    "js",
    "behavioral",
    "captcha",
    "fingerprint",
    "integrity",
    "rate_limit",
    "waf",
]

ANALYZER_MESSAGES = {
    "robots": "Robots.txt check failed.",
    "tls": "TLS fingerprint analysis failed.",
    "js": "JavaScript rendering analysis failed.",
    "behavioral": "Honeypot detection failed.",
    "captcha": "Captcha detection failed.",
    "fingerprint": "Fingerprint analysis failed.",
    "integrity": "Function integrity analysis failed.",
    "rate_limit": "Rate limit profiling failed.",
    "waf": "WAF detection failed.",
}


def _runner() -> Any:
    return importlib.import_module("tldw_Server_API.app.core.Web_Scraping.preflight.runner")


def _decision(*, allowed: bool = True) -> PolicyDecision:
    return PolicyDecision(
        allowed=allowed,
        mode="compat",
        reason="allowed" if allowed else "robots_disallowed",
        stage="preflight",
        source="task-11",
    )


def _target(*, allowed: bool = True) -> PreflightTarget:
    return PreflightTarget(
        url="https://example.com/path",
        decision=_decision(allowed=allowed),
        request_context=RuntimeRequestContext(
            source="test",
            stage="preflight",
            request_id="task-11",
        ),
    )


class _Controls:
    def __init__(self, remaining_s: float | None) -> None:
        self.remaining_s = remaining_s
        self.calls = 0

    def remaining_seconds(self) -> float | None:
        self.calls += 1
        return self.remaining_s


class _Context:
    def __init__(
        self,
        *,
        remaining_s: float | None = None,
        close_error: BaseException | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.controls = _Controls(remaining_s)
        self.close_error = close_error
        self.close_calls = 0
        self.events = events

    async def close(self) -> None:
        self.close_calls += 1
        if self.events is not None:
            self.events.append("close")
        if self.close_error is not None:
            raise self.close_error


def _context(
    *,
    remaining_s: float | None = None,
    close_error: BaseException | None = None,
    events: list[str] | None = None,
) -> PreflightExecutionContext:
    return cast(
        PreflightExecutionContext,
        _Context(
            remaining_s=remaining_s,
            close_error=close_error,
            events=events,
        ),
    )


def _patch_private_analyzers(
    monkeypatch: pytest.MonkeyPatch,
    runner: Any,
    *,
    events: list[str] | None = None,
) -> None:
    async def result(name: str, *_args: Any) -> dict[str, Any]:
        if events is not None:
            events.append(name)
        payload: dict[str, Any] = {"status": "success", "name": name}
        if name == "robots":
            payload["crawl_delay"] = 2.5
        return payload

    for key, private_name in (
        ("robots", "_check_robots_txt"),
        ("tls", "_analyze_tls_fingerprint"),
        ("js", "_analyze_js_rendering"),
        ("behavioral", "_detect_honeypots"),
        ("captcha", "_detect_captcha"),
        ("fingerprint", "_analyze_fingerprinting"),
        ("integrity", "_analyze_function_integrity"),
        ("rate_limit", "_profile_rate_limits"),
        ("waf", "_detect_waf"),
    ):
        monkeypatch.setattr(
            runner,
            private_name,
            lambda *args, _key=key: result(_key, *args),
        )


@pytest.mark.asyncio
async def test_internal_runner_calls_only_private_analyzers_in_exact_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    context = _context()
    target = _target()
    options = PreflightOptions(
        scan_depth="deep",
        find_all_waf=True,
        impersonate=True,
        external_tools_enabled=True,
    )
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def private(name: str, payload: dict[str, Any]) -> Callable[..., Awaitable[dict[str, Any]]]:
        async def call(*args: Any) -> dict[str, Any]:
            calls.append((name, args))
            return payload

        return call

    monkeypatch.setattr(
        runner,
        "_check_robots_txt",
        private("robots", {"status": "success", "crawl_delay": 2.5}),
    )
    monkeypatch.setattr(
        runner,
        "_analyze_tls_fingerprint",
        private("tls", {"status": "inactive"}),
    )
    monkeypatch.setattr(
        runner,
        "_analyze_js_rendering",
        private("js", {"status": "success"}),
    )
    monkeypatch.setattr(
        runner,
        "_detect_honeypots",
        private("behavioral", {"status": "success"}),
    )
    monkeypatch.setattr(
        runner,
        "_detect_captcha",
        private("captcha", {"status": "success"}),
    )
    monkeypatch.setattr(
        runner,
        "_analyze_fingerprinting",
        private("fingerprint", {"status": "success"}),
    )
    monkeypatch.setattr(
        runner,
        "_analyze_function_integrity",
        private("integrity", {"status": "success"}),
    )
    monkeypatch.setattr(
        runner,
        "_profile_rate_limits",
        private("rate_limit", {"status": "success"}),
    )
    monkeypatch.setattr(
        runner,
        "_detect_waf",
        private("waf", {"status": "success"}),
    )

    for public_name in (
        "check_robots_txt",
        "analyze_tls_fingerprint",
        "analyze_js_rendering",
        "detect_honeypots",
        "detect_captcha",
        "analyze_fingerprinting",
        "analyze_function_integrity",
        "profile_rate_limits",
        "detect_waf",
    ):
        monkeypatch.setattr(
            runner,
            public_name,
            lambda *_args, _name=public_name, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"public analyzer called: {_name}")
            ),
        )

    monkeypatch.setattr(
        runner,
        "calculate_difficulty_score",
        lambda results: calls.append(("score", (results,))) or {"score": 0, "label": "Easy"},
    )
    monkeypatch.setattr(
        runner,
        "generate_recommendations",
        lambda results: calls.append(("recommend", (results,))) or {"tools": [], "strategy": []},
    )

    output = await runner.gather_analysis_with_context(target, options, context)

    assert list(output) == ["results", "score", "recommendations"]
    assert list(output["results"]) == ANALYZER_KEYS
    assert [name for name, _args in calls] == [*ANALYZER_KEYS, "score", "recommend"]
    assert calls[:9] == [
        ("robots", (target.url, context)),
        ("tls", (target.url, context)),
        ("js", (target.url, context)),
        ("behavioral", (target.url, context, "deep")),
        ("captcha", (target.url, context)),
        ("fingerprint", (target.url, context)),
        ("integrity", (target.url, context)),
        ("rate_limit", (target.url, context, 2.5, True)),
        ("waf", (target.url, context, True, True)),
    ]
    assert calls[9][1][0] is output["results"]
    assert calls[10][1][0] is output["results"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "message"), ANALYZER_MESSAGES.items())
async def test_isolated_maps_each_unexpected_failure_to_its_exact_safe_payload(
    name: str,
    message: str,
) -> None:
    runner = _runner()

    def fail_during_setup() -> Awaitable[dict[str, Any]]:
        raise RuntimeError("credential-bearing analyzer setup failure")

    result = await runner._isolated(name, fail_during_setup)

    assert result == {
        "status": "error",
        "message": message,
        "error_code": "analyzer_error",
    }


@pytest.mark.asyncio
async def test_analyzer_setup_failure_emits_no_unawaited_coroutine_warning() -> None:
    runner = _runner()

    def fail_before_coroutine_creation() -> Awaitable[dict[str, Any]]:
        raise RuntimeError("setup failed")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = await runner._isolated("js", fail_before_coroutine_creation)
        gc.collect()

    assert result["error_code"] == "analyzer_error"


@pytest.mark.asyncio
async def test_isolated_normalizes_probe_error_without_stopping_later_analyzers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    _patch_private_analyzers(monkeypatch, runner, events=events)

    async def local_timeout(*_args: Any) -> dict[str, Any]:
        events.append("js")
        raise ProbeTimeout

    monkeypatch.setattr(runner, "_analyze_js_rendering", local_timeout)

    result = await runner.gather_analysis_with_context(
        _target(),
        PreflightOptions(),
        _context(),
    )

    assert events == ANALYZER_KEYS
    assert result["results"]["js"] == {
        "status": "error",
        "message": "Probe timed out.",
        "error_code": "timeout",
    }


@pytest.mark.asyncio
async def test_unexpected_analyzer_failure_is_isolated_and_remaining_analyzers_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    _patch_private_analyzers(monkeypatch, runner, events=events)

    def fail_during_setup(*_args: Any) -> Awaitable[dict[str, Any]]:
        events.append("js")
        raise RuntimeError("secret analyzer failure")

    monkeypatch.setattr(runner, "_analyze_js_rendering", fail_during_setup)

    result = await runner.gather_analysis_with_context(
        _target(),
        PreflightOptions(),
        _context(),
    )

    assert events == ANALYZER_KEYS
    assert result["results"]["js"] == {
        "status": "error",
        "message": "JavaScript rendering analysis failed.",
        "error_code": "analyzer_error",
    }


@pytest.mark.asyncio
async def test_unexpected_analyzer_failure_logs_only_sanitized_context_and_exception_class() -> None:
    runner = _runner()
    records: list[str] = []

    def fail_during_setup() -> Awaitable[dict[str, Any]]:
        raise RuntimeError("https://user:password@example.com/path?token=secret-token")

    sink_id = logger.add(
        lambda message: records.append(str(message.record["message"])),
        level="WARNING",
    )
    try:
        result = await runner._isolated("js", fail_during_setup)
    finally:
        logger.remove(sink_id)

    assert result == {
        "status": "error",
        "message": "JavaScript rendering analysis failed.",
        "error_code": "analyzer_error",
    }
    assert records == ["Preflight analyzer failure: analyzer=js exception=RuntimeError"]


@pytest.mark.asyncio
async def test_legacy_policy_failure_logs_only_sanitized_context_and_exception_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    records: list[str] = []
    sink_id = logger.add(
        lambda message: records.append(str(message.record["message"])),
        level="WARNING",
    )
    monkeypatch.setattr(
        runner,
        "_default_policy_checker",
        lambda: _PolicyChecker(error=RuntimeError("https://user:password@example.com/path?token=secret-token")),
    )
    try:
        result = await runner.gather_analysis("https://example.com/path")
    finally:
        logger.remove(sink_id)

    assert result["results"] == {
        key: {
            "status": "error",
            "message": "Probe destination was denied.",
            "error_code": "policy_error",
        }
        for key in ANALYZER_KEYS
    }
    assert records == ["Legacy preflight policy failure: exception=RuntimeError"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [asyncio.CancelledError(), PreflightDeadlineExceeded()],
    ids=["cancelled", "deadline"],
)
async def test_cancellation_and_deadline_stop_the_runner_immediately(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    runner = _runner()
    events: list[str] = []
    _patch_private_analyzers(monkeypatch, runner, events=events)

    async def fail(*_args: Any) -> dict[str, Any]:
        events.append("js")
        raise failure

    monkeypatch.setattr(runner, "_analyze_js_rendering", fail)

    with pytest.raises(type(failure)) as caught:
        await runner.gather_analysis_with_context(
            _target(),
            PreflightOptions(),
            _context(),
        )

    assert caught.value is failure
    assert events == ["robots", "tls", "js"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["score", "recommendations"])
async def test_scoring_and_recommendation_failures_are_overall_failures(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    runner = _runner()
    _patch_private_analyzers(monkeypatch, runner)
    failure = RuntimeError(f"{stage} failed")
    if stage == "score":
        monkeypatch.setattr(
            runner,
            "calculate_difficulty_score",
            lambda _results: (_ for _ in ()).throw(failure),
        )
    else:
        monkeypatch.setattr(
            runner,
            "generate_recommendations",
            lambda _results: (_ for _ in ()).throw(failure),
        )

    with pytest.raises(RuntimeError) as caught:
        await runner.gather_analysis_with_context(
            _target(),
            PreflightOptions(),
            _context(),
        )

    assert caught.value is failure


@pytest.mark.asyncio
async def test_run_preflight_disabled_and_denied_do_not_start_or_close_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    context = _context()
    monkeypatch.setattr(
        runner,
        "gather_analysis_with_context",
        lambda *_args: (_ for _ in ()).throw(AssertionError("runner started")),
    )

    assert await facade.run_preflight(_target(), PreflightOptions(enabled=False), context) is None
    assert cast(_Context, context).close_calls == 0

    with pytest.raises(ValueError, match="^run_preflight requires an allowed target$"):
        await facade.run_preflight(
            _target(allowed=False),
            PreflightOptions(enabled=True),
            context,
        )
    assert cast(_Context, context).close_calls == 0


@pytest.mark.asyncio
async def test_run_preflight_success_uses_analysis_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    analysis = {
        "results": {
            "js": {"status": "success", "js_required": False, "is_spa": True},
            "tls": {"status": "active"},
        },
        "score": {"score": 2, "label": "Moderate"},
        "recommendations": {"tools": [], "strategy": []},
    }
    context = _context()
    monkeypatch.setattr(
        runner,
        "gather_analysis_with_context",
        lambda *_args: asyncio.sleep(0, result=analysis),
    )

    result = await facade.run_preflight(
        _target(),
        PreflightOptions(enabled=True, timeout_s=0.0001),
        context,
    )

    assert result == PreflightResult(
        analysis=analysis,
        advice=PreflightAdvice(
            backend="curl",
            method="playwright",
            notes=("js_required", "tls_active"),
        ),
    )
    assert cast(_Context, context).controls.calls == 1
    assert cast(_Context, context).close_calls == 1


@pytest.mark.asyncio
async def test_run_preflight_uses_context_deadline_not_options_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    context = _context(remaining_s=None)

    async def delayed_success(*_args: Any) -> dict[str, Any]:
        await asyncio.sleep(0.02)
        return {"results": {}, "score": {}, "recommendations": {}}

    monkeypatch.setattr(runner, "gather_analysis_with_context", delayed_success)

    result = await facade.run_preflight(
        _target(),
        PreflightOptions(enabled=True, timeout_s=0.001),
        context,
    )

    assert result is not None
    assert result.status is WebScrapingStatus.OK
    assert cast(_Context, context).controls.calls == 1


@pytest.mark.asyncio
async def test_run_preflight_expired_deadline_never_starts_runner_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    context = _context(remaining_s=0.0)
    monkeypatch.setattr(
        runner,
        "gather_analysis_with_context",
        lambda *_args: (_ for _ in ()).throw(AssertionError("runner started")),
    )

    result = await facade.run_preflight(
        _target(),
        PreflightOptions(enabled=True),
        context,
    )

    assert result == PreflightResult(
        status=WebScrapingStatus.TIMEOUT,
        failure=RuntimeFailure(
            status=WebScrapingStatus.TIMEOUT,
            public_message="Preflight analysis timed out.",
        ),
    )
    assert cast(_Context, context).close_calls == 1


@pytest.mark.asyncio
async def test_run_preflight_timeout_retires_runner_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    started = asyncio.Event()
    context = _context(remaining_s=0.01, events=events)

    async def wait_forever(*_args: Any) -> dict[str, Any]:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            events.append("runner-retired")

    monkeypatch.setattr(runner, "gather_analysis_with_context", wait_forever)

    result = await facade.run_preflight(
        _target(),
        PreflightOptions(enabled=True),
        context,
    )

    assert started.is_set()
    assert result is not None
    assert result.status is WebScrapingStatus.TIMEOUT
    assert result.failure == RuntimeFailure(
        status=WebScrapingStatus.TIMEOUT,
        public_message="Preflight analysis timed out.",
    )
    assert events == ["runner-retired", "close"]


@pytest.mark.asyncio
async def test_expired_deadline_stays_timeout_when_runner_suppresses_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    context = _context(remaining_s=0.01, events=events)

    async def suppress_cancellation(*_args: Any) -> dict[str, Any]:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("runner-retired")
            return {"results": {}, "score": {}, "recommendations": {}}

    monkeypatch.setattr(
        runner,
        "gather_analysis_with_context",
        suppress_cancellation,
    )

    result = await facade.run_preflight(
        _target(),
        PreflightOptions(enabled=True),
        context,
    )

    assert result is not None
    assert result.status is WebScrapingStatus.TIMEOUT
    assert events == ["runner-retired", "close"]


@pytest.mark.asyncio
async def test_deadline_timeout_survives_retired_child_failure_without_unhandled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    started = asyncio.Event()
    child_tasks: list[asyncio.Task[Any]] = []
    unhandled: list[dict[str, Any]] = []
    context = _context(remaining_s=0.01, events=events)

    async def fail_after_cancellation(*_args: Any) -> dict[str, Any]:
        child = asyncio.current_task()
        assert child is not None
        child_tasks.append(child)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("runner-cancelled")
            events.append("runner-failed")
            raise RuntimeError("retired child failed") from None

    monkeypatch.setattr(
        runner,
        "gather_analysis_with_context",
        fail_after_cancellation,
    )
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, details: unhandled.append(details))
    try:
        result = await facade.run_preflight(
            _target(),
            PreflightOptions(enabled=True),
            context,
        )
        assert started.is_set()
        assert len(child_tasks) == 1
        child = child_tasks.pop()
        assert child.done()
        del child
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert not any(task.get_name() == "preflight-runner" for task in asyncio.all_tasks())
    assert result is not None
    assert events == ["runner-cancelled", "runner-failed", "close"]
    assert cast(_Context, context).close_calls == 1
    problems: list[str] = []
    if result.status is not WebScrapingStatus.TIMEOUT:
        problems.append(f"run_preflight returned {result.status.name} instead of TIMEOUT")
    if unhandled:
        problems.append(f"unhandled loop exceptions: {[item.get('message') for item in unhandled]}")
    if problems:
        pytest.fail("; ".join(problems))


@pytest.mark.asyncio
async def test_no_deadline_caller_cancellation_survives_runner_suppression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    started = asyncio.Event()
    child_tasks: list[asyncio.Task[Any]] = []
    context = _context(events=events)

    async def suppress_cancellation(*_args: Any) -> dict[str, Any]:
        child = asyncio.current_task()
        assert child is not None
        child_tasks.append(child)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("runner-cancelled")
        events.append("runner-returned")
        return {"results": {}, "score": {}, "recommendations": {}}

    monkeypatch.setattr(
        runner,
        "gather_analysis_with_context",
        suppress_cancellation,
    )
    task = asyncio.create_task(
        facade.run_preflight(
            _target(),
            PreflightOptions(enabled=True),
            context,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel("caller cancelled")

    try:
        result = await task
    except asyncio.CancelledError as cancellation:
        assert cancellation.args == ("caller cancelled",)
    else:
        pytest.fail(f"run_preflight returned {result.status.name} instead of raising caller cancellation")

    assert len(child_tasks) == 1
    assert child_tasks[0].done()
    assert events == ["runner-cancelled", "runner-returned", "close"]
    assert cast(_Context, context).close_calls == 1


@pytest.mark.asyncio
async def test_no_deadline_cancellation_survives_retired_child_failure_without_unhandled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    started = asyncio.Event()
    child_tasks: list[asyncio.Task[Any]] = []
    unhandled: list[dict[str, Any]] = []
    context = _context(events=events)

    async def fail_after_cancellation(*_args: Any) -> dict[str, Any]:
        child = asyncio.current_task()
        assert child is not None
        child_tasks.append(child)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("runner-cancelled")
            events.append("runner-failed")
            raise RuntimeError("retired child failed") from None

    monkeypatch.setattr(
        runner,
        "gather_analysis_with_context",
        fail_after_cancellation,
    )
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, details: unhandled.append(details))
    cancellation: asyncio.CancelledError | None = None
    result: PreflightResult | None = None
    try:
        task = asyncio.create_task(
            facade.run_preflight(
                _target(),
                PreflightOptions(enabled=True),
                context,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel("caller cancelled")
        try:
            result = await task
        except asyncio.CancelledError as exc:
            cancellation = exc

        assert len(child_tasks) == 1
        child = child_tasks.pop()
        assert child.done()
        del child
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert not any(task.get_name() == "preflight-runner" for task in asyncio.all_tasks())
    assert events == ["runner-cancelled", "runner-failed", "close"]
    assert cast(_Context, context).close_calls == 1
    problems: list[str] = []
    if cancellation is None:
        assert result is not None
        problems.append(f"run_preflight returned {result.status.name} instead of raising caller cancellation")
    if unhandled:
        problems.append(f"unhandled loop exceptions: {[item.get('message') for item in unhandled]}")
    if problems:
        pytest.fail("; ".join(problems))
    assert cancellation is not None
    assert cancellation.args == ("caller cancelled",)


@pytest.mark.asyncio
async def test_run_preflight_cancellation_retires_runner_and_cleans_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    started = asyncio.Event()
    context = _context(events=events)

    async def wait_forever(*_args: Any) -> dict[str, Any]:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            events.append("runner-retired")

    monkeypatch.setattr(runner, "gather_analysis_with_context", wait_forever)
    task = asyncio.create_task(
        facade.run_preflight(
            _target(),
            PreflightOptions(enabled=True),
            context,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert events == ["runner-retired", "close"]
    assert cast(_Context, context).close_calls == 1


@pytest.mark.asyncio
async def test_observed_caller_cancellation_wins_deadline_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    first_cancellation = asyncio.Event()
    retired = asyncio.Event()
    context = _context(remaining_s=0.01)

    async def suppress_deadline_cancellation(*_args: Any) -> dict[str, Any]:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            first_cancellation.set()
            try:
                await asyncio.Event().wait()
            finally:
                retired.set()
        raise AssertionError("unreachable")

    monkeypatch.setattr(
        runner,
        "gather_analysis_with_context",
        suppress_deadline_cancellation,
    )
    task = asyncio.create_task(
        facade.run_preflight(
            _target(),
            PreflightOptions(enabled=True),
            context,
        )
    )
    await asyncio.wait_for(first_cancellation.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert retired.is_set()
    assert cast(_Context, context).close_calls == 1


@pytest.mark.asyncio
async def test_runner_failure_maps_to_exact_safe_overall_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    context = _context()

    async def fail(*_args: Any) -> dict[str, Any]:
        raise RuntimeError("secret scoring failure")

    monkeypatch.setattr(runner, "gather_analysis_with_context", fail)

    result = await facade.run_preflight(
        _target(),
        PreflightOptions(enabled=True),
        context,
    )

    assert result == PreflightResult(
        status=WebScrapingStatus.ERROR,
        failure=RuntimeFailure(
            status=WebScrapingStatus.ERROR,
            public_message="Preflight analysis failed.",
        ),
    )
    assert cast(_Context, context).close_calls == 1


@pytest.mark.asyncio
async def test_runner_timeout_error_without_deadline_expiry_is_overall_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    context = _context(remaining_s=10.0)

    async def fail(*_args: Any) -> dict[str, Any]:
        raise TimeoutError("scoring timed out internally")

    monkeypatch.setattr(runner, "gather_analysis_with_context", fail)

    result = await facade.run_preflight(
        _target(),
        PreflightOptions(enabled=True),
        context,
    )

    assert result == PreflightResult(
        status=WebScrapingStatus.ERROR,
        failure=RuntimeFailure(
            status=WebScrapingStatus.ERROR,
            public_message="Preflight analysis failed.",
        ),
    )
    assert cast(_Context, context).close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "timeout", "error"])
async def test_cleanup_failure_never_replaces_established_non_cancel_outcome(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    runner = _runner()
    remaining_s = 0.0 if outcome == "timeout" else None
    context = _context(
        remaining_s=remaining_s,
        close_error=RuntimeError("secret cleanup failure"),
    )

    async def run(*_args: Any) -> dict[str, Any]:
        if outcome == "error":
            raise RuntimeError("runner failed")
        return {"results": {}, "score": {}, "recommendations": {}}

    monkeypatch.setattr(runner, "gather_analysis_with_context", run)

    result = await facade.run_preflight(
        _target(),
        PreflightOptions(enabled=True),
        context,
    )

    assert result is not None
    assert (
        result.status
        is {
            "success": WebScrapingStatus.OK,
            "timeout": WebScrapingStatus.TIMEOUT,
            "error": WebScrapingStatus.ERROR,
        }[outcome]
    )
    assert cast(_Context, context).close_calls == 1


@pytest.mark.asyncio
async def test_cleanup_cancellation_propagates_over_success() -> None:
    runner = _runner()
    context = _context(close_error=asyncio.CancelledError())

    async def run(*_args: Any) -> dict[str, Any]:
        return {"results": {}, "score": {}, "recommendations": {}}

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(runner, "gather_analysis_with_context", run)
        with pytest.raises(asyncio.CancelledError):
            await facade.run_preflight(
                _target(),
                PreflightOptions(enabled=True),
                context,
            )


@pytest.mark.parametrize(
    ("analysis", "method", "backend_setting", "expected_backend", "expected_method", "notes"),
    [
        (
            {
                "results": {
                    "js": {"status": "success", "js_required": True},
                    "tls": {"status": "active"},
                }
            },
            "auto",
            "auto",
            "curl",
            "playwright",
            ("js_required", "tls_active"),
        ),
        (
            {"results": {"js": {"status": "success", "is_spa": True}}},
            "auto",
            "httpx",
            "httpx",
            "playwright",
            ("js_required",),
        ),
        (
            {"results": {"js": {"status": "error", "js_required": True}}},
            "auto",
            "auto",
            "httpx",
            "auto",
            (),
        ),
        (
            {"results": {"tls": {"status": "active"}}},
            "beautifulsoup",
            "auto",
            "curl",
            "beautifulsoup",
            ("tls_active",),
        ),
        (
            {"results": {"tls": {"status": "error"}}},
            "auto",
            "auto",
            "httpx",
            "auto",
            (),
        ),
        (
            {"results": {"tls": {"status": "active"}}},
            "auto",
            "curl",
            "httpx",
            "auto",
            (),
        ),
    ],
)
def test_apply_preflight_advice_routes_only_from_explicit_success_signals(
    analysis: dict[str, Any],
    method: str,
    backend_setting: str,
    expected_backend: str,
    expected_method: str,
    notes: tuple[str, ...],
) -> None:
    failure = RuntimeFailure(
        status=WebScrapingStatus.OK,
        public_message="retained test failure metadata",
    )
    original = PreflightResult(
        analysis=analysis,
        advice=PreflightAdvice(backend="old", method="old", notes=("old",)),
        failure=failure,
    )

    backend, final_method, updated = facade.apply_preflight_advice(
        original,
        backend="httpx",
        method=method,
        backend_setting=backend_setting,
    )

    assert backend == expected_backend
    assert final_method == expected_method
    assert updated == replace(
        original,
        advice=PreflightAdvice(
            backend=expected_backend,
            method=expected_method,
            notes=notes,
        ),
    )
    assert updated is not None
    assert updated.analysis == original.analysis
    assert updated.status is original.status
    assert updated.failure is failure


def test_apply_preflight_advice_preserves_inputs_for_none_and_non_ok_result() -> None:
    assert facade.apply_preflight_advice(
        None,
        backend="httpx",
        method="auto",
        backend_setting="auto",
    ) == ("httpx", "auto", None)

    failure = RuntimeFailure(
        status=WebScrapingStatus.TIMEOUT,
        public_message="Preflight analysis timed out.",
    )
    original = PreflightResult(
        analysis={
            "results": {
                "js": {"status": "success", "js_required": True},
                "tls": {"status": "active"},
            }
        },
        status=WebScrapingStatus.TIMEOUT,
        failure=failure,
    )

    backend, method, updated = facade.apply_preflight_advice(
        original,
        backend="httpx",
        method="auto",
        backend_setting="auto",
    )

    assert (backend, method) == ("httpx", "auto")
    assert updated == replace(
        original,
        advice=PreflightAdvice(backend="httpx", method="auto"),
    )


def test_public_preflight_payload_is_the_only_status_and_inclusion_gate() -> None:
    success = PreflightResult(
        analysis={"results": {}, "score": {}, "recommendations": {}},
        advice=PreflightAdvice(backend="curl", method="playwright", notes=("tls_active",)),
    )
    failure = PreflightResult(
        status=WebScrapingStatus.ERROR,
        failure=RuntimeFailure(
            status=WebScrapingStatus.ERROR,
            public_message="Preflight analysis failed.",
        ),
    )

    assert facade.public_preflight_payload(success, True) == {
        "analysis": {"results": {}, "score": {}, "recommendations": {}},
        "advice": {
            "backend": "curl",
            "method": "playwright",
            "notes": ["tls_active"],
        },
    }
    assert facade.public_preflight_payload(success, False) is None
    assert facade.public_preflight_payload(None, True) is None
    assert facade.public_preflight_payload(failure, True) is None


class _PolicyChecker:
    def __init__(
        self,
        *,
        decision: PolicyDecision | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.decision = decision or _decision()
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def decide(self, url: str, **kwargs: Any) -> PolicyDecision:
        self.calls.append((url, dict(kwargs)))
        if self.error is not None:
            raise self.error
        return self.decision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("checker", "error_code"),
    [
        (_PolicyChecker(decision=_decision(allowed=False)), "policy_denied"),
        (_PolicyChecker(error=RuntimeError("secret policy failure")), "policy_error"),
    ],
    ids=["denied", "checker-error"],
)
async def test_legacy_gather_returns_all_safe_policy_entries_without_context_or_probes(
    monkeypatch: pytest.MonkeyPatch,
    checker: _PolicyChecker,
    error_code: str,
) -> None:
    runner = _runner()
    monkeypatch.setattr(runner, "_default_policy_checker", lambda: checker)
    monkeypatch.setattr(
        runner,
        "build_execution_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("context constructed")),
    )
    for private_name in (
        "_check_robots_txt",
        "_analyze_tls_fingerprint",
        "_analyze_js_rendering",
        "_detect_honeypots",
        "_detect_captcha",
        "_analyze_fingerprinting",
        "_analyze_function_integrity",
        "_profile_rate_limits",
        "_detect_waf",
    ):
        monkeypatch.setattr(
            runner,
            private_name,
            lambda *_args, _name=private_name: (_ for _ in ()).throw(AssertionError(f"probe ran: {_name}")),
        )

    result = await runner.gather_analysis("https://example.com/path")

    assert list(result) == ["results", "score", "recommendations"]
    assert list(result["results"]) == ANALYZER_KEYS
    assert result["results"] == {
        key: {
            "status": "error",
            "message": "Probe destination was denied.",
            "error_code": error_code,
        }
        for key in ANALYZER_KEYS
    }
    assert len(checker.calls) == 1
    assert checker.calls[0][1]["respect_robots"] is False


@pytest.mark.asyncio
async def test_legacy_gather_allowed_target_uses_one_context_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    checker = _PolicyChecker()
    context = _context()
    context_calls: list[tuple[Any, ...]] = []
    _patch_private_analyzers(monkeypatch, runner)
    monkeypatch.setattr(runner, "_default_policy_checker", lambda: checker)

    def build(*args: Any, **kwargs: Any) -> PreflightExecutionContext:
        context_calls.append((args, kwargs))
        return context

    monkeypatch.setattr(runner, "build_execution_context", build)

    result = await runner.gather_analysis(
        "https://example.com/path",
        find_all=True,
        impersonate=True,
        scan_depth="deep",
    )

    assert list(result["results"]) == ANALYZER_KEYS
    assert len(context_calls) == 1
    built_target, options = context_calls[0][0]
    assert built_target.url == "https://example.com/path"
    assert options == PreflightOptions(
        enabled=True,
        scan_depth="deep",
        find_all_waf=True,
        impersonate=True,
    )
    assert context_calls[0][1] == {"policy_checker": checker}
    assert cast(_Context, context).close_calls == 1


@pytest.mark.asyncio
async def test_legacy_gather_converts_deadline_to_timeout_aggregate_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    checker = _PolicyChecker()
    context = _context()

    async def deadline(*_args: Any, **_kwargs: Any) -> Any:
        raise PreflightDeadlineExceeded()

    monkeypatch.setattr(runner, "_default_policy_checker", lambda: checker)
    monkeypatch.setattr(runner, "build_execution_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(runner, "gather_analysis_with_context", deadline)

    result = await runner.gather_analysis("https://example.com/path")

    assert result["results"] == {
        key: {
            "status": "error",
            "message": "Probe timed out.",
            "error_code": "timeout",
        }
        for key in ANALYZER_KEYS
    }
    assert cast(_Context, context).close_calls == 1


def test_sync_legacy_gather_converts_deadline_to_timeout_aggregate_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    checker = _PolicyChecker()
    context = _context()

    async def deadline(*_args: Any, **_kwargs: Any) -> Any:
        raise PreflightDeadlineExceeded()

    monkeypatch.setattr(runner, "_default_policy_checker", lambda: checker)
    monkeypatch.setattr(runner, "build_execution_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(runner, "gather_analysis_with_context", deadline)

    result = runner.run_analysis("https://example.com/path")

    assert result["results"] == {
        key: {
            "status": "error",
            "message": "Probe timed out.",
            "error_code": "timeout",
        }
        for key in ANALYZER_KEYS
    }
    assert cast(_Context, context).close_calls == 1


@pytest.mark.asyncio
async def test_run_analysis_rejects_active_loop_before_coroutine_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    gather_calls = 0

    async def gather(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal gather_calls
        gather_calls += 1
        return {"results": {}, "score": {}, "recommendations": {}}

    monkeypatch.setattr(runner, "gather_analysis", gather)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(
            RuntimeError,
            match=(
                "^run_analysis cannot be used inside an active event loop; " "use 'await gather_analysis' instead\\.$"
            ),
        ):
            runner.run_analysis("https://example.com")
        gc.collect()

    assert gather_calls == 0
