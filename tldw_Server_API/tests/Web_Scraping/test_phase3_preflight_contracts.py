from __future__ import annotations

import asyncio
import math
from dataclasses import fields
from types import MappingProxyType

import pytest
from hypothesis import given
from hypothesis import strategies as st

import tldw_Server_API.app.core.Web_Scraping.preflight.context as context_module
import tldw_Server_API.app.core.Web_Scraping.preflight.options as options_module
from tldw_Server_API.app.core.Web_Scraping.contracts import (
    PreflightAdvice as SharedPreflightAdvice,
)
from tldw_Server_API.app.core.Web_Scraping.contracts import (
    PreflightResult as SharedPreflightResult,
)
from tldw_Server_API.app.core.Web_Scraping.contracts import RuntimeFailure as SharedRuntimeFailure
from tldw_Server_API.app.core.Web_Scraping.contracts import (
    WebScrapingStatus as SharedWebScrapingStatus,
)
from tldw_Server_API.app.core.Web_Scraping.preflight import (
    BrowserProbeOptions,
    ExternalToolResult,
    PreflightAdvice,
    PreflightConsumed,
    PreflightDeadlineExceeded,
    PreflightExecutionContext,
    PreflightLimits,
    PreflightOptions,
    PreflightResult,
    PreflightRuntimeControls,
    PreflightTarget,
    ProbeBudgetExhausted,
    ProbeError,
    ProbeHttpRequest,
    ProbeHttpResponse,
    ProbeTimeout,
    ProbeUnavailable,
    RuntimeFailure,
    WebScrapingStatus,
)
from tldw_Server_API.app.core.Web_Scraping.runtime import PolicyDecision, RuntimeRequestContext
from tldw_Server_API.tests.Web_Scraping.preflight_fakes import (
    EventSleep,
    FakeBrowserProbe,
    FakeCleanupHandle,
    FakeClock,
    FakeEgressGuard,
    FakeExternalToolProbe,
    FakeHttpProbe,
    FakeIdentitySelector,
    FakePolicyChecker,
    FakeSleep,
)

_LEGACY_BOOLEAN_KEYS = (
    ("web_scraper_preflight_analyzers", "enabled"),
    ("web_scraper_preflight_find_all_waf", "find_all_waf"),
    ("web_scraper_preflight_impersonate", "impersonate"),
    ("web_scraper_preflight_include_results", "include_results"),
    ("web_scraper_playwright_no_sandbox", "playwright_no_sandbox"),
)
_APPROVED_PROBE_ERROR_PAIRS = (
    ("policy_denied", "Probe destination was denied."),
    ("policy_error", "Probe destination was denied."),
    ("budget_exhausted", "Probe budget exhausted."),
    ("timeout", "Probe timed out."),
    ("unavailable", "Probe capability is unavailable."),
    ("missing_dependency", "Probe dependency is unavailable."),
    ("external_tool_disabled", "External tool probing is disabled."),
    ("redirect_loop", "Redirect loop detected."),
    ("invalid_redirect", "Redirect target is invalid."),
    ("too_many_redirects", "Redirect limit exceeded."),
    ("probe_error", "Probe failed."),
    ("probe_error", "HTTP probe failed."),
)


def _request_context() -> RuntimeRequestContext:
    return RuntimeRequestContext(source="test", stage="preflight")


def _policy_decision() -> PolicyDecision:
    return PolicyDecision(
        allowed=True,
        mode="strict",
        reason="allowed",
        stage="preflight",
        source="test",
    )


def _execution_context(
    controls: PreflightRuntimeControls,
    identity_selector: FakeIdentitySelector,
) -> PreflightExecutionContext:
    return PreflightExecutionContext(
        request_context=controls.request_context,
        policy_checker=FakePolicyChecker(),
        egress_guard=FakeEgressGuard(),
        controls=controls,
        http=FakeHttpProbe(),
        browser=FakeBrowserProbe(),
        external_tools=FakeExternalToolProbe(),
        identity_selector=identity_selector,
    )


async def _complete_close_tasks_with_fallback(
    tasks: set[asyncio.Task[None]],
    cleanup: FakeCleanupHandle,
    *,
    timeout_s: float = 0.35,
) -> bool:
    """Keep a RED failure bounded while releasing deliberately stuck fakes."""
    _, pending = await asyncio.wait(tasks, timeout=timeout_s)
    completed_in_time = not pending
    if pending:
        cleanup.release_close()
        cleanup.release_force_close()
        await asyncio.gather(*tasks, return_exceptions=True)
    return completed_in_time


@pytest.mark.unit
def test_preflight_package_reuses_shared_result_contracts() -> None:
    assert PreflightAdvice is SharedPreflightAdvice
    assert PreflightResult is SharedPreflightResult
    assert RuntimeFailure is SharedRuntimeFailure
    assert WebScrapingStatus is SharedWebScrapingStatus


@pytest.mark.unit
def test_options_preserve_legacy_defaults() -> None:
    options = PreflightOptions.from_mapping({})

    assert options == PreflightOptions(
        enabled=False,
        timeout_s=None,
        scan_depth="default",
        find_all_waf=False,
        impersonate=False,
        include_results=False,
        external_tools_enabled=None,
        playwright_no_sandbox=False,
    )


@pytest.mark.unit
def test_options_have_exact_public_fields() -> None:
    assert [item.name for item in fields(PreflightOptions)] == [
        "enabled",
        "timeout_s",
        "scan_depth",
        "find_all_waf",
        "impersonate",
        "include_results",
        "external_tools_enabled",
        "playwright_no_sandbox",
    ]


@pytest.mark.unit
@pytest.mark.parametrize(("key", "attribute"), _LEGACY_BOOLEAN_KEYS)
@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), (1, True), (0, False), (" YES ", True), ("off", False)],
)
def test_legacy_boolean_options_normalize_valid_values(
    key: str,
    attribute: str,
    value: object,
    expected: bool,
) -> None:
    options = PreflightOptions.from_mapping({key: value})

    assert getattr(options, attribute) is expected


@pytest.mark.property
@pytest.mark.parametrize(("key", "attribute"), _LEGACY_BOOLEAN_KEYS)
@given(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
    )
)
def test_legacy_boolean_options_preserve_boolean_invariant(
    key: str,
    attribute: str,
    value: object,
) -> None:
    options = PreflightOptions.from_mapping({key: value})

    assert isinstance(getattr(options, attribute), bool)


@pytest.mark.unit
@pytest.mark.parametrize("value", [object(), [], {}])
@pytest.mark.parametrize(("key", "attribute"), _LEGACY_BOOLEAN_KEYS)
def test_malformed_legacy_boolean_options_use_false(
    key: str,
    attribute: str,
    value: object,
) -> None:
    options = PreflightOptions.from_mapping({key: value})

    assert getattr(options, attribute) is False


@pytest.mark.unit
@pytest.mark.parametrize("value", ["default", " THOROUGH ", "Deep"])
def test_scan_depth_normalizes_approved_values(value: str) -> None:
    options = PreflightOptions.from_mapping({"web_scraper_preflight_scan_depth": value})

    assert options.scan_depth == value.strip().lower()


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, "", "wide", 3, object()])
def test_malformed_scan_depth_uses_default(value: object) -> None:
    options = PreflightOptions.from_mapping({"web_scraper_preflight_scan_depth": value})

    assert options.scan_depth == "default"


@pytest.mark.unit
@pytest.mark.parametrize(("value", "expected"), [(2, 2.0), (3.5, 3.5), (" 4.25 ", 4.25), (True, 1.0)])
def test_timeout_normalizes_positive_finite_values(value: object, expected: float) -> None:
    options = PreflightOptions.from_mapping({"web_scraper_preflight_timeout_s": value})

    assert options.timeout_s == expected


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, False, 0, -1, float("nan"), float("inf"), "never", object()])
def test_malformed_or_nonpositive_timeout_is_unbounded(value: object) -> None:
    options = PreflightOptions.from_mapping({"web_scraper_preflight_timeout_s": value})

    assert options.timeout_s is None


@pytest.mark.property
@given(st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True), st.text(), st.booleans()))
def test_timeout_is_none_or_positive_finite(value: object) -> None:
    options = PreflightOptions.from_mapping({"web_scraper_preflight_timeout_s": value})

    assert options.timeout_s is None or (math.isfinite(options.timeout_s) and options.timeout_s > 0)


@pytest.mark.unit
def test_absent_external_tool_config_remains_unspecified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(options_module.logger, "warning", warnings.append)

    options = PreflightOptions.from_mapping({})

    assert options.external_tools_enabled is None
    assert warnings == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), (1, True), (0, False), (" yes ", True), ("OFF", False)],
)
def test_explicit_external_tool_booleans_normalize_without_warning(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
    expected: bool,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(options_module.logger, "warning", warnings.append)

    options = PreflightOptions.from_mapping({"web_scraper_preflight_enable_external_tools": value})

    assert options.external_tools_enabled is expected
    assert warnings == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [None, 2, -1, 0.5, "https://user:password@example.com/?token=secret", object()],
)
def test_malformed_external_tool_config_fails_closed_with_one_sanitized_warning(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(options_module.logger, "warning", warnings.append)

    options = PreflightOptions.from_mapping({"web_scraper_preflight_enable_external_tools": value})

    assert options.external_tools_enabled is False
    assert warnings == ["Invalid preflight external-tool setting; external tools disabled."]
    assert str(value) not in warnings[0]
    assert "http://" not in warnings[0]
    assert "https://" not in warnings[0]


@pytest.mark.unit
def test_target_has_exact_fields_and_normalizes_url() -> None:
    target = PreflightTarget(
        url="  https://example.com/path  ",
        decision=_policy_decision(),
        request_context=_request_context(),
    )

    assert [item.name for item in fields(PreflightTarget)] == [
        "url",
        "decision",
        "request_context",
    ]
    assert target.url == "https://example.com/path"


@pytest.mark.unit
def test_target_rejects_blank_url() -> None:
    with pytest.raises(ValueError, match="url is required"):
        PreflightTarget(url="  ", decision=_policy_decision(), request_context=_request_context())


@pytest.mark.unit
def test_probe_http_contracts_copy_and_freeze_mappings() -> None:
    headers = {"X-Test": "before"}
    cookies = {"session": "before"}
    proxies = {"https": "https://proxy.example"}
    request = ProbeHttpRequest(
        url=" https://example.com ",
        headers=headers,
        cookies=cookies,
        timeout_s=2,
        proxies=proxies,
    )
    response = ProbeHttpResponse(
        url=" https://example.com/final ",
        status=200,
        headers=headers,
        text="body",
    )

    headers["X-Test"] = "after"
    cookies["session"] = "after"
    proxies["https"] = "https://changed.example"

    assert request.url == "https://example.com"
    assert request.timeout_s == 2.0
    assert request.headers == {"X-Test": "before"}
    assert request.cookies == {"session": "before"}
    assert request.proxies == {"https": "https://proxy.example"}
    assert response.url == "https://example.com/final"
    assert response.headers == {"X-Test": "before"}
    assert isinstance(request.headers, MappingProxyType)
    with pytest.raises(TypeError):
        request.headers["X-Test"] = "blocked"


@pytest.mark.unit
@pytest.mark.parametrize("contract", [ProbeHttpRequest, lambda **kwargs: ProbeHttpResponse(status=200, **kwargs)])
def test_probe_http_contracts_reject_blank_urls(contract: object) -> None:
    with pytest.raises(ValueError, match="url is required"):
        contract(url=" ")  # type: ignore[operator]


@pytest.mark.unit
@pytest.mark.parametrize("timeout", [False, 0, -1, float("nan"), float("inf")])
def test_probe_http_request_rejects_invalid_explicit_timeout(timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        ProbeHttpRequest(url="https://example.com", timeout_s=timeout)  # type: ignore[arg-type]


@pytest.mark.unit
def test_browser_options_copy_and_freeze_collections() -> None:
    headers = {"X-Test": "before"}
    resources = ["image", "font"]
    scripts = ["window.ready = true"]
    options = BrowserProbeOptions(
        extra_headers=headers,
        block_resource_types=resources,  # type: ignore[arg-type]
        init_scripts=scripts,  # type: ignore[arg-type]
        capture_requests=True,
    )

    headers["X-Test"] = "after"
    resources.append("media")
    scripts.append("window.changed = true")

    assert options.extra_headers == {"X-Test": "before"}
    assert options.block_resource_types == ("image", "font")
    assert options.init_scripts == ("window.ready = true",)
    assert isinstance(options.extra_headers, MappingProxyType)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"viewport_width": 0},
        {"viewport_width": -1},
        {"viewport_width": True},
        {"viewport_width": 12.5},
        {"viewport_height": 0},
        {"viewport_height": False},
    ],
)
def test_browser_options_reject_invalid_viewport_dimensions(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="viewport"):
        BrowserProbeOptions(**kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
def test_external_tool_result_normalizes_scalar_values() -> None:
    result = ExternalToolResult(returncode="2", stdout=3, stderr=None)  # type: ignore[arg-type]

    assert result == ExternalToolResult(returncode=2, stdout="3", stderr="")


@pytest.mark.unit
@pytest.mark.parametrize(("error_code", "public_message"), _APPROVED_PROBE_ERROR_PAIRS)
def test_probe_error_accepts_approved_stable_public_payload(
    error_code: str,
    public_message: str,
) -> None:
    error = ProbeError(error_code, public_message)

    assert error.error_code == error_code
    assert error.public_message == public_message
    assert str(error) == public_message


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error_code", "public_message"),
    [
        ("probe_error", "Connection refused by 10.0.0.8:8443"),
        ("probe_error", "https://example.com/?token=secret"),
        ("policy_denied", "HTTP probe failed."),
        ("unsupported", "Probe failed."),
    ],
)
def test_probe_error_rejects_unapproved_public_payloads(
    error_code: str,
    public_message: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported probe error payload") as caught:
        ProbeError(error_code, public_message)

    assert "secret" not in str(caught.value)
    assert "10.0.0.8" not in str(caught.value)


@pytest.mark.unit
def test_probe_error_public_fields_are_immutable() -> None:
    error = ProbeError("probe_error", "Probe failed.")

    with pytest.raises(AttributeError):
        error.error_code = "policy_denied"
    with pytest.raises(AttributeError):
        error.public_message = "Probe destination was denied."

    assert (error.error_code, error.public_message) == ("probe_error", "Probe failed.")


@pytest.mark.unit
def test_specialized_probe_errors_have_fixed_safe_defaults() -> None:
    budget = ProbeBudgetExhausted()
    timeout = ProbeTimeout()
    unavailable = ProbeUnavailable()
    missing = ProbeUnavailable(error_code="missing_dependency")

    assert (budget.error_code, budget.public_message) == (
        "budget_exhausted",
        "Probe budget exhausted.",
    )
    assert (timeout.error_code, timeout.public_message) == ("timeout", "Probe timed out.")
    assert (unavailable.error_code, unavailable.public_message) == (
        "unavailable",
        "Probe capability is unavailable.",
    )
    assert (missing.error_code, missing.public_message) == (
        "missing_dependency",
        "Probe dependency is unavailable.",
    )
    with pytest.raises(TypeError):
        ProbeTimeout("https://example.com/?token=secret")  # type: ignore[call-arg]


@pytest.mark.unit
def test_deadline_exception_is_distinct_from_analyzer_probe_timeout() -> None:
    deadline = PreflightDeadlineExceeded()

    assert not isinstance(deadline, ProbeError)
    assert not hasattr(deadline, "error_code")
    assert isinstance(ProbeTimeout(), ProbeError)


@pytest.mark.unit
@pytest.mark.parametrize("field_name", ["requests", "browsers", "active_probes"])
@pytest.mark.parametrize("value", [False, -1, 1.5, "2"])
def test_limits_reject_invalid_values(field_name: str, value: object) -> None:
    with pytest.raises(ValueError, match=field_name):
        PreflightLimits(**{field_name: value})  # type: ignore[arg-type]


@pytest.mark.unit
def test_limits_allow_zero_and_none() -> None:
    assert PreflightLimits(requests=0, browsers=None, active_probes=2) == PreflightLimits(
        requests=0,
        browsers=None,
        active_probes=2,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "field_name"),
    [("request", "requests"), ("browser", "browsers"), ("active_probe", "active_probes")],
)
async def test_budget_reservation_is_atomic(kind: str, field_name: str) -> None:
    controls = PreflightRuntimeControls(
        request_context=_request_context(),
        limits=PreflightLimits(**{field_name: 8}),
        clock=lambda: 10.0,
    )

    outcomes = await asyncio.gather(
        *(controls.reserve(kind) for _ in range(12)),  # type: ignore[arg-type]
        return_exceptions=True,
    )

    assert sum(result is None for result in outcomes) == 8
    assert sum(isinstance(result, ProbeBudgetExhausted) for result in outcomes) == 4
    assert getattr(controls.consumed, field_name) == 8


@pytest.mark.asyncio
async def test_budget_reservation_validates_amount_and_preserves_counter() -> None:
    controls = PreflightRuntimeControls(
        request_context=_request_context(),
        limits=PreflightLimits(requests=3),
    )

    for amount in (True, False, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            await controls.reserve("request", amount)  # type: ignore[arg-type]
    await controls.reserve("request", 3)
    with pytest.raises(ProbeBudgetExhausted):
        await controls.reserve("request")

    assert controls.consumed == PreflightConsumed(requests=3)


@pytest.mark.asyncio
async def test_unbounded_budget_still_tracks_consumption() -> None:
    controls = PreflightRuntimeControls(request_context=_request_context())

    await controls.reserve("request", 4)
    await controls.reserve("browser", 2)
    await controls.reserve("active_probe", 3)

    assert controls.consumed == PreflightConsumed(requests=4, browsers=2, active_probes=3)


@pytest.mark.unit
def test_deadline_uses_injected_monotonic_clock_and_caps_local_timeout() -> None:
    clock = FakeClock(10.0)
    controls = PreflightRuntimeControls(
        request_context=_request_context(),
        deadline=15.0,
        clock=clock,
    )

    assert controls.remaining_seconds() == 5.0
    assert controls.cap_timeout(None) == 5.0
    assert controls.cap_timeout(3.0) == 3.0
    assert controls.deadline_exhausted() is False

    clock.advance(5.0)

    assert controls.remaining_seconds() == 0.0
    assert controls.deadline_exhausted() is True
    with pytest.raises(PreflightDeadlineExceeded):
        controls.cap_timeout(3.0)


@pytest.mark.unit
def test_local_timeout_remains_independent_without_overall_deadline() -> None:
    controls = PreflightRuntimeControls(request_context=_request_context())

    assert controls.remaining_seconds() is None
    assert controls.cap_timeout(None) is None
    assert controls.cap_timeout(2.5) == 2.5
    assert controls.deadline_exhausted() is False


@pytest.mark.asyncio
async def test_zero_delay_sleep_invokes_injected_sleep_once() -> None:
    clock = FakeClock()
    sleep = FakeSleep(clock)
    controls = PreflightRuntimeControls(
        request_context=_request_context(),
        clock=clock,
        sleep=sleep,
    )

    await controls.sleep(0)

    assert sleep.delays == [0]


@pytest.mark.asyncio
async def test_sleep_uses_deadline_cap_and_raises_overall_deadline() -> None:
    clock = FakeClock(10.0)
    sleep = FakeSleep(clock)
    controls = PreflightRuntimeControls(
        request_context=_request_context(),
        deadline=12.0,
        clock=clock,
        sleep=sleep,
    )

    with pytest.raises(PreflightDeadlineExceeded):
        await controls.sleep(5.0)

    assert sleep.delays == [2.0]


@pytest.mark.asyncio
async def test_observed_caller_cancellation_wins_deadline_race() -> None:
    clock = FakeClock(10.0)
    sleep = EventSleep(clock)
    controls = PreflightRuntimeControls(
        request_context=_request_context(),
        deadline=12.0,
        clock=clock,
        sleep=sleep,
    )
    caller = asyncio.create_task(controls.sleep(5.0))
    await sleep.started.wait()

    clock.advance(2.0)
    caller.cancel()

    with pytest.raises(asyncio.CancelledError):
        await caller

    assert sleep.delays == [2.0]
    assert controls.deadline_exhausted() is True


@pytest.mark.unit
def test_execution_context_copies_and_caches_one_browser_identity() -> None:
    selector = FakeIdentitySelector({"User-Agent": "test-agent", "Platform": "desktop"})
    controls = PreflightRuntimeControls(request_context=_request_context())
    context = _execution_context(controls, selector)

    first = context.browser_identity()
    first["User-Agent"] = "mutated"
    second = context.browser_identity()

    assert selector.calls == 1
    assert first is not second
    assert second == {"User-Agent": "test-agent", "Platform": "desktop"}


@pytest.mark.asyncio
async def test_cleanup_stack_closes_gracefully_in_reverse_registration_order() -> None:
    events: list[str] = []
    controls = PreflightRuntimeControls(request_context=_request_context())
    controls.register_cleanup(FakeCleanupHandle(events=events, name="first"))
    controls.register_cleanup(FakeCleanupHandle(events=events, name="second"))

    await controls.close()

    assert events == ["close:second", "close:first"]


@pytest.mark.asyncio
async def test_cleanup_uses_one_shared_grace_for_all_remaining_handles() -> None:
    events: list[str] = []
    first = FakeCleanupHandle(block_close=True, events=events, name="first")
    second = FakeCleanupHandle(block_close=True, events=events, name="second")
    controls = PreflightRuntimeControls(request_context=_request_context())
    controls.register_cleanup(first)
    controls.register_cleanup(second)

    await controls.close(grace_s=0.01)

    assert second.close_calls == 1
    assert first.close_calls == 0
    assert second.force_close_calls == 1
    assert first.force_close_calls == 1
    assert events == ["close:second", "force:second", "force:first"]


@pytest.mark.asyncio
async def test_cleanup_supervisor_bounds_cancellation_suppressing_workers() -> None:
    cleanup = FakeCleanupHandle(
        block_close=True,
        suppress_close_cancellation=True,
        block_force_close=True,
    )
    controls = PreflightRuntimeControls(request_context=_request_context())
    controls.register_cleanup(cleanup)
    task = asyncio.create_task(controls.close(grace_s=0.01))
    started_at = asyncio.get_running_loop().time()

    completed_in_time = await _complete_close_tasks_with_fallback({task}, cleanup)
    elapsed_s = asyncio.get_running_loop().time() - started_at

    assert completed_in_time, f"cleanup exceeded scheduling bound: {elapsed_s:.3f}s"
    assert cleanup.close_cancellations == 1
    assert cleanup.force_close_calls == 1
    assert cleanup.force_close_cancellations == 1
    assert cleanup.close_finished.is_set()
    assert cleanup.force_close_finished.is_set()
    assert all(task.done() for task in cleanup.close_tasks + cleanup.force_close_tasks)
    assert not {
        pending
        for pending in asyncio.all_tasks()
        if pending is not asyncio.current_task() and pending.get_name().startswith("preflight-cleanup")
    }


@pytest.mark.asyncio
async def test_cleanup_consumes_force_task_that_suppresses_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumed_tasks: list[asyncio.Task[None]] = []
    consume_cleanup_task = PreflightRuntimeControls._consume_cleanup_task

    def record_consumed_task(task: asyncio.Task[None]) -> None:
        consumed_tasks.append(task)
        consume_cleanup_task(task)

    monkeypatch.setattr(
        PreflightRuntimeControls,
        "_consume_cleanup_task",
        staticmethod(record_consumed_task),
    )
    cleanup = FakeCleanupHandle(
        block_close=True,
        block_force_close=True,
        suppress_force_cancellation=True,
    )
    controls = PreflightRuntimeControls(request_context=_request_context())
    controls.register_cleanup(cleanup)
    close_task = asyncio.create_task(controls.close(grace_s=0.01))
    started_at = asyncio.get_running_loop().time()
    await cleanup.force_close_started.wait()

    completed_in_time = await _complete_close_tasks_with_fallback({close_task}, cleanup)
    elapsed_s = asyncio.get_running_loop().time() - started_at
    force_task = cleanup.force_close_tasks[0]

    assert completed_in_time, f"cleanup exceeded scheduling bound: {elapsed_s:.3f}s"
    assert close_task.result() is None
    assert cleanup.force_close_calls == 1
    assert cleanup.force_close_cancellations == 1
    assert not cleanup.force_close_finished.is_set()
    assert not force_task.done()

    cleanup.release_force_close()
    await cleanup.force_close_finished.wait()
    await asyncio.sleep(0)

    assert force_task.done()
    assert force_task in consumed_tasks
    assert not {
        pending
        for pending in asyncio.all_tasks()
        if pending is not asyncio.current_task() and pending.get_name().startswith("preflight-cleanup")
    }


@pytest.mark.asyncio
async def test_cleanup_preserves_caller_cancellation_after_bounded_force_close() -> None:
    cleanup = FakeCleanupHandle(
        block_close=True,
        suppress_close_cancellation=True,
    )
    controls = PreflightRuntimeControls(request_context=_request_context())
    controls.register_cleanup(cleanup)
    task = asyncio.create_task(controls.close(grace_s=0.01))
    await cleanup.close_started.wait()

    task.cancel()
    completed_in_time = await _complete_close_tasks_with_fallback({task}, cleanup)

    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed_in_time
    assert cleanup.force_close_calls == 1
    assert cleanup.close_finished.is_set()
    assert all(task.done() for task in cleanup.close_tasks + cleanup.force_close_tasks)


@pytest.mark.asyncio
async def test_cleanup_close_is_idempotent_for_repeated_and_concurrent_callers() -> None:
    cleanup = FakeCleanupHandle()
    controls = PreflightRuntimeControls(request_context=_request_context())
    controls.register_cleanup(cleanup)

    outcomes = await asyncio.gather(
        controls.close(grace_s=0.05),
        controls.close(grace_s=0.05),
        controls.close(grace_s=0.05),
        return_exceptions=True,
    )
    await controls.close(grace_s=0.05)

    assert outcomes == [None, None, None]
    assert cleanup.close_calls == 1
    assert cleanup.force_close_calls == 0


@pytest.mark.asyncio
async def test_concurrent_cleanup_callers_share_force_outcome_and_own_cancellation() -> None:
    cleanup = FakeCleanupHandle(
        block_close=True,
        suppress_close_cancellation=True,
    )
    controls = PreflightRuntimeControls(request_context=_request_context())
    controls.register_cleanup(cleanup)
    cancelled_caller = asyncio.create_task(controls.close(grace_s=0.01))
    await cleanup.close_started.wait()
    other_callers = {
        asyncio.create_task(controls.close(grace_s=1.0)),
        asyncio.create_task(controls.close(grace_s=1.0)),
    }

    cancelled_caller.cancel()
    callers = {cancelled_caller, *other_callers}
    completed_in_time = await _complete_close_tasks_with_fallback(callers, cleanup)
    outcomes = await asyncio.gather(
        cancelled_caller,
        *other_callers,
        return_exceptions=True,
    )
    await controls.close(grace_s=1.0)

    assert completed_in_time
    assert isinstance(outcomes[0], asyncio.CancelledError)
    assert outcomes[1:] == [None, None]
    assert cleanup.close_calls == 1
    assert cleanup.force_close_calls == 1
    assert all(task.done() for task in cleanup.close_tasks + cleanup.force_close_tasks)


@pytest.mark.asyncio
async def test_cleanup_errors_are_sanitized_and_do_not_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(context_module.logger, "warning", warnings.append)
    cleanup = FakeCleanupHandle(close_error=RuntimeError("https://user:password@example.com/?token=secret"))
    controls = PreflightRuntimeControls(request_context=_request_context())
    controls.register_cleanup(cleanup)

    await controls.close()

    assert warnings == ["Preflight cleanup failed for FakeCleanupHandle."]
    assert cleanup.force_close_calls == 1
    assert "password" not in warnings[0]
    assert "https://" not in warnings[0]


@pytest.mark.asyncio
async def test_execution_context_close_delegates_to_controls() -> None:
    cleanup = FakeCleanupHandle()
    controls = PreflightRuntimeControls(request_context=_request_context())
    controls.register_cleanup(cleanup)
    context = _execution_context(
        controls,
        FakeIdentitySelector({"User-Agent": "test-agent"}),
    )

    await context.close()

    assert cleanup.close_calls == 1
