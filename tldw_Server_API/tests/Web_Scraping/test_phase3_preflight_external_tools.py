from __future__ import annotations

import asyncio
import builtins
import importlib
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.Web_Scraping.preflight import (
    ExternalToolResult,
    PreflightDeadlineExceeded,
    PreflightLimits,
    PreflightOptions,
    PreflightRuntimeControls,
    ProbeBudgetExhausted,
    ProbeError,
    ProbeTimeout,
    ProbeUnavailable,
)
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext
from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.utils.waf_result_parser import (
    parse_wafw00f_output,
)
from tldw_Server_API.tests.Web_Scraping.preflight_fakes import (
    FakeCallRecorder,
    FakeClock,
    FakeCoercionValue,
    FakeExternalProcess,
    FakeExternalToolDecision,
    FakeProbeEgressGuard,
    FakeProcessFactory,
    FakeWhich,
)

pytestmark = pytest.mark.unit

_ADAPTER_MODULE = "tldw_Server_API.app.core.Web_Scraping.preflight.adapters.external_tools"
_ADAPTER_PACKAGE = "tldw_Server_API.app.core.Web_Scraping.preflight.adapters"
_METRICS_MODULE = "tldw_Server_API.app.core.Metrics"
_URL = "https://example.com/path?token=secret"
_EXECUTABLE = "/opt/private/bin/wafw00f"
_MISSING = object()


def _adapter_module() -> Any | None:
    try:
        return importlib.import_module(_ADAPTER_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in {_ADAPTER_MODULE, _ADAPTER_MODULE.rpartition(".")[0]}:
            return None
        raise


def _required(name: str) -> Any:
    module = _adapter_module()
    assert module is not None, "Task 6 governed external-tool adapter module is missing"
    assert hasattr(module, name), f"Task 6 external-tool symbol {name} is missing"
    return getattr(module, name)


@contextmanager
def _fresh_adapter_package() -> Iterator[Any]:
    parent_name, _, attribute = _ADAPTER_PACKAGE.rpartition(".")
    parent = importlib.import_module(parent_name)
    saved_attribute = getattr(parent, attribute, _MISSING)
    saved_modules = {name: sys.modules.pop(name) for name in (_ADAPTER_MODULE, _ADAPTER_PACKAGE) if name in sys.modules}
    if saved_attribute is not _MISSING:
        delattr(parent, attribute)
    try:
        yield importlib.import_module(_ADAPTER_PACKAGE)
    finally:
        sys.modules.pop(_ADAPTER_MODULE, None)
        sys.modules.pop(_ADAPTER_PACKAGE, None)
        sys.modules.update(saved_modules)
        if saved_attribute is _MISSING:
            if hasattr(parent, attribute):
                delattr(parent, attribute)
        else:
            setattr(parent, attribute, saved_attribute)


@pytest.fixture(autouse=True)
def _short_adapter_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _adapter_module()
    if module is None:
        return
    monkeypatch.setattr(module, "_WAF_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(module, "_PROCESS_CLEANUP_GRACE_SECONDS", 0.02)


def _controls(
    *,
    active_probes: int | None = None,
    deadline: float | None = None,
    clock: FakeClock | None = None,
) -> PreflightRuntimeControls:
    return PreflightRuntimeControls(
        RuntimeRequestContext(
            source="preflight",
            stage="preflight",
            user_id="7",
            request_id="request-task-6",
            metadata={"scope": "task-6"},
        ),
        limits=PreflightLimits(active_probes=active_probes),
        deadline=deadline,
        clock=clock or FakeClock(),
    )


def _probe(
    *,
    controls: PreflightRuntimeControls | None = None,
    guard: FakeProbeEgressGuard | None = None,
    which: FakeWhich | None = None,
    process_factory: FakeProcessFactory | None = None,
    observer: Any | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "controls": controls or _controls(),
        "egress_guard": guard or FakeProbeEgressGuard([True]),
        "which": which or FakeWhich(_EXECUTABLE),
        "process_factory": process_factory or FakeProcessFactory([FakeExternalProcess()]),
    }
    if observer is not None:
        kwargs["legacy_default_observer"] = observer
    return _required("GuardedExternalToolProbe")(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "installed", "starts", "warns", "error_type"),
    [
        (None, True, True, True, None),
        (None, False, False, False, ProbeUnavailable),
        (True, True, True, False, None),
        (True, False, False, False, ProbeUnavailable),
        (False, True, False, False, ProbeError),
    ],
)
async def test_external_tool_enablement_matrix(
    enabled: bool | None,
    installed: bool,
    starts: bool,
    warns: bool,
    error_type: type[BaseException] | None,
) -> None:
    warning = FakeCallRecorder()
    metric = FakeCallRecorder()
    observer = _required("_LegacyExternalToolDefaultObserver")(
        warning=warning,
        increment_counter=metric,
    )
    which = FakeWhich(_EXECUTABLE if installed else None)
    factory = FakeProcessFactory([FakeExternalProcess()])
    probe = _probe(
        which=which,
        process_factory=factory,
        observer=observer,
    )

    if error_type is None:
        result = await probe.run_waf(_URL, find_all=True, enabled=enabled)
        assert result == ExternalToolResult(returncode=0, stdout="", stderr="")
    else:
        with pytest.raises(error_type) as raised:
            await probe.run_waf(_URL, find_all=True, enabled=enabled)
        if enabled is False:
            assert raised.value.error_code == "external_tool_disabled"
        else:
            assert raised.value.error_code == "missing_dependency"

    assert bool(factory.calls) is starts
    assert len(warning.calls) == int(warns)
    assert len(metric.calls) == int(warns)


@pytest.mark.asyncio
async def test_absent_installed_transition_signal_is_once_under_concurrency() -> None:
    warning = FakeCallRecorder()
    metric = FakeCallRecorder()
    observer = _required("_LegacyExternalToolDefaultObserver")(
        warning=warning,
        increment_counter=metric,
    )
    processes = [FakeExternalProcess() for _ in range(20)]
    factory = FakeProcessFactory(processes)
    controls = _controls(active_probes=20)
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([True] * 20),
        process_factory=factory,
        observer=observer,
    )

    await asyncio.gather(*(probe.run_waf(_URL, find_all=False, enabled=None) for _ in range(20)))

    assert len(warning.calls) == 1
    assert warning.calls[0][0] == ("Preflight external tool used because its config key is absent",)
    assert len(metric.calls) == 1
    assert metric.calls[0] == (
        ("web_scraping_preflight_legacy_external_tool_default_total",),
        {"labels": {"tool": "wafw00f"}},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_observer", ["warning", "metric"])
async def test_transition_observer_failure_does_not_block_execution(
    failed_observer: str,
) -> None:
    warning = FakeCallRecorder(
        error=RuntimeError("logger included sensitive output") if failed_observer == "warning" else None
    )
    metric = FakeCallRecorder(
        error=RuntimeError("metric included sensitive output") if failed_observer == "metric" else None
    )
    observer = _required("_LegacyExternalToolDefaultObserver")(
        warning=warning,
        increment_counter=metric,
    )
    factory = FakeProcessFactory([FakeExternalProcess(stdout=b"ok")])

    result = await _probe(
        process_factory=factory,
        observer=observer,
    ).run_waf(_URL, find_all=False, enabled=None)

    assert result.stdout == "ok"
    assert len(factory.calls) == 1
    assert len(warning.calls) == 1
    assert len(metric.calls) == 1


@pytest.mark.asyncio
async def test_raising_injected_observer_does_not_block_governed_execution() -> None:
    observer = FakeCallRecorder(error=RuntimeError("observer raw secret"))
    factory = FakeProcessFactory([FakeExternalProcess(stdout=b"ok")])

    result = await _probe(
        process_factory=factory,
        observer=observer,
    ).run_waf(_URL, find_all=False, enabled=None)

    assert result.stdout == "ok"
    assert len(observer.calls) == 1
    assert len(factory.calls) == 1


@pytest.mark.asyncio
async def test_import_disabled_and_missing_paths_do_not_load_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_imports: list[str] = []
    real_import = builtins.__import__

    def reject_metrics_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == _METRICS_MODULE or name.startswith(f"{_METRICS_MODULE}."):
            metrics_imports.append(name)
            raise ImportError("Metrics unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_metrics_import)
    try:
        with _fresh_adapter_package() as package:
            assert _ADAPTER_MODULE in sys.modules
            for enabled, dependency in ((False, _EXECUTABLE), (None, None), (True, None)):
                probe = package.GuardedExternalToolProbe(
                    controls=_controls(),
                    egress_guard=FakeProbeEgressGuard([]),
                    which=FakeWhich(dependency),
                    process_factory=FakeProcessFactory(),
                )
                with pytest.raises(ProbeError):
                    await probe.run_waf(_URL, find_all=False, enabled=enabled)
    except ImportError:
        pytest.fail("adapter import eagerly loaded Metrics", pytrace=False)

    assert metrics_imports == []


@pytest.mark.asyncio
async def test_lazy_metric_import_failure_does_not_block_governed_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_imports: list[str] = []
    real_import = builtins.__import__

    def reject_metrics_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == _METRICS_MODULE or name.startswith(f"{_METRICS_MODULE}."):
            metrics_imports.append(name)
            raise ImportError("Metrics unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_metrics_import)
    try:
        with _fresh_adapter_package() as package:
            warning = FakeCallRecorder()
            module = sys.modules[_ADAPTER_MODULE]
            observer = module._LegacyExternalToolDefaultObserver(warning=warning)
            process = FakeExternalProcess(stdout=b"governed")
            probe = package.GuardedExternalToolProbe(
                controls=_controls(),
                egress_guard=FakeProbeEgressGuard([True]),
                which=FakeWhich(_EXECUTABLE),
                process_factory=FakeProcessFactory([process]),
                legacy_default_observer=observer,
            )
            result = await probe.run_waf(_URL, find_all=False, enabled=None)
    except ImportError:
        pytest.fail("adapter import eagerly loaded Metrics", pytrace=False)

    assert result.stdout == "governed"
    assert len(warning.calls) == 1
    assert metrics_imports == [_METRICS_MODULE]


@pytest.mark.asyncio
async def test_disabled_and_missing_dependency_have_no_governed_side_effects() -> None:
    for enabled, installed in ((False, True), (None, False), (True, False)):
        events: list[str] = []
        observer = FakeCallRecorder(events=events, name="observe")
        controls = _controls(active_probes=0)
        guard = FakeProbeEgressGuard([], events=events)
        which = FakeWhich(_EXECUTABLE if installed else None, events=events)
        factory = FakeProcessFactory(events=events)

        with pytest.raises(ProbeError):
            await _probe(
                controls=controls,
                guard=guard,
                which=which,
                process_factory=factory,
                observer=observer,
            ).run_waf(_URL, find_all=False, enabled=enabled)

        assert events == ([] if enabled is False else ["which"])
        assert controls.consumed.active_probes == 0


@pytest.mark.asyncio
async def test_falsey_dependency_lookup_is_missing_without_side_effects() -> None:
    controls = _controls(active_probes=0)
    guard = FakeProbeEgressGuard([])
    factory = FakeProcessFactory()

    with pytest.raises(ProbeUnavailable) as raised:
        await _probe(
            controls=controls,
            guard=guard,
            which=FakeWhich(""),
            process_factory=factory,
        ).run_waf(_URL, find_all=False, enabled=True)

    assert raised.value.error_code == "missing_dependency"
    assert controls.consumed.active_probes == 0
    assert guard.urls == []
    assert factory.calls == []


def test_malformed_external_tool_option_remains_explicitly_disabled() -> None:
    options = PreflightOptions.from_mapping({"web_scraper_preflight_enable_external_tools": "maybe"})

    assert options.external_tools_enabled is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "error_code"),
    [
        (False, "policy_denied"),
        ("policy_error", "policy_error"),
        (RuntimeError("secret guard details"), "policy_error"),
    ],
)
async def test_guard_denial_and_error_fail_closed_before_process_creation(
    decision: bool | str | BaseException,
    error_code: str,
) -> None:
    controls = _controls(active_probes=1)
    factory = FakeProcessFactory()

    with pytest.raises(ProbeError) as raised:
        await _probe(
            controls=controls,
            guard=FakeProbeEgressGuard([decision]),
            process_factory=factory,
        ).run_waf(_URL, find_all=False, enabled=True)

    assert raised.value.error_code == error_code
    assert raised.value.public_message == "Probe destination was denied."
    assert controls.consumed.active_probes == 1
    assert factory.calls == []
    assert "secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_guard_cancellation_propagates_before_process_creation() -> None:
    cancellation = asyncio.CancelledError("caller cancellation")
    factory = FakeProcessFactory()

    with pytest.raises(asyncio.CancelledError) as raised:
        await _probe(
            guard=FakeProbeEgressGuard([cancellation]),
            process_factory=factory,
        ).run_waf(_URL, find_all=False, enabled=True)

    assert raised.value is cancellation
    assert factory.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision",
    [
        FakeExternalToolDecision(allowed_error=RuntimeError("raw allowed accessor secret")),
        FakeExternalToolDecision(allowed=FakeCoercionValue(bool_error=RuntimeError("raw allowed coercion secret"))),
        FakeExternalToolDecision(
            allowed=False,
            reason_error=RuntimeError("raw reason accessor secret"),
        ),
        FakeExternalToolDecision(
            allowed=False,
            reason=FakeCoercionValue(str_error=RuntimeError("raw reason coercion secret")),
        ),
    ],
)
async def test_decision_access_and_coercion_fail_closed_without_raw_details(
    decision: FakeExternalToolDecision,
) -> None:
    factory = FakeProcessFactory()

    with pytest.raises(ProbeError) as raised:
        await _probe(
            guard=FakeProbeEgressGuard([decision]),
            process_factory=factory,
        ).run_waf(_URL, find_all=False, enabled=True)

    assert raised.value.error_code == "policy_error"
    assert raised.value.public_message == "Probe destination was denied."
    assert "raw" not in str(raised.value)
    assert factory.calls == []


@pytest.mark.asyncio
async def test_active_probe_budget_exhaustion_precedes_guard_and_process() -> None:
    controls = _controls(active_probes=0)
    guard = FakeProbeEgressGuard([])
    factory = FakeProcessFactory()

    with pytest.raises(ProbeBudgetExhausted):
        await _probe(
            controls=controls,
            guard=guard,
            process_factory=factory,
        ).run_waf(_URL, find_all=False, enabled=True)

    assert guard.urls == []
    assert factory.calls == []


@pytest.mark.asyncio
async def test_governed_order_and_exact_exec_argv_have_no_shell_options() -> None:
    events: list[str] = []
    process = FakeExternalProcess(events=events)
    factory = FakeProcessFactory([process], events=events)
    observer = FakeCallRecorder(events=events, name="observe")
    guard = FakeProbeEgressGuard([True], events=events)
    which = FakeWhich(_EXECUTABLE, events=events)

    await _probe(
        controls=_controls(active_probes=1),
        guard=guard,
        which=which,
        process_factory=factory,
        observer=observer,
    ).run_waf(_URL, find_all=True, enabled=None)

    assert which.calls == ["wafw00f"]
    assert factory.calls == [
        (
            (_EXECUTABLE, _URL, "-a"),
            {"stdout": asyncio.subprocess.PIPE, "stderr": asyncio.subprocess.PIPE},
        )
    ]
    assert events[:5] == [
        "which",
        f"guard:{_URL}",
        "process:create",
        "observe",
        "process:communicate",
    ]
    assert guard.contexts[0].stage == "preflight_subrequest"


@pytest.mark.asyncio
async def test_find_all_false_omits_only_optional_argv() -> None:
    factory = FakeProcessFactory([FakeExternalProcess()])

    await _probe(process_factory=factory).run_waf(
        _URL,
        find_all=False,
        enabled=True,
    )

    assert factory.calls[0][0] == (_EXECUTABLE, _URL)


@pytest.mark.asyncio
async def test_bytes_decode_with_replacement_and_nonzero_result_is_unchanged() -> None:
    process = FakeExternalProcess(
        returncode=9,
        stdout=b"Cloudflare \xff WAF",
        stderr=b"failure \xfe detail",
    )

    result = await _probe(process_factory=FakeProcessFactory([process])).run_waf(_URL, find_all=False, enabled=True)

    assert result == ExternalToolResult(
        returncode=9,
        stdout="Cloudflare \ufffd WAF",
        stderr="failure \ufffd detail",
    )


@pytest.mark.asyncio
async def test_decoded_result_is_compatible_with_the_legacy_waf_parser() -> None:
    process = FakeExternalProcess(
        stdout=b"The site is behind Cloudflare WAF (Cloudflare Inc)",
    )

    result = await _probe(process_factory=FakeProcessFactory([process])).run_waf(_URL, find_all=False, enabled=True)

    assert parse_wafw00f_output(result.stdout, result.stderr) == [("behind Cloudflare WAF", "Cloudflare Inc")]


@pytest.mark.asyncio
async def test_communicate_error_and_captured_output_never_enter_logs() -> None:
    raw_error = "raw communicate error with token=secret"
    logged: list[str] = []
    sink_id = logger.add(lambda message: logged.append(str(message)))
    try:
        process = FakeExternalProcess(
            communicate_error=RuntimeError(raw_error),
            stdout=b"raw stdout secret",
            stderr=b"raw stderr secret",
        )
        with pytest.raises(ProbeError) as raised:
            await _probe(process_factory=FakeProcessFactory([process])).run_waf(_URL, find_all=False, enabled=True)
    finally:
        logger.remove(sink_id)

    assert raised.value.error_code == "probe_error"
    assert raised.value.public_message == "Probe failed."
    assert process.terminate_calls == 1
    assert all(raw_error not in message for message in logged)
    assert all("raw stdout secret" not in message for message in logged)
    assert all("raw stderr secret" not in message for message in logged)


@pytest.mark.asyncio
async def test_process_creation_is_bounded_by_overall_deadline() -> None:
    factory = FakeProcessFactory(block_creation=True)
    loop = asyncio.get_running_loop()
    controls = _controls(deadline=loop.time() + 0.02, clock=loop.time)

    with pytest.raises(PreflightDeadlineExceeded):
        await _probe(
            controls=controls,
            process_factory=factory,
        ).run_waf(_URL, find_all=False, enabled=True)

    assert factory.creation_cancellations == 1


@pytest.mark.asyncio
async def test_process_creation_is_bounded_by_analyzer_local_timeout() -> None:
    factory = FakeProcessFactory(block_creation=True)

    with pytest.raises(ProbeTimeout):
        async with asyncio.timeout(0.2):
            await _probe(
                process_factory=factory,
            ).run_waf(_URL, find_all=False, enabled=True)

    assert factory.creation_cancellations == 1


@pytest.mark.asyncio
async def test_spawn_elapsed_time_reduces_communicate_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _adapter_module()
    assert module is not None
    monkeypatch.setattr(module, "_WAF_TIMEOUT_SECONDS", 10.0)
    clock = FakeClock()
    monkeypatch.setattr(module, "_monotonic", clock, raising=False)
    process = FakeExternalProcess()

    class ElapsedFactory(FakeProcessFactory):
        async def __call__(self, *args: Any, **kwargs: Any) -> FakeExternalProcess:
            created = await super().__call__(*args, **kwargs)
            clock.advance(4.0)
            return created

    creation_timeouts: list[float | None] = []
    communicate_timeouts: list[float | None] = []

    class RecordingTimeout:
        def __init__(self, timeout_s: float | None) -> None:
            creation_timeouts.append(timeout_s)

        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args: Any) -> None:
            return None

    async def recording_wait_for(awaitable: Any, timeout: float | None) -> Any:
        communicate_timeouts.append(timeout)
        return await awaitable

    monkeypatch.setattr(module.asyncio, "timeout", RecordingTimeout)
    monkeypatch.setattr(module.asyncio, "wait_for", recording_wait_for)

    result = await _probe(
        process_factory=ElapsedFactory([process]),
    ).run_waf(_URL, find_all=False, enabled=True)

    assert result.returncode == 0
    assert creation_timeouts == [10.0]
    assert communicate_timeouts == [6.0]


@pytest.mark.asyncio
async def test_exhausted_deadline_prevents_process_creation_after_reservation_and_guard() -> None:
    clock = FakeClock(10.0)
    controls = _controls(active_probes=1, deadline=10.0, clock=clock)
    guard = FakeProbeEgressGuard([True])
    factory = FakeProcessFactory()

    with pytest.raises(PreflightDeadlineExceeded):
        await _probe(
            controls=controls,
            guard=guard,
            process_factory=factory,
        ).run_waf(_URL, find_all=False, enabled=True)

    assert controls.consumed.active_probes == 1
    assert guard.urls == [_URL]
    assert factory.calls == []


@pytest.mark.asyncio
async def test_deadline_is_rechecked_after_communicate_before_success() -> None:
    clock = FakeClock()
    controls = _controls(deadline=1.0, clock=clock)
    process = FakeExternalProcess(
        stdout=b"late success",
        communicate_hook=lambda: clock.advance(1.0),
    )

    with pytest.raises(PreflightDeadlineExceeded):
        await _probe(
            controls=controls,
            process_factory=FakeProcessFactory([process]),
        ).run_waf(_URL, find_all=False, enabled=True)

    assert process.returncode == 0
    assert process.terminate_calls == 0
    assert process.kill_calls == 0


@pytest.mark.asyncio
async def test_communicate_uses_analyzer_local_timeout_and_cleans_process() -> None:
    process = FakeExternalProcess(block_communicate=True)

    with pytest.raises(ProbeTimeout):
        await _probe(
            process_factory=FakeProcessFactory([process]),
        ).run_waf(_URL, find_all=False, enabled=True)

    assert process.communicate_cancellations == 1
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_overall_deadline_timeout_wins_over_local_timeout() -> None:
    process = FakeExternalProcess(block_communicate=True)
    loop = asyncio.get_running_loop()
    controls = _controls(deadline=loop.time() + 0.02, clock=loop.time)

    with pytest.raises(PreflightDeadlineExceeded):
        await _probe(
            controls=controls,
            process_factory=FakeProcessFactory([process]),
        ).run_waf(_URL, find_all=False, enabled=True)

    assert process.terminate_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_caller_cancellation_wins_after_shielded_process_cleanup() -> None:
    process = FakeExternalProcess(block_communicate=True)
    probe = _probe(
        process_factory=FakeProcessFactory([process]),
    )
    task = asyncio.create_task(
        probe.run_waf(_URL, find_all=False, enabled=True),
    )
    await process.communicate_started.wait()

    task.cancel("caller requested cancellation")
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_stubborn_process_is_terminated_killed_and_waited_once() -> None:
    process = FakeExternalProcess(
        block_communicate=True,
        terminate_completes=False,
    )
    probe = _probe(
        process_factory=FakeProcessFactory([process]),
    )
    task = asyncio.create_task(
        probe.run_waf(_URL, find_all=False, enabled=True),
    )
    await process.communicate_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_terminate_process_lookup_error_still_awaits_shared_wait() -> None:
    process = FakeExternalProcess(
        block_wait=True,
        terminate_error=ProcessLookupError(),
    )
    handle = _required("_ProcessCleanupHandle")(process)
    cleanup = asyncio.create_task(handle.close())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert process.wait_started.is_set()
    assert cleanup.done() is False
    process.release_wait()
    await cleanup
    await handle.close()

    assert process.terminate_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_kill_process_lookup_error_still_awaits_shared_wait() -> None:
    process = FakeExternalProcess(
        block_wait=True,
        kill_error=ProcessLookupError(),
    )
    handle = _required("_ProcessCleanupHandle")(process)
    cleanup = asyncio.create_task(handle.force_close())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert process.wait_started.is_set()
    assert cleanup.done() is False
    process.release_wait()
    await cleanup
    await handle.force_close()

    assert process.kill_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_local_and_request_cleanup_race_is_idempotent() -> None:
    process = FakeExternalProcess(
        block_communicate=True,
        terminate_completes=False,
    )
    controls = _controls()
    probe = _probe(
        controls=controls,
        process_factory=FakeProcessFactory([process]),
    )
    run_task = asyncio.create_task(
        probe.run_waf(_URL, find_all=False, enabled=True),
    )
    await process.communicate_started.wait()

    run_task.cancel()
    request_cleanup = asyncio.create_task(controls.close(grace_s=0.02))
    results = await asyncio.gather(
        run_task,
        request_cleanup,
        return_exceptions=True,
    )

    assert isinstance(results[0], asyncio.CancelledError)
    assert results[1] is None
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_success_cleanup_is_idempotent_with_later_request_close() -> None:
    process = FakeExternalProcess(stdout=b"success")
    controls = _controls()

    result = await _probe(
        controls=controls,
        process_factory=FakeProcessFactory([process]),
    ).run_waf(_URL, find_all=False, enabled=True)
    await controls.close(grace_s=0.02)

    assert result.stdout == "success"
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.wait_calls == 0


@pytest.mark.asyncio
async def test_process_failures_and_logs_redact_url_argv_path_and_output() -> None:
    sensitive_values = (_URL, _EXECUTABLE, "raw stdout secret", "raw stderr secret")
    factory = FakeProcessFactory(
        error=RuntimeError(" ".join(sensitive_values)),
    )

    with pytest.raises(ProbeError) as raised:
        await _probe(process_factory=factory).run_waf(
            _URL,
            find_all=False,
            enabled=True,
        )

    assert raised.value.error_code == "probe_error"
    assert raised.value.public_message == "Probe failed."
    assert all(value not in str(raised.value) for value in sensitive_values)


@pytest.mark.asyncio
async def test_package_exports_only_the_approved_task_6_adapter_addition() -> None:
    package = importlib.import_module("tldw_Server_API.app.core.Web_Scraping.preflight.adapters")

    assert "GuardedExternalToolProbe" in package.__all__
    assert package.GuardedExternalToolProbe is _required("GuardedExternalToolProbe")
    assert "GuardedPlaywrightBrowserProbe" not in package.__all__


def test_external_adapter_uses_injected_boundaries_only() -> None:
    constructor: Callable[..., Any] = _required("GuardedExternalToolProbe")

    probe = constructor(
        controls=_controls(),
        egress_guard=FakeProbeEgressGuard([True]),
        which=FakeWhich(None),
        process_factory=FakeProcessFactory(),
    )

    assert probe is not None
