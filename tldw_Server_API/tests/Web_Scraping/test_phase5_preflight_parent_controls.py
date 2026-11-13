"""Local preflight and parent allowances compose without partial admission."""

import asyncio
import time

import pytest

from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.preflight.context import PreflightLimits, PreflightRuntimeControls
from tldw_Server_API.app.core.Web_Scraping.preflight.probes import ProbeBudgetExhausted
from tldw_Server_API.app.core.Web_Scraping.preflight.runner import _isolated
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop

pytestmark = pytest.mark.asyncio


async def test_local_rejection_does_not_debit_parent():
    parent = CrawlContext(CrawlLimits(), clock=time.monotonic)
    local = PreflightRuntimeControls(RuntimeRequestContext(), limits=PreflightLimits(requests=0), parent=parent)
    with pytest.raises(ProbeBudgetExhausted):
        await local.reserve("request")
    assert parent.snapshot().network_requests == 0


async def test_parent_rejection_does_not_debit_local():
    parent = CrawlContext(CrawlLimits(network_requests=1), clock=time.monotonic)
    parent.reserve("network_requests").commit()
    local = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)
    with pytest.raises(ParentStop):
        await local.reserve("request")
    assert local.consumed.requests == 0


async def test_concurrent_reservations_do_not_overshoot():
    parent = CrawlContext(CrawlLimits(network_requests=3), clock=time.monotonic)
    local = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)
    results = await asyncio.gather(*(local.reserve("request") for _ in range(10)), return_exceptions=True)
    assert sum(result is None for result in results) == 3
    assert local.consumed.requests == parent.snapshot().network_requests == 3


async def test_analyzer_parent_stop_is_not_isolated():
    stop = ParentStop("budget_exhausted")

    async def fail():
        raise stop

    with pytest.raises(ParentStop) as caught:
        await _isolated("robots", fail)
    assert caught.value is stop


async def test_robots_parent_requests_and_bytes_are_accounted(monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.policy.adapters import DefaultWebOutboundPolicyChecker
    from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.http import HttpxProbeTransport
    from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressDecision

    parent = CrawlContext(CrawlLimits(), clock=time.monotonic)
    checker = DefaultWebOutboundPolicyChecker(parent=parent)

    async def send(self, request):
        self._controls.charge_bytes(10)
        return SimpleNamespace(status_code=200, headers={}, text="User-agent: *\nAllow: /", url=request.url)

    async def allowed(*_args, **_kwargs):
        return ProbeEgressDecision(True, "allowed", ("93.184.216.34",))

    from tldw_Server_API.app.core.Web_Scraping import outbound_policy
    from tldw_Server_API.app.core.Web_Scraping.policy.probe import DefaultProbeEgressGuard

    monkeypatch.setattr(DefaultProbeEgressGuard, "decide", allowed)
    monkeypatch.setattr(HttpxProbeTransport, "send", send)
    monkeypatch.setattr(
        outbound_policy, "evaluate_url_policy", lambda _url: SimpleNamespace(allowed=True, reason="allowed")
    )
    await checker.decide(
        "https://example.com/private",
        respect_robots=True,
        user_agent="agent",
        context=RuntimeRequestContext(),
        config={},
    )
    assert parent.snapshot().network_requests == 1
    assert parent.snapshot().transfer_bytes == 10


async def test_external_tool_parent_output_abort_keeps_primary_and_waits():
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.external_tools import GuardedExternalToolProbe
    from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressDecision

    parent = CrawlContext(CrawlLimits(transfer_bytes=3), clock=time.monotonic)
    controls = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)
    stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
    stdout.feed_data(b"over budget")
    stdout.feed_eof()
    stderr.feed_eof()

    class Process:
        returncode = None

        def __init__(self):
            self.stdout, self.stderr = stdout, stderr
            self.waited = False

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        async def wait(self):
            self.waited = True
            return self.returncode

    process = Process()

    async def create(*_args, **_kwargs):
        return process

    async def allowed(*_args, **_kwargs):
        return ProbeEgressDecision(True, "allowed")

    probe = GuardedExternalToolProbe(
        controls=controls,
        egress_guard=SimpleNamespace(decide=allowed),
        which=lambda _: "wafw00f",
        process_factory=create,
    )
    with pytest.raises(ParentStop, match="budget_exhausted"):
        await probe.run_waf("https://example.test", find_all=False, enabled=True)
    assert process.waited
    assert parent.snapshot().active_probes == 1


async def test_preflight_browser_parent_bytes_cannot_fail_open():
    from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.browser import GuardedPlaywrightBrowserProbe

    parent = CrawlContext(CrawlLimits(transfer_bytes=3), clock=time.monotonic)
    probe = GuardedPlaywrightBrowserProbe(
        controls=PreflightRuntimeControls(RuntimeRequestContext(), parent=parent), egress_guard=None
    )
    with pytest.raises(ParentStop, match="budget_exhausted"):
        probe._on_transfer({"encodedDataLength": 4})
    with pytest.raises(ParentStop, match="budget_exhausted"):
        parent.check()


async def test_preflight_http_redirects_charge_at_each_started_hop():
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.http import GuardedHttpProbe
    from tldw_Server_API.app.core.Web_Scraping.preflight.probes import ProbeHttpRequest
    from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressDecision

    parent = CrawlContext(CrawlLimits(network_requests=1), clock=time.monotonic)
    calls = []

    async def allowed(*_args, **_kwargs):
        return ProbeEgressDecision(True, "allowed")

    async def send(request):
        calls.append(request.url)
        return SimpleNamespace(status_code=302, headers={"location": "/final"}, text="", url=request.url)

    probe = GuardedHttpProbe(
        controls=PreflightRuntimeControls(RuntimeRequestContext(), parent=parent),
        egress_guard=SimpleNamespace(decide=allowed),
        transport=SimpleNamespace(send=send),
    )
    with pytest.raises(ParentStop, match="budget_exhausted") as caught:
        await probe.get(ProbeHttpRequest(url="https://example.test/start", allow_redirects=True))
    assert calls == ["https://example.test/start"]
    assert caught.value.reason == "budget_exhausted"
    assert parent.snapshot().network_requests == 1


async def test_preflight_browser_stop_aborts_route_without_losing_reason():
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.browser import GuardedPlaywrightBrowserProbe
    from tldw_Server_API.app.core.Web_Scraping.preflight.probes import BrowserProbeOptions
    from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressDecision

    parent = CrawlContext(CrawlLimits(network_requests=1), clock=time.monotonic)
    parent.reserve("network_requests").commit()
    aborted = []

    async def allowed(*_args, **_kwargs):
        return ProbeEgressDecision(True, "allowed")

    async def abort():
        aborted.append(True)

    probe = GuardedPlaywrightBrowserProbe(
        controls=PreflightRuntimeControls(RuntimeRequestContext(), parent=parent),
        egress_guard=SimpleNamespace(decide=allowed),
    )
    route = SimpleNamespace(request=SimpleNamespace(resource_type="document", url="https://example.test"), abort=abort)
    with pytest.raises(ParentStop, match="budget_exhausted") as caught:
        await probe._http_handler(BrowserProbeOptions())(route)
    assert aborted == [True]
    assert caught.value.reason == "budget_exhausted"
    assert parent.snapshot().network_requests == 1


async def test_cancel_during_tool_spawn_owns_late_process_cleanup():
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.external_tools import GuardedExternalToolProbe
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import PhysicalWork, physical_work

    entered, release, reaped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    parent = CrawlContext(CrawlLimits(), clock=time.monotonic)
    controls = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)

    class Process:
        returncode = None

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        async def wait(self):
            reaped.set()
            return self.returncode

    async def create(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return Process()

    owner = PhysicalWork()
    token = physical_work.set(owner)
    try:
        probe = GuardedExternalToolProbe(controls=controls, egress_guard=SimpleNamespace(), process_factory=create)
        task = asyncio.create_task(
            probe._create_process("wafw00f", "https://example.test", find_all=False, timeout_s=1)
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        owner.finish()
        assert not owner.drained.done()
        release.set()
        await asyncio.wait_for(asyncio.shield(owner.drained), 2)
        assert reaped.is_set()
    finally:
        release.set()
        physical_work.reset(token)


async def test_preflight_late_runner_exit_keeps_physical_ownership(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.preflight import PreflightOptions, facade, runner
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import PhysicalWork, physical_work
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import _allowed_target

    entered, release = asyncio.Event(), asyncio.Event()

    async def analyze(*_args):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
        return {}

    monkeypatch.setattr(runner, "gather_analysis_with_context", analyze)
    parent = CrawlContext(CrawlLimits(), clock=time.monotonic)
    context = facade.build_execution_context(_allowed_target(), PreflightOptions(), parent=parent)
    owner = PhysicalWork()
    token = physical_work.set(owner)
    try:
        task = asyncio.create_task(facade._run_before_deadline(_allowed_target(), PreflightOptions(), context))
        await entered.wait()
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=0.05)
        assert task in done
        with pytest.raises(asyncio.CancelledError):
            await task
        owner.finish()
        assert not owner.drained.done()
        release.set()
        await asyncio.wait_for(asyncio.shield(owner.drained), 2)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        physical_work.reset(token)


async def test_page_cancel_before_preflight_commit_rolls_back_both_ledgers(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import PageControls

    parent = CrawlContext(CrawlLimits(), clock=lambda: 0)
    page = PageControls(parent)
    reserve = page.reserve
    local = PreflightRuntimeControls(RuntimeRequestContext(), parent=page)

    def cancel_after_reserve(*args, **kwargs):
        reservation = reserve(*args, **kwargs)
        page.cancel()
        return reservation

    monkeypatch.setattr(page, "reserve", cancel_after_reserve)
    with pytest.raises(asyncio.CancelledError):
        await local.reserve("request")
    assert local.consumed.requests == 0
    assert parent.snapshot().network_requests == 0
    parent.check()


async def test_page_cancel_rejects_reserved_commit_and_refunds_once():
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import PageControls

    parent = CrawlContext(CrawlLimits(), clock=lambda: 0)
    page = PageControls(parent)
    reservation = page.reserve("browser_attempts")
    page.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            reservation.commit()
    finally:
        reservation.close()
        reservation.close()
    assert parent.snapshot().browser_attempts == 0


async def test_page_reserved_commit_preserves_crawl_deadline_check_and_rollback():
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import PageControls

    now = [0]
    parent = CrawlContext(CrawlLimits(execution_seconds=1), clock=lambda: now[0])
    reservation = PageControls(parent).reserve("active_probes")
    now[0] = 1
    try:
        with pytest.raises(ParentStop, match="deadline_exhausted"):
            reservation.commit()
    finally:
        reservation.close()
    assert parent.snapshot().active_probes == 0


async def test_page_commit_before_cancel_remains_charged_and_idempotent():
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import PageControls

    parent = CrawlContext(CrawlLimits(), clock=lambda: 0)
    page = PageControls(parent)
    reservation = page.reserve("network_requests")
    reservation.commit()
    page.cancel()
    reservation.commit()
    reservation.close()
    reservation.close()
    assert parent.snapshot().network_requests == 1
