"""Finite event admission and required public browser-session capability."""

import asyncio

import pytest

from tldw_Server_API.app.core.Web_Scraping.preflight import (
    BrowserProbeOptions,
    PreflightRuntimeControls,
    ProbeUnavailable,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.browser import GuardedPlaywrightBrowserProbe, _FetchGuard
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import PhysicalWork, physical_work
from tldw_Server_API.tests.Web_Scraping.preflight_fakes import FakePlaywrightLauncher, FakeProbeEgressGuard

pytestmark = pytest.mark.asyncio


async def test_preflight_fetch_callback_flood_is_bounded_before_scheduling():
    launcher = FakePlaywrightLauncher()
    browser = launcher.browser
    controls = PreflightRuntimeControls(RuntimeRequestContext())
    probe = GuardedPlaywrightBrowserProbe(controls=controls, egress_guard=FakeProbeEgressGuard([]))
    probe._context = browser
    owner = PhysicalWork()
    token = physical_work.set(owner)
    try:
        guard = _FetchGuard(browser, probe, BrowserProbeOptions())
        await guard.install()
        callback = browser.cdp.handlers["Fetch.requestPaused"]
        for index in range(1000):
            callback({"requestId": str(index), "request": {"url": "https://e.test/"}, "resourceType": "Document"})
        assert len(guard._callbacks) == 64 and len(guard._commands) <= 64
        stop = probe._stop_task
        assert stop is not None and not probe._admission_open
        await stop
        await guard.close()
        assert browser.close_calls == 1 and browser.cdp.detached
        assert len([method for method, _ in browser.cdp.commands if method == "Fetch.failRequest"]) == 64
        owner.finish()
        await asyncio.wait_for(asyncio.shield(owner.drained), 1)
    finally:
        physical_work.reset(token)


async def test_missing_public_browser_fetch_session_fails_closed_before_context():
    from tldw_Server_API.app.core.Web_Scraping.browser_transport import decide_browser_transport

    launcher = FakePlaywrightLauncher()
    launcher.browser.new_browser_cdp_session = None
    controls = PreflightRuntimeControls(RuntimeRequestContext())
    probe = GuardedPlaywrightBrowserProbe(
        controls=controls,
        egress_guard=FakeProbeEgressGuard([]),
        launcher=launcher,
        capability_check=lambda: True,
        transport_decision=lambda: decide_browser_transport(
            configured_mode="url_guarded", auth_mode="single_user", outbound_policy_mode="compat"
        ),
    )
    with pytest.raises(ProbeUnavailable):
        async with probe.open_page(BrowserProbeOptions()):
            pytest.fail("missing browser Fetch capability must never fall back to context routing")
    await controls.close()
    assert launcher.browser.contexts == [] and not launcher.browser.is_connected()


async def test_terminal_guard_close_does_not_allow_connection_stop_before_actual_retirement():
    from tldw_Server_API.app.core.Web_Scraping.browser_transport import decide_browser_transport

    launcher = FakePlaywrightLauncher()
    controls = PreflightRuntimeControls(RuntimeRequestContext())
    probe = GuardedPlaywrightBrowserProbe(
        controls=controls,
        egress_guard=FakeProbeEgressGuard([]),
        launcher=launcher,
        capability_check=lambda: True,
        transport_decision=lambda: decide_browser_transport(
            configured_mode="url_guarded", auth_mode="single_user", outbound_policy_mode="compat"
        ),
    )
    manager = probe.open_page(BrowserProbeOptions())
    await manager.__aenter__()
    guard = next(handle for handle in controls._cleanup_handles if handle._kind == "fetch")
    connection = next(handle for handle in controls._cleanup_handles if handle._kind == "playwright")
    forced = asyncio.create_task(guard.force_close())
    await asyncio.sleep(0)
    assert guard._terminal and not guard._operation_task.done()
    stopped = asyncio.create_task(connection.close())
    try:
        for _ in range(3):
            await asyncio.sleep(0)
        assert launcher.playwright.stop_calls == 0, "terminal is not native guard retirement"
        assert not stopped.done() and not launcher.browser.cdp.detached
    finally:
        launcher.browser.disconnect()
        await asyncio.gather(forced, stopped, return_exceptions=True)
        await manager.__aexit__(None, None, None)
        await controls.close()
    assert launcher.playwright.stop_calls == 1 and launcher.browser.cdp.detached


@pytest.mark.parametrize("failure", ["malformed", "scheduling", "command_capacity"])
async def test_preflight_fetch_failures_close_admission_without_losing_owned_work(monkeypatch, failure):
    from tldw_Server_API.app.core.Web_Scraping.preflight.probes import ProbeError

    browser = FakePlaywrightLauncher().browser
    controls = PreflightRuntimeControls(RuntimeRequestContext())
    probe = GuardedPlaywrightBrowserProbe(controls=controls, egress_guard=FakeProbeEgressGuard([True]))
    probe._context = browser
    guard = _FetchGuard(browser, probe, BrowserProbeOptions())
    await guard.install()
    event = {"requestId": 1 if failure == "malformed" else "one", "request": {"url": "https://e.test/"}}
    if failure == "scheduling":
        create = asyncio.create_task

        def fail_once(operation, **kwargs):
            if kwargs.get("name") == "preflight-browser-fetch-callback":
                raise RuntimeError("scheduler unavailable")
            return create(operation, **kwargs)

        monkeypatch.setattr(asyncio, "create_task", fail_once)
    elif failure == "command_capacity":
        future = asyncio.get_running_loop().create_future()
        guard._commands = {future for _ in range(1)}
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.preflight.adapters.browser._FETCH_CALLBACK_CAPACITY", 1
        )
        with pytest.raises(ProbeError):
            await guard.send("Fetch.continueRequest", {"requestId": "one"})
        future.set_result(None)
        guard._commands.clear()
        browser.disconnect()
        await guard.close()
        return
    browser.cdp.handlers["Fetch.requestPaused"](event)
    await asyncio.gather(*tuple(guard._callbacks))
    await probe._stop_task
    assert not probe._admission_open and isinstance(controls._callback_error, ProbeError)
    await guard.close()
    assert browser.cdp.detached and guard._callbacks == set() and guard._commands == set()
