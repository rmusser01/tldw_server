"""Exact parent-boundary regressions for preflight HTTP and transfer events."""

import time
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.browser import GuardedPlaywrightBrowserProbe
from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.http import GuardedHttpProbe
from tldw_Server_API.app.core.Web_Scraping.preflight.context import PreflightRuntimeControls
from tldw_Server_API.app.core.Web_Scraping.preflight.probes import ProbeHttpRequest
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop
from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressDecision

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("boundary", ["initial", "raw", "padded_raw", "resolved", "effective", "utf8"])
async def test_preflight_http_url_overflow_latches_before_next_policy_or_dispatch(boundary):
    base = "https://example.test/"
    limit = 64
    initial = base + ("a" * 70 if boundary == "initial" else "start")
    location = "a" * 65 if boundary == "raw" else "a" * 50 if boundary == "resolved" else "\u00e9" * 30
    if boundary == "padded_raw":
        location = " " * 64 + "/ok"
    parent = CrawlContext(CrawlLimits(max_url_bytes=limit), clock=time.monotonic)
    controls = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)
    calls, checks, closed = [], [], []

    async def decide(url, **kwargs):
        checks.append(url)
        return ProbeEgressDecision(True, "allowed")

    async def send(request):
        calls.append(request.url)

        async def close():
            closed.append(request.url)

        return SimpleNamespace(
            status_code=200 if boundary == "effective" or len(calls) > 1 else 302,
            url=base + "x" * 70 if boundary == "effective" else request.url,
            headers={"Location": location},
            text="",
            aclose=close,
        )

    probe = GuardedHttpProbe(
        controls=controls, egress_guard=SimpleNamespace(decide=decide), transport=SimpleNamespace(send=send)
    )
    with pytest.raises(ParentStop, match="budget_exhausted"):
        await probe.get(ProbeHttpRequest(initial))
    with pytest.raises(ParentStop, match="budget_exhausted"):
        await probe.get(ProbeHttpRequest(base + "sibling"))
    assert calls == ([] if boundary == "initial" else [initial])
    assert checks == calls and closed == calls


async def test_preflight_zero_encoded_transfer_uses_positive_data_length():
    parent = CrawlContext(CrawlLimits(transfer_bytes=3), clock=time.monotonic)
    controls = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)
    probe = GuardedPlaywrightBrowserProbe(controls=controls, egress_guard=None)
    with pytest.raises(ParentStop, match="budget_exhausted"):
        probe._on_transfer({"requestId": "one", "encodedDataLength": 0, "dataLength": 4})
    with pytest.raises(ParentStop):
        parent.check()


@pytest.mark.parametrize("boundary", ["initial", "raw", "resolved", "effective", "utf8"])
async def test_preflight_http_exact_url_byte_limit_is_admitted(boundary):
    base = "https://example.test/"
    initial = base + "a" * (64 - len(base)) if boundary == "initial" else base + "start"
    target = base + "b" * (64 - len(base))
    location = target if boundary == "raw" else "b" * (64 - len(base))
    if boundary == "utf8":
        location = "\u00e9" * ((64 - len(base)) // 2) + "x" * ((64 - len(base)) % 2)
    parent = CrawlContext(CrawlLimits(max_url_bytes=64), clock=time.monotonic)
    controls = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)
    calls = []

    async def decide(url, **kwargs):
        return ProbeEgressDecision(True, "allowed")

    async def send(request):
        calls.append(request.url)
        return SimpleNamespace(
            status_code=200 if boundary in {"initial", "effective"} or len(calls) > 1 else 302,
            url=target if boundary == "effective" else request.url,
            headers={"Location": location},
            text="kept",
        )

    probe = GuardedHttpProbe(
        controls=controls, egress_guard=SimpleNamespace(decide=decide), transport=SimpleNamespace(send=send)
    )
    response = await probe.get(ProbeHttpRequest(initial))
    assert response.text == "kept" and len(response.url.encode("utf-8")) == 64
    assert parent.snapshot().network_requests == len(calls)


async def test_preflight_cumulative_finish_charges_only_unobserved_remainder():
    parent = CrawlContext(CrawlLimits(transfer_bytes=10), clock=time.monotonic)
    controls = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)
    probe = GuardedPlaywrightBrowserProbe(controls=controls, egress_guard=None)
    handlers = {}

    async def send(*args):
        return {}

    async def session(page):
        return SimpleNamespace(on=lambda event, handler: handlers.setdefault(event, handler), send=send, detach=send)

    await probe._install_parent_transfer(SimpleNamespace(new_cdp_session=session), None, [])
    handlers["Network.dataReceived"]({"requestId": "one", "encodedDataLength": 4, "dataLength": 4})
    assert parent.snapshot().transfer_bytes == 4
    handlers["Network.loadingFinished"]({"requestId": "one", "encodedDataLength": 10})
    handlers["Network.loadingFinished"]({"requestId": "one", "encodedDataLength": 10})
    assert parent.snapshot().transfer_bytes == 10
    handlers["Network.dataReceived"]({"requestId": "two", "encodedDataLength": 1})
    with pytest.raises(ParentStop, match="budget_exhausted"):
        controls.cap_timeout(None)
