from __future__ import annotations

import asyncio
import ipaddress
from urllib.parse import urlsplit

import pytest

from tldw_Server_API.app.core.Web_Scraping.preflight import (
    BrowserProbeOptions,
    PreflightRuntimeControls,
    ProbeUnavailable,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.browser import (
    GuardedPlaywrightBrowserProbe,
)
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    ProbeEgressDecision,
    RuntimeRequestContext,
)

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


class _LoopbackOnlyGuard:
    async def decide(
        self,
        url: str,
        *,
        context: RuntimeRequestContext,
    ) -> ProbeEgressDecision:
        del context
        try:
            allowed = ipaddress.ip_address(urlsplit(url).hostname or "").is_loopback
        except ValueError:
            allowed = False
        return ProbeEgressDecision(
            allowed=allowed,
            reason="allowed" if allowed else "address_forbidden",
        )


async def _serve_marker(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        await reader.readuntil(b"\r\n\r\n")
        body = b"<html><body><main id='phase3-marker'>loopback-browser-ok</main></body></html>"
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/html; charset=utf-8\r\n"
            + f"Content-Length: {len(body)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n"
            + body
        )
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_guarded_browser_renders_loopback_marker() -> None:
    pytest.importorskip("playwright.async_api")
    try:
        server = await asyncio.start_server(_serve_marker, "127.0.0.1", 0)
    except OSError as exc:
        pytest.skip(f"Loopback server is unavailable: errno={exc.errno}")
    sockets = server.sockets or []
    assert sockets
    port = int(sockets[0].getsockname()[1])
    target = f"http://127.0.0.1:{port}/marker"
    controls = PreflightRuntimeControls(
        RuntimeRequestContext(
            source="preflight",
            stage="preflight",
            user_id="smoke",
            request_id="phase3-browser-smoke",
        )
    )
    probe = GuardedPlaywrightBrowserProbe(
        controls=controls,
        egress_guard=_LoopbackOnlyGuard(),
    )

    try:
        try:
            async with probe.open_page(BrowserProbeOptions()) as page:
                await page.goto(
                    target,
                    wait_until="domcontentloaded",
                    timeout_ms=10_000,
                )
                content = await page.content()
        except ProbeUnavailable as exc:
            pytest.skip(exc.public_message)
        assert "loopback-browser-ok" in content
        assert content.strip()
    finally:
        await controls.close()
        server.close()
        await server.wait_closed()
