from __future__ import annotations

import asyncio
import importlib

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


async def _serve_fixture(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        await reader.readuntil(b"\r\n\r\n")
        body = b"<!DOCTYPE html><html><head><title>phase4</title></head><body><main>loopback</main></body></html>"
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
async def test_browser_side_serialization_matches_page_content_on_loopback() -> None:
    playwright = pytest.importorskip("playwright.async_api")
    browser_module = importlib.import_module("tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser")
    expression = getattr(browser_module, "_HTML_SERIALIZATION_EXPRESSION", None)
    assert isinstance(expression, str) and expression

    server = await asyncio.start_server(_serve_fixture, "127.0.0.1", 0)
    sockets = server.sockets or []
    assert sockets
    target = f"http://127.0.0.1:{int(sockets[0].getsockname()[1])}/fixture"
    manager = await playwright.async_playwright().start()
    browser = None
    try:
        try:
            browser = await manager.chromium.launch(headless=True)
        except playwright.Error as exc:
            pytest.skip(f"Playwright Chromium is unavailable: {type(exc).__name__}")
        page = await browser.new_page()
        await page.goto(target, wait_until="domcontentloaded", timeout=10_000)
        page_content = await page.content()
        page_content_size = len(page_content.encode("utf-8"))
        over_result = await page.evaluate(expression, page_content_size)
        expected = page_content.replace("<!DOCTYPE html>", "<!DOCTYPE html>\n", 1)
        result = await page.evaluate(expression, len(expected.encode("utf-8")))

        assert over_result == {"ok": False, "size": page_content_size + 1}
        assert result == {"ok": True, "html": expected}
    finally:
        if browser is not None:
            await browser.close()
        await manager.stop()
        server.close()
        await server.wait_closed()
