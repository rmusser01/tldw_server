from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest

from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import ArticleFailure

pytestmark = [pytest.mark.integration, pytest.mark.smoke]

_NORMAL_BODY = b"<!DOCTYPE html><html><head><title>phase4</title></head>" b"<body><main>loopback</main></body></html>"
_HOSTILE_BODY = b"""<!DOCTYPE html><html><head><title>hostile</title></head><body>
<script>
globalThis.TextEncoder = class { encode() { return new Uint8Array(0); } };
globalThis.XMLSerializer = class { serializeToString() { return "HOSTILE_DOCTYPE"; } };
Object.defineProperty(Document.prototype, "doctype", {
  configurable: true,
  get() { return null; }
});
Object.defineProperty(Element.prototype, "outerHTML", {
  configurable: true,
  get() { return "<html>HOSTILE_HTML</html>"; }
});
</script>
<main id="trusted-marker">hostile loopback</main></body></html>"""


async def _serve_fixture(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        request = await reader.readuntil(b"\r\n\r\n")
        body = _HOSTILE_BODY if request.startswith(b"GET /hostile ") else _NORMAL_BODY
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


async def _isolated_html(
    browser_module: Any,
    context: Any,
    page: Any,
    limit: int,
) -> str:
    session = await context.new_cdp_session(page)
    try:
        return await browser_module.GuardedArticleBrowser._serialize_html(session, limit)
    finally:
        await session.detach()


def _assert_limit_failure(exc: ArticleFailure) -> None:
    assert exc.code == "response_too_large"
    assert exc.stage == "rendered_html"


@pytest.fixture
async def loopback_server() -> Any:
    server = await asyncio.start_server(_serve_fixture, "127.0.0.1", 0)
    sockets = server.sockets or []
    assert sockets
    try:
        yield f"http://127.0.0.1:{int(sockets[0].getsockname()[1])}"
    finally:
        server.close()
        await server.wait_closed()


@pytest.fixture
async def chromium_page() -> Any:
    playwright = pytest.importorskip("playwright.async_api")
    manager = await playwright.async_playwright().start()
    browser = None
    try:
        try:
            browser = await manager.chromium.launch(headless=True)
        except playwright.Error as exc:
            pytest.skip(f"Playwright Chromium is unavailable: {type(exc).__name__}")
        context = await browser.new_context()
        page = await context.new_page()
        yield context, page
    finally:
        if browser is not None:
            await browser.close()
        await manager.stop()


@pytest.mark.asyncio
async def test_browser_side_serialization_matches_page_content_on_loopback(
    loopback_server: str,
    chromium_page: tuple[Any, Any],
) -> None:
    browser_module = importlib.import_module("tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser")
    context, page = chromium_page
    await page.goto(f"{loopback_server}/normal", wait_until="domcontentloaded", timeout=10_000)

    page_content = await page.content()
    expected = page_content.replace("<!DOCTYPE html>", "<!DOCTYPE html>\n", 1)
    expected_size = len(expected.encode("utf-8"))
    result = await _isolated_html(browser_module, context, page, expected_size)

    assert expected_size == len(page_content.encode("utf-8")) + 1
    assert result == expected
    with pytest.raises(ArticleFailure) as raised:
        await _isolated_html(browser_module, context, page, expected_size - 1)
    _assert_limit_failure(raised.value)


@pytest.mark.asyncio
async def test_isolated_world_measurement_resists_hostile_main_world_globals(
    loopback_server: str,
    chromium_page: tuple[Any, Any],
) -> None:
    browser_module = importlib.import_module("tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser")
    context, page = chromium_page
    await page.goto(f"{loopback_server}/hostile", wait_until="domcontentloaded", timeout=10_000)

    poisoned = await page.evaluate("""() => ({
          encoded: new TextEncoder().encode("abc").length,
          serialized: new XMLSerializer().serializeToString(document),
          doctype: document.doctype,
          html: document.documentElement.outerHTML
        })""")
    assert poisoned == {
        "encoded": 0,
        "serialized": "HOSTILE_DOCTYPE",
        "doctype": None,
        "html": "<html>HOSTILE_HTML</html>",
    }

    trusted = await _isolated_html(browser_module, context, page, 1_000_000)
    assert "<!DOCTYPE html>\n" in trusted
    assert 'id="trusted-marker"' in trusted
    assert trusted != "<!DOCTYPE html>\n<html>HOSTILE_HTML</html>"
    trusted_size = len(trusted.encode("utf-8"))
    assert await _isolated_html(browser_module, context, page, trusted_size) == trusted
    with pytest.raises(ArticleFailure) as raised:
        await _isolated_html(browser_module, context, page, trusted_size - 1)
    _assert_limit_failure(raised.value)
