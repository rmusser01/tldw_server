# Web Scraping Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the web scraping ingest remediation complete by stabilizing the `/process-web-scraping` endpoint tests, hardening centralized curl redirect and egress enforcement, rerouting the enhanced curl path through the centralized HTTP client, and locking in the current best-effort robots contract with direct regression coverage.

**Architecture:** Keep the existing friendly-ingest compatibility seam in `media.__init__` and `web_scraping_service.py`, but stop relying on it for the `/process-web-scraping` endpoint tests. Centralize curl request-safety in `http_client.fetch()`, then make `EnhancedWebScraper._fetch_html_curl()` delegate to that path instead of owning its own curl request logic. Cover the enhanced robots branch directly with focused unit tests rather than changing runtime behavior.

**Tech Stack:** Python 3.14, FastAPI, pytest, httpx, curl_cffi adapter seam, Bandit

---

## File Map

**Primary code files**
- `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py`
  Endpoint-local task resolution seam for `/process-web-scraping`.
- `tldw_Server_API/app/core/http_client.py`
  Central simple fetch redirect and egress behavior for `backend="curl"`.
- `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
  Enhanced scraper curl delegation and robots branch clarification.

**Primary test files**
- `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
  Usage-events endpoint test updated to patch the endpoint-local seam.
- `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`
  Deterministic request-path custom-headers test using the same client style as usage-events.
- `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
  Curl redirect and egress policy regression tests.
- `tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py`
  New focused enhanced-scraper guard tests for curl delegation and robots behavior.
- `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_compat_patchpoints.py`
  Compatibility assertion updated for the new endpoint-local seam.

**Validation-only test files**
- `tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py`
- `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`

**Inspection-only compatibility file**
- `tldw_Server_API/app/services/web_scraping_service.py`
  Preserve existing `media.process_web_scraping_task` compatibility behavior for friendly-ingest callers.

## Stage 1: Endpoint Determinism
**Goal:** Remove the order-sensitive `/process-web-scraping` test failure by standardizing the endpoint tests on one lifecycle-safe client strategy and patch target.
**Success Criteria:** The usage-events and custom-header endpoint tests pass together in one pytest invocation, and friendly-ingest tests that still patch `media.process_web_scraping_task` remain green.
**Tests:** `test_webscraping_usage_events.py`, `test_custom_headers_support.py`, `test_friendly_ingest_crawl_flags.py`, `test_media_compat_patchpoints.py`
**Status:** Not Started

## Stage 2: Central Curl Policy
**Goal:** Make the centralized simple curl fetch path enforce redirect and egress policy in the same shape as the existing simple httpx path.
**Success Criteria:** Same-host redirects still succeed, blocked redirect hops are not followed, and curl requests never rely on implicit redirect following.
**Tests:** `test_http_client_fetch.py`
**Status:** Not Started

## Stage 3: Enhanced Scraper Delegation And Robots Coverage
**Goal:** Route the enhanced curl path through the centralized HTTP client and add direct enhanced-scraper regression coverage for curl delegation and the current robots best-effort contract.
**Success Criteria:** `_fetch_html_curl()` uses `http_client.fetch()` with `backend="curl"`, enhanced robots fail-open and explicit-disallow behavior is covered, Bandit passes on the touched scope, and the final targeted batch is green.
**Tests:** `test_enhanced_web_scraping_guards.py`, `test_http_client_fetch.py`, `test_robots_enforcement.py`, final targeted batch
**Status:** Not Started

### Task 1: Stabilize `/process-web-scraping` Endpoint Tests

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py`
- Modify: `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
- Modify: `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`
- Modify: `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_compat_patchpoints.py`
- Test: `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
- Test: `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py`
- Test: `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_compat_patchpoints.py`

- [ ] **Step 1: Write failing deterministic endpoint tests against an endpoint-local seam**

Update the two endpoint test files so they patch a resolver on `process_web_scraping.py` and so the custom-headers request uses the same `TestClient` lifecycle style as the usage-events test.

```python
# tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture()
def client_with_ws_overrides(monkeypatch, client_with_single_user):
    client, usage_logger = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media.process_web_scraping as endpoint_mod

    async def stub_process_web_scraping_task(**kwargs):
        return {"status": "ok", "results": []}

    monkeypatch.setattr(
        endpoint_mod,
        "_resolve_process_web_scraping_task",
        lambda: stub_process_web_scraping_task,
    )

    yield client, usage_logger
```

```python
# tldw_Server_API/tests/WebScraping/test_custom_headers_support.py
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import CookieManager


@pytest.mark.asyncio
async def test_cookie_manager_stores_and_returns_cookies(tmp_path):
    manager = CookieManager(storage_path=tmp_path / "cookies.json")
    manager.add_cookies(
        "example.com",
        [{"name": "foo", "value": "bar"}, {"name": "session", "value": "trace-id"}],
    )
    cookies = manager.get_cookies("https://example.com/some/path")
    assert cookies == [{"name": "foo", "value": "bar"}, {"name": "session", "value": "trace-id"}]
    assert manager.get_cookies("https://other.example") is None


def test_process_web_scraping_endpoint_receives_custom_headers(
    client_with_single_user,
    monkeypatch,
):
    client, _ = client_with_single_user

    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.endpoints.media import process_web_scraping as endpoint_mod
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.main import app

    fake_db = MagicMock(spec=MediaDatabase)
    app.dependency_overrides[get_media_db_for_user] = lambda: fake_db

    mocked_result = {"status": "success", "message": "ok", "count": 0, "results": []}
    payload = {
        "scrape_method": "Individual URLs",
        "url_input": "https://example.com/article",
        "max_pages": 5,
        "max_depth": 1,
        "summarize_checkbox": False,
        "mode": "ephemeral",
        "user_agent": "IntegrationAgent/1.0",
        "custom_headers": {"X-Test": "true"},
    }

    async def fake_process_web_scraping_task(**kwargs):
        return mocked_result

    monkeypatch.setattr(
        endpoint_mod,
        "_resolve_process_web_scraping_task",
        lambda: fake_process_web_scraping_task,
    )

    try:
        response = client.post("/api/v1/media/process-web-scraping", json=payload)
    finally:
        app.dependency_overrides.pop(get_media_db_for_user, None)

    assert response.status_code == 200
    assert response.json() == mocked_result
```

- [ ] **Step 2: Run the endpoint tests to verify the new seam does not exist yet**

Run:
```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -vv \
  tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py \
  tldw_Server_API/tests/WebScraping/test_custom_headers_support.py
```

Expected: FAIL because `process_web_scraping.py` does not yet expose `_resolve_process_web_scraping_task`.

- [ ] **Step 3: Implement the endpoint-local resolver seam**

Add a module-local resolver in `process_web_scraping.py` and switch the route to use it. Remove the runtime dependency on `media_mod` from this endpoint only.

```python
# tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py
from tldw_Server_API.app.api.v1.schemas.media_request_models import WebScrapingRequest
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE
from tldw_Server_API.app.services.web_scraping_service import process_web_scraping_task


def _resolve_process_web_scraping_task():
    return process_web_scraping_task
```

```python
# tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py
        task = _resolve_process_web_scraping_task()
        result = await task(
            scrape_method=payload.scrape_method,
            url_input=payload.url_input,
            url_level=payload.url_level,
            max_pages=payload.max_pages,
            max_depth=payload.max_depth,
            summarize_checkbox=payload.summarize_checkbox,
            custom_prompt=payload.custom_prompt,
            api_name=payload.api_name,
            api_key=None,
            keywords=payload.keywords or "",
            custom_titles=payload.custom_titles,
            system_prompt=payload.system_prompt,
            temperature=payload.temperature,
            custom_cookies=payload.custom_cookies,
            mode=payload.mode,
            user_id=(
                getattr(getattr(db, "user", None), "id", None)
                if db is not None
                else None
            ),
            user_agent=payload.user_agent,
            custom_headers=payload.custom_headers,
            crawl_strategy=payload.crawl_strategy,
            include_external=payload.include_external,
            score_threshold=payload.score_threshold,
        )
```

- [ ] **Step 4: Update the compat assertion to reflect the new endpoint seam**

```python
# tldw_Server_API/tests/Media_Ingestion_Modification/test_media_compat_patchpoints.py
def test_web_scraping_endpoint_resolves_task_without_compat_module():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "api"
        / "v1"
        / "endpoints"
        / "media"
        / "process_web_scraping.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "compat_patchpoints" not in source
    assert "_resolve_process_web_scraping_task" in source
    assert 'getattr(media_mod, "process_web_scraping_task"' not in source
```

- [ ] **Step 5: Run the endpoint and compatibility validation batch**

Run:
```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py \
  tldw_Server_API/tests/WebScraping/test_custom_headers_support.py \
  tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_media_compat_patchpoints.py
```

Expected: `11 passed`

- [ ] **Step 6: Commit the endpoint determinism changes**

Run:
```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py \
  tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py \
  tldw_Server_API/tests/WebScraping/test_custom_headers_support.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_media_compat_patchpoints.py
git commit -m "test: stabilize web scraping endpoint request-path tests"
```

Expected: one commit captures the endpoint-local seam and deterministic request-path tests.

### Task 2: Harden Centralized Curl Redirect And Egress Enforcement

**Files:**
- Modify: `tldw_Server_API/app/core/http_client.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`

- [ ] **Step 1: Write failing curl redirect policy tests**

Extend `test_http_client_fetch.py` so the curl path must manually follow allowed same-host redirects and refuse blocked redirect hops before a second request is issued.

```python
# tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py
class DummyResp:
    def __init__(
        self,
        url: str,
        headers: dict,
        *,
        status_code: int = 200,
        text: str = "<html><body>ok</body></html>",
    ):
        self.status_code = status_code
        self.headers = headers
        self.text = text
        self.url = url


def test_curl_fetch_follows_same_host_redirects_under_policy(monkeypatch):
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: True)
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda proxies: None)

    calls = []

    class DummyCurlSession:
        def __init__(self, impersonate=None):
            self.impersonate = impersonate

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            if url == "https://example.com/start":
                return DummyResp(
                    url,
                    {"Location": "/final"},
                    status_code=302,
                    text="",
                )
            return DummyResp("https://example.com/final", {}, status_code=200, text="<html>final</html>")

    monkeypatch.setattr(hc, "_resolve_curl_session", lambda: DummyCurlSession)

    resp = hc.fetch(
        "https://example.com/start",
        headers={"Accept-Encoding": "gzip, br, zstd"},
        backend="curl",
        follow_redirects=True,
    )

    assert [url for url, _ in calls] == [
        "https://example.com/start",
        "https://example.com/final",
    ]
    assert calls[0][1]["allow_redirects"] is False
    assert calls[1][1]["allow_redirects"] is False
    assert resp["status"] == 200
    assert resp["url"] == "https://example.com/final"


def test_curl_fetch_denies_blocked_redirect_hop(monkeypatch):
    allowed = {
        "https://example.com/start": True,
        "https://example.com/final": False,
    }
    monkeypatch.setattr(hc, "_is_url_allowed", lambda url: allowed.get(url, True))
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda proxies: None)

    calls = []

    class DummyCurlSession:
        def __init__(self, impersonate=None):
            self.impersonate = impersonate

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return DummyResp(
                url,
                {"Location": "/final"},
                status_code=302,
                text="",
            )

    monkeypatch.setattr(hc, "_resolve_curl_session", lambda: DummyCurlSession)

    with pytest.raises(ValueError):
        hc.fetch("https://example.com/start", backend="curl", follow_redirects=True)

    assert [url for url, _ in calls] == ["https://example.com/start"]
    assert calls[0][1]["allow_redirects"] is False
```

- [ ] **Step 2: Run the curl-policy tests to verify the current behavior fails**

Run:
```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -vv tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py
```

Expected: FAIL because the current curl path still passes `allow_redirects=True` through to the curl session and does not manually apply the same redirect loop used by the simple httpx path.

- [ ] **Step 3: Implement manual redirect handling for the simple curl path**

Keep `_fetch_curl_simple()` as a one-request helper and move redirect policy into the `backend="curl"` branch inside `fetch()`.

```python
# tldw_Server_API/app/core/http_client.py
def _fetch_curl_simple(
    *,
    url: str,
    headers: dict[str, str],
    cookies: dict[str, str] | None,
    timeout: float | None,
    impersonate: str | None,
    proxies: dict[str, str] | None,
    allow_redirects: bool,
) -> HttpResponse:
    CurlSession = _resolve_curl_session()
    if CurlSession is None:
        raise RuntimeError("curl_cffi is not installed")  # noqa: TRY003
    if proxies:
        _validate_proxies_or_raise(proxies)

    req_kwargs: dict[str, Any] = {
        "headers": headers,
        "cookies": cookies,
        "allow_redirects": False,
    }
    if timeout is not None:
        req_kwargs["timeout"] = timeout
    if proxies:
        req_kwargs["proxies"] = proxies

    with CurlSession(impersonate=impersonate) as session:
        resp = session.get(url, **req_kwargs)
        return HttpResponse(
            status=int(getattr(resp, "status_code", 0)),
            headers=dict(getattr(resp, "headers", {}) or {}),
            text=str(getattr(resp, "text", "")),
            url=str(getattr(resp, "url", url)),
            backend="curl",
        )

```

```python
# tldw_Server_API/app/core/http_client.py
    if backend_norm == "curl":
        cur_url = url
        redirects = 0

        while True:
            if not _is_url_allowed(cur_url):
                raise ValueError("Egress denied for URL")  # noqa: TRY003

            resp = _fetch_curl_simple(
                url=cur_url,
                headers=req_headers,
                cookies=cookies,
                timeout=timeout,
                impersonate=impersonate,
                proxies=proxies,
                allow_redirects=False,
            )
            status = int(resp["status"])

            if not follow_redirects or status not in (301, 302, 303, 307, 308):
                return resp

            location = (resp["headers"] or {}).get("location") or (resp["headers"] or {}).get("Location")
            if not location:
                return resp

            try:
                base_url = str(resp["url"] or cur_url)
                next_url = str(httpx.URL(base_url).join(httpx.URL(location)))
            except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                try:
                    next_url = str(httpx.URL(location))
                except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
                    return resp

            if not _redirect_allowed(cur_url, next_url):
                return resp

            redirects += 1
            if redirects > DEFAULT_MAX_REDIRECTS:
                return resp

            cur_url = next_url
```

- [ ] **Step 4: Run the centralized curl-policy tests**

Run:
```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -vv tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py
```

Expected: `8 passed`

- [ ] **Step 5: Commit the centralized curl hardening**

Run:
```bash
git add \
  tldw_Server_API/app/core/http_client.py \
  tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py
git commit -m "fix: harden curl fetch redirect enforcement"
```

Expected: one commit captures the centralized curl redirect and egress hardening.

### Task 3: Reroute Enhanced Curl Fetches And Cover Robots Contract

**Files:**
- Modify: `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`
- Test: `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
- Test: `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py`
- Test: `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_compat_patchpoints.py`

- [ ] **Step 1: Write a failing enhanced-scraper guard test module**

Create a focused unit test file that proves `_fetch_html_curl()` must go through `http_client.fetch()` and directly covers the current enhanced robots branch.

```python
# tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping as ews


def test_fetch_html_curl_routes_through_http_client_fetch(monkeypatch):
    calls = {}

    def fake_fetch(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return {
            "status": 200,
            "headers": {"Content-Type": "text/html"},
            "text": "<html>ok</html>",
            "url": url,
            "backend": "curl",
        }

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)

    scraper = ews.EnhancedWebScraper(config={})
    html = scraper._fetch_html_curl(
        "https://example.com/article",
        headers={"X-Test": "true"},
        cookies={"session": "abc"},
        timeout=5.0,
        impersonate="chrome120",
        proxies=None,
    )

    assert html == "<html>ok</html>"
    assert calls["url"] == "https://example.com/article"
    assert calls["kwargs"]["backend"] == "curl"
    assert calls["kwargs"]["headers"]["X-Test"] == "true"
    assert calls["kwargs"]["cookies"] == {"session": "abc"}


def _build_scraper(monkeypatch):
    scraper = ews.EnhancedWebScraper(config={})

    async def _acquire():
        return None

    scraper.rate_limiter.acquire = _acquire

    plan = SimpleNamespace(
        respect_robots=True,
        ua_profile="chrome_120_win",
        extra_headers={},
        cookies={},
        impersonate=None,
        proxies=None,
        strategy_order=None,
        schema_rules=None,
        llm_settings=None,
        regex_settings=None,
        cluster_settings=None,
        backend="auto",
    )

    monkeypatch.setattr(scraper, "_resolve_scrape_plan", lambda url: (plan, "httpx", ""))
    monkeypatch.setattr(scraper, "_run_preflight_analysis", AsyncMock(return_value=None))
    monkeypatch.setattr(scraper, "_apply_preflight_advice", lambda *args: ("httpx", "trafilatura", []))
    monkeypatch.setattr("tldw_Server_API.app.core.Security.egress.evaluate_url_policy", lambda url: SimpleNamespace(allowed=True))
    monkeypatch.setattr(ews, "increment_counter", lambda *args, **kwargs: None)

    return scraper


@pytest.mark.asyncio
async def test_scrape_article_allows_when_robots_check_errors(monkeypatch):
    scraper = _build_scraper(monkeypatch)
    fake_scrape = AsyncMock(
        return_value={
            "url": "https://example.com/article",
            "content": "ok",
            "extraction_successful": True,
        }
    )
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", fake_scrape)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib.is_allowed_by_robots_async",
        AsyncMock(side_effect=RuntimeError("robots unavailable")),
    )

    result = await scraper.scrape_article("https://example.com/article")

    assert result["extraction_successful"] is True
    fake_scrape.assert_awaited_once()


@pytest.mark.asyncio
async def test_scrape_article_blocks_when_robots_disallows(monkeypatch):
    scraper = _build_scraper(monkeypatch)
    fake_scrape = AsyncMock()
    monkeypatch.setattr(scraper, "_scrape_with_trafilatura", fake_scrape)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib.is_allowed_by_robots_async",
        AsyncMock(return_value=False),
    )

    result = await scraper.scrape_article("https://example.com/article")

    assert result["extraction_successful"] is False
    assert result["error"] == "Blocked by robots policy"
    fake_scrape.assert_not_awaited()
```

- [ ] **Step 2: Run the enhanced-scraper guard tests to verify the curl delegation is still missing**

Run:
```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -vv tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py
```

Expected: `1 failed, 2 passed` because `_fetch_html_curl()` still imports and uses `CurlCffiSession` directly instead of routing through `http_client.fetch()`.

- [ ] **Step 3: Reroute the enhanced curl branch through the centralized HTTP client and clarify robots semantics**

Replace the direct `curl_cffi` call in `enhanced_web_scraping.py` with a delegation to `http_client.fetch()`, and add a short comment above the robots branch to make the fail-open contract explicit.

```python
# tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py
    @staticmethod
    def _fetch_html_curl(
        url: str,
        *,
        headers: dict[str, str],
        cookies: Optional[dict[str, str]],
        timeout: float,
        impersonate: Optional[str],
        proxies: Optional[dict[str, str]],
    ) -> str:
        from tldw_Server_API.app.core import http_client as _http_client

        resp = _http_client.fetch(
            url,
            headers=headers,
            cookies=cookies,
            timeout=timeout,
            proxies=proxies,
            backend="curl",
            impersonate=impersonate,
        )
        return resp["text"]
```

```python
# tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py
            # robots.txt enforcement is best-effort: explicit disallow blocks
            # the scrape, but transient robots retrieval errors do not.
            if getattr(plan, "respect_robots", True):
                try:
                    from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import (
                        is_allowed_by_robots_async,
                    )
                    if not await is_allowed_by_robots_async(url, headers.get("User-Agent", DEFAULT_USER_AGENT)):
                        try:
                            parsed = urlparse(url)
                            increment_counter("scrape_blocked_by_robots_total", labels={"domain": parsed.netloc})
                        except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
                            increment_counter("scrape_blocked_by_robots_total", labels={})
                        return _attach_preflight({
                            "url": url,
                            "error": "Blocked by robots policy",
                            "extraction_successful": False,
                        })
                except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS:
                    pass
```

- [ ] **Step 4: Run the final targeted remediation batch**

Run:
```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py \
  tldw_Server_API/tests/WebScraping/test_custom_headers_support.py \
  tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py \
  tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py \
  tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py \
  tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_media_compat_patchpoints.py
```

Expected: `24 passed`

- [ ] **Step 5: Run Bandit on the touched scope**

Run:
```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py \
  tldw_Server_API/app/core/http_client.py \
  tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py \
  -f json -o /tmp/bandit_web_scraping_remediation.json
```

Expected: command exits successfully and writes `/tmp/bandit_web_scraping_remediation.json`; address any new findings in the touched code before committing.

- [ ] **Step 6: Commit the enhanced-scraper and robots changes**

Run:
```bash
git add \
  tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py \
  tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py
git add \
  tldw_Server_API/app/core/http_client.py \
  tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py \
  tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py \
  tldw_Server_API/tests/WebScraping/test_custom_headers_support.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_media_compat_patchpoints.py
git commit -m "fix: route enhanced curl scraping through http client"
```

Expected: final remediation commit captures the enhanced curl delegation, robots clarification, and any remaining test adjustments needed to leave the targeted batch green.
