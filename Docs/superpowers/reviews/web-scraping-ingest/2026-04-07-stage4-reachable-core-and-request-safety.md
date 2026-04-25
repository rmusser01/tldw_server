# Stage 4 Reachable Core and Request Safety

## Scope
- Reachable outbound fetch and request-safety path for the ingest flow only.
- Reviewed service entry points and the core helpers they actually call:
  - `tldw_Server_API/app/services/web_scraping_service.py`
  - `tldw_Server_API/app/services/enhanced_web_scraping_service.py`
  - `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
  - `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
  - `tldw_Server_API/app/core/http_client.py`
  - `tldw_Server_API/app/core/Security/egress.py`
- Follow-up inspection included `scraper_router.py` and `filters.py` because Stage 4 claims depend on router-controlled `respect_robots` and recursive-discovery `RobotsFilter` semantics.

## Code Paths Reviewed
- Legacy service individual scrape path:
  - `web_scraping_service.py:681` -> `Article_Extractor_Lib.scrape_article()`
- Legacy service sitemap path:
  - `web_scraping_service.py:703-707` -> `Article_Extractor_Lib.scrape_from_sitemap()`
- Enhanced service URL-level and recursive paths:
  - `enhanced_web_scraping_service.py:432-445` and `489-502` -> `EnhancedWebScraper.recursive_scrape()`
- Legacy core single-article path:
  - `Article_Extractor_Lib.py:2715-2729` pre-checks `evaluate_url_policy()`
  - `Article_Extractor_Lib.py:2860-2877` applies robots gating when `plan.respect_robots` is true
  - `Article_Extractor_Lib.py:2888-2920` fetches through `_fetch_with_curl()` or `http_fetch()`
- Legacy sitemap/core crawl helpers:
  - `Article_Extractor_Lib.py:3445-3469` pre-checks sitemap URL then fetches with `http_fetch()`
  - `Article_Extractor_Lib.py:3518-3535` pre-checks base URL then fetches linked pages with `http_fetch()`
  - `Article_Extractor_Lib.py:3582-3587` async link collection fetches with `afetch()`
- Enhanced single-article path:
  - `enhanced_web_scraping.py:1307-1321` pre-checks `evaluate_url_policy()`
  - `enhanced_web_scraping.py:1338-1349` builds effective headers from plan + request headers
  - `enhanced_web_scraping.py:1380-1397` applies robots gating when `plan.respect_robots` is true
  - `enhanced_web_scraping.py:1406-1413` can route into `_scrape_with_trafilatura(..., backend=backend_choice, custom_headers=headers)`
  - `enhanced_web_scraping.py:1497-1504` calls `_fetch_html()`
  - `enhanced_web_scraping.py:1170-1185` routes `backend="curl"` into `_fetch_html_curl()`
  - `enhanced_web_scraping.py:1208-1236` uses `CurlCffiSession.get()` directly and only clearly reuses proxy validation
- Enhanced sitemap and recursive crawl path:
  - `enhanced_web_scraping.py:1950-1975` pre-checks sitemap URL and fetches with `afetch()`
  - `enhanced_web_scraping.py:2029-2038` pre-checks recursive base URL
  - `enhanced_web_scraping.py:2143-2150` enables discovery-time `RobotsFilter` from config
  - `enhanced_web_scraping.py:2266-2267` and `2474-2475` recurse via `scrape_multiple()` -> `scrape_article()`
- Central egress enforcement:
  - `egress.py:192-297` rejects unsupported schemes, bad ports, denylisted hosts, strict-profile non-allowlisted hosts, and private/reserved IP resolutions
  - `http_client.py:947-1010` raises on egress denial and validates proxy hosts
  - `http_client.py:2072-2183`, `2396-2484`, and `2731-2831` re-check egress before the first request and on each redirect hop
  - `http_client.py:3058-3164` simple `fetch()` path also re-checks egress and redirect hops

## Tests Reviewed
- Direct request-safety tests:
  - `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
  - `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`
- Follow-up semantics tests executed because Stage 4 conclusions depend on router/filter behavior:
  - `tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py`
  - `tldw_Server_API/tests/Web_Scraping/test_router_validation.py`

## Validation Commands
- Reachable path trace:
```bash
rg -n "scrape_article|recursive_scrape|scrape_from_sitemap|scrape_by_url_level|http_fetch|fetch\(|evaluate_url_policy|is_allowed_by_robots" \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py \
  tldw_Server_API/app/core/http_client.py \
  tldw_Server_API/app/core/Security/egress.py
```
- Enforcement inspection:
```bash
sed -n '1,260p' tldw_Server_API/app/core/http_client.py
sed -n '1,260p' tldw_Server_API/app/core/Security/egress.py
rg -n "robots|cookie|custom_headers|user_agent|evaluate_url_policy|http_fetch|fetch\(" \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py
```
- Direct tests:
```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py \
  tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py
```
- Follow-up tests:
```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py \
  tldw_Server_API/tests/Web_Scraping/test_router_validation.py
```
- Bandit:
```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py \
  tldw_Server_API/app/api/v1/schemas/media_request_models.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/core/http_client.py \
  tldw_Server_API/app/core/Web_Scraping \
  -f json -o /tmp/bandit_web_scraping_ingest_review.json
```
- Results:
  - `test_http_client_fetch.py` + `test_robots_enforcement.py`: `8 passed`
  - `test_filters_and_robots.py` + `test_router_validation.py`: `3 passed`
  - Bandit JSON written to `/tmp/bandit_web_scraping_ingest_review.json`
  - No Stage 4 test directly exercised `EnhancedWebScraper._fetch_html_curl()`

## Findings
1. **Low | High | Robots compliance is not a hard request-safety guarantee on the reachable ingest path**
   - Files: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py:2860`, `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py:1380`, `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py:2143`, `tldw_Server_API/app/core/Web_Scraping/scraper_router.py:245`, `tldw_Server_API/app/core/Web_Scraping/filters.py:281`
   - Evidence: direct code-path proof plus passing tests in `test_robots_enforcement.py`, `test_filters_and_robots.py`, and `test_router_validation.py`
   - Why it matters: the reachable fetch path honors robots only when `respect_robots` remains enabled, and both page-level and discovery-time checks fail open when robots is unreachable or unreadable, so any claim of unconditional robots enforcement should be downgraded to configurable best-effort behavior.
   - Confidence downgrade note: this is not a confirmed egress bypass; it is a policy-strength limitation caused by explicit configuration and fail-open error handling.
2. **Medium | Medium | `http_client`-backed paths have strong request-safety evidence, but the reachable enhanced curl branch remains unproven**
   - Files: `tldw_Server_API/app/core/http_client.py:947`, `tldw_Server_API/app/core/http_client.py:2083`, `tldw_Server_API/app/core/http_client.py:2182`, `tldw_Server_API/app/core/http_client.py:2407`, `tldw_Server_API/app/core/http_client.py:2483`, `tldw_Server_API/app/core/http_client.py:2759`, `tldw_Server_API/app/core/http_client.py:3058`, `tldw_Server_API/app/core/http_client.py:3133`, `tldw_Server_API/app/core/Security/egress.py:192`, `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py:1406`, `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py:1497`, `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py:1170`, `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py:1208`, `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py:1222`, `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py:1235`
   - Evidence: direct code-path proof plus passing `test_http_client_fetch.py`
   - Why it matters: the legacy helpers and enhanced `afetch()`/`httpx` branches have strong in-code evidence for pre-request and redirect-hop egress checks, but `scrape_article()` can also reach `_scrape_with_trafilatura()` -> `_fetch_html()` -> `_fetch_html_curl()`, which uses `CurlCffiSession.get()` directly; this file clearly validates proxies but does not clearly route through `http_client.fetch()` or visibly re-check redirect hops, so Stage 4 cannot generalize the tested `http_client` conclusion across that reachable branch.
   - Confidence downgrade note: this is a probable enforcement gap or at minimum an unproven one, not a confirmed exploitable `custom_headers` security bug from the current evidence.
3. **Info | Medium | Bandit did not surface a reachable Stage 4 request-safety hit**
   - Files: `tldw_Server_API/app/core/http_client.py`, `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`, `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`, `tldw_Server_API/app/core/Web_Scraping/filters.py`, `tldw_Server_API/app/core/Web_Scraping/scraper_router.py`
   - Evidence: Bandit JSON at `/tmp/bandit_web_scraping_ingest_review.json`
   - Why it matters: the broader recursive scan reported 16 low-severity findings, but they were outside this reachable path or not request-safety relevant here; the only touched-file hit in reviewed scope was `B112 try_except_continue` in `scraper_router.py:179`, which does not weaken outbound request enforcement.

## Coverage Gaps
- `playwright` execution was not dynamically exercised in this stage. The reachable code reviewed here still pre-checks URL policy before that path is selected, but this stage does not include an integration test proving equivalent enforcement after browser launch.
- The enhanced curl branch was not dynamically exercised in this stage. `test_http_client_fetch.py` proves redirect and egress behavior for `http_client.fetch()`, but no Stage 4 test proves equivalent redirect-hop enforcement for `EnhancedWebScraper._fetch_html_curl()`.
- `scrape_from_sitemap()` has a deliberate test-mode relaxation at `Article_Extractor_Lib.py:3445-3465` that proceeds after egress denial when pytest/test mode is active. That is not a production-path bypass, but it does mean test behavior is intentionally weaker than runtime behavior for this helper.
- Bandit scope was broader than the exact Stage 4 path because `-r tldw_Server_API/app/core/Web_Scraping` includes unrelated analyzers and legacy web-search modules. Findings from those files were triaged out of the Stage 4 conclusion.

## Improvements
- Add an integration test that exercises the enhanced scraper `playwright` fallback and proves the same egress denial behavior before navigation.
- Add an integration test that exercises the enhanced curl/trafilatura path and verifies redirect-hop behavior against blocked targets.
- Add an explicit regression test asserting that custom request headers, including `User-Agent`, do not alter egress outcomes on the enhanced service path, and that any curl branch remains policy-equivalent.
- If stronger compliance is required, change robots handling from fail-open to fail-closed for selected deployments and record that as a configuration contract rather than an implicit assumption.

## Exit Note
- Stage 4 reachable-path review does not confirm a request-safety bypass in the ingest flow.
- Confidence is high for the `http_client`-backed paths reviewed here, lower for any claim that robots policy is universally enforced, and downgraded to medium for the reachable enhanced curl branch because equivalent redirect-safe enforcement is not proven by the inspected code or current tests.
