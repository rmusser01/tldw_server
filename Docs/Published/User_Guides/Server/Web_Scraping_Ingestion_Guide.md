# Web Scraping and Ingestion Pipeline Guide

This guide explains how web scraping works in tldw_server, which endpoints to use, and which knobs you can tune.

Auth note: examples use `X-API-KEY`. For multi-user JWTs, use `Authorization: Bearer <token>`.

Compatibility note: when scraping contracts evolve, the API can emit
deprecation headers (`Deprecation`, `Sunset`, `Link`) alongside normal
responses so clients can migrate before removal windows close.

## Pipeline overview

1. Request intake: URL(s) + scrape strategy + optional crawl overrides.
2. Egress + robots checks: outbound URLs are validated and robots.txt can be enforced.
3. Scraper routing: auto-selects backend or uses per-domain rules.
4. Extraction: content parsed via trafilatura / Playwright / BeautifulSoup.
5. Optional summarization: LLM summary of each article if requested.
6. Output: either persisted in Media DB or returned as ephemeral results.

## Choose the right endpoint

- `POST /api/v1/media/process-web-scraping`
  - Primary ingestion path.
  - Uses the enhanced scraping service when available.
  - Supports persistence (`mode: "persist"`) or ephemeral output (`mode: "ephemeral"`).
  - Accepts advanced crawl overrides (`crawl_strategy`, `include_external`, `score_threshold`).

- `POST /api/v1/media/ingest-web-content`
  - URL-list oriented, good for quick batch ingestion or simple pipelines.
  - For `individual` and `sitemap`, uses direct extraction + optional analysis.
  - For `url_level` and `recursive_scraping`, delegates to the enhanced service in ephemeral mode.
  - Current implementation returns scraped results; it does not persist to Media DB. Use `/process-web-scraping` if you need storage.

- Optional management endpoints (may be feature-flagged):
  - `GET /api/v1/web-scraping/status`
  - `GET /api/v1/web-scraping/progress/{task_id}`
  - `POST /api/v1/web-scraping/service/initialize`
  - `GET/POST /api/v1/web-scraping/cookies/{domain}`

If `/api/v1/web-scraping/*` returns 404, enable the `web-scraping` route group in config (see `tldw_Server_API/Config_Files/README.md`).

## Scrape methods and inputs

`/process-web-scraping` expects UI-style labels:

- `Individual URLs`: `url_input` is a newline-separated list.
- `Sitemap`: `url_input` is a sitemap URL.
- `URL Level`: `url_input` is a base URL and `url_level` is the path depth.
- `Recursive Scraping`: `url_input` is a base URL plus `max_pages` and `max_depth`.

`/ingest-web-content` uses enum-style values:

- `individual`
- `sitemap`
- `url_level`
- `recursive_scraping`

## Example requests

### Persist scraped pages into Media DB

```bash
curl -X POST http://127.0.0.1:8000/api/v1/media/process-web-scraping \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "scrape_method": "Individual URLs",
        "url_input": "https://example.com/page-1\nhttps://example.com/page-2",
        "summarize_checkbox": true,
        "keywords": "example,docs",
        "mode": "persist"
      }'
```

### Recursive crawl with overrides (ephemeral preview)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/media/process-web-scraping \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "scrape_method": "Recursive Scraping",
        "url_input": "https://docs.example.com",
        "max_pages": 50,
        "max_depth": 3,
        "crawl_strategy": "best_first",
        "include_external": false,
        "score_threshold": 0.25,
        "mode": "ephemeral"
      }'
```

### Batch ingest (results only)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/media/ingest-web-content \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "urls": ["https://example.com/a", "https://example.com/b"],
        "titles": ["Article A", "Article B"],
        "scrape_method": "individual",
        "perform_analysis": true
      }'
```

## Scheduled scraping with Watchlists

Watchlists are for recurring scraping (RSS or site lists) with runs you can trigger manually or schedule.

How it works:

1. Create sources (`rss` or `site`).
2. Create a job that selects sources and sets a schedule.
3. Preview or run the job.
4. Inspect runs and scraped items.

Key endpoints:

- `POST /api/v1/watchlists/sources`
- `POST /api/v1/watchlists/jobs`
- `POST /api/v1/watchlists/jobs/{job_id}/run`
- `GET /api/v1/watchlists/runs/{run_id}`
- `GET /api/v1/watchlists/items`

Persistence behavior:
- Watchlists always record run stats and scraped items.
- Set `ingest_prefs.persist_to_media_db=true` to also write items into the Media DB.

### Queue items into a run-specific report (WebUI `/watchlists`)

Use this flow when you want to hand-pick scraped items and generate a report from only those queued items:

1. Open `/watchlists` and go to the **Articles** tab.
2. Open an item and click **Include in next briefing** to toggle `queued_for_briefing=true`.
3. In **Smart Feeds**, switch to **Queued for briefing**.
4. Select the run in the run selector, then click **Generate report from queue**.
5. The UI switches to **Reports** and filters to that run so you can open/download the generated output.

Notes:
- Queueing is independent of item `status` (`ingested`/`filtered`) and `reviewed`; it only controls report candidate selection.
- The generated report request is run-specific and includes explicit queued `item_ids`.

### Example: create a site source with scrape rules

```bash
curl -X POST http://127.0.0.1:8000/api/v1/watchlists/sources \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "name": "Docs changelog",
        "url": "https://docs.example.com/changelog",
        "source_type": "site",
        "settings": {
          "top_n": 10,
          "scrape_rules": {
            "list_url": "https://docs.example.com/changelog",
            "item_selector": ".entry",
            "title_selector": "h2 a",
            "link_selector": "h2 a",
            "summary_selector": ".summary",
            "date_selector": "time"
          }
        }
      }'
```

Notes on `scrape_rules`:

- Use either CSS selectors (`*_selector`) or XPath (`*_xpath`) fields.
- Common fields: `item_selector`/`entry_selector`, `title_selector`, `link_selector` or `url_selector`, `summary_selector`, `content_selector`, `published_xpath` or `date_selector`.
- Advanced options include `limit`, `pagination`, and `alternates` (see `tldw_Server_API/app/core/Watchlists/fetchers.py` for the full schema).

### Example: create a scheduled job

```bash
curl -X POST http://127.0.0.1:8000/api/v1/watchlists/jobs \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "name": "Docs updates",
        "scope": {"sources": [123]},
        "schedule_expr": "0 */6 * * *",
        "timezone": "UTC",
        "ingest_prefs": {"persist_to_media_db": true}
      }'
```

Trigger a run:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/watchlists/jobs/456/run \
  -H "X-API-KEY: $SINGLE_USER_API_KEY"
```

## WebSearch integration (research endpoint)

`POST /api/v1/research/websearch` is for discovery and aggregation; it does not ingest content into the Media DB.
Use it to find URLs, then pass those URLs into `/process-web-scraping` or a Watchlists source.

Example flow:

1. Call `POST /api/v1/research/websearch` with your query.
2. Extract result URLs.
3. Submit those URLs to `POST /api/v1/media/process-web-scraping` with `scrape_method: "Individual URLs"`.

### 4chan engine (`engine: "4chan"`)

Use the `4chan` engine when you want thread-level discovery from selected boards.

Supported request fields:

- `engine`: must be `4chan`.
- `boards`: optional list of board names (example: `["g", "tv", "pol"]`).
  - Default: `["g", "tv", "pol"]` unless overridden by server config/env.
- `max_threads_per_board`: optional scan cap for live catalog threads per board.
  - Range: `1..1000`.
  - Default: `250` (or server override).
- `include_archived`: include archived thread scan per board.
  - Default: `false`.
- `max_archived_threads_per_board`: optional scan cap for archived threads per board.
  - Range: `1..500`.
  - Default: `min(max_threads_per_board, 50)` (or server override).
- `result_count`: optional global cap on the final number of results returned.
  - This limit is applied **after** per-board scans and deduplication/merge, not per-board. Unless explicitly specified as a per-board override, `result_count` truncates the combined global result set across all boards.

Behavior notes:

- Per-board scans (live catalog and, optionally, archived threads) run independently for each board.
- Results from all boards are then combined and deduped globally by `(board, thread_no)`: when a duplicate exists, non-archived items are preferred and metadata is merged from both sources.
- After deduplication, the global result set is truncated to `result_count`. Because this truncation happens after the cross-board merge, the final count may include threads from any combination of the requested boards.
- Board failures are fail-soft: one board can fail while others still return results.
- Warning metadata is returned when a board fails; if all boards fail, result set is empty and `error` is populated.

Minimal example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/research/websearch \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "rust memory safety",
        "engine": "4chan",
        "result_count": 5
      }'
```

Advanced example (board + archive tuning):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/research/websearch \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "rust ownership borrowing",
        "engine": "4chan",
        "result_count": 20,
        "boards": ["g", "tv"],
        "max_threads_per_board": 300,
        "include_archived": true,
        "max_archived_threads_per_board": 80
      }'
```

## Request knobs (what you can tweak)

### `/process-web-scraping` (WebScrapingRequest)

- `scrape_method`: `Individual URLs|Sitemap|URL Level|Recursive Scraping`.
- `url_input`: base URL or newline-separated list.
- `url_level`: required for `URL Level`.
- `max_pages`, `max_depth`: crawl bounds.
- `summarize_checkbox`: enable per-article LLM summary.
- `custom_prompt`, `system_prompt`, `temperature`: summary prompt tuning.
- `api_name`: LLM provider name (keys are read from server config).
- `keywords`: comma-separated keywords to store in the Media DB.
- `custom_titles`: optional titles for individual URLs.
- `custom_cookies`: list of cookie dicts (Playwright-style or `{name,value}` pairs).
- `user_agent`, `custom_headers`: per-request UA and headers.
- `crawl_strategy`: `best_first|best-first|bestfirst`.
- `include_external`: allow following off-domain links.
- `score_threshold`: float in [0.0, 1.0] for crawl scoring.
- `mode`: `persist` or `ephemeral`.

### `/ingest-web-content` (IngestWebContentRequest)

- `urls`: list of URLs (first URL is used for sitemap/recursive).
- `titles`, `authors`, `keywords`: per-URL metadata (optional).
- `scrape_method`: `individual|sitemap|url_level|recursive_scraping`.
- `url_level`, `max_pages`, `max_depth`: crawl bounds.
- `perform_analysis`: add `analysis` to results.
- `custom_prompt`, `system_prompt`: analysis prompt tuning.
- `use_cookies`: enable cookie parsing from `cookies`.
- `cookies`: JSON string of cookie dicts or list of dicts.
- `crawl_strategy`, `include_external`, `score_threshold`: crawl overrides.

Notes:
- Per-request API keys are not supported. The server uses provider keys from config.
- `perform_translation`, `perform_chunking`, and `overwrite_existing` are present in the schema but are currently placeholders in this endpoint.

## What gets stored (persist mode)

When `mode: "persist"` is used, scraped pages are stored as `web_document` records in the Media DB:

- `content`: extracted text (with crawl metadata for recursive runs).
- `analysis_content`: summary output if enabled.
- `keywords`: stored as tags.
- `safe_metadata`: includes URL, title, author, date, crawl depth, parent URL, and crawl score.
- Chunking: the pipeline generates sentence-based hierarchical chunks for FTS.

## Tuning via config

Edit `tldw_Server_API/Config_Files/config.txt` or set env vars.

### Web scraper defaults

Section `[Web-Scraper]`:

- `web_scraper_default_backend`: `auto|curl|httpx|playwright`
- `web_scraper_ua_mode`: `fixed|rotate`
- `web_scraper_stealth_playwright`: enable stealth mode when available
- `web_scraper_respect_robots`: honor robots.txt (default true)
- `web_scraper_retry_count`: retry count for fetch failures
- `custom_scrapers_yaml_path`: custom scraper routing rules file

Advanced (add these keys to `[Web-Scraper]` if needed):

- `max_rps`, `max_rpm`, `max_rph`: rate limiting
- `max_concurrent`: max concurrent workers
- `connector_limit`, `connector_limit_per_host`: connection pool limits

### Crawl defaults and scoring

- `web_crawl_strategy` (env `WEB_CRAWL_STRATEGY`)
- `web_crawl_include_external` (env `WEB_CRAWL_INCLUDE_EXTERNAL`)
- `web_crawl_score_threshold` (env `WEB_CRAWL_SCORE_THRESHOLD`)
- `web_crawl_max_pages` (env `WEB_CRAWL_MAX_PAGES`)
- `web_crawl_allowed_domains`, `web_crawl_blocked_domains`
- `web_crawl_enable_keyword_scorer`, `web_crawl_keywords`
- `web_crawl_enable_domain_map`, `web_crawl_domain_map`

### Egress and HTTP policy

Outbound HTTP requests are gated by the central egress policy:

- `EGRESS_ALLOWLIST`, `EGRESS_DENYLIST`
- `PROXY_ALLOWLIST`
- `HTTP_*` timeouts and retry settings (see `tldw_Server_API/Config_Files/README.md`)

## Custom scrapers (per-domain rules)

Custom scraper rules let you override backend choice, headers, and extraction strategies per domain.

### Create a rules file

1. Copy `tldw_Server_API/Config_Files/custom_scrapers.example.yaml` to `tldw_Server_API/Config_Files/custom_scrapers.yaml`.
2. Add domain rules under `domains:`.
3. Ensure `[Web-Scraper] custom_scrapers_yaml_path` points at the file (or leave default).
4. Test with `POST /api/v1/media/process-web-scraping`.

Edits to the YAML apply on the next request; changing the file path requires a restart.

### Matching order

- Exact domain match first (`example.com`).
- Wildcard match next (`*.example.com`).
- If `url_patterns` is present, the rule applies only when a regex matches the URL; otherwise it falls back to defaults.

### Rule fields (common)

- `backend`: `auto|curl|httpx|playwright`
- `handler`: extraction handler (allowlist enforced, defaults to `handle_generic_html`)
- `ua_profile`: UA profile name from `ua_profiles.py`
- `impersonate`: curl-cffi impersonation profile (optional)
- `extra_headers`: request headers applied to the scrape
- `cookies`: cookie map injected for the domain
- `respect_robots`: override robots.txt behavior for this rule
- `url_patterns`: regex list to constrain when the rule applies
- `strategy_order`: extraction strategy order (e.g. `jsonld`, `schema`, `regex`, `llm`, `cluster`, `trafilatura`)
- `schema_rules`: XPath/CSS rules for schema-based extraction
- `llm_settings`, `regex_settings`, `cluster_settings`: advanced extractors (optional)
- `proxies`: per-domain proxy map (subject to `PROXY_ALLOWLIST`)

Notes:
- Invalid handler paths are replaced with the safe default handler.
- Invalid `url_patterns` entries are ignored.
- Unknown keys are dropped during validation.

### Example: force Playwright for a JS-heavy site

```yaml
domains:
  app.example.com:
    backend: playwright
    handler: tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html
    ua_profile: chrome_120_win
    respect_robots: true
```

### Example: wildcard + URL pattern + schema rules

```yaml
domains:
  "*.substack.com":
    backend: curl
    handler: tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html
    ua_profile: chrome_120_win
    impersonate: chrome120
    url_patterns:
      - ".*\\?output=1$"
    strategy_order: [schema, trafilatura]
    schema_rules:
      title_xpath: "//article//h1"
      content_xpath: "//article//p"
    extra_headers:
      Referer: https://www.google.com
    respect_robots: true
```

### Testing tips

- Start with a single URL and `scrape_method: "Individual URLs"`.
- Enable debug logging to see rule selection and backend choice.
- If a rule is not applied, double-check domain match and `url_patterns`.

## Cookies and authenticated pages

You can provide cookies per request or store them for a domain:

- Per request: `custom_cookies` in `/process-web-scraping`
- Per request (JSON string): `use_cookies` + `cookies` in `/ingest-web-content`
- Stored cookies: `POST /api/v1/web-scraping/cookies/{domain}` (if enabled)

Cookie format: list of dicts with `name` and `value` (extra fields like `domain`, `path`, `expires` are accepted).

## Troubleshooting

- JS-heavy pages return empty content: install Playwright and set backend to `playwright` or add a custom scraper rule.
- 403/blocked content: lower rate limits, rotate UA, add cookies, or use the scraper analyzers (`Docs/Scraper_Analyzers.md`).
- Recursive crawl yields few pages: increase `max_pages`, lower `score_threshold`, or allow external links.
- Advanced crawl overrides rejected: ensure the enhanced service is available (fallback path does not support advanced flags on recursive crawl).

## Related docs

- `Docs/Scraper_Analyzers.md`
- `tldw_Server_API/Config_Files/Prompts/webscraping.prompts.yaml`
- `Docs/Design/WebScraping.md`
- `Docs/Product/Collections_Feeds_Ingestion.md`
- `Docs/Design/WebSearch.md`
