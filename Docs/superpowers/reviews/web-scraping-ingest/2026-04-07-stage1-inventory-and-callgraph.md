# Stage 1 Inventory and Call Graph

## Scope
Create the review output directory, freeze the report structure, and record the initial Web_Scraping source/test inventory plus the recent-history baseline.

## Code Paths Reviewed
### Scope Snapshot
- `tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py`
- `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py`
- `tldw_Server_API/app/api/v1/schemas/media_request_models.py`
- `tldw_Server_API/app/services/web_scraping_service.py`
- `tldw_Server_API/app/services/enhanced_web_scraping_service.py`
- `tldw_Server_API/app/services/ephemeral_store.py`
- `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- `tldw_Server_API/app/core/Web_Scraping/scraper_router.py`
- `tldw_Server_API/app/core/Web_Scraping/filters.py`
- `tldw_Server_API/app/core/Web_Scraping/scoring.py`
- `tldw_Server_API/app/core/http_client.py`
- `tldw_Server_API/app/core/Security/egress.py`
- `tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py`
- `tldw_Server_API/tests/Web_Scraping/test_ingest_cookie_parsing.py`
- `tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py`
- `tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py`
- `tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py`
- `tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py`
- `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
- `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`
- `tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py`
- `tldw_Server_API/tests/Web_Scraping/test_router_validation.py`
- `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
- `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`
- `tldw_Server_API/tests/server_e2e_tests/test_web_scraping_workflow.py`

## Tests Reviewed
Stage 1 does not review test behavior in depth. This section freezes the targeted test inventory for later validation stages.

- `tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py`
- `tldw_Server_API/tests/Web_Scraping/test_ingest_cookie_parsing.py`
- `tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py`
- `tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py`
- `tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py`
- `tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py`
- `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
- `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`
- `tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py`
- `tldw_Server_API/tests/Web_Scraping/test_router_validation.py`
- `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
- `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`
- `tldw_Server_API/tests/server_e2e_tests/test_web_scraping_workflow.py`

## Validation Commands
- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate`
- `python -m pytest --version` -> `pytest 9.0.2`
- `python -m bandit --version` -> `No module named bandit`
- `rg --files ... | sort` -> scoped inventory captured in this report
- `git log --oneline -n 20 -- ...` -> recent churn baseline captured below

## Findings
- None recorded at scaffold time.

## Coverage Gaps
- Bandit is unavailable in the shared project venv, so security validation is limited to static review until that dependency is installed.

## Improvements
- Use the fixed stage template to capture subsequent evidence and findings without expanding the scope.

## Exit Note
- Stage 1 scaffold and baseline inventory captured; deeper review is deferred to later stages.

## Recent Churn Baseline
```text
91806e5c3 fix: address non-threaded pr review followups
8f0279df1 refactor: trim remaining media endpoint shim imports
b6dd1e7a3 fix media session wrapper listing and trim media endpoint shim imports
021980919 refactor: manage article extractor db lifecycle
f2fb4dba1 refactor: use managed media db helper in web scraping services
eb0852342 refactor: migrate more service callers to media db api
166fbc2fe refactor: migrate ingestion callers to media db api factory
41d4f59d4 refactor: expose media db api factory import path
c31c779a5 refactor: route workflow and scrape ingest through media api
c2ae77e49 refactor(compat): enforce runtime deprecation registry and fallback gate
d10e600db refactor(webscraping): remove cookie manager close_all shim
e2aeac723 refactor(websearch): remove deprecated simple breaker shim
4769c3dd2 docs(media): remove stale legacy shim wording
547f51e0c cleanup
7973e411d refactor(media): reduce legacy ingestion compatibility surface
40beb3561 bandit+personas
58312fd53 bandit fixes + UI/UX in webui/extension + Claims fixups
20bfc9cf4 bugfixes
7f2596979 fixes
d0654d0cf colors, themes + splashscreens
```

## Initial Ingest Call Graph
- `/api/v1/media/ingest-web-content` -> `ingest_web_content_orchestrate()` -> reachable scrape helpers
- `/api/v1/media/process-web-scraping` -> `process_web_scraping_task()` -> enhanced service or legacy fallback

## Final Review Output Shape
```markdown
## Findings
1. Severity: concise finding with file references and impact

## Open Questions
- only unresolved assumptions
```
