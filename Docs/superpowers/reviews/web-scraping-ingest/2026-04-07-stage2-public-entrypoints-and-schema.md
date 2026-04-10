# Stage 2 Public Entrypoints and Schema

## Scope
- Public request boundary review for `/api/v1/media/ingest-web-content` and `/api/v1/media/process-web-scraping`.
- Request-model inspection for `IngestWebContentRequest` and `WebScrapingRequest`.
- Route-to-service glue inspection for `ingest_web_content_orchestrate()` and `process_web_scraping_task()`.

## Code Paths Reviewed
- `tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py`
- `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py`
- `tldw_Server_API/app/api/v1/schemas/media_request_models.py`
- `tldw_Server_API/app/services/web_scraping_service.py`

## Tests Reviewed
- `tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py`
- `tldw_Server_API/tests/Web_Scraping/test_ingest_cookie_parsing.py`
- `tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py`
- `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
- `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`

## Validation Commands
- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py tldw_Server_API/tests/Web_Scraping/test_ingest_cookie_parsing.py tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`
- Result: `15 passed, 1 failed`

## Public Contract Table
| Route | Request model | Key coercions or validators | Downstream service | Error translation notes |
| --- | --- | --- | --- | --- |
| `/api/v1/media/ingest-web-content` | `IngestWebContentRequest` | `scrape_method` binds to `ScrapeMethod`; `max_pages` enforces `ge=1`; route rejects empty `urls`; orchestrator parses `cookies` into a cookie list and raises `400` on malformed payloads | `ingest_web_content_orchestrate()` | Route preserves downstream `HTTPException` and wraps unexpected failures in `500 Failed to ingest web content` |
| `/api/v1/media/process-web-scraping` | `WebScrapingRequest` | `max_pages` enforces `ge=1`; service normalizes `crawl_strategy` aliases to `best_first` and validates `score_threshold` into `[0.0, 1.0]`; typed forwarding for `custom_headers`, `custom_cookies`, and `mode` | `process_web_scraping_task()` | Route preserves downstream `HTTPException` and wraps unexpected failures in `500 Web scraping failed due to an internal error.` |

## Findings
1. **Medium | High | `/api/v1/media/process-web-scraping` end-to-end custom-header contract**
   - Files: `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py:24`, `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py:25`, `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py:63`
   - Evidence: failing test
   - Why it matters: the public route currently fails with `503 Service Unavailable` in the scoped custom-header integration test before the patched task result can be returned, so the endpoint-level custom-header forwarding contract is not presently proven end-to-end.
   - Recommended fix direction: trace the `503` dependency or pre-route failure in the request path used by `test_process_web_scraping_endpoint_receives_custom_headers` and restore a `200` response so the forwarding assertions can execute.

## Validated Behaviors
- Cookie parsing for friendly ingest preserves `400` semantics for malformed JSON, invalid cookie object types, and non-dict list members.
  Files: `tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py:66`, `tldw_Server_API/app/services/web_scraping_service.py:606`
  Evidence: runtime proof from `tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py` and `tldw_Server_API/tests/Web_Scraping/test_ingest_cookie_parsing.py`
- `process-web-scraping` preserves downstream `400`s for invalid `crawl_strategy` values and invalid `score_threshold` values instead of translating them to generic `500`s.
  Files: `tldw_Server_API/app/services/web_scraping_service.py:187`, `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py:91`
  Evidence: runtime proof from `tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py`
- Usage logging for the public `/api/v1/media/process-web-scraping` route is covered by a passing request-path test.
  Files: `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py:47`
  Evidence: runtime proof from `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`

## Coverage Gaps
- The failing async integration test prevents a clean runtime confirmation that `/api/v1/media/process-web-scraping` reaches the patched task and forwards `custom_headers` under the current test fixture.
- `IngestWebContentRequest` itself does not perform cookie-shape validation in the schema; cookie coercion remains service-layer behavior in `ingest_web_content_orchestrate()`.

## Improvements
- Move cookie parsing and shape validation closer to `IngestWebContentRequest` if the project wants schema-level `422`s instead of service-level `400`s for malformed cookie payloads.
- Add an endpoint-level regression test that records the actual `503` response body or dependency source when the custom-header path fails, so the failure stays attributable.

## Exit Note
- Verified cookie parsing behavior for friendly ingest.
- Verified `process-web-scraping` preserves `400`s for invalid crawl strategy.
- Verified usage logging coverage; custom-header forwarding coverage is currently blocked by a reproducible `503` in `test_process_web_scraping_endpoint_receives_custom_headers`.
