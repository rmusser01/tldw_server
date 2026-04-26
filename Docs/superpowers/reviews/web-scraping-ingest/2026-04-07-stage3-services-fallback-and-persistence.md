# Stage 3 Services Fallback and Persistence

## Scope
- Service-layer review only.
- Read path focused on `process_web_scraping_task()`, `ingest_web_content_orchestrate()`, enhanced-service dispatch, fallback gating, and mode-specific storage behavior.
- No application-code or test changes were made.

## Code Paths Reviewed
- `tldw_Server_API/app/services/web_scraping_service.py`
- `tldw_Server_API/app/services/enhanced_web_scraping_service.py`
- `tldw_Server_API/app/services/ephemeral_store.py`
- `tldw_Server_API/app/core/DB_Management/media_db/api.py`

## Tests Reviewed
- `tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py`
- `tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py`
- `tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py`

## Validation Commands
```bash
rg -n "async def process_web_scraping_task|async def ingest_web_content_orchestrate|_store_ephemeral|_store_persistent|_legacy_web_scraping_fallback_enabled|_collect_fallback_unsupported_controls" \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py

sed -n '133,940p' tldw_Server_API/app/services/web_scraping_service.py
sed -n '1,980p' tldw_Server_API/app/services/enhanced_web_scraping_service.py

source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py \
  tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py

rg -n "class EphemeralStorage|store_data|get_data|managed_media_database|get_media_repository" \
  tldw_Server_API/app/services/ephemeral_store.py \
  tldw_Server_API/app/core/DB_Management/media_db/api.py

sed -n '1,260p' tldw_Server_API/app/services/ephemeral_store.py
sed -n '1,260p' tldw_Server_API/app/core/DB_Management/media_db/api.py
```

Result: all 13 targeted tests passed.

## Service Control Flow
- `process_web_scraping_task()` in `web_scraping_service.py` normalizes `crawl_strategy` and `score_threshold`, computes a coarse priority, and forwards the request to the enhanced service.
- `WebScrapingService.process_web_scraping_task()` resolves config defaults from `load_and_log_configs()` with explicit request overrides winning for `max_pages`, `crawl_strategy`, `include_external`, and `score_threshold`.
- Enhanced dispatch generates a `task_id`, routes by scrape method, annotates the result with `crawl_config`, then diverges by mode:
  - `mode="ephemeral"` calls `_store_ephemeral()` and returns an ephemeral handle plus a preview payload.
  - `mode="persist"` calls `_store_persistent()` and returns stored media IDs, article counts, and the generated `task_id`.
- `ingest_web_content_orchestrate()` uses the same service entrypoint for URL-level and recursive ingest, hardcodes `mode="ephemeral"` for those friendly-ingest paths, then maps returned `summary` fields into `analysis`.
- Enhanced-service failures may trigger the legacy fallback depending on environment gating; successful fallback payloads include `engine="legacy_fallback"` plus `fallback_context`, while disabled fallback raises a contract `400`.

## Failure-Seam Checklist
- `input normalization`: wrapper-level validation rejects bad `crawl_strategy` and out-of-range `score_threshold` before enhanced dispatch.
- `fallback eligibility`: legacy fallback is opt-in and rejects unsupported advanced controls instead of silently degrading them.
- `degraded-control handling`: fallback applies a manual `max_pages` cap for `Sitemap` and `URL Level` results and records that degradation in `fallback_context["degraded_controls_applied"]`.
- `metadata persistence`: enhanced `persist` mode normalizes crawl traversal metadata and writes it into `safe_metadata`.
- `error translation`: enhanced-service noncritical failures become `HTTPException(500)` inside the enhanced service; the outer wrapper either falls back or returns an explicit `400` if fallback is disabled or unsupported.

## Findings
- No Stage 3 contract failure was reproduced in the targeted service-layer tests.
- Confirmed: config-default versus request-override precedence is explicit and observable through the returned `crawl_config` payload in [`enhanced_web_scraping_service.py:139`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/enhanced_web_scraping_service.py#L139) and [`enhanced_web_scraping_service.py:266`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/enhanced_web_scraping_service.py#L266). This matches the passing precedence tests in `test_crawl_config_precedence.py`.
- Confirmed: legacy fallback does not silently ignore unsupported advanced controls. `_collect_fallback_unsupported_controls()` marks `custom_headers`, non-default crawl strategy, `include_external`, and active `score_threshold` as unsupported in [`web_scraping_service.py:70`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/web_scraping_service.py#L70), and the wrapper converts that into a `400` in [`web_scraping_service.py:299`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/web_scraping_service.py#L299). This matches the passing fallback-contract tests in `test_legacy_fallback_behavior.py`.
- Confirmed: fallback degradation is explicit rather than silent for page-count control. The legacy path truncates `Sitemap` and `URL Level` results and records `"max_pages"` in `fallback_context["degraded_controls_applied"]` in [`web_scraping_service.py:365`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/web_scraping_service.py#L365).
- Confirmed: enhanced `persist` mode uses `managed_media_database(...)` for lifecycle ownership in [`enhanced_web_scraping_service.py:589`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/enhanced_web_scraping_service.py#L589) and the context manager guarantees `close_connection()` on exit in [`api.py:105`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/core/DB_Management/media_db/api.py#L105).
- Confirmed: enhanced `persist` mode carries crawl traversal fields (`crawl_depth`, `crawl_parent_url`, `crawl_score`) into `safe_metadata`, with best-effort numeric normalization before write, in [`enhanced_web_scraping_service.py:627`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/enhanced_web_scraping_service.py#L627) and [`enhanced_web_scraping_service.py:683`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/enhanced_web_scraping_service.py#L683). This matches the passing persistence test in `test_persistence_crawl_metadata.py`.
- Confirmed: enhanced ephemeral storage is process-local and TTL-bound. `_store_ephemeral()` writes a wrapper object into `ephemeral_storage` in [`enhanced_web_scraping_service.py:555`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/enhanced_web_scraping_service.py#L555), and the storage implementation prunes expired entries on both write and read in [`ephemeral_store.py:105`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/ephemeral_store.py#L105) and [`ephemeral_store.py:134`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/ephemeral_store.py#L134).

## Probable Risks
- The earlier Stage 2 custom-header `503` is consistent with a narrow service seam where custom headers require the enhanced scraper path. Enhanced dispatch forwards `custom_headers` into scraper calls in [`enhanced_web_scraping_service.py:323`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/enhanced_web_scraping_service.py#L323), [`enhanced_web_scraping_service.py:377`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/enhanced_web_scraping_service.py#L377), and [`enhanced_web_scraping_service.py:489`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/enhanced_web_scraping_service.py#L489), but the legacy fallback explicitly treats `custom_headers` as unsupported in [`web_scraping_service.py:89`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-ingest-review/tldw_Server_API/app/services/web_scraping_service.py#L89). This review did not reproduce the exact endpoint-level `503`, so this remains a bounded inference rather than a confirmed defect.

## Coverage Gaps
- This pass did not audit the underlying Playwright scraper, queue internals, or endpoint-layer translation around the Stage 2 `503` custom-header failure.
- This pass did not verify on-disk lifecycle for recursive progress files such as `scrape_progress_<id>.json`.
- This pass did not perform a broader Media DB audit beyond confirming context-manager ownership and repository adaptation.

## Improvements
- Add a targeted service or endpoint test that fixes the exact expected status code and payload when `custom_headers` are supplied but the enhanced scraper is unavailable.
- Add a service test that asserts `task_id` propagation in successful `persist` responses for each enhanced crawl mode.
- Add a persistence-path test covering traversal metadata sourced from both top-level article fields and nested `metadata` fields across all supported crawl methods.

## Exit Note
- Confirmed behavior for request-vs-config precedence, fallback gating/degraded controls, and crawl metadata persistence in `persist` mode.
- Kept the earlier Stage 2 custom-header `503` as a probable-risk note only; this Stage 3 pass did not reproduce or localize the exact endpoint-level failure.
