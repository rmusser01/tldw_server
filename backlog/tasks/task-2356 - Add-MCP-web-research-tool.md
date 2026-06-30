---
id: TASK-2356
title: Add MCP web.research tool
status: Done
updated_date: '2026-06-14'
labels:
- mcp
- tools
- web
- research
references:
- Docs/Design/MCP_Web_Research_Tool_Design.md
dependencies:
- TASK-2354
- TASK-2355
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a built-in read-only MCP `web.research` tool that composes the merged `web.search` and `web.fetch` tools: run one search query, then fetch + extract the top N results into a single bounded research bundle. Per-result fetches inherit web.fetch's per-hop outbound policy; individual fetch failures are tolerated and recorded per source.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A `web.research` MCP tool searches a query then fetches the top `fetch_top_n` results, returning `{ok, query, engine, result_count, fetched_count, truncated, sources:[{title,url,snippet,fetched,status_code?,content?,reason_code?}], eval}` with sources in search order.
- [x] #2 It composes WebSearchModule + WebFetchModule via their execute_tool contracts (no re-implemented search/fetch/egress); a search-stage failure surfaces as the tool error, while individual fetch failures are tolerated (`fetched: false` + the fetch reason_code) and do not fail the call.
- [x] #3 Bounds: `max_results` 1..25 (search result_count), `fetch_top_n` 0..10 clamped to max_results, bounded fetch concurrency (3); `truncated` is true when search truncated OR any fetched source was clipped; results without a url are skipped for fetching.
- [x] #4 The module is registered only when `MCP_ENABLE_WEB_RESEARCH_MODULE` is set, the composed modules are injectable for tests (no network/config), a permissive `sanitize_input` override allows `--`/punycode queries, and the `deep-researcher` preset enables `web.research`.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
In-server module `tldw_Server_API/app/core/MCP_unified/modules/implementations/web_research_module.py` (`WebResearchModule(BaseModule)`, single `web.research` tool). Composition over re-implementation: holds a `WebSearchModule` + `WebFetchModule` (injectable via `search_module`/`fetch_module` ctor kwargs; defaults lazily construct the real modules) and drives them through `execute_tool`. So every fetched URL re-runs web.fetch's per-hop SSRF/egress + redirect policy, and search provider egress runs inside web.search — no new egress surface.

Flow: validate/clamp → `web.search {query, result_count: max_results, engine?, site_*}` → on error surface its reason_code/message → take first `fetch_top_n` results with a url → `asyncio.gather` fetch with `Semaphore(3)` → assemble sources in search order. Engine-value validation is DELEGATED to web.search (web.research only checks structural types/bounds), so an invalid engine surfaces as web.search's `invalid_engine`. Permissive `sanitize_input` override (strip control chars only) mirrors web.search/web.fetch since the protocol sanitizes before execute.

Registration: optional env-flag block in `server.py` gated by `MCP_ENABLE_WEB_RESEARCH_MODULE` (off by default), id `web_research`, department `research`. Preset: `_WEB_READ_TOOLS = ["web.fetch", "web.search", "web.research"]`, enabled in `deep-researcher`.

TDD: RED `pytest test_web_research_module.py` → collection error (module absent). GREEN: 21 module + 2 registration tests; 107 with web search/fetch/preset regression. ruff/compileall/bandit clean.

Deferred to the web-tools hardening slice (task TBD): extract a shared `_WebToolBase` (sanitize_input override, eval helpers, control-char constants, domain-list validator) across web.fetch/web.search/web.research; per-domain rate limiting; response caching; richer citation metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
