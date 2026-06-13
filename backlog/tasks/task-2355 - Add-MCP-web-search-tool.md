---
id: TASK-2355
title: Add MCP web.search tool
status: Done
updated_date: '2026-06-12'
labels:
- mcp
- tools
- web
- research
references:
- Docs/Design/MCP_Web_Search_Tool_Design.md
dependencies:
- TASK-2354
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a built-in read-only MCP `web.search` tool that runs a query against a configured multi-provider web search backend and returns a bounded list of normalized results, complementing `web.fetch`. The provider call enforces the centralized outbound (SSRF/egress) policy; the tool is gated by the `external_network`/`research.web` preset capability and tool-level permission rules.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A `web.search` MCP tool runs a query against a configured provider and returns `{ok, engine, query, result_count, total_results_found, results:[{title,url,content,metadata}], eval}`; empty query and unsupported engine return structured `invalid_arguments`/`invalid_engine` errors without calling the backend.
- [x] #2 The provider call goes through `WebSearch_APIs.perform_websearch` (which enforces per-provider outbound policy); a `processing_error` mentioning the outbound policy maps to `outbound_policy_denied`, any other error/exception maps to `search_failed`.
- [x] #3 Results are bounded: list truncated to `result_count` (clamped to 25, default 10) and per-result `content` truncated to 4000 chars; site_whitelist/site_blacklist accept only lists of non-empty strings and are forwarded to the backend.
- [x] #4 The module is registered only when `MCP_ENABLE_WEB_SEARCH_MODULE` is set, the search backend is injectable for tests (no network/config in unit tests, blocking call offloaded via asyncio.to_thread), and the `deep-researcher` gateway preset enables `web.search`.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
In-server module `tldw_Server_API/app/core/MCP_unified/modules/implementations/web_search_module.py` (`WebSearchModule(BaseModule)`, single `web.search` tool), mirroring the `web.fetch` slice (task-2354). Default backend `PerformWebSearchBackend` delegates to `Web_Scraping.WebSearch_APIs.perform_websearch` (sync, multi-provider, returns a normalized `{results:[{title,url,content,metadata}], processing_error, error, total_results_found}` dict). Offloaded via `asyncio.to_thread`.

Governance: `web.search` takes a `query` (no `url`), so domain *subjects* are not extracted from the call — it is gated as a whole tool via tool-level rules and the `external_network`/`research.web` capability. Outbound (SSRF/egress) policy is enforced inside `perform_websearch._enforce_provider_outbound_policy`; the module maps an "outbound policy" processing_error to `outbound_policy_denied`.

Bounds: engine allow-list (google/duckduckgo/brave/kagi/serper/tavily/exa/firecrawl/searx/yandex/baidu), default `duckduckgo`; result_count default 10 / max 25 with defensive list truncation; per-result content capped at 4000 chars; site_whitelist/site_blacklist validated as list[str].

Registration: optional env-flag block in `server.py` gated by `MCP_ENABLE_WEB_SEARCH_MODULE` (off by default), id `web_search`, department `research`. Preset wiring: `_WEB_READ_TOOLS = ["web.fetch", "web.search"]` in `presets.py`, enabled in the `deep-researcher` tooling_metadata_document.

Stacked on the `web.fetch` branch (PR base = codex/mcp-webfetch-tool; retarget to dev after #2348 merges) because both touch `server.py` and `presets.py`.

TDD evidence:
- RED: `pytest test_web_search_module.py` → collection error (module absent).
- GREEN: `test_web_search_module.py` 19 passed; `test_web_search_module_registration.py` 2 passed; `test_profile_presets.py` 27 passed (incl. updated `test_deep_researcher_enables_web_tools`); web.fetch suites still green (67 combined).
- ruff (new/touched files) clean; `compileall` rc=0; `bandit -r web_search_module.py` no findings; `git diff --check` clean.

Deferred: query sub-generation/aggregation (the heavier `generate_and_search`/`aggregate_results` pipeline), result re-ranking, and per-result fetch+extract chaining with `web.fetch`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
