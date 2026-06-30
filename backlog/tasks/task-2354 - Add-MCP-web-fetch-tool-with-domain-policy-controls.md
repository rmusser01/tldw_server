---
id: TASK-2354
title: Add MCP web.fetch tool with domain policy controls
status: Done
updated_date: '2026-06-12'
labels:
- mcp
- tools
- web
- policy
references:
- Docs/Design/MCP_Web_Fetch_Tool_Design.md
dependencies:
- TASK-2344
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a built-in MCP `web.fetch` tool that retrieves a single user-specified URL, applies the centralized outbound (SSRF/egress) policy, and returns bounded extracted content. The tool takes a `url` argument so it is automatically governed by the gateway's existing domain permission subjects (`WebFetch(<domain>)` allow/deny/ask rules) without new policy plumbing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A `web.fetch` MCP tool fetches an http/https URL and returns `{ok, url, final_url, status_code, content_type, title, format, content, bytes_fetched, truncated, eval}`; non-http(s) schemes return a structured `invalid_url` error.
- [x] #2 Every fetch passes through `decide_web_outbound_policy` (SSRF/egress, optional robots); a denied target returns `outbound_policy_denied` and performs no network request beyond policy evaluation.
- [x] #3 Responses are bounded by `max_bytes` (truncation flagged) and `timeout_seconds`, both clamped to safe maxima; HTML is extracted to markdown/text via trafilatura, text/plain and json pass through, other content types return `empty_content`.
- [x] #4 The module is registered only when `MCP_ENABLE_WEB_FETCH_MODULE` is set, the HTTP client is injectable for tests (no network in unit tests), and the `deep-researcher` gateway preset enables `web.fetch`.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
In-server module `tldw_Server_API/app/core/MCP_unified/modules/implementations/web_fetch_module.py` (`WebFetchModule(BaseModule)`, single `web.fetch` tool). The implementation lives in-server (not the standalone `mcp_unified/` package) so it may import `Web_Scraping.outbound_policy.decide_web_outbound_policy` and trafilatura without adding heavy deps to the standalone gateway. Domain allow/deny/ask is NOT re-implemented: the gateway already extracts a `domain` subject from the `url` argument (`mcp_unified/profiles/subjects.py`) and compiles Claude-style `WebFetch(<domain>)` rule specifiers into domain rules (`permission_rules.py`), so runtime domain policy applies automatically.

Security/bounds: scheme allow-list (http/https → else `invalid_url`); mandatory `decide_web_outbound_policy(... source="mcp.web_fetch", stage="web.fetch")` SSRF/egress gate before any fetch (denied → `outbound_policy_denied`, no network call); `max_bytes` (default 1MB, max 5MB) with `truncated` flag; `timeout_seconds` (default 15, max 30); HTML → trafilatura markdown/txt with a regex tag-strip fallback so extraction is deterministic; text/plain·markdown·json·xml pass through; other content types → `empty_content`; status ≥400 → `fetch_failed`; client exceptions → `fetch_failed`.

Testability via injectable `WebFetchHttpClient` Protocol + `WebFetchResponse` dataclass (default `HttpxWebFetchClient` streams with a byte cap); outbound policy monkeypatched at the module reference in tests — no network in unit tests.

Registration: optional env-flag block in `server.py` gated by `MCP_ENABLE_WEB_FETCH_MODULE` (disabled by default, mirroring git/sandbox), id `web_fetch`, department `research`. Gateway preset: `_WEB_READ_TOOLS = ["web.fetch"]` wired into the `deep-researcher` preset's new tooling_metadata_document (the only preset carrying `external_network`/`research.web`).

TDD evidence:
- RED: `pytest test_web_fetch_module.py` → collection error (module absent), then 16/17 (one test-harness `RequestContext` arg bug) before module existed.
- GREEN: `test_web_fetch_module.py` 17 passed; `test_web_fetch_module_registration.py` 2 passed; `test_profile_presets.py` 27 passed (incl. new `test_deep_researcher_enables_web_fetch`).
- Regression: `test_federation_shell_contracts.py`, `test_storage_contracts.py`, `test_profile_permission_rules.py`, `test_profile_policy_decisions.py` → 91 passed.
- ruff (new files) clean; `compileall` rc=0; `bandit -r web_fetch_module.py` no findings; `git diff --check` clean. (Pre-existing SIM300 findings in `test_profile_presets.py` left at dev baseline.)

Deferred: streaming/range fetch, per-domain rate limiting, caching, and a companion `web.search` tool (separate backlog item).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
