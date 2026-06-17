---
id: TASK-2357
title: Harden MCP web tools with a shared WebToolBase
status: Done
updated_date: '2026-06-14'
labels:
- mcp
- tools
- web
- refactor
dependencies:
- TASK-2354
- TASK-2355
- TASK-2356
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extract a shared `WebToolBase` for the three read-only web MCP tools (web.fetch, web.search, web.research) to remove the duplicated `sanitize_input` override, execution eval metadata, structured error-result shape, profile-id context reader, and domain-list validator. Behavior-preserving refactor guarded by the existing web-tool test suites.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A `web_tool_base.py` provides `WebToolBase(BaseModule)` + shared `WebToolError`, with one implementation each of the permissive `sanitize_input` override, `_structured_error`, `_eval_metadata`, `_profile_id_from_context_metadata`, and `_validate_domain_list`.
- [x] #2 web.fetch, web.search, and web.research extend `WebToolBase`, set the `_ACTION_FAMILY`/`_RESULT_KIND`/`_TOOL_PROMPT_VERSION` class attributes, and drop their per-module copies of the shared helpers (incl. the three private `_WebXError` classes, now unified as `WebToolError`).
- [x] #3 The refactor is behavior-preserving: all existing web-tool, registration, and preset tests stay green (111) and a new `test_web_tool_base.py` (6) directly covers the base.
- [x] #4 ruff/compileall/bandit clean; no public result-shape or error-code changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
New `tldw_Server_API/app/core/MCP_unified/modules/implementations/web_tool_base.py`:
- `WebToolError(reason_code, message)` — single control-flow error replacing `_WebFetchError`/`_WebSearchError`/`_WebResearchError`.
- `WebToolBase(BaseModule)`:
  - `sanitize_input` permissive override (strip NUL/control chars only via shared `CONTROL_CHARS_RE`, depth guard) — the MCP protocol sanitizes every tool call before execute, so the SQL denylist would otherwise reject `--`/`/*`/punycode.
  - `_structured_error(tool_name, reason_code, message, *, context, truncated=False, **extra)` — extra non-None fields (e.g. web.fetch `status_code`) merge into the result.
  - `_eval_metadata` driven by class attrs `_ACTION_FAMILY` / `_RESULT_KIND` / `_TOOL_PROMPT_VERSION` (lazy import of `build_execution_eval_metadata`).
  - `_profile_id_from_context_metadata`, `_validate_domain_list` (empty list → None).
  - Exports `CONTROL_CHARS_RE` so web.fetch reuses it for URL control-char rejection.

Each module now: extends `WebToolBase`, sets the three class attrs, deletes its duplicated helpers + private error class + control-char/depth constants, and uses `WebToolError` / `self._structured_error`. web.fetch keeps its fetch/redirect/extraction logic; web.search keeps normalization; web.research keeps orchestration.

Net: ~150 lines of duplication removed; eval metadata, sanitize semantics, and error shape now have a single source of truth.

Tests: 117 green (6 new base + 29 fetch + 28 search + 21 research + 2×3 registration + 27 presets, overlapping). ruff/compileall/bandit clean; `git diff --check` clean.

Deferred (next web-tools hardening slice): per-domain rate limiting (shared limiter consulted in web.fetch per hop), response caching, and richer citation metadata on web.research sources.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
