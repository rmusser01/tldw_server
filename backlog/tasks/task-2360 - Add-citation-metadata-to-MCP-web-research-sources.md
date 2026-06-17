---
id: TASK-2360
title: Add citation metadata to MCP web.research sources
status: Done
updated_date: '2026-06-14'
labels:
- mcp
- tools
- web
- research
dependencies:
- TASK-2356
references:
- Docs/Design/MCP_Web_Research_Tool_Design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enrich each web.research source with citation-oriented metadata: a 1-based rank, the domain, the search provider's own per-result metadata (author/date/source/...), and, when fetched, the final URL, content type, and a retrieval timestamp.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every source carries `rank` (1-based, sequential over search results), `domain` (host of the result url), and `search_metadata` (the provider's per-result metadata dict, `{}` when absent) — available whether or not the source was fetched.
- [x] #2 Fetched sources additionally carry `final_url` (post-redirect), `content_type`, and `retrieved_at` (ISO-8601 UTC); unfetched sources omit those retrieval fields.
- [x] #3 The enrichment is additive (existing source fields and bundle shape unchanged); all existing web.research tests stay green.
- [x] #4 New tests cover citation fields present, 1-based sequential rank, and that unfetched sources omit retrieval fields; ruff/compileall/bandit clean.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
`WebResearchModule._assemble_sources` now emits, per source: `rank` (incremented over dict entries), `domain` (`_safe_host(url)`), and `search_metadata` (carried through from the search result's `metadata`, previously dropped). When the sub-fetch succeeded it also sets `final_url` (fetch `final_url` or url), `content_type`, and `retrieved_at` (`datetime.now(UTC).isoformat()`); unfetched/denied sources keep `reason_code` and omit the retrieval fields. Purely additive — existing keys (title/url/snippet/fetched/status_code/content) unchanged.

Tests: +3 (citation fields present incl. passed-through provider metadata; 1-based sequential rank; unfetched omits retrieval fields); 30 web.research tests green. ruff/compileall/bandit clean.

This completes the deferred web-tools follow-ups (per-domain rate limiting #2357, response caching #2358, citation metadata). Remaining nice-to-haves: optional shared/process-global limiter+cache from gateway settings; conditional revalidation (ETag/Last-Modified).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
