---
id: TASK-2359
title: Add response caching to MCP web.fetch
status: Done
updated_date: '2026-06-14'
labels:
- mcp
- tools
- web
- performance
references:
- Docs/Design/MCP_Web_Fetch_Response_Cache.md
dependencies:
- TASK-2354
- TASK-2358
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an opt-in in-memory TTL+LRU response cache to web.fetch so repeated fetches of the same resource (common inside a web.research bundle or across closely-spaced calls) skip the network.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A `ResponseCache` provides thread-safe TTL + LRU get/put keyed via `make_cache_key(url, format, max_bytes)`, with injectable clock; `ttl_seconds<=0` or `max_entries<=0` disables it.
- [x] #2 `WebFetchModule` (opt-in `response_cache=None`) returns a cache hit as `{...cached: true}` with no network request; fresh successful results are tagged `cached: false` and stored; errors are never cached.
- [x] #3 The cache key distinguishes format and max_bytes; expired entries miss and are dropped; LRU eviction past `max_entries`. web.research can de-dupe sub-fetches by supplying its WebFetchModule a cache.
- [x] #4 Unit tests for the cache (put/get, miss, key fields, TTL expiry, LRU, disabled) and web.fetch integration (2nd fetch served from cache, format miss, errors not cached); existing web suites stay green.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
New `web_cache.py`: `ResponseCache` (OrderedDict TTL+LRU, threading.Lock, injectable clock, default 300s/256 entries; `enabled` False when ttl/max<=0) + `make_cache_key(url, fmt, max_bytes)`.

`WebFetchModule.__init__` takes `response_cache=None` (opt-in, off by default since caching trades freshness). In `execute_tool` after validation: lookup by `(requested_url, fmt, max_bytes)` → hit returns `{**cached, "cached": True}` with no network (gateway already enforced permission rules; a hit hits no network so SSRF/egress is moot). Fresh successes are tagged `cached: False` and `put` into the cache; errors are not cached.

web.research composes a WebFetchModule, so passing it a cache de-dupes repeated sub-fetches across a bundle (no web.research change needed).

Tests: `test_web_cache.py` (6) + 3 web.fetch integration; 96 web tests green. ruff/compileall/bandit clean.

Deferred: web.research citation metadata; optional shared/process-global cache from gateway settings; conditional revalidation (ETag/Last-Modified).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
