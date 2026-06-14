---
id: TASK-2358
title: Add per-domain rate limiting to MCP web tools
status: Done
updated_date: '2026-06-14'
labels:
- mcp
- tools
- web
- hardening
references:
- Docs/Design/MCP_Web_Tools_Rate_Limiting.md
dependencies:
- TASK-2354
- TASK-2356
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a soft per-domain request-rate control to the read-only web MCP tools so a single web.fetch (and transitively web.research) instance cannot hammer one host. Distinct from the always-on SSRF/egress outbound policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A `DomainRateLimiter` provides a thread-safe in-memory sliding-window `try_acquire(domain) -> bool` keyed by host, with injectable clock, bounded domain tracking, and case-insensitive host keys; `max_requests <= 0`/`None` disables it.
- [x] #2 `WebFetchModule` consults the limiter per hop (after the outbound policy, before the client fetch); over-limit returns a structured `rate_limited` error and performs no network request; redirects count as hops.
- [x] #3 The limiter is on by default (generous 60/60s) and injectable; passing `DomainRateLimiter(max_requests=0)` disables it. web.research's sub-fetches inherit throttling via the composed WebFetchModule.
- [x] #4 Unit tests for the limiter (within/over limit, per-domain, window expiry, case-insensitive, disabled) and web.fetch integration tests (blocks 2nd same-domain fetch, per-domain independence, disabled allows many); existing web suites stay green.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
New `tldw_Server_API/app/core/MCP_unified/modules/implementations/web_rate_limit.py`: `DomainRateLimiter` — `dict[host] -> deque[timestamps]`, `threading.Lock`, `time.monotonic` (injectable `clock`), trailing-window eviction on each `try_acquire`, `_MAX_TRACKED_DOMAINS=1024` LRU-evicted (OrderedDict) cap, `enabled` property. Defaults 60 req / 60 s.

`WebFetchModule.__init__` takes `rate_limiter=None` → default-constructs an enabled limiter (on by default). In the redirect loop, after `decide_web_outbound_policy` passes and before `self._client.fetch`, `_safe_host(current_url)` is checked via `try_acquire`; over-limit → `_structured_error(... "rate_limited" ...)` (logged with host). web.research is unchanged: its composed `WebFetchModule` carries the limiter so sub-fetches throttle per-domain across a bundle.

To disable: `WebFetchModule(..., rate_limiter=DomainRateLimiter(max_requests=0))`.

Tests: `test_web_rate_limit.py` (7) + 3 web.fetch integration tests; 136 web/preset tests green. ruff/compileall/bandit clean.

Deferred: response caching (TTL/LRU) for web.fetch; richer citation metadata on web.research; optional process-global limiter wired from gateway settings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
