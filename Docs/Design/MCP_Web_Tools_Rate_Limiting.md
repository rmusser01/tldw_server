# MCP Web Tools — Per-Domain Rate Limiting

Status: Implementing (June 2026)
Owner: standalone MCP gateway backlog
Companions: `MCP_Web_Fetch_Tool_Design.md`, `MCP_Web_Research_Tool_Design.md`

## Goal

Add a soft, per-domain request-rate control to the read-only web tools so a
single `web.fetch` (and, transitively, `web.research`) instance cannot hammer one
host. This is politeness / basic abuse protection — distinct from the always-on
SSRF/egress outbound policy, which remains the hard security boundary.

## `DomainRateLimiter`

`web_rate_limit.py` — an in-memory, thread-safe **sliding-window** limiter keyed
by destination host:

- `try_acquire(domain) -> bool`: records a request; returns `False` when the host
  already has `max_requests` hits inside the trailing `window_seconds`.
- `max_requests <= 0` (or `None`) disables limiting (always allows) — the explicit
  opt-out.
- Injectable `clock` for deterministic tests; bounded domain tracking
  (`_MAX_TRACKED_DOMAINS`, LRU-evicted (OrderedDict)) so a long-lived limiter can't grow
  without bound.
- Defaults: 60 requests / 60 s per domain — enough headroom for normal
  research/redirect flows, low enough to curb a runaway loop.

## Integration

`WebFetchModule.__init__` gains `rate_limiter: DomainRateLimiter | None`; when
`None` a default (enabled) limiter is constructed, so the control is **on by
default**. In the redirect loop, after the outbound policy passes and **before**
each `client.fetch`, the hop's host is checked: over-limit → structured
`rate_limited` error (the request never goes out, and redirects count as hops).

`web.research` composes a `WebFetchModule`, so its sub-fetches share that
module's limiter and are throttled per-domain across the bundle automatically —
no `web.research` change required.

## Result / errors

A throttled call returns the standard `{ ok: false, reason_code: "rate_limited",
message, eval }`. To disable in a given deployment, wire
`WebFetchModule(..., rate_limiter=DomainRateLimiter(max_requests=0))`.

## Tests

- `test_web_rate_limit.py` (7) — within/over limit, per-domain isolation, window
  expiry (fake clock), case-insensitive host key, disabled via `0`/`None`.
- `test_web_fetch_module.py` — rate limit blocks the second same-domain fetch
  (client not called), different domains independent, disabled limiter allows many.

## Deferred (next slices)

- Response caching (TTL/LRU keyed by url+format+max_bytes) for `web.fetch`.
- Richer citation metadata on `web.research` sources.
- Optional shared/process-global limiter wired from gateway settings.
