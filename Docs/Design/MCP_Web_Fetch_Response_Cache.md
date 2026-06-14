# MCP `web.fetch` Response Cache

Status: Implementing (June 2026)
Owner: standalone MCP gateway backlog
Companion: `MCP_Web_Tools_Rate_Limiting.md`

## Goal

Cache successful `web.fetch` results so repeated fetches of the same resource —
common inside a `web.research` bundle or across closely-spaced calls — skip the
network. Opt-in (off unless a cache is supplied), since caching trades content
freshness.

## `ResponseCache` (`web_cache.py`)

In-memory, thread-safe **TTL + LRU** cache:

- `make_cache_key(url, fmt, max_bytes)` — the request inputs that determine the
  output. Different format or byte cap is a distinct entry.
- `get(key)` returns the value or `None` (miss/expired; expired entries are
  dropped on access); a hit is moved to most-recently-used.
- `put(key, value)` stores with the configured TTL and evicts the
  least-recently-used entry (`OrderedDict.popitem(last=False)`) past `max_entries`.
- `ttl_seconds <= 0` or `max_entries <= 0` disables the cache (`enabled` False).
- Injectable `clock` for deterministic tests. Defaults: 300 s TTL, 256 entries.

## Integration

`WebFetchModule.__init__` gains `response_cache: ResponseCache | None` (default
`None` → no caching). In `execute_tool`, after validation:

- **Lookup** by `(requested_url, format, max_bytes)`. A hit returns the stored
  result with `cached: true` and performs **no** network request. This is safe:
  the gateway already enforced the call's permission rules before `execute_tool`,
  and a cache hit hits no network (so SSRF/egress is moot).
- **Store** only successful results (`ok: true`), tagged `cached: false` when
  fresh. Errors are never cached.

`web.research` composes a `WebFetchModule`; supplying it a cache transparently
de-dupes repeated sub-fetches across a bundle.

## Result

Successful results now carry a `cached: bool` field. To enable in a deployment,
wire `WebFetchModule(..., response_cache=ResponseCache(...))`.

## Tests

- `test_web_cache.py` (6) — put/get, miss, key distinguishes format/max_bytes,
  TTL expiry (fake clock), LRU eviction, disabled.
- `test_web_fetch_module.py` — second identical fetch served from cache (client
  not called, `cached: true`); different format is a miss; errors not cached.

## Deferred

Richer citation metadata on `web.research`; an optional process-global / shared
cache wired from gateway settings; conditional revalidation (ETag/Last-Modified).
