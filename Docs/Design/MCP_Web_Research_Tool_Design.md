# MCP `web.research` Tool Design

Status: Implementing (June 2026)
Owner: standalone MCP gateway backlog
Companions: `MCP_Web_Fetch_Tool_Design.md`, `MCP_Web_Search_Tool_Design.md`

## Goal

Add a built-in, read-only MCP tool, `web.research`, that **composes** the
already-merged `web.search` and `web.fetch` tools into one call: run a search
query, then fetch + extract the top N results into a single bounded research
bundle. It is the natural "do my web research" primitive for the
`deep-researcher` preset.

## Why composition (not a new pipeline)

`web.research` does not re-implement search or fetch. It instantiates a
`WebSearchModule` and a `WebFetchModule` (injectable for tests) and drives them
via their existing `execute_tool` contracts. This means:

- Every fetched URL re-runs the per-hop outbound (SSRF/egress) policy inside
  `web.fetch`, including redirect re-checks — no new egress surface.
- Search provider egress is enforced inside `web.search`.
- Bounds (byte caps, content truncation, result caps) and structured error
  codes are inherited from the two tools.

## Tool: `web.research`

Read-only. Arguments (additionalProperties: false):

| arg | type | default | bounds |
|---|---|---|---|
| `query` | string (required) | — | non-empty |
| `engine` | string | search default | search engine allow-list |
| `max_results` | integer | 5 | 1 .. 25 (search `result_count`) |
| `fetch_top_n` | integer | 3 | 0 .. 10, clamped to `max_results` |
| `format` | enum markdown\|text\|html | markdown | fetch format |
| `max_bytes` | integer | — | per-fetch byte cap (fetch bounds) |
| `site_whitelist` | string[] | — | forwarded to search |
| `site_blacklist` | string[] | — | forwarded to search |

Success result:

```
{
  "ok": true,
  "query": "...",
  "engine": "duckduckgo",
  "result_count": 5,        // results returned by search
  "fetched_count": 3,       // sources actually fetched+extracted ok
  "truncated": false,       // search truncated OR any source clipped
  "sources": [
    {
      "title": "...",
      "url": "https://...",
      "snippet": "<search result content>",
      "fetched": true,
      "status_code": 200,
      "content": "<extracted, bounded>"
    },
    { "title": "...", "url": "https://...", "snippet": "...",
      "fetched": false, "reason_code": "outbound_policy_denied" }
  ],
  "eval": { ... }
}
```

Error result: `{ ok: false, reason_code, message, eval }`.

Top-level reason codes (search stage): `invalid_arguments`, `invalid_engine`,
`outbound_policy_denied`, `search_failed`, `unknown_tool`. A search failure fails
the whole call; **individual fetch failures do not** — they are recorded on the
source with `fetched: false` and the fetch's `reason_code`.

## Behavior

1. Permissive `sanitize_input` override (strip control chars only; queries
   contain `--`/punycode), mirroring `web.search`/`web.fetch`.
2. Validate + clamp (`fetch_top_n` ≤ `max_results` ≤ 25; `fetch_top_n` ≤ 10).
3. `web.search` with `{query, engine, result_count: max_results, site_*}`. On
   error → return it as the tool error.
4. Take the first `fetch_top_n` results that carry a `url`; fetch them with
   **bounded concurrency** (semaphore = 3) via `web.fetch`
   `{url, format, max_bytes}`. Partial-failure tolerant.
5. Assemble `sources` preserving search order; `fetched_count` = number of ok
   fetches; `truncated` = search truncated OR any fetched source truncated.

## Testability

`WebResearchModule(config, *, search_module=None, fetch_module=None)` — tests
inject fakes exposing `async execute_tool(name, args, context)` that return
canned `web.search` / `web.fetch` dicts. No network, no provider config.

## Registration & presets

- `server.py`: optional block gated by `MCP_ENABLE_WEB_RESEARCH_MODULE` (off by
  default), id `web_research`, department `research`.
- `presets.py`: add `web.research` to `_WEB_READ_TOOLS`; enabled in the
  `deep-researcher` preset.

## Tests

- `test_web_research_module.py`: contract; happy path (search → top-N fetch →
  bundle with order preserved); search error short-circuits; one fetch failing
  is tolerated (`fetched: false` + reason_code); `fetch_top_n` clamped to
  results/max; `fetch_top_n: 0` returns search-only sources; truncated flag from
  search or fetch; results without a url skipped; sanitize override allows `--`;
  arg validation; unknown tool; eval profile metadata.
- `test_web_research_module_registration.py`: registered iff flag set.
- `test_profile_presets.py`: `deep-researcher` enables `web.research`.
