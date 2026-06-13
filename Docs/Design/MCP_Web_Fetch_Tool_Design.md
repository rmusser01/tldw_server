# MCP `web.fetch` Tool Design

Status: Implementing (June 2026)
Owner: standalone MCP gateway backlog

## Goal

Add a built-in MCP tool, `web.fetch`, that retrieves a single user-specified
URL, applies the centralized outbound (SSRF/egress) policy, and returns bounded,
extracted page content (markdown/text/html). The tool is gated by the gateway's
existing **domain** permission subjects so profile authors can allow/deny/ask on
`WebFetch(<domain>)` rules without any new policy plumbing.

## Why this shape

- **In-server module, not standalone-package code.** Built-in tool
  *implementations* live in
  `tldw_Server_API/app/core/MCP_unified/modules/implementations/` (e.g.
  `git_module.py`, `filesystem_module.py`). They may import server internals
  (`Web_Scraping/`), which keeps the standalone `mcp_unified/` gateway package
  free of heavy web deps (httpx/trafilatura). The gateway only learns the tool
  exists via `mcp_unified/profiles/presets.py`.
- **Domain policy is already wired.** `mcp_unified/profiles/subjects.py` extracts
  a `domain` subject from any `url`/`uri`/`domain` argument, and
  `permission_rules.py` compiles `WebFetch(<pattern>)` rule specifiers into
  `domain` rules. A tool that takes a `url` argument therefore gets
  allow/deny/ask enforcement for free at the gateway runtime. The module does not
  re-implement domain policy — it only enforces the always-on SSRF/egress guard.

## Tool: `web.fetch`

Read-only. Arguments (additionalProperties: false):

| arg | type | default | bounds |
|---|---|---|---|
| `url` | string (required) | — | http/https only |
| `format` | enum `markdown`\|`text`\|`html` | `markdown` | — |
| `max_bytes` | integer | 1_000_000 | 1 .. 5_000_000 |
| `timeout_seconds` | integer | 15 | 1 .. 30 |
| `respect_robots` | boolean | `false` | explicit fetch defaults off |

Success result:

```
{
  "ok": true,
  "url": "<requested>",
  "final_url": "<after redirects>",
  "status_code": 200,
  "content_type": "text/html; charset=utf-8",
  "title": "<extracted or null>",
  "format": "markdown",
  "content": "<bounded extracted text>",
  "bytes_fetched": 12345,
  "truncated": false,
  "eval": { ... }
}
```

Error result: `{ ok: false, reason_code, message, eval }`.

Reason codes: `invalid_url` (bad/unsupported scheme), `outbound_policy_denied`
(SSRF/egress/robots), `fetch_failed` (network/timeout/status), `empty_content`,
`unknown_tool`.

## Security & bounds

1. Validate scheme is `http`/`https`; reject everything else (`invalid_url`).
2. `decide_web_outbound_policy(url, respect_robots=..., user_agent=...,
   source="mcp.web_fetch", stage="web.fetch")` — denies private/loopback/link-local
   targets and (optionally) robots-disallowed paths. Denial → `outbound_policy_denied`.
3. Bounded streaming download: stop after `max_bytes`; set `truncated=true`.
4. `timeout_seconds` hard cap (30s).
5. HTML extracted with `trafilatura.extract(..., output_format=...)`; text/plain,
   text/markdown, application/json returned as decoded bounded body; other
   content types rejected with `empty_content`.

## Testability

`WebFetchHttpClient` Protocol + `WebFetchResponse` dataclass so tests inject a
fake fetcher (no network). Default `HttpxWebFetchClient` uses
`httpx.AsyncClient` with streaming + byte cap. Outbound policy is monkeypatched
in tests via `evaluate_url_policy` (same hook `Web_Scraping` tests use).

## Registration

`server.py`: optional block gated by `MCP_ENABLE_WEB_FETCH_MODULE` (disabled by
default, mirroring git/sandbox), id `web_fetch`, department `research`.

## Gateway presets

Add `_WEB_READ_TOOLS = ["web.fetch"]` in `presets.py`; wire into the
`deep-researcher` preset (the only preset carrying the `external_network` risk
class and `research.web` capability) via a `tooling_metadata_document`.

## Tests

- `test_web_fetch_module.py`: invalid scheme; outbound denial; happy-path html→
  markdown extraction with injected client; byte cap → truncated; text/plain
  passthrough; unsupported content type; arg validation (unknown arg, bad bounds);
  fetch failure mapping.
- `test_web_fetch_module_registration.py`: registered when flag set, absent when unset.
- preset assertion: `deep-researcher` enables `web.fetch`.
