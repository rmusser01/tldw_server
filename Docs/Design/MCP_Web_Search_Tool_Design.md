# MCP `web.search` Tool Design

Status: Implementing (June 2026)
Owner: standalone MCP gateway backlog
Companion: `Docs/Design/MCP_Web_Fetch_Tool_Design.md`

## Goal

Add a built-in, read-only MCP tool, `web.search`, that runs a query against a
configured multi-provider web search backend and returns a bounded list of
normalized results. It complements `web.fetch` (which retrieves one URL) and is
the natural follow-up search primitive for the `deep-researcher` preset.

## Why this shape

- **In-server module.** Like `web.fetch`/`git`, the tool implementation lives in
  `tldw_Server_API/app/core/MCP_unified/modules/implementations/web_search_module.py`
  so it can call `Web_Scraping.WebSearch_APIs.perform_websearch` without pulling
  provider SDKs into the standalone `mcp_unified/` gateway package.
- **Outbound policy is enforced inside the provider call.** `perform_websearch`
  runs each provider request through `_enforce_provider_outbound_policy`
  (SSRF/egress). The module surfaces a denial as `outbound_policy_denied` rather
  than re-implementing the gate.
- **Governance.** `web.search` is gated as a whole tool via tool-level permission
  rules and the `external_network` / `research.web` preset capability. Unlike
  `web.fetch`, the input is a `query` (no `url`), so domain *subjects* are not
  extracted from the call — domain rules don't apply per-result, by design. The
  Claude-style `WebSearch(<domain>)` rule keyword remains available for authors
  but only matches when a domain subject is present.

## Tool: `web.search`

Read-only. Arguments (additionalProperties: false):

| arg | type | default | bounds |
|---|---|---|---|
| `query` | string (required) | — | non-empty |
| `engine` | string | configured/`duckduckgo` | allow-list of supported providers |
| `result_count` | integer | 10 | 1 .. 25 |
| `content_country` | string | `US` | — |
| `search_lang` | string | `en` | — |
| `output_lang` | string | `en` | — |
| `safesearch` | string | provider default | — |
| `site_whitelist` | string[] | — | non-empty strings |
| `site_blacklist` | string[] | — | non-empty strings |
| `date_range` | string | — | — |

Supported engines: google, duckduckgo, brave, kagi, serper, tavily, exa,
firecrawl, searx, yandex, baidu.

Success result:

```
{
  "ok": true,
  "engine": "duckduckgo",
  "query": "python asyncio",
  "result_count": 2,
  "total_results_found": 2,
  "results": [
    {"title": "...", "url": "https://...", "content": "<<=4000 chars>>", "metadata": {...}}
  ],
  "eval": { ... }
}
```

Error result: `{ ok: false, reason_code, message, eval }`.

Reason codes: `invalid_arguments`, `invalid_engine`, `outbound_policy_denied`
(provider blocked by egress policy), `search_failed` (provider/network/parse
error), `unknown_tool`.

## Bounds & normalization

- `result_count` clamped to 25; the result list is also truncated to the
  requested count defensively.
- Per-result `content` truncated to 4000 chars.
- `perform_websearch` is synchronous/blocking → run via `asyncio.to_thread`.
- A `processing_error`/`error` field containing "outbound policy" maps to
  `outbound_policy_denied`; any other error maps to `search_failed`.

## Testability

`WebSearchBackend` Protocol (default `PerformWebSearchBackend`) is injected so
unit tests provide a fake returning a normalized `perform_websearch` payload —
no provider config, no network.

## Registration & presets

- `server.py`: optional block gated by `MCP_ENABLE_WEB_SEARCH_MODULE` (off by
  default), id `web_search`, department `research`.
- `presets.py`: `_WEB_READ_TOOLS = ["web.fetch", "web.search"]`, wired into the
  `deep-researcher` preset.

## Tests

- `test_web_search_module.py`: contract, happy path, content bounding, result
  cap, empty query, invalid engine, argument validation, processing-error →
  search_failed, outbound-policy → denied, backend exception → search_failed,
  unknown tool, site-filter passthrough, eval profile metadata.
- `test_web_search_module_registration.py`: registered iff flag set.
- `test_profile_presets.py`: `deep-researcher` enables `web.search`.
