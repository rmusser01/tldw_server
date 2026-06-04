# MCP CDP Browser Inspection Read Tools Design

## Goal

Add the first native browser-inspection MCP tools for front-end and QA profiles using Chrome DevTools Protocol (CDP) as the initial backend. This slice is read-only: it can inspect pages, collect bounded diagnostic output, and capture screenshots, but it must not navigate, click, type, select, evaluate caller-provided JavaScript, or mutate browser/page state.

## Scope

Included:

- `browser.status` for CDP availability and target summary.
- `browser.pages.list` for visible inspectable page targets.
- `browser.snapshot` for a bounded read-only page snapshot.
- `browser.page_state` for URL/title/viewport/document/readiness/performance state.
- `browser.screenshot` for a bounded screenshot capture.
- `browser.console` for console/log events observed during a bounded window.
- `browser.network` for network events observed during a bounded window.
- Profile metadata/policy wiring so browser-capable presets can discover installed read-only browser tools when the module is enabled.

Excluded:

- Browser mutation tools such as navigate, click, type, select, reload, focus, local-storage mutation, or arbitrary `Runtime.evaluate` supplied by a model.
- Persisting screenshots to workspace files.
- External MCP installation/update templates.
- Playwright or host-provided browser adapters. Those remain future adapter targets behind the same module-facing interface.

## Architecture

The implementation adds a host-side MCP module under `tldw_Server_API/app/core/MCP_unified/modules/implementations/browser_cdp_module.py` and a focused CDP client seam under `tldw_Server_API/app/core/MCP_unified/browser_cdp/`. The module owns MCP tool schemas, argument validation, metadata, and result shaping. The CDP client owns target discovery, HTTP `/json/*` calls, WebSocket command dispatch, bounded event observation, and result normalization.

The browser module is optional. It should register when an operator explicitly sets `MCP_ENABLE_BROWSER_CDP_MODULE=true` or provides `MCP_BROWSER_CDP_URL`. Tool calls never accept a debugger URL. They only use module configuration resolved from environment/config, which avoids exposing arbitrary model-controlled network access.

## Configuration

Initial settings:

- `debugger_url`: CDP HTTP base URL, defaulting from `MCP_BROWSER_CDP_URL`.
- `request_timeout_seconds`: bounded HTTP/WebSocket timeout, default 3 seconds.
- `observation_window_ms`: default console/network observation window, default 250 ms.
- `max_events`: default maximum console/network events per call, default 100.
- `max_snapshot_nodes`: maximum page snapshot nodes/entries, default 200.
- `screenshot_max_bytes`: maximum returned screenshot payload bytes after base64 decode estimate, default 2 MB.
- `allow_non_loopback`: default false. Unless explicitly enabled, only literal loopback hosts are accepted: `localhost`, `127.0.0.0/8`, and `::1`. Do not perform DNS resolution to prove a hostname is loopback.

## Tool Behavior

`browser.status` returns whether CDP is configured and reachable, version metadata when available, page count, and a stable reason code when unavailable.

`browser.pages.list` returns bounded page targets with `target_id`, `title`, `url`, `type`, and `attached`/metadata when CDP provides it. Browser extension, service worker, and non-page targets are excluded by default.

`browser.snapshot` uses fixed read-only CDP operations. It returns URL/title plus a bounded representation of accessibility or document state. It must never execute model-provided JavaScript and must truncate large results with `truncated: true`.

`browser.page_state` returns fixed read-only state such as URL, title, document ready state, viewport dimensions, scroll dimensions, focused element summary, and performance timing where available. Any `Runtime.evaluate` usage must be hardcoded and side-effect-free.

`browser.screenshot` calls CDP screenshot capture with bounded options. It returns MIME type, base64 data, byte estimate, dimensions when available, and truncation/error metadata when payload limits are exceeded.

`browser.console` and `browser.network` do not claim historical completeness. They enable the relevant CDP domains and collect events during a bounded observation window. Results include `observed_for_ms`, `events`, `truncated`, and counts by type/status where useful.

## Security And Safety

- CDP endpoint is operator-owned configuration, not a tool argument.
- Default endpoint validation is loopback-only and rejects arbitrary hostnames without resolving them.
- No arbitrary JavaScript input is accepted.
- No browser mutation CDP methods are exposed.
- Event observation windows and result sizes are capped.
- Tool metadata must mark tools as read-only and category `browser`.
- `browser.screenshot` returns in-memory data only in this slice.
- Errors should use stable reason codes such as `cdp_not_configured`, `cdp_unreachable`, `target_not_found`, `payload_too_large`, and `cdp_protocol_error`.

## Profile Integration

Browser read tools should be discoverable for profiles that already express browser/debug intent, primarily Frontend Engineer and QA Engineer, and optionally SDET where browser inspection is a deferred category. Profile policy remains authoritative: discovery only shows installed tools when exact tool names or read-only browser capabilities match the effective policy.

The read-only browser tools use capabilities such as:

- `browser.inspect`
- `browser.debug`
- `screenshots.capture`
- `app_state.read`

No `browser_mutation` risk class is introduced for this read-only slice.

## Testing

Use TDD with a fake CDP client/factory for module behavior and small fake transport tests for the client seam. Do not require a real Chrome process in unit tests. A live CDP smoke is optional and should be recorded as manual/local verification when available.

Required coverage:

- Tool descriptors and read-only metadata.
- Argument validation and unknown-key rejection.
- Missing/unconfigured CDP behavior.
- Page target selection.
- Snapshot/page-state/screenshot/console/network result shaping.
- Bounded event/payload truncation.
- Server registration gating.
- Profile discovery for browser-capable presets.
- Package boundary checks proving `mcp_unified` does not import host CDP code.

## Open Follow-Ups

- Browser interaction tools with approval-gated `browser_mutation` risk.
- Playwright and host-provided browser adapters behind the same module seam.
- External MCP install/update template for `ChromeDevTools/chrome-devtools-mcp`.
- Screenshot artifact persistence through workspace-scoped file tools.
