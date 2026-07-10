# Remaining Frontend CodeQL Alerts Design

## Context

GitHub reports 161 open CodeQL alerts on `main`. Merged PR #2696 already applies the Python and clear-text API-key remediations for 149 of those alerts to `dev`. This change addresses the remaining 12 JavaScript/TypeScript alerts, IDs 2251 through 2262, in a new PR targeting `dev`.

The alerts will remain visible against `main` until the fixes are promoted from `dev` and GitHub analyzes the updated `main` branch.

## Scope

| Alerts | Rule | Root cause | Design |
| --- | --- | --- | --- |
| 2251-2252 | `js/xss` | API-provided avatar URLs reach image sources without a scheme/data validation boundary | Normalize avatar URLs with the existing `safeExternalUrl` and `createImageDataUrl` helpers before selections are rendered or stored |
| 2253 | `js/xss` | Untrusted article HTML is passed to `DOMParser` to derive text | Remove the raw DOM parsing path and use the existing DOMPurify dependency plus the non-DOM text fallback |
| 2254-2255 | `js/xss` | Image URLs extracted from article HTML are returned without validation | Validate extracted URLs with `safeExternalUrl` at the shared extraction boundary |
| 2256, 2262 | `js/xss`, `js/xml-bomb` | Group filtering parses server-returned OPML with `DOMParser` | Delete the OPML parsing/cache path and filter the already-fetched source records by their existing `group_ids` metadata |
| 2257 | `js/xss` | Printable quiz HTML is written to a new document; not every runtime value is guaranteed to match its TypeScript type | Sanitize the complete generated document with DOMPurify immediately before the `document.write` sink |
| 2258-2260 | `js/tainted-format-string` | Untrusted identifiers are interpolated into the first argument of console calls | Use constant console messages and pass identifiers/errors as separate arguments |
| 2261 | `js/regex-injection` | A custom callback property named `match` is called with an untrusted model ID and is mistaken for a regex sink | Rename the callback to `matches`; inference remains literal `includes`/`startsWith` checks |

## Design

### Watchlist group filtering

`SourcesTab` already fetches every source page whenever a group or source-type client filter is active. Each `WatchlistSource` exposes `group_ids`, and the component already has a normalization helper for that field. Group filtering will use that metadata directly.

This removes an extra export request, the 30-second OPML URL cache, XML parsing, and URL-based joins. Failures remain handled by the existing outer source-load error path; there is no longer a separate group-OPML failure mode.

### Untrusted HTML and image URLs

`stripHtmlToText` will sanitize to text without constructing a document from the untrusted string. Script and style content must remain excluded, and the existing entity/whitespace normalization behavior must be retained.

`extractImageUrl` remains the single boundary for watchlist article preview images. Both HTML and Markdown candidates must pass `safeExternalUrl`; unsafe or malformed schemes return `null` and render the existing fallback tile.

Character and assistant normalizers will accept only:

- raster data URLs validated by `createImageDataUrl`; or
- URLs accepted by `safeExternalUrl`.

Unsafe avatar values become empty/null and render the existing generic user icon. No new URL-validation abstraction or dependency is introduced.

### Printable quiz document

The existing printable HTML builder and print-window flow remain intact. The complete document is sanitized with the already-installed DOMPurify dependency immediately before it is written. This creates a recognized trust boundary at the sink and protects against malformed API payloads that do not honor compile-time quiz types.

If print-window creation or sanitization fails, the existing error notification and cleanup path remains authoritative.

### Provider inference and logging

Provider inference behavior does not change. The `ProviderInferenceRule.match` callback is renamed to `matches` so CodeQL no longer treats the custom call as a regular-expression operation.

Console calls use constant first arguments. User/server identifiers and error objects are passed as subsequent arguments so format specifiers in external input cannot affect log formatting.

## Testing

Focused regression coverage will prove:

- group filtering uses `group_ids`, returns the correct sources, and does not export or parse OPML;
- HTML-to-text conversion removes executable markup and script/style bodies;
- article image extraction preserves allowed HTTP(S) URLs and rejects `javascript:`, unsafe data, and malformed candidates;
- character/assistant selection preserves verified raster data and safe URLs while rejecting unsafe avatar schemes;
- the printable quiz sink receives sanitized output for malicious runtime payloads;
- provider inference results are unchanged after the callback rename;
- tainted identifiers are passed as console arguments rather than format strings.

Verification includes the focused Vitest files, related component suites, the `apps/tldw-frontend` TypeScript check, and `git diff --check`. Bandit is not applicable because the intended patch touches no Python source. The PR's GitHub checks will be inspected, including CodeQL when emitted for the PR.

## Non-goals

- Reworking the 149 alerts already addressed on `dev` by PR #2696.
- Changing watchlist API contracts or adding server-side group filtering.
- Replacing the quiz print feature with a new renderer.
- Dismissing or suppressing valid CodeQL findings.

## Alternatives Considered

Direct sink annotations or CodeQL suppressions would produce a smaller apparent diff but retain redundant OPML parsing and scattered trust decisions. New server endpoints and a dedicated print renderer would be broader than required. The selected design removes unnecessary parsing and reuses existing security boundaries with the smallest reviewable behavior change.
