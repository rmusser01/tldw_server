# Remaining Frontend CodeQL Alerts Design

## Context

GitHub reports 161 open CodeQL alerts on `main`. Merged PR #2696 already applies the Python and clear-text API-key remediations for 149 of those alerts to `dev`. This change addresses the remaining 12 JavaScript/TypeScript alerts, IDs 2251 through 2262, in a new PR targeting `dev`.

The alerts will remain visible against `main` until the fixes are promoted from `dev` and GitHub analyzes the updated `main` branch.

## Scope

| Alerts | Rule | Alerted files | Root cause | Design |
| --- | --- | --- | --- | --- |
| 2251-2252 | `js/xss` | `AssistantSelect.tsx`, `CharacterSelect.tsx` | API-provided avatar URLs reach image sources without an image-specific validation boundary | Normalize avatar URLs with a shared image-source validator before selections are rendered or stored |
| 2253 | `js/xss` | `ItemsTab/items-utils.ts` | Untrusted article HTML is passed to `DOMParser` to derive text | Remove the raw DOM parsing path; sanitize away active/non-text content with DOMPurify, then reuse the non-DOM text scanner |
| 2254-2255 | `js/xss` | `ItemsTab/ItemsTab.tsx` | Image URLs extracted from article HTML are returned without validation | Apply the shared image-source validator at the common `extractImageUrl` boundary |
| 2256, 2262 | `js/xss`, `js/xml-bomb` | `SourcesTab/SourcesTab.tsx` | Group filtering parses server-returned OPML with `DOMParser` | Delete the OPML parsing/cache path and send the backend's existing `groups` query parameter when fetching sources |
| 2257 | `js/xss` | `Quiz/tabs/ManageTab.tsx` | Printable quiz HTML is written to a new document; not every runtime value is guaranteed to match its TypeScript type | Sanitize the complete generated document with DOMPurify immediately before the `document.write` sink |
| 2258 | `js/tainted-format-string` | `DocumentGeneratorDrawer.tsx` | An untrusted identifier is interpolated into the first argument of a console call | Use a constant console message and pass identifiers/errors as separate arguments |
| 2259-2260 | `js/tainted-format-string` | `services/timeline/api.ts` | Untrusted identifiers are interpolated into the first argument of console calls | Use constant console messages and pass identifiers/errors as separate arguments |
| 2261 | `js/regex-injection` | `utils/provider-registry.ts` | A custom callback property named `match` is called with an untrusted model ID and is mistaken for a regex sink | Rename the callback to `matches`; inference remains literal `includes`/`startsWith` checks |

## Design

### Watchlist group filtering

The existing `GET /api/v1/watchlists/sources` endpoint already accepts repeated `groups` parameters and performs the group membership join in the database. The frontend service type will expose this existing parameter, and `SourcesTab` will send `groups: [selectedGroupId]` as part of its base query.

Group-only filtering will therefore use the API's normal pagination and exact total. When a source-type filter is also active, the existing client-side pagination loop remains only for that unsupported type filter, but each page is already narrowed to the selected group by the server. This avoids the current client cap for group-only queries and prevents inactive or non-RSS group members from disappearing merely because the OPML export excludes them.

This removes an extra export request, the 30-second OPML URL cache, XML parsing, and URL-based joins. Failures remain handled by the existing outer source-load error path; there is no longer a separate group-OPML failure mode. No backend implementation or API behavior changes are required.

### Untrusted HTML and image URLs

`stripHtmlToText` will sanitize the untrusted string with the existing DOMPurify dependency, using the HTML profile and explicitly forbidding `script` and `style`. The sanitized markup then passes through the existing non-DOM tag scanner, entity decoding, and whitespace normalization. This preserves spaces at block/tag boundaries while excluding script and style bodies, without constructing a document from the untrusted string.

An image-specific helper will compose the existing `createImageDataUrl` and `safeExternalUrl` behavior, but narrow the result to verified raster data URLs, HTTP(S) URLs, and relative URLs. In particular, navigation-safe but image-inappropriate schemes such as `mailto:` remain rejected. Accepted output must be normalized with a known safe prefix rather than returning an unchanged attacker-controlled absolute string after a boolean check; this makes the scheme guarantee explicit to both the browser and CodeQL's tainted-prefix analysis. This small shared helper is justified by the three affected image boundaries and introduces no dependency.

`extractImageUrl` remains the single boundary for watchlist article preview images. Both HTML and Markdown candidates must pass the image-source validator; unsafe or malformed schemes return `null` and render the existing fallback tile.

Character and assistant normalizers will accept only:

- raster data URLs validated by `createImageDataUrl`;
- HTTP(S) URLs; or
- relative URLs.

Unsafe avatar values become empty/null and render the existing generic user icon. When both an external avatar and embedded base64 are present, a rejected external candidate must not mask a valid embedded raster fallback.

### Printable quiz document

The existing printable HTML builder and print-window flow remain intact. The complete document is sanitized with the already-installed DOMPurify dependency in `WHOLE_DOCUMENT` mode immediately before it is written. This creates a recognized trust boundary at the sink and protects against malformed API payloads that do not honor compile-time quiz types.

DOMPurify preserves the trusted `<html>`, `<head>`, `<title>`, `<style>`, and `<body>` shell in whole-document mode but omits the doctype. The builder will therefore prepend one constant `<!doctype html>` after sanitization. Tests must assert that the printable title, stylesheet, and body content survive, malicious elements/attributes are removed, and exactly one trusted doctype reaches the sink.

If print-window creation or sanitization fails, the existing error notification and cleanup path remains authoritative.

### Provider inference and logging

Provider inference behavior does not change. The `ProviderInferenceRule.match` callback is renamed to `matches` so CodeQL no longer treats the custom call as a regular-expression operation.

Console calls use constant first arguments. User/server identifiers and error objects are passed as subsequent arguments so format specifiers in external input cannot affect log formatting.

## Testing

Focused regression coverage will prove:

- group filtering sends the existing `groups` query parameter, returns the API-paginated result when no type filter is active, and does not export or parse OPML;
- combined group/type filtering keeps server-side group narrowing while retaining the existing client-side type behavior;
- HTML-to-text conversion removes executable markup and script/style bodies while retaining tag-boundary spacing and entity normalization;
- the shared image-source validator preserves verified raster data, HTTP(S), and relative URLs while rejecting `javascript:`, `mailto:`, SVG/other data URLs, and malformed candidates;
- article preview and character/assistant selection use that validator and retain their existing fallback UI;
- a rejected external avatar still falls back to a valid embedded raster image;
- the printable quiz sink receives sanitized output for malicious runtime payloads while preserving one trusted doctype plus the title, stylesheet, and body shell;
- provider inference results are unchanged after the callback rename;
- tainted identifiers are passed as console arguments rather than format strings.

Verification includes the focused Vitest files, related component suites, the `apps/tldw-frontend` TypeScript check, exact review of all 12 alert source/sink paths, and `git diff --check`. Bandit is not applicable because the intended patch touches no Python source.

At review time, the repository's advanced CodeQL workflow analyzes Python only, while GitHub default setup analyzes JavaScript/TypeScript on the default or protected branches. `dev` is neither, so a PR targeting `dev` is not expected to emit a JavaScript CodeQL result. The PR's available checks will still be inspected, but behavioral regressions and source-path review are the pre-merge evidence for this patch. The alert state can be confirmed by GitHub only after the fixes reach `main` (or the repository's code-scanning branch configuration changes). This limitation must be stated in the PR rather than represented as a successful JavaScript scan.

## Non-goals

- Reworking the 149 alerts already addressed on `dev` by PR #2696.
- Changing the backend watchlist API or group-filter semantics; the frontend only adopts the already-supported `groups` parameter.
- Replacing the quiz print feature with a new renderer.
- Reconfiguring CodeQL, branch protection, or GitHub default setup.
- Dismissing or suppressing valid CodeQL findings.

## Alternatives Considered

Direct sink annotations or CodeQL suppressions would produce a smaller apparent diff but retain redundant OPML parsing and scattered trust decisions. Filtering already-fetched records by `group_ids` would remove XML parsing, but it would keep the group-only client cap and duplicate a database filter the API already provides. A new server endpoint and a dedicated print renderer would be broader than required. The selected design removes unnecessary parsing, uses the existing server contract, and centralizes the image trust boundary with the smallest reviewable behavior change.
