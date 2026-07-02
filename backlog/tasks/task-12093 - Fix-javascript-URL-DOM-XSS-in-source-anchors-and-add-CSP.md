---
id: TASK-12093
title: Fix javascript URL DOM-XSS in source anchors and add a CSP
status: Done
labels:
- bug
- high
- security
- xss
- frontend
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: High (DOM-XSS on the web app origin).** From the 2026-07-02 frontend audit (finding H1).

The Next.js web app ships **no Content-Security-Policy** (verified: no `headers()` in `apps/tldw-frontend/next.config.mjs`, no `middleware`). So a `javascript:` URL that reaches the DOM executes on click on the app origin.

Multiple hand-rolled anchors render an untrusted URL with no protocol allowlist (`target`/`rel` do not block `javascript:`). Confirmed sink: `apps/packages/ui/src/components/Common/Playground/MessageSource.tsx:80` (`const url = source?.url`) → `:183` (`<a href={url} target="_blank" rel="noopener noreferrer">`). Same pattern at: `Option/ResearchWorkspace/SourcesPane/index.tsx:2411`; `Option/Watchlists/SourcesTab/SourcesTab.tsx:1249,1517`; `Watchlists/RunsTab/RunDetailDrawer.tsx:755,949`; `Watchlists/OutputsTab/ReportEvidencePanel.tsx:141,225`; `Watchlists/AlertsTab/AlertsTab.tsx:644,650`; `Option/Collections/ReadingList/ReadingItemDetail.tsx:1030`; `Option/Items/ItemsWorkspace.tsx:851`; `Option/Processed/index.tsx:62`. Also `window.open(url)` without an allowlist at `useRagResultsDisplay.tsx:193`, `KnowledgeQA/SourceCard.tsx:279`, `SourceViewerModal.tsx:99`, `Watchlists/ItemsTab/ItemsTab.tsx:1948`, `ReadingList/ReadingItemCard.tsx:113`, `useKnowledgeSearch.ts:620`, `useFileSearch.ts:222`.

`source.url` is attacker-influenceable via poisoned ingested-page metadata (yt-dlp title/URL, scraped feeds), a malicious web-search/research citation, or a crafted API/JSON response. The markdown renderer already blocks this via `urlTransform`; these hand-rolled anchors bypass it.

Two related same-family issues (fold in or split as preferred):
- `OutputPreviewDrawer.tsx:325` — `safeHtml = sanitizedHtml || content` re-injects raw HTML into a same-origin `blob:` tab when DOMPurify returns empty; should be `|| ""` (sibling `:638` already does).
- `notes-manager-utils.ts:834-842` `sanitizeUrl` — a control-char scheme (`java\tscript:`) matches neither branch and is emitted verbatim; strip control chars before scheme matching.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A shared `safeExternalUrl()` helper (allowlist `http`/`https`/`mailto`, else return null/no-op) exists and is applied at every hand-rolled `<a href={...url}>` source/citation anchor listed above.
- [x] #2 The same helper (or equivalent guard) is applied at the `window.open(url)` sites listed above.
- [x] #3 A Content-Security-Policy is served for the web app (at minimum blocking inline/`javascript:` script execution), verified against a page load.
- [x] #4 `OutputPreviewDrawer.tsx:325` fallback is `|| ""`; `sanitizeUrl` strips control characters before scheme matching.
- [x] #5 Tests cover: a `source.url` of `javascript:alert(1)` renders as inert (no navigation/exec), and a `java\tscript:` note link is neutralized.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
### Shared helper
`apps/packages/ui/src/utils/safe-external-url.ts`
- `safeExternalUrl(url: unknown): string | null` — allowlists `http:`/`https:`/`mailto:` (and relative paths/anchors); strips C0 control chars + DEL and trims before scheme detection (so `java\tscript:` can't slip through); resolves via `new URL(url, base)` so casing/whitespace can't obfuscate the scheme; returns `null` when unsafe.
- `openExternalUrl(url, target = "_blank", features = "noopener,noreferrer"): Window | null` — `window.open` guarded by `safeExternalUrl`; no-ops on unsafe URL or SSR.

Reuse note: existing per-component sanitizers (`transformMarkdownUrl` in `Common/Markdown.tsx`, `sanitizeUrl` in `Notes/notes-manager-utils.ts`) were not exported/shared, so a single shared util was added rather than duplicating logic. `notes-manager-utils.ts` `sanitizeUrl` was hardened in place (control-char stripping) rather than replaced, since it also allows `tel:`/`note:`/`#`.

### Anchor sinks guarded (inert/dropped href when unsafe)
`MessageSource.tsx` (safeUrl gates both anchors + the no-content link → falls back to inert `<span>`); `ResearchWorkspace/SourcesPane/index.tsx`; `Watchlists/SourcesTab/SourcesTab.tsx` (x2); `Watchlists/RunsTab/RunDetailDrawer.tsx` (x2); `Watchlists/OutputsTab/ReportEvidencePanel.tsx` (x2); `Watchlists/AlertsTab/AlertsTab.tsx` (x2, guarded at the `sourceUrl`/`itemUrl` derivation); `Collections/ReadingList/ReadingItemDetail.tsx` (href → `undefined` when unsafe, gate is `domain`); `Items/ItemsWorkspace.tsx`; `Processed/index.tsx` (antd `Button` → `href` undefined + `disabled`).

### window.open sites guarded (openExternalUrl)
`Sidepanel/Chat/hooks/useRagResultsDisplay.tsx`; `KnowledgeQA/SourceCard.tsx`; `KnowledgeQA/SourceViewerModal.tsx`; `Watchlists/ItemsTab/ItemsTab.tsx`; `Collections/ReadingList/ReadingItemCard.tsx`; `Knowledge/hooks/useKnowledgeSearch.ts`; `Knowledge/hooks/useFileSearch.ts`. (`SourceCard.tsx:698` opens a hardcoded `/document-workspace` path — not attacker-controlled, left as-is.)

### CSP (next.config.mjs `async headers()`, applied to `/:path*`)
```
default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none';
script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; style-src 'self' 'unsafe-inline';
img-src 'self' data: blob: https: http:; font-src 'self' data:;
media-src 'self' data: blob: https: http:; connect-src 'self' https: http: ws: wss: data: blob:;
worker-src 'self' blob:; frame-src 'self' blob: data: https: http:
```
`object-src 'none'` + `base-uri 'self'` + `frame-ancestors 'none'` are the hard protections. `script-src` keeps `'unsafe-inline'`/`'unsafe-eval'` (required by the `_document` theme-bootstrap inline script, antd runtime, Next.js) and `blob:` (Web Workers: OCR/diff/tokenizer). Resource directives (img/media/connect/frame) stay broad so external thumbnails, remote backends, realtime-audio WebSockets, and blob:/PDF iframe previews keep working.
**Follow-up:** tighten `script-src` to nonce/hash-based (drop `'unsafe-inline'`/`'unsafe-eval'`) — this also lets CSP block `javascript:` navigation directly. Consider adding `X-Content-Type-Options: nosniff` and `Referrer-Policy`.

### Same-family fixes
- `OutputPreviewDrawer.tsx` `handleOpenInNewTab`: `safeHtml = sanitizedHtml || content` → `|| ""` (matches sibling rendered view; no raw HTML into the blob: tab).
- `notes-manager-utils.ts` `sanitizeUrl`: strips C0 controls + DEL before scheme matching.

### Tests
- `apps/packages/ui/src/utils/__tests__/safe-external-url.test.ts` (7 tests) — javascript:/`java\tscript:`/data:/vbscript:/file: rejected; http/https/mailto/relative allowed; non-string/empty rejected.
- `apps/packages/ui/src/components/Notes/__tests__/notes-manager-utils.sanitize-url.test.ts` (4 tests) — control-char/newline obfuscated schemes neutralized via `markdownInlineToHtml`.
- Both suites green (11/11). All 19 changed source files pass an esbuild syntax check; `next.config.mjs` `headers()` verified to emit the CSP.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
