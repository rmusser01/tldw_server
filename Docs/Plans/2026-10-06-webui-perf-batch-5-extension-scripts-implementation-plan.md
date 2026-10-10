# WebUI Perf Batch W5 — Extension Background & Content Scripts Implementation Plan

**Backlog task:** TASK-13525 · **Index:** [coordination index](2026-10-06-webui-perf-remediation-coordination-index.md)
**Verified against:** `dev` @ `7ba48f251e`
**Findings:** WP-56..WP-63
**Note:** the extension entrypoints are thin shims; the real code is `apps/packages/ui/src/entries/*` and `apps/packages/ui/src/{services,utils}/*`. Disjoint from W1–W4 files — can run any time after W0.

**Goal:** Content scripts stop doing eager full-document work on the host page's main thread; the MV3 service worker stops waking for no-op refreshes; per-chunk hot paths stop being O(total²).

---

## Stage 1: Web-clipper tiered capture (WP-56, WP-57)

**Goal:** Selection clips never pay for full-page serialize + forced layout + heavy parse.
**Files:** Modify `services/web-clipper/content-extract.ts:67-79` (anchor: `grep -n "documentElement?.outerHTML" services/web-clipper/content-extract.ts`), `parser/default.ts:22-35`.
**Approach:**
- Thread `requestedType` into `extractClipPageTextFromDocument`; return selection text immediately when `requestedType === "selection"` and a selection exists; compute `articleText` lazily only when the higher-priority source is empty; compute `fullPageText` only as the final fallback.
- `defaultExtractContent` fallback: replace the `$("*").each` + per-attribute `removeAttr` loop with one cheerio filter/select pass per attribute family (or drop attributes via `cheerio`'s `parseOptions` with `lowerCaseAttributeNames` and a single traversal).
**Success Criteria:** Selection clip on a 5 MB DOM: no `outerHTML` serialize, no `innerText` forced layout (assert via spies/mocks); capture main-thread time < 10 ms for selection path.
**Tests:** Unit: selection-clip extraction with a stubbed document — assert `outerHTML`/`innerText`/DOMParser never touched; article fallback still produced when selection empty; parity of extracted text vs golden fixtures for full-page mode.
**Status:** Not Started

## Stage 2: Copilot popup streaming + audio base64 (WP-58, WP-59)

**Files / fixes:**
- `entries/copilot-popup.content.tsx:415-418, 626-630`: stop assigning the full accumulated string per chunk. Append `document.createTextNode(chunk)` (or buffer chunks and flush on rAF — the position updates at 368-374 already rAF-coalesce; reuse that scheduler). Keep the existing per-frame style writes as-is (`grep -n "responseEl.textContent" entries/copilot-popup.content.tsx`).
- `utils/compress.ts:36-43`: chunked base64 (`String.fromCharCode.apply` over 8 KB slices) or send `ArrayBuffer` over the port (structured clone) instead of JSON+base64 at `entries/background.ts:4021-4026` (`grep -n "String.fromCharCode" utils/compress.ts`).
**Success Criteria:** A 2,000-chunk stream performs O(chunks) DOM text-node appends and 0 full-text rewrites; per-audio-chunk encode is O(n) with no per-byte string concat.
**Tests:** Unit: stream 1000 chunks into a fake element — assert appendChild called 1000×, textContent assigned ≤ once per rAF flush; base64 helper output identical to `btoa` for random byte arrays across sizes (property test).
**Status:** Not Started

## Stage 3: Background warm/keep-alive hygiene (WP-60, WP-61, WP-62, WP-63)

**Files / fixes:**
- `entries/background.ts:971-976`: `ensureSidepanelOpen` once per funnel (guard `Set<funnelId>`), not per broadcast (`grep -n "ensureSidepanelOpen(tabId)" entries/background.ts`).
- `entries/shared/background-init.ts`: hourly alarm drops `force=true` (TTL cache makes warm a no-op when fresh — overlaps W2 Stage 3); `syncWebClipperContextMenu` gate on `nextUrl !== prevUrl` instead of every config write, and let it use the 5-min capabilities TTL (anchor: `grep -n "forceRefresh: true" entries/shared/background-init.ts`).
- Funnel metrics (m1): keep the in-memory array; flush through the existing coalescing writer instead of read-modify-write per event (bounded 200 — low priority; fold in only if touching the file anyway).
- Effective-config per-request reads: covered by W4 Stage 5 (`single-user-credential.ts`); no duplicate work here.
**Success Criteria:** Idle hour with sidepanel already open: 0 forced catalog fetches, 0 redundant `sidePanel.setOptions/open` calls, config-watcher refreshes only on URL change.
**Tests:** Unit: alarm handler with warm cache performs 0 network calls; two broadcasts for one funnel call `setOptions` once; context-menu sync no-ops on unrelated config key change.
**Status:** Not Started

## Verification & DoD

- [ ] Each stage: failing test first, then implementation, scoped `vitest run` green; commit referencing TASK-13525.
- [ ] `bun run test:e2e:perf` green; W0 copilot/clipper manual measurements re-run and recorded in the baseline doc.
- [ ] Host-page main-thread long-task count during clip + copilot stream recorded before/after.
- [ ] `bun run lint` clean on touched paths.
