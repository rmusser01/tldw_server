# `/media` Page — Senior UX UAT Review

**Date:** 2026-06-14
**Reviewer role:** Senior UX specialist (User Acceptance Test)
**Build under test:** dev branch (`origin/dev` @ `62e3ba3298`), Next.js frontend on `:8080`, FastAPI backend on `:8000` (single-user, **~922 media items** ingested). Scope per request: deep on `/media` (`ViewMediaPage`); light smoke of `/media-multi` and `/media-trash`.
**Method:** Live browser, driven through real Chromium via Playwright (`apps/tldw-frontend/scripts/media-uat-driver.mjs` + focused probes). Auth via the env-driven smoke config (`seedAuth`). Personas: **first-time** and **power user**.
**Artifacts:** screenshots + `observations.json` in `./assets/media-uat-2026-06-14/`.

> **Note on fixes:** Unlike `/characters` (which had a clean one-line root-cause fix), every concrete `/media` finding below resists a clean, low-risk *frontend-only* fix — they need either a backend batch endpoint, a deeper consolidation of duplicate fetch paths, or are not reliably reproducible. They are documented here with evidence and recommended approaches rather than patched under time pressure. See the remediation plan.

---

## Summary

**First-time user:** Capable but heavy. `/media` ("Media Inspector") is a three-pane research tool: a deep faceted **filter rail** (search, full-text/metadata mode, sort, date range, favorites, media types, keywords/exclude-keywords, collections), a results list, and a reading/content pane that starts in a clean empty state. The breadth is impressive but front-loads a lot of controls before a first result is opened.

**Power user:** Strong feature set — full-text + metadata search, faceted filters with active-filter chips, bulk mode, collections, a rich `ContentViewer` (metadata, document intelligence, find-bar, read-along), ingest jobs, library stats, and well-labeled keyboard navigation. The page is stable (0 uncaught page errors). The concerns are **performance/correctness under the hood**: redundant list fetches, an N+1 per-item progress fetch, and a transient render-loop warning.

---

## What works well (keep)
- **Faceted search/filter** with active-filter chips and full-text vs metadata modes.
- **Clean content-viewer empty state** (`content-viewer-empty`) before an item is selected.
- **Well-labeled keyboard affordances** (tab order: Clear search → Clear search and filters → Search syntax help → … → Search).
- **Stability**: no uncaught page errors across load/search/detail/mobile.

---

## Findings

| # | Area | Issue | Severity | Status |
|---|------|-------|----------|--------|
| 1 | Redundant list fetch (shared transport) | On passive load the media list is fetched **~10×**, with **every request duplicated as both `/media?…` and `/media/?…`**. **Root cause traced to the shared `bgRequest` transport, not a `/media` caller:** `bgRequest` races the background `sendMessage` against a timeout (`background-proxy.ts:619`) and, on timeout, fires a **direct fallback** (`:660`) — but the background fetch is already in flight, so **both** hit the network (background sends the raw `/media/?…` path; direct sends the normalized `/media?…`). Confirmed: removing the trailing slash from `useMediaSearch`'s paths did **not** remove the `/media/?…` variant, and those requests are invisible to page `fetch`/XHR (they run in the background/SW transport). This is **app-wide** (affects any request that hits the fallback timeout), not `/media`-specific. | **High** (perf, app-wide) | Open |
| 2 | N+1 per-item fetch | The reading-progress badges trigger **one `/api/v1/media/{id}/progress` request per visible result** — deterministically **37** requests across a load+search (≈20 per list). Sequential `await` per item in `useMediaSelection.ts:190`. | **High** (perf) | Open |
| 3 | Render loop | A **"Maximum update depth exceeded"** React warning was observed **once** in `MediaPageContent` during a load+open-item sequence, but **did not reproduce** (0/3 in follow-up runs). Likely candidate: the reading-progress effect (`setReadingProgressMap(new Map())`) re-running on an unstable `displayResults` reference. | Medium (intermittent) | Open |
| 4 | No-match feedback | Searching a string with no matches did not surface an obvious "no results for…" message in the content area; the header still showed the library total. Empty-vs-no-match states should be distinct. *(Needs a focused repro; flagged by code exploration too.)* | Low/Medium | Open (verify) |

---

## Recommended fixes (for owned follow-up)

1. **#1 — Fix the `bgRequest` timeout-race double-send (app-wide).** When the background `sendMessage` is slow and the timeout wins, abort/ignore the in-flight background request before issuing the direct fallback (or don't race a fallback while the original may still land) so a single logical GET produces one network request. This is shared-request-layer surgery (`services/background-proxy.ts` ~538–681) used by the entire app — it needs an owned change with broad regression testing, not a `/media`-local patch. (Attempted in this UAT: removing the trailing slash in `useMediaSearch` and normalizing the coalescing key both had **no effect**, confirming the duplication is in the transport, not the caller.)
2. **#2 — Batch reading-progress.** Add/await a batch progress endpoint (e.g. `POST /media/progress` with the visible ids) instead of N sequential `/media/{id}/progress` calls; or defer progress to the selected item only. Frontend source: `useMediaSelection.ts:190`.
3. **#3 — Stabilize the reading-progress effect.** Gate the effect on a derived stable ID-key (e.g. `ids.join(',')`) rather than the `displayResults` array reference, so it re-runs only when the set of media ids changes — removing both the redundant re-fetch and the setState-on-every-render that risks the update-depth loop.
4. **#4 — Distinct no-match state.** Ensure a clear "No results match your filters" message (with a clear-filters action) distinct from "library empty".

A regression test scaffold is included: `apps/tldw-frontend/e2e/media-render-loop.spec.ts` asserts no "Maximum update depth" error on `/media` load (currently passes — guards against a regression).

---

## Coverage & limitations
- **Covered live:** load (922 items, filter rail, empty content pane), full-text search, item open, no-match search, keyboard tab order, mobile (390px), console/page-error + `/api` request-count capture, and focused probes for the render loop (3×) and list-fetch URLs/timing.
- **Light smoke only (per scope):** `/media-multi` and `/media-trash` were not deep-dived.
- **Not exhausted:** the 86K-line `ContentViewer` sub-features (intelligence, annotations, read-along, diff/versions, comparison) beyond primary surfaces.
- **Fix posture:** no code fix landed — the findings need careful, owned fixes (backend batch endpoint, fetch-path consolidation) rather than a rushed patch.

## Reproduction
```bash
cd apps/tldw-frontend
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev -- -p 8080
node scripts/media-uat-driver.mjs    # full walkthrough + screenshots
node scripts/media-list-probe.mjs    # shows the ~10 list requests + slash-variant duplication
node scripts/media-loop-probe.mjs    # progress N+1 (37) + render-loop occurrence over 3 runs
```
