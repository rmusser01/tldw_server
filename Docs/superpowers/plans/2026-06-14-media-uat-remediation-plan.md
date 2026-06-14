# Media Page UAT Remediation Plan

**Created:** 2026-06-14
**Source review:** `Docs/Design/2026-06-14-media-page-uat-review.md`
**Scope:** `/media` (`ViewMediaPage` + `useMediaSearch`/`useMediaSelection`).

`/media` is a large, feature-rich page. The concrete findings are real but each needs a careful, owned fix (backend support or fetch-path consolidation) rather than a quick patch — so this PR ships the **documented findings + a regression scaffold**, not a code fix. Each stage below is a sequenced follow-up.

---

## Stage 1: Fix the `bgRequest` timeout-race double-send — **High (perf, app-wide), Open**
**Finding:** #1. On passive `/media` load the list is fetched ~10×, every request duplicated as `/media?…` and `/media/?…`.
**Root cause (traced during this UAT):** the duplication is **not** a `/media` caller — it is in the shared `bgRequest` transport. `bgRequest` races the background `sendMessage` against a timeout (`services/background-proxy.ts:619`); on timeout it fires a **direct fallback** (`:660`) while the background fetch is still in flight, so both reach the network (background = raw `/media/?…`, direct = normalized `/media?…`). Two attempts confirmed the caller is not at fault: removing the trailing slash in `useMediaSearch` (lines 404/854/867) and normalizing the coalescing key (`normalizeKnownPathQuirks(init.path)`) both had **no effect**; the `/media/?…` requests are invisible to page `fetch`/XHR (they run in the background transport).
**Approach (owned, app-wide):** when the timeout wins the race, abort/ignore the in-flight `sendMessage` request before issuing the direct fallback (e.g. propagate an AbortController, or have the background handler short-circuit when a fallback is taken), so one logical GET = one network request. This is core-transport surgery used by **every** page — it needs broad regression testing, not a `/media`-local change.
**Success:** `scripts/media-list-probe.mjs` shows each logical query fetched once (no `/media/?…` twin); reduced request counts app-wide.
**Tests:** unit tests in `background-proxy.test.ts` (timeout fallback does not double-send); e2e network assertion.
**Status:** Not Started (deferred — shared transport, out of scope for a `/media`-only PR).

## Stage 2: Batch the per-item reading-progress fetch — **High (perf), Open**
**Finding:** #2. `useMediaSelection.ts:190` fetches `/api/v1/media/{id}/progress` sequentially per visible result (~20/list, 37 across load+search). N+1.
**Approach:** add/await a **batch** progress endpoint (e.g. `POST /api/v1/media/progress` with the visible ids) — needs backend support — and replace the per-item loop; or defer progress to only the selected item. Guard with `readingProgressUnavailableRef` as today.
**Success:** ≤1 progress request per list render (or per selected item) instead of N.
**Tests:** e2e network count assertion on `/media/{id}/progress`.
**Status:** Not Started (needs backend endpoint).

## Stage 3: Stabilize the reading-progress effect (render-loop hardening) — **Medium, Open**
**Finding:** #3. "Maximum update depth exceeded" observed once (not reliably reproducible). The reading-progress effect re-runs on every `displayResults` reference change and `setReadingProgressMap(new Map())`.
**Approach:** depend on a derived **stable ID-key** (`ids.join(',')`) rather than the `displayResults` array reference, and reconstruct ids inside the effect — so it re-runs only when the id set changes. This removes redundant re-fetch and the setState-on-every-render that risks the loop. Low-risk and complements Stage 2.
**Success:** `e2e/media-render-loop.spec.ts` stays green; progress effect re-runs only on id-set change.
**Status:** Not Started (regression scaffold `media-render-loop.spec.ts` already added).

## Stage 4: Distinct no-match empty state — **Low/Medium, Open (verify first)**
**Finding:** #4. No-match search did not surface an obvious "no results" message distinct from an empty library.
**Approach:** verify with a focused repro (search a no-match term, inspect the results region); if confirmed, render a clear "No results match your filters" state with a clear-filters action, distinct from the empty-library first-ingest tutorial.
**Status:** Not Started (verify).

---

## Notes
- The `bgRequest` GET-coalescing key was briefly tested with `normalizeKnownPathQuirks(init.path)` (CodeRabbit's deferred #2350 suggestion) to collapse the slash variants — it had **no effect** here because the duplicate path bypasses `bgRequest` entirely, so it was reverted. The real fix is Stage 1 (consolidate the callers).
- `/media-multi` and `/media-trash` were only smoke-checked per the agreed scope.
