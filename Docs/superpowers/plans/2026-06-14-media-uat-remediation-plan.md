# Media Page UAT Remediation Plan

**Created:** 2026-06-14
**Source review:** `Docs/Design/2026-06-14-media-page-uat-review.md`
**Scope:** `/media` (`ViewMediaPage` + `useMediaSearch`/`useMediaSelection`).

`/media` is a large, feature-rich page. The concrete findings are real but each needs a careful, owned fix (backend support or fetch-path consolidation) rather than a quick patch — so this PR ships the **documented findings + a regression scaffold**, not a code fix. Each stage below is a sequenced follow-up.

---

## Stage 1: Serve the media list without a 307 trailing-slash redirect — **Done (PR #2353)**
**Finding:** #1. On passive `/media` load the list is fetched ~10×, every request duplicated as `/media?…` and `/media/?…`.
**Root cause (corrected):** a **307 trailing-slash redirect** — **not** a `bgRequest` double-send (an earlier draft said that; the web shim's `sendMessage` is a no-op, so that path never runs). The list route was registered only for the slash form (`@router.get("/")` in `tldw_Server_API/app/api/v1/endpoints/media/listing.py`), so FastAPI's default `redirect_slashes` 307-redirects `/api/v1/media` → `/api/v1/media/`. The client normalizes media-list URLs to the no-slash form (`normalizeKnownPathQuirks`, for the proxy-404 case), so each call is `/media?…` (307) → followed to `/media/?…` (200). Confirmed via CDP initiator (the `/media/?…` are `type:other`/no-stack redirect-follows) and `curl` (307 no-slash → 200 slash). Two ruled-out attempts: removing the trailing slash in `useMediaSearch` and normalizing the coalescing key both had **no effect** (the redirect is server-side).
**Fix:** register both `""` and `"/"` on the `list_media_endpoint` handler so the no-slash form serves 200 directly. Client unchanged.
**Tests:** `tests/MediaIngestion_NEW/unit/test_media_list_no_slash_redirect.py` (both forms registered; neither 307s) — red before, green after.
**Status:** Completed (PR #2353).

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
- The `bgRequest` GET-coalescing key was briefly tested with `normalizeKnownPathQuirks(init.path)` (CodeRabbit's deferred #2350 suggestion) to collapse the slash variants — it had **no effect** because the duplicate `/media/?…` is a server-side **307 redirect-follow**, not a client request, so it was reverted. The real fix was the backend 307 redirect (Stage 1, PR #2353).
- `/media-multi` and `/media-trash` were only smoke-checked per the agreed scope.
