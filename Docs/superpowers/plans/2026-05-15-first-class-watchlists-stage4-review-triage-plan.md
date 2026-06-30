# First-Class Watchlists Stage 4 Review Queue And Triage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the selected Watchlist Items/Updates surface efficient for CTI/OSINT and news triage by turning the current local review controls into a server-authoritative, alert-aware, Watchlist-scoped review queue.

**Architecture:** Preserve the existing `scraped_items` review fields and current ItemsTab reader as the foundation. Add only the backend contract needed for reliable filtering, sorting, saved views, alert context, and batch triage at Watchlist scale. Keep generated report construction in Stage 5; Stage 4 only prepares and hands off reviewed or queued evidence.

**Tech Stack:** FastAPI, Pydantic, WatchlistsDatabase, SQLite/Postgres migrations where needed, React, Ant Design, Zustand Watchlists store, Vitest, pytest, Bandit, Playwright/CDP for real-browser verification.

---

## Source Spec

- Design spec: `Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md`
- Stage 1 plan: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md`
- Stage 2 plan: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md`
- Stage 3 plan: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage3-content-alerts-plan.md`
- Stage 4 planning task: `TASK-349.1`
- API docs: `Docs/API-related/Watchlists_API.md`

## Current Evidence

- `WatchlistsDatabase.list_items()` currently filters by `run_id`, `job_id`, `source_id`, `watchlist_id`, `status`, `reviewed`, `queued_for_briefing`, text query, `since`, and `until`, but always orders by `si.created_at DESC`.
- `ScrapedItem` currently exposes item identity, source/run/job IDs, URL, title, summary, content, published timestamp, tags, `status`, `reviewed`, `queued_for_briefing`, and `created_at`.
- `PATCH /api/v1/watchlists/items/{item_id}` updates one item at a time and records companion activity when `reviewed`, `status`, or `queued_for_briefing` changes.
- `GET /api/v1/watchlists/items/smart-counts` already provides all/today/today unread/unread/reviewed/queued counts for the same item filters.
- Stage 3 added `watchlist_content_alerts` with `watchlist_id`, `rule_id`, `item_id`, `run_id`, `job_id`, `source_id`, `severity`, `status`, `matched_text`, `snippet`, and evidence JSON.
- The frontend `ItemsTab` already has source filtering, search, status filters, smart filters, local sorting, localStorage saved views, per-item review toggles, queued-for-briefing toggles, selected/page/all-filtered batch review controls, keyboard shortcuts, chat handoff, and report generation from queued items.
- Current batch review is client-driven. The all-filtered path fetches matching item IDs page by page and sends one `PATCH` request per item.
- Current saved views are localStorage-only through `ITEMS_VIEW_PRESETS_STORAGE_KEY`; they are not persisted per Watchlist and source IDs can collide across Watchlists.
- Current item sorting is applied client-side to the current page only, so sort order is not authoritative across pages.
- Current item rows do not show content alert context even though Stage 3 alerts now reference item IDs.
- Current UX copy still uses "Articles" in many Watchlists labels. Stage 4 should contextually move the selected Watchlist review surface toward "Updates" while preserving route/API compatibility.

## Product Decisions For Stage 4

- "Updates" is the user-facing label for the selected Watchlist review queue. Existing API names and compatibility aliases can continue to say `items` or `articles` where changing them would create churn.
- Server-side sorting and filtering are required before adding more triage controls. Client-side sorting may remain as a fallback only for already loaded data, not as the primary truth for paginated review.
- Stage 4 should not invent confidence or novelty scores when no enrichment field exists. It should add explicit placeholders and filters only when backed by persisted data or Stage 3 alert severity/evidence.
- Alert-aware triage should be based on Stage 3 content alert records. Items can be filtered by alert presence/status/severity/rule and can display a compact alert summary, but full alert management remains in the Alerts tab.
- Batch triage must move to a backend endpoint for selected IDs and filter scopes so large Watchlists do not require hundreds or thousands of client-side PATCH requests.
- Saved views must be scoped by `user_id` and `watchlist_id`. The first migration can preserve localStorage custom views as a client-side import fallback, but server-saved views are the product contract.
- Stage 4 report handoff means queueing items for a future report/briefing and generating from the existing queue path. Defensible report building, evidence tables, immutable snapshots, and weak-evidence warnings are Stage 5.
- Existing `reviewed` and `queued_for_briefing` fields remain the first triage states. Do not overload ingestion `status` with report inclusion semantics unless the API explicitly documents that choice.

## Implementation Boundaries

- Do not rename `/watchlists` or break existing `/api/v1/watchlists/items` clients.
- Do not remove existing `reviewed`, `queued_for_briefing`, smart counts, or per-item PATCH behavior.
- Do not replace the current ItemsTab reader in the first slice. Improve it in place.
- Do not add LLM-based novelty/confidence scoring in Stage 4.
- Do not build the Stage 5 report builder.
- Do not make saved views browser-local as the final contract.
- Do not use Computer Use for browser QA; use Playwright/CDP against the real WebUI/server when doing browser verification.

## Proposed File Responsibilities

Backend:

- `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
  - Add item list sort modes and alert-aware filters.
  - Add optional alert summary aggregation for items.
  - Add `watchlist_item_saved_views` persistence for per-Watchlist saved views.
  - Add batch item update helper for explicit item IDs and filter-based scopes.
- `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
  - Add item sort/filter enum-style literals.
  - Add item alert summary response model.
  - Add saved view create/update/response schemas.
  - Add batch item update request/response schemas.
- `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Extend `GET /items` and `/items/smart-counts` without breaking existing query parameters.
  - Add static batch route before `/items/{item_id}`.
  - Add nested saved-view routes under a selected Watchlist.
- `tldw_Server_API/app/core/Personalization/companion_activity.py`
  - Preserve current item update activity behavior. Extend only if batch updates need a compact activity record.

Frontend:

- `apps/packages/ui/src/types/watchlists.ts`
  - Add alert summary, item sort mode, saved view, and batch update types.
- `apps/packages/ui/src/services/watchlists.ts`
  - Add query serialization for new item filters/sort.
  - Add saved view CRUD methods.
  - Add batch item update method.
- `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/items-utils.ts`
  - Normalize server-backed sort/filter/saved view state.
  - Keep localStorage migration helpers for legacy saved views.
- `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/ItemsTab.tsx`
  - Rename visible selected-Watchlist review copy toward Updates.
  - Use server sort/filter parameters instead of current-page sort for primary order.
  - Display alert match context on item rows and reader.
  - Use backend batch endpoint for selected/page/all-filtered review actions.
  - Load and persist saved views per Watchlist through the API.
- `apps/packages/ui/src/store/watchlists.tsx`
  - Keep global tab state, but reset or scope item filters and active saved views by selected Watchlist where needed.
- `apps/packages/ui/src/assets/locale/en/watchlists.json`
  - Add Updates, alert-match, saved-view, batch-result, and server-filter copy.
- `apps/packages/ui/src/public/_locales/en/watchlists.json`
  - Mirror locale changes if still maintained manually.

Docs:

- `Docs/API-related/Watchlists_API.md`
  - Document item sort/filter additions, item alert summary, batch triage, and saved views.
- `Docs/Published/API-related/Watchlists_API.md`
  - Mirror API docs if this repo keeps the published copy in sync.

Tests:

- `tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py`
  - API coverage for sorting, alert-aware filters, saved views, batch updates, and scoping.
- `tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py`
  - DB coverage for sort stability, alert aggregation, saved view persistence, and batch helper behavior.
- `apps/packages/ui/src/services/__tests__/watchlists-items-triage.test.ts`
  - Service query serialization, saved view routes, and batch endpoint payloads.
- `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.stage4-triage.test.tsx`
  - Server-backed filters/sort, alert chips, saved view load/save, and backend batch action calls.
- `apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlists-stage4-copy-contract.test.ts`
  - Static copy contract for Updates, content alert matches, health issue separation, and Stage 5 report boundary.

## Backlog Task Map For Implementation

Create implementation tasks before code changes:

- Stage 4A: Backend item triage query contract and alert summary.
- Stage 4B: Backend batch triage and saved views API.
- Stage 4C: Frontend service/types and saved-view migration.
- Stage 4D: Items/Updates tab alert-aware triage refresh.
- Stage 4E: API docs, real-server CDP smoke, and closeout.

Keep commits aligned to these task groups.

## Task 0: Baseline And Task Setup

**Files:**
- Reference: `Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md`
- Reference: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage4-review-triage-plan.md`
- Reference: `backlog/tasks/task-349.1 - Plan-Stage-4-Watchlist-review-queue-and-triage.md`

- [ ] **Step 1: Create Stage 4 implementation Backlog tasks**

Use Backlog from the active worktree to create Stage 4A-4E tasks listed above. Each task must reference this plan, the design spec, the Stage 3 plan, and the API docs.

- [ ] **Step 2: Capture backend baseline**

Run:

```bash
rg -n "list_items\\(|get_item_smart_counts|update_item_flags|watchlist_content_alerts|items/smart-counts|/items" \
  tldw_Server_API/app/core/DB_Management/Watchlists_DB.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/tests/Watchlists
```

Expected: current item filters, single-item update behavior, smart counts, and Stage 3 alert records are identified before edits.

- [ ] **Step 3: Capture frontend baseline**

Run:

```bash
rg -n "ITEMS_VIEW_PRESETS_STORAGE_KEY|fetchScrapedItems|fetchScrapedItemSmartCounts|updateScrapedItem|queued_for_briefing|sortMode|savedViews|Articles|Updates" \
  apps/packages/ui/src/components/Option/Watchlists/ItemsTab \
  apps/packages/ui/src/services/watchlists.ts \
  apps/packages/ui/src/types/watchlists.ts \
  apps/packages/ui/src/store/watchlists.tsx \
  apps/packages/ui/src/assets/locale/en/watchlists.json
```

Expected: current local sorting, localStorage saved views, batch review behavior, queue handoff, and visible copy are identified before edits.

- [ ] **Step 4: Run focused baseline tests**

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.batch-controls.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.keyboard-shortcuts.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts \
  src/services/__tests__/watchlists-first-class.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Run from the repo root:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_api.py \
  -q
```

Expected: establish current pass/fail baseline before Stage 4 changes.

- [ ] **Step 5: Commit task records**

Run:

```bash
git add backlog/tasks/<stage-4-task-files>
git commit -m "chore: task watchlists stage 4 setup"
```

Expected: only task records are committed.

## Task 1: Backend Item Triage Query Contract And Alert Summary

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Create: `tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py`
- Create or modify: `tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py`

- [ ] **Step 1: Add failing DB/API tests first**

Cover:

- Stable server-side sort modes: `created_desc`, `created_asc`, `published_desc`, `published_asc`, `unread_first`, `source_asc`, and `alert_severity_desc`.
- Existing filters continue to work with `watchlist_id`.
- Alert filters: `has_alert`, `alert_status`, `alert_severity`, and `alert_rule_id`.
- Optional `include_alert_summary=true` returns compact counts/highest severity/latest alert for each item without duplicating rows.
- Missing confidence/novelty fields are not exposed as fake filters.
- Static route ordering still protects `/items/smart-counts` and future `/items/batch-update` from `/items/{item_id}`.

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py \
  -q
```

Expected: tests fail before implementation.

- [ ] **Step 2: Implement DB sort/filter contract**

Extend `list_items()` and `get_item_smart_counts()` narrowly:

- Add validated sort enum input.
- Add alert-aware joins or correlated aggregate helpers behind explicit filter/include flags.
- Keep default behavior identical to today: `created_desc`, no alert summary.
- Avoid unbounded joins that multiply item rows.

- [ ] **Step 3: Extend schemas and endpoint query params**

Add response models such as:

- `ScrapedItemAlertSummary`
- Optional `alert_summary` on `ScrapedItem`

Add query params to `GET /items`:

- `sort`
- `has_alert`
- `alert_status`
- `alert_severity`
- `alert_rule_id`
- `include_alert_summary`

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_api.py \
  -q
```

Commit:

```bash
git add \
  tldw_Server_API/app/core/DB_Management/Watchlists_DB.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py \
  backlog/tasks/<stage-4a-task-file>
git commit -m "feat: add watchlist item triage query contract"
```

## Task 2: Backend Batch Triage And Saved Views API

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Modify: `tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py`
- Modify: `tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py`

- [ ] **Step 1: Add failing tests first**

Cover:

- Batch update by explicit `item_ids`.
- Batch update by current filter scope under a required `watchlist_id`.
- Batch result reports matched, changed, unchanged, failed IDs, and capped/exhausted state when a limit is hit.
- Batch updates can set `reviewed` and `queued_for_briefing`; `status` updates remain explicit and validated.
- Batch updates do not cross users or Watchlists.
- Batch updates record companion activity in a bounded way.
- Saved views CRUD is scoped by `user_id` and `watchlist_id`.
- Saved views reject invalid filter/sort payloads and source IDs outside the selected Watchlist.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py -q
```

Expected: tests fail before implementation.

- [ ] **Step 2: Add saved-view persistence**

Add a table such as `watchlist_item_saved_views`:

- `id`
- `user_id`
- `watchlist_id`
- `name`
- `filters_json`
- `sort`
- `is_default`
- `created_at`
- `updated_at`

Keep system defaults client-defined for now unless a backend default row is explicitly needed. Persist only custom user views.

- [ ] **Step 3: Add batch update helper and endpoint**

Add a static route before `/items/{item_id}`, for example:

- `POST /api/v1/watchlists/items/batch-update`

Request modes:

- `item_ids`: explicit selected/page IDs.
- `scope`: filter payload plus required `watchlist_id` for all-filtered operations.

The endpoint should return a deterministic summary and avoid fetching item content unnecessarily.

- [ ] **Step 4: Add saved view routes**

Suggested routes:

- `GET /api/v1/watchlists/{watchlist_id}/item-views`
- `POST /api/v1/watchlists/{watchlist_id}/item-views`
- `PATCH /api/v1/watchlists/{watchlist_id}/item-views/{view_id}`
- `DELETE /api/v1/watchlists/{watchlist_id}/item-views/{view_id}`

Keep these routes after static `/items/...` routes but before any ambiguous dynamic Watchlist routes if route order requires it.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py \
  -q
```

Commit:

```bash
git add \
  tldw_Server_API/app/core/DB_Management/Watchlists_DB.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py \
  backlog/tasks/<stage-4b-task-file>
git commit -m "feat: add watchlist item batch triage api"
```

## Task 3: Frontend Services Types And Saved-View Migration

**Files:**
- Modify: `apps/packages/ui/src/types/watchlists.ts`
- Modify: `apps/packages/ui/src/services/watchlists.ts`
- Create: `apps/packages/ui/src/services/__tests__/watchlists-items-triage.test.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/items-utils.ts`
- Modify or create tests under `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/`

- [ ] **Step 1: Add failing service and utility tests first**

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/services/__tests__/watchlists-items-triage.test.ts \
  src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Expected: tests fail until new service methods and utility behavior are added.

- [ ] **Step 2: Add TypeScript contracts**

Add:

- Item sort/filter types matching backend literals.
- Optional `alert_summary` on `ScrapedItem`.
- Saved view request/response types.
- Batch update request/response types.

- [ ] **Step 3: Add service methods**

Add:

- New query params to `fetchScrapedItems()` and `fetchScrapedItemSmartCounts()`.
- `batchUpdateScrapedItems()`.
- `fetchWatchlistItemViews()`.
- `createWatchlistItemView()`.
- `updateWatchlistItemView()`.
- `deleteWatchlistItemView()`.

- [ ] **Step 4: Add localStorage migration helpers**

Keep current localStorage views readable, but normalize them into server-create payloads for the selected Watchlist. Migration should be explicit and recoverable; do not silently delete local views until a server save succeeds.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/services/__tests__/watchlists-items-triage.test.ts \
  src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Commit:

```bash
git add \
  apps/packages/ui/src/types/watchlists.ts \
  apps/packages/ui/src/services/watchlists.ts \
  apps/packages/ui/src/services/__tests__/watchlists-items-triage.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/ItemsTab/items-utils.ts \
  apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts \
  backlog/tasks/<stage-4c-task-file>
git commit -m "feat: add watchlist item triage client contract"
```

## Task 4: Items/Updates Tab Alert-Aware Triage Refresh

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/ItemsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/items-utils.ts`
- Modify: `apps/packages/ui/src/store/watchlists.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/watchlists.json`
- Modify if mirrored: `apps/packages/ui/src/public/_locales/en/watchlists.json`
- Create: `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.stage4-triage.test.tsx`
- Create: `apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlists-stage4-copy-contract.test.ts`
- Modify existing ItemsTab tests where behavior intentionally moves from local loops to backend batch calls.

- [ ] **Step 1: Add failing component and copy tests first**

Cover:

- Visible selected-Watchlist copy uses Updates/review queue language where appropriate.
- Existing Articles aliases do not break route or old test assumptions outside the selected review context.
- Sort/filter controls send backend query params.
- Alert summary chips appear on rows and reader when `alert_summary` is present.
- Alert-match filters request `has_alert`, `alert_status`, `alert_severity`, and `alert_rule_id`.
- Saved views load from the selected Watchlist and save through the API.
- Legacy localStorage saved views can be imported without losing them on failed save.
- Batch selected/page/all-filtered actions call the backend batch endpoint and show returned success/partial/failure counts.
- Extension-sized layout keeps primary review, filter, batch, and reader controls reachable.

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.stage4-triage.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.batch-controls.test.tsx \
  src/components/Option/Watchlists/__tests__/watchlists-stage4-copy-contract.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Expected: tests fail until the tab uses the new contract.

- [ ] **Step 2: Replace primary page sort with server sort**

Thread selected sort mode into `fetchScrapedItems()`. Keep `sortItemsForReader()` only for legacy fallback or local tests that intentionally pass unsorted data.

- [ ] **Step 3: Add alert-aware filters and row context**

Add compact content-alert context:

- Highest severity.
- Unread/dismissed count where available.
- Matched rule or matched text snippet when provided.
- One action to open the Alerts tab filtered by the selected item or Watchlist context if the Alerts tab supports it.

- [ ] **Step 4: Use backend batch endpoint**

Replace selected/page/all-filtered review loops with `batchUpdateScrapedItems()`. Preserve progress and partial-failure feedback, but base counts on the server response.

- [ ] **Step 5: Move saved views to the backend contract**

Load custom saved views from the selected Watchlist. Show an import affordance when legacy localStorage views exist. Save/update/delete custom views through the API.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.stage4-triage.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.batch-controls.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.keyboard-shortcuts.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.accessibility-baseline.test.tsx \
  src/components/Option/Watchlists/__tests__/watchlists-stage4-copy-contract.test.ts \
  src/services/__tests__/watchlists-items-triage.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Commit:

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/ItemsTab \
  apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlists-stage4-copy-contract.test.ts \
  apps/packages/ui/src/store/watchlists.tsx \
  apps/packages/ui/src/assets/locale/en/watchlists.json \
  apps/packages/ui/src/public/_locales/en/watchlists.json \
  backlog/tasks/<stage-4d-task-file>
git commit -m "feat: refresh watchlist updates triage"
```

## Task 5: Documentation Real-Server QA And Closeout

**Files:**
- Modify: `Docs/API-related/Watchlists_API.md`
- Modify if mirrored: `Docs/Published/API-related/Watchlists_API.md`
- Modify: relevant Stage 4 Backlog task files.

- [ ] **Step 1: Update API docs**

Document:

- Item sort modes and alert-aware filters.
- Optional item alert summary.
- Batch item triage endpoint.
- Saved views endpoints.
- Stage 4 boundary: review queue and report handoff, not defensible report artifacts.

- [ ] **Step 2: Run backend verification**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_db.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_api.py \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py \
  -q
```

Run Bandit on touched backend files:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/DB_Management/Watchlists_DB.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  -f json -o /tmp/bandit_watchlists_stage4_review_triage.json
```

- [ ] **Step 3: Run frontend verification**

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/services/__tests__/watchlists-items-triage.test.ts \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.stage4-triage.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.batch-controls.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.keyboard-shortcuts.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.accessibility-baseline.test.tsx \
  src/components/Option/Watchlists/__tests__/watchlists-stage4-copy-contract.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Run `git diff --check`.

- [ ] **Step 4: Real-server CDP smoke**

Use the real FastAPI server and real WebUI. Do not mock the server.

Smoke flows:

- Open `/watchlists?tab=items` or the current selected-Watchlist route alias.
- Verify Updates/Items list loads from the real API.
- Apply an alert-match filter against seeded content alerts.
- Save or import a saved view for the selected Watchlist.
- Batch mark selected/page items reviewed through the backend batch endpoint.
- Toggle report queue state and verify the queued view.
- Capture desktop and extension-sized screenshots.

- [ ] **Step 5: Close tasks and commit**

Update Backlog Stage 4 tasks with verification notes, known skips, and final summaries.

Commit:

```bash
git add \
  Docs/API-related/Watchlists_API.md \
  Docs/Published/API-related/Watchlists_API.md \
  backlog/tasks/<stage-4-task-files>
git commit -m "docs: close watchlist updates triage stage"
```

## Stage 4 Exit Criteria

- `/watchlists` still opens and preserves the selected Watchlist shell.
- Existing unscoped item list clients still work.
- Item filtering/sorting is server-authoritative across pages.
- Users can filter Updates by content alert match context.
- Users can batch-review selected/page/all-filtered items without one request per item.
- Users can save review views per Watchlist and recover legacy local views.
- CTI/OSINT users can prioritize alert-bearing, severe, source-specific, and unread Updates.
- News users can prioritize recent, unread, source-specific, and queued Updates.
- Report handoff through `queued_for_briefing` remains intact.
- Stage 5 report builder remains a separate follow-up.

## Known Deferrals To Later Stages

- Novelty scoring and confidence scoring require enrichment data that does not currently exist.
- Defensible report builder, evidence tables, immutable published report snapshots, and weak-evidence warnings are Stage 5.
- Full constrained-viewport management across every Watchlists tab is Stage 6, though Stage 4 changed surfaces must pass extension-sized smoke.
- Broader trust/calibration copy around why an item matched and alert dedupe explanations is Stage 7 unless Stage 3 alert evidence already supports a cheap display.
