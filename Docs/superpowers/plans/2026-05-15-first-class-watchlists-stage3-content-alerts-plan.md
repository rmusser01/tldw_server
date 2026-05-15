# First-Class Watchlists Stage 3 Content Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class Watchlist content-match alerts for user-defined descriptors, classifications, entities, keywords, and source constraints, with an alert inbox that is clearly separate from pipeline health issues.

**Architecture:** Keep Watchlist as the product-owned container from Stages 1 and 2. Add Watchlists-owned content alert rules and alert records in the Watchlists data model so alerts can reference Watchlist items, runs, jobs, sources, and reports directly. Reuse Topic Monitoring behavior where it is already a good fit, especially regex matching patterns, snippets, dedupe ideas, and notification delivery, but do not persist product Watchlists in the separate `monitoring_watchlists` model. Preserve existing run-stat alert rules as backward-compatible health-rule behavior and expose them as health issues in the Watchlists UX.

**Tech Stack:** FastAPI, Pydantic, WatchlistsDatabase, SQLite/Postgres migrations, Watchlists pipeline, Monitoring notification service, React, Ant Design, Zustand Watchlists store, Vitest, pytest, Bandit, Playwright/CDP for browser verification.

---

## Source Spec

- Design spec: `Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md`
- Stage 1 plan: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md`
- Stage 2 plan: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md`
- Stage 3 planning task: `TASK-378`
- API docs: `Docs/API-related/Watchlists_API.md`

## Current Evidence

- Current Watchlists router is included through `tldw_Server_API/app/api/v1/router_groups/content.py` as `tldw_Server_API.app.api.v1.endpoints.watchlists` under `/api/v1`.
- Current run-stat alert-rule endpoint exists at `tldw_Server_API/app/api/v1/endpoints/watchlist_alert_rules.py` with prefix `/watchlists/alert-rules`, but the implementation is explicitly based on run statistics: `no_items`, `error_rate_above`, `items_below`, `items_above`, and `run_failed`.
- Current run-stat rule evaluator lives in `tldw_Server_API/app/core/Watchlists/alert_rules.py` and emits notification payloads with link type `watchlist_run`.
- Current run-stat alert-rule storage lives in `tldw_Server_API/app/core/DB_Management/watchlist_alert_rules_db.py` and is scoped by `user_id` plus optional `job_id`. It does not model content evidence or Watchlist item references.
- Topic Monitoring already has content alert primitives in `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py` and `tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py`, including `TopicAlert`, rule patterns, snippets, dedupe windows, read state, and notification delivery.
- Topic Monitoring uses a separate `monitoring_watchlists` model with string watchlist IDs and administrative monitoring scopes. That model should be treated as a reusable dependency, not the user-facing Watchlist storage source of truth.
- Watchlist items already persist content evidence through `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py` `ScrapedItemRow`, including `run_id`, `job_id`, `source_id`, `url`, `title`, `summary`, `content`, `published_at`, `tags_json`, `status`, `reviewed`, and `queued_for_briefing`.
- The pipeline records items through a local `_record_scraped` helper in `tldw_Server_API/app/core/Watchlists/pipeline.py`, which is the narrowest candidate hook for item-level content alert evaluation.
- The current frontend Watchlists page has Overview, Feeds/Sources, Monitors/Jobs, Activity/Runs, Articles/Items, Reports/Outputs, Templates, and Settings tabs, but no Alerts tab.

## Product Decisions For Stage 3

- "Alert" means a user-configured content match against newly collected Watchlist items.
- Run failures, no-item runs, high error rate, and source failures are "Health issues" or "Health rules", not generic alerts in the user-facing Watchlists UI.
- Content alert rules belong to a Watchlist. They may optionally constrain matching to specific source IDs, source labels/tags, item status, or source type.
- Content alerts must store evidence at creation time: matched rule, matched item, snippet, source, URL, run/job/source IDs, published timestamp when available, and enough metadata to reconstruct why the alert fired.
- A first implementation may support deterministic text/regex matching before richer NLP classification. The data model should leave room for descriptors/classifications/entities/IOCs without requiring the first slice to ship extraction models.
- Alert review state should support at least unread/read and dismissed. Acknowledge or snooze can be added in the same table if inexpensive, but the first UI can expose read/dismiss only if scope needs to stay tight.
- Notifications should use `NotificationService.notify_or_batch()` with a `type` such as `watchlist_content_alert`. Do not force content alerts into the `TopicAlert` dataclass unless a small adapter is created deliberately.
- Existing `/watchlists/alert-rules` run-stat behavior must not be broken. Add health-rule aliases or copy/docs changes first, then consider deprecation later.
- Stage 3 does not build full CTI entity extraction, novelty scoring, source-diversity analytics, or defensible report generation. It prepares evidence that later stages can use for reporting.

## Implementation Boundaries

- Do not rename `/watchlists`.
- Do not store product Watchlist content rules in `monitoring_watchlists`.
- Do not overload `watchlist_alert_rules` with content-matching fields.
- Do not remove or break existing run-stat alert-rule tests.
- Do not redesign the full Watchlists page. Add the Alerts surface inside the current selected-Watchlist shell.
- Do not require network access or external LLM providers for matching.
- Do not introduce a new frontend state library.
- Do not use Computer Use for browser QA; use Playwright/CDP.

## Proposed File Responsibilities

Backend:

- `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
  - Add content alert rule and content alert dataclasses.
  - Add `watchlist_content_alert_rules` and `watchlist_content_alerts` migrations for SQLite and Postgres.
  - Add rule CRUD helpers scoped by `user_id` and `watchlist_id`.
  - Add alert insert/list/get/update helpers scoped by `user_id` and `watchlist_id`.
  - Add dedupe helpers keyed by `watchlist_id`, `rule_id`, and `item_id` or item evidence hash.
- `tldw_Server_API/app/core/Watchlists/content_alerts.py`
  - Implement rule validation, matching, snippet generation, evidence payload construction, dedupe coordination, and notification dispatch.
  - Reuse or extract Topic Monitoring matching helpers only when doing so keeps Topic Monitoring behavior unchanged.
- `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
  - Add content alert rule create/update/response schemas.
  - Add content alert list/update/response schemas.
  - Add enum-style literals for rule kind, severity, review state, and match mode.
- `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Add nested endpoints under the already-registered Watchlists router.
  - Preserve route ordering so static collection paths are not shadowed.
  - Suggested routes:
    - `GET /api/v1/watchlists/{watchlist_id}/content-alert-rules`
    - `POST /api/v1/watchlists/{watchlist_id}/content-alert-rules`
    - `PATCH /api/v1/watchlists/{watchlist_id}/content-alert-rules/{rule_id}`
    - `DELETE /api/v1/watchlists/{watchlist_id}/content-alert-rules/{rule_id}`
    - `GET /api/v1/watchlists/{watchlist_id}/alerts`
    - `GET /api/v1/watchlists/{watchlist_id}/alerts/{alert_id}`
    - `PATCH /api/v1/watchlists/{watchlist_id}/alerts/{alert_id}`
- `tldw_Server_API/app/core/Watchlists/pipeline.py`
  - Evaluate enabled content alert rules immediately after a scraped item is recorded.
  - Pass item evidence and source/job/run context into `content_alerts.py`.
  - Treat alert creation as non-critical pipeline work and log failures without failing ingestion.
- `tldw_Server_API/app/api/v1/endpoints/watchlist_alert_rules.py`
  - Preserve behavior.
  - Add docs/copy or route alias work only if the endpoint is registered as part of the implementation slice.
- `tldw_Server_API/app/core/Watchlists/alert_rules.py`
  - Preserve run-stat evaluation behavior.
  - Rename emitted notification type or add a compatibility field so downstream consumers can distinguish health issues from content alerts.

Frontend:

- `apps/packages/ui/src/types/watchlists.ts`
  - Add `WatchlistContentAlertRule`, `WatchlistContentAlert`, rule payloads, alert update payloads, and list query types.
- `apps/packages/ui/src/services/watchlists.ts`
  - Add content alert rule CRUD methods.
  - Add alert list/detail/update methods.
  - Add query serialization for severity, review state, source ID, rule ID, since/until, and search text.
- `apps/packages/ui/src/store/watchlists.tsx`
  - Add selected-Watchlist content alert state only if the store already owns comparable tab state. Otherwise keep fetching local to the Alerts tab.
- `apps/packages/ui/src/components/Option/Watchlists/AlertsTab/AlertsTab.tsx`
  - Add content alert inbox, rule list, rule create/edit form, filters, and empty states.
  - Show matched evidence, source, item, severity, created time, review state, and actions.
  - Keep controls usable in extension-sized viewports.
- `apps/packages/ui/src/components/Option/Watchlists/AlertsTab/index.ts`
  - Re-export the tab.
- `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
  - Add Alerts tab to selected-Watchlist navigation.
  - Surface unread alert count in the tab label when available.
- `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
  - Add a small unread content alerts summary and a health issue summary, with distinct labels.
- `apps/packages/ui/src/assets/locale/en/watchlists.json`
  - Add Alerts, content rule, evidence, empty state, health issue, and review action copy.
- `apps/packages/ui/src/public/_locales/en/watchlists.json`
  - Mirror locale changes if still maintained manually.

Docs:

- `Docs/API-related/Watchlists_API.md`
  - Document content alert rules, content alerts, and the health-rule boundary.
- `Docs/Published/API-related/Watchlists_API.md`
  - Mirror API docs if this repo keeps the published copy in sync.

Tests:

- `tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_db.py`
  - DB migration, rule CRUD, alert insert/list/update, evidence JSON, dedupe, and user/watchlist scoping.
- `tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_api.py`
  - Nested endpoint CRUD, filters, review updates, 404/403-style scoping behavior, and validation errors.
- `tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_pipeline.py`
  - Item recording triggers alerts, source constraints work, dedupe prevents repeated alerts, and matcher errors do not fail ingestion.
- `tldw_Server_API/tests/test_watchlist_alert_rules.py`
  - Add or adjust tests only to preserve run-stat health-rule compatibility.
- `tldw_Server_API/tests/Monitoring/test_topic_monitoring.py`
  - Update only if matching helpers are extracted.
- `apps/packages/ui/src/services/__tests__/watchlists-content-alerts.test.ts`
  - Service route and query serialization coverage.
- `apps/packages/ui/src/components/Option/Watchlists/AlertsTab/__tests__/AlertsTab.test.tsx`
  - Rules, inbox, filters, evidence display, empty/error/loading states, and read/dismiss actions.
- `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx`
  - Alerts tab integration and selected-Watchlist scoping.
- `apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlists-stage3-copy-contract.test.ts`
  - Static copy contract that keeps "alert" for content matches and "health issue" for pipeline failures.

## Backlog Task Map For Implementation

Create implementation tasks before code changes:

- Stage 3A: Backend content alert persistence and matcher service.
- Stage 3B: Content alert API endpoints and docs.
- Stage 3C: Pipeline integration, notification dispatch, and health-rule separation.
- Stage 3D: Frontend services, types, Alerts tab, and copy.
- Stage 3E: Overview integration, constrained viewport CDP smoke, and closeout.

Keep commits aligned to these task groups.

## Task 0: Baseline And Task Setup

**Files:**
- Reference: `Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md`
- Reference: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage3-content-alerts-plan.md`
- Reference: `backlog/tasks/task-378 - Plan-Stage-3-Watchlist-content-match-alerts.md`

- [ ] **Step 1: Create Stage 3 implementation Backlog tasks**

Use Backlog from the active worktree to create Stage 3A-3E tasks listed above. Each task must reference this plan, the design spec, and the API docs.

- [ ] **Step 2: Capture backend baseline**

Run:

```bash
rg -n "watchlist_alert_rules|ALERT_CONDITION_TYPE_VALUES|evaluate_rules_for_run|record_scraped_item|evaluate_and_alert|TopicAlert|notify_or_batch" \
  tldw_Server_API/app \
  tldw_Server_API/tests
```

Expected: current health-rule implementation, Topic Monitoring primitives, notification service, and item-recording pipeline hook are identified before edits.

- [ ] **Step 3: Capture frontend baseline**

Run:

```bash
rg -n "alerts|Alert|health|tabs|WatchlistsTab|selectedWatchlist" \
  apps/packages/ui/src/components/Option/Watchlists \
  apps/packages/ui/src/services/watchlists.ts \
  apps/packages/ui/src/types/watchlists.ts \
  apps/packages/ui/src/assets/locale/en/watchlists.json
```

Expected: no existing content alert tab is present; any existing alert/health copy is identified before edits.

- [ ] **Step 4: Run focused baseline tests**

Run from the repo root:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/test_watchlist_alert_rules.py \
  tldw_Server_API/tests/Monitoring/test_topic_monitoring.py \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_db.py \
  -q
```

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx \
  src/services/__tests__/watchlists-first-class.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Expected: establish current pass/fail baseline before content alert changes.

- [ ] **Step 5: Commit task records**

Run:

```bash
git add backlog/tasks/<stage-3-task-files>
git commit -m "chore: task watchlists stage 3 setup"
```

Expected: only task records are committed.

## Task 1: Backend Content Alert Persistence And Matcher

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
- Create: `tldw_Server_API/app/core/Watchlists/content_alerts.py`
- Create: `tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_db.py`
- Create: `tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_pipeline.py`

- [ ] **Step 1: Add failing DB tests first**

Cover:

- Creating, listing, updating, disabling, and deleting content alert rules by `user_id` and `watchlist_id`.
- Rule validation for empty pattern, invalid regex, invalid severity, invalid kind, and missing Watchlist.
- Alert insert/list/get/update with evidence fields.
- Dedupe behavior for the same `watchlist_id`, `rule_id`, and `item_id`.
- User and Watchlist scoping.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_db.py -q
```

Expected: tests fail before implementation.

- [ ] **Step 2: Add Watchlists DB schema and helpers**

Add tables equivalent to:

- `watchlist_content_alert_rules`
  - `id`
  - `user_id`
  - `watchlist_id`
  - `name`
  - `enabled`
  - `rule_kind`
  - `match_mode`
  - `pattern`
  - `classification`
  - `descriptor`
  - `entity_type`
  - `source_constraints_json`
  - `severity`
  - `metadata_json`
  - `created_at`
  - `updated_at`
- `watchlist_content_alerts`
  - `id`
  - `user_id`
  - `watchlist_id`
  - `rule_id`
  - `item_id`
  - `run_id`
  - `job_id`
  - `source_id`
  - `severity`
  - `status`
  - `title`
  - `snippet`
  - `matched_text`
  - `evidence_json`
  - `dedupe_key`
  - `created_at`
  - `read_at`
  - `dismissed_at`

Use the existing Watchlists DB migration style for SQLite and Postgres. Add indexes for:

- `user_id`, `watchlist_id`, `enabled`
- `user_id`, `watchlist_id`, `status`, `created_at`
- `user_id`, `watchlist_id`, `rule_id`
- unique or conflict-safe `dedupe_key`

- [ ] **Step 3: Implement matcher service**

In `content_alerts.py`, implement:

- Rule compilation for literal, keyword, regex, descriptor, classification, and entity-style rule kinds.
- Safe regex handling with validation at rule save time.
- Source constraint matching against source ID, source type, tags, and URL when available.
- Text assembly from item title, summary, content, tags, and URL.
- Snippet generation around the matched span.
- Evidence payload construction with item/source/run/job references.
- Notification payload generation via `NotificationService.notify_or_batch()`.

Keep richer NLP classification as a later extension. For Stage 3, descriptor/classification/entity rules can map to deterministic text or metadata fields if no extractor is present.

- [ ] **Step 4: Add pipeline-level tests**

Cover:

- A newly recorded item matching a rule creates exactly one alert.
- Non-matching item creates no alert.
- Disabled rules are skipped.
- Source constraints work.
- Duplicate item/rule evaluation does not create duplicate alerts.
- Matcher or notification failures are logged and do not fail the scrape run.

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_db.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_pipeline.py \
  -q
```

Expected: new backend persistence and matcher tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add \
  tldw_Server_API/app/core/DB_Management/Watchlists_DB.py \
  tldw_Server_API/app/core/Watchlists/content_alerts.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_db.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_pipeline.py \
  backlog/tasks/<stage-3a-task-file>
git commit -m "feat: add watchlist content alert matching"
```

## Task 2: Content Alert API And Documentation

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Create: `tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_api.py`
- Modify: `Docs/API-related/Watchlists_API.md`
- Modify if mirrored: `Docs/Published/API-related/Watchlists_API.md`

- [ ] **Step 1: Add failing API tests first**

Cover:

- Rule create/list/update/delete under a selected Watchlist.
- Alert list filters: status, severity, rule ID, source ID, since/until, and text query.
- Alert detail includes evidence and item linkage.
- Alert update can mark read/unread and dismiss.
- Requests for another user's Watchlist or alert do not leak data.
- Invalid regex/source constraints return clear validation errors.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_api.py -q
```

Expected: tests fail before endpoint implementation.

- [ ] **Step 2: Add schemas**

Add schemas for:

- `WatchlistContentAlertRuleCreate`
- `WatchlistContentAlertRuleUpdate`
- `WatchlistContentAlertRule`
- `WatchlistContentAlertRuleList`
- `WatchlistContentAlert`
- `WatchlistContentAlertList`
- `WatchlistContentAlertUpdate`
- `WatchlistContentAlertEvidence`

Validation expectations:

- `name` and `pattern` must be non-empty after trimming.
- `severity` must use the current Watchlists severity set.
- `match_mode` must be explicit enough to avoid surprising regex behavior.
- Source constraints must be structured JSON, not free-form strings.

- [ ] **Step 3: Add nested routes**

Add nested routes to `watchlists.py` under the existing router so no new router group registration is needed for the first implementation.

Keep static routes before dynamic `/{watchlist_id}` routes. If this becomes unwieldy, split to a new module only after adding an inclusion test that proves the router is registered.

- [ ] **Step 4: Update API docs**

Update docs to clearly separate:

- Content alert rules and content alert inbox.
- Health rules and health issues.
- Topic Monitoring as an internal dependency, not the user-facing Watchlist model.

Run:

```bash
rg -n "content-alert|Health issue|alert-rules|watchlist_content" Docs/API-related/Watchlists_API.md Docs/Published/API-related/Watchlists_API.md
```

Expected: docs contain the new content alert contract and the health-rule boundary.

- [ ] **Step 5: Run API tests and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_api.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_db.py \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py \
  -q
```

Commit:

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_api.py \
  Docs/API-related/Watchlists_API.md \
  Docs/Published/API-related/Watchlists_API.md \
  backlog/tasks/<stage-3b-task-file>
git commit -m "feat: expose watchlist content alert api"
```

## Task 3: Pipeline Integration And Health-Rule Separation

**Files:**
- Modify: `tldw_Server_API/app/core/Watchlists/pipeline.py`
- Modify: `tldw_Server_API/app/core/Watchlists/alert_rules.py`
- Modify if registered: `tldw_Server_API/app/api/v1/endpoints/watchlist_alert_rules.py`
- Modify: `tldw_Server_API/tests/test_watchlist_alert_rules.py`
- Modify: `tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_pipeline.py`

- [ ] **Step 1: Add failing health-boundary tests**

Cover:

- Run-stat rule notifications include a health-oriented type or metadata marker.
- Content alert notifications use a different type.
- Content alerts link to items, while health issues link to runs.
- Existing `/watchlists/alert-rules` behavior remains backward compatible if the route is active.

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/test_watchlist_alert_rules.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_pipeline.py \
  -q
```

Expected: tests fail only on the new health/content distinction until implementation is added.

- [ ] **Step 2: Wire item-level evaluation into the pipeline**

Update `_record_scraped` in `pipeline.py` so it:

- Receives the `ScrapedItemRow` returned by `record_scraped_item`.
- Calls the content alert matcher only for recorded items with enough text or metadata to evaluate.
- Passes `watchlist_id`, `run_id`, `job_id`, source information, item evidence, and user scope.
- Treats matcher failures as non-critical.

- [ ] **Step 3: Separate health-rule naming**

Keep the old run-stat rule engine but ensure emitted payloads and docs/copy support:

- `watchlist_health_issue` or equivalent type for run-stat events.
- `watchlist_content_alert` for content matches.
- Compatibility fields where existing consumers expect `watchlist_alert`.

If `/watchlists/alert-rules` is not currently registered in the router group, do not add a breaking public contract accidentally. Prefer docs plus a future alias task unless the implementation task explicitly registers it.

- [ ] **Step 4: Run focused backend tests and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/test_watchlist_alert_rules.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_pipeline.py \
  tldw_Server_API/tests/Monitoring/test_topic_monitoring.py \
  -q
```

Commit:

```bash
git add \
  tldw_Server_API/app/core/Watchlists/pipeline.py \
  tldw_Server_API/app/core/Watchlists/alert_rules.py \
  tldw_Server_API/app/api/v1/endpoints/watchlist_alert_rules.py \
  tldw_Server_API/tests/test_watchlist_alert_rules.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_pipeline.py \
  backlog/tasks/<stage-3c-task-file>
git commit -m "feat: trigger watchlist content alerts from pipeline"
```

## Task 4: Frontend Alerts Tab And Copy

**Files:**
- Modify: `apps/packages/ui/src/types/watchlists.ts`
- Modify: `apps/packages/ui/src/services/watchlists.ts`
- Create: `apps/packages/ui/src/services/__tests__/watchlists-content-alerts.test.ts`
- Create: `apps/packages/ui/src/components/Option/Watchlists/AlertsTab/AlertsTab.tsx`
- Create: `apps/packages/ui/src/components/Option/Watchlists/AlertsTab/index.ts`
- Create: `apps/packages/ui/src/components/Option/Watchlists/AlertsTab/__tests__/AlertsTab.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/watchlists.json`
- Modify if mirrored: `apps/packages/ui/src/public/_locales/en/watchlists.json`
- Create: `apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlists-stage3-copy-contract.test.ts`

- [ ] **Step 1: Add failing service and copy tests first**

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/services/__tests__/watchlists-content-alerts.test.ts \
  src/components/Option/Watchlists/__tests__/watchlists-stage3-copy-contract.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Expected: tests fail until types, services, and copy are added.

- [ ] **Step 2: Add frontend types and service methods**

Add service coverage for:

- Rule list/create/update/delete.
- Alert list/detail/update.
- Query serialization for status, severity, rule ID, source ID, date window, and text query.

- [ ] **Step 3: Add Alerts tab component tests**

Cover:

- Empty state when no rules or alerts exist.
- Rule creation validation.
- Inbox rendering with severity, source, item title, snippet, rule name, and timestamp.
- Filter controls.
- Mark read/unread and dismiss actions.
- Loading and API error states.
- Extension-width layout without requiring hover-only controls.

- [ ] **Step 4: Implement Alerts tab**

The tab should provide:

- Rule management: list, create, edit, disable, delete.
- Inbox: unread/read/dismissed filters, severity filters, source/rule filters, text search.
- Evidence view: snippet, matched text, source URL, item link or item title, run/job/source IDs when useful.
- Clear health boundary copy: content alerts here, pipeline failures in Health/Activity.

Suggested copy:

- Tab label: `Alerts`
- Rule section heading: `Content alert rules`
- Inbox heading: `Alert inbox`
- Empty rules: `Create a rule to be notified when new Watchlist items match a descriptor, keyword, classification, entity, or source constraint.`
- Empty inbox: `No content alerts match these filters.`
- Health boundary helper: `Run failures and source problems are health issues, not content alerts.`

- [ ] **Step 5: Integrate tab into the Watchlists shell**

Update `WatchlistsPlaygroundPage.tsx` to add the tab inside the selected-Watchlist experience. If unread counts are cheap through the alert list response, show a count badge. If not, leave count wiring to Task 5.

- [ ] **Step 6: Run focused frontend tests and commit**

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/services/__tests__/watchlists-content-alerts.test.ts \
  src/components/Option/Watchlists/AlertsTab/__tests__/AlertsTab.test.tsx \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx \
  src/components/Option/Watchlists/__tests__/watchlists-stage3-copy-contract.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Commit:

```bash
git add \
  apps/packages/ui/src/types/watchlists.ts \
  apps/packages/ui/src/services/watchlists.ts \
  apps/packages/ui/src/services/__tests__/watchlists-content-alerts.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/AlertsTab \
  apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx \
  apps/packages/ui/src/assets/locale/en/watchlists.json \
  apps/packages/ui/src/public/_locales/en/watchlists.json \
  apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlists-stage3-copy-contract.test.ts \
  backlog/tasks/<stage-3d-task-file>
git commit -m "feat: add watchlist alerts tab"
```

## Task 5: Overview Integration, Browser QA, And Closeout

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
- Modify relevant Overview tests under `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/`
- Modify: `Docs/API-related/Watchlists_API.md`
- Modify if mirrored: `Docs/Published/API-related/Watchlists_API.md`
- Modify: Stage 3 Backlog tasks

- [ ] **Step 1: Add Overview alert and health summary tests**

Cover:

- Overview shows unread content alert count separately from health issues.
- Empty state offers "Create content alert rule" only when a Watchlist is selected.
- Health copy does not call run failures generic alerts.

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab*.test.tsx \
  src/components/Option/Watchlists/__tests__/watchlists-stage3-copy-contract.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

- [ ] **Step 2: Implement Overview integration**

Add a compact section or summary row that links to:

- Alerts tab for unread content alerts.
- Activity/Health view for pipeline health issues.

Avoid adding another large dashboard card if the current layout is already dense.

- [ ] **Step 3: Run full focused verification**

Backend:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_db.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_api.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_pipeline.py \
  tldw_Server_API/tests/test_watchlist_alert_rules.py \
  tldw_Server_API/tests/Monitoring/test_topic_monitoring.py \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_db.py \
  -q
```

Frontend:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/services/__tests__/watchlists-content-alerts.test.ts \
  src/components/Option/Watchlists/AlertsTab/__tests__/AlertsTab.test.tsx \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx \
  src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab*.test.tsx \
  src/components/Option/Watchlists/__tests__/watchlists-stage3-copy-contract.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Formatting and security:

```bash
git diff --check
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/DB_Management/Watchlists_DB.py \
  tldw_Server_API/app/core/Watchlists/content_alerts.py \
  tldw_Server_API/app/core/Watchlists/pipeline.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  -f json -o /tmp/bandit_watchlists_stage3_content_alerts.json
```

- [ ] **Step 4: Run CDP browser smoke**

Use the existing dev-server pattern for this repo. Then use Playwright/CDP, not Computer Use, to test:

- Desktop `/watchlists` opens with selected Watchlist.
- Alerts tab renders.
- Rule create validation works.
- Mocked or test-backed alert inbox shows evidence.
- Mark read/dismiss changes visible state.
- Overview separates content alerts and health issues.
- `390x844` viewport supports full management without document-level horizontal overflow.

Save screenshots to `/tmp/watchlists-stage3-alerts-desktop-cdp.png` and `/tmp/watchlists-stage3-alerts-mobile-cdp.png`.

- [ ] **Step 5: Final docs and Backlog closeout**

Update Stage 3 Backlog task records with:

- Touched files.
- Verification commands and results.
- Bandit output path.
- CDP screenshots.
- Known skips or blockers.

- [ ] **Step 6: Commit closeout**

Run:

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab \
  Docs/API-related/Watchlists_API.md \
  Docs/Published/API-related/Watchlists_API.md \
  backlog/tasks/<stage-3-task-files>
git commit -m "chore: close watchlists content alerts stage"
```

## UX Acceptance Criteria

- Users can create a content alert rule from inside the selected Watchlist without understanding jobs, runs, or Topic Monitoring.
- CTI and OSINT users can express rules for CVEs, IOCs, actors, malware names, advisory terms, source constraints, and severity labels, even if the first implementation uses deterministic text matching.
- News users can express rules for people, organizations, topics, events, and source constraints without CTI-specific jargon.
- The alert inbox explains why an alert fired by showing matched evidence and source context.
- Content alerts are not visually or verbally mixed with pipeline failures.
- Rule validation prevents invalid regex and empty patterns before save.
- Users can triage alerts with at least read/unread and dismiss states.
- Extension-sized viewport supports rule management and inbox triage.

## Technical Acceptance Criteria

- Content alert persistence is scoped by `user_id` and `watchlist_id`.
- Content alert rules and run-stat health rules use separate tables or clearly separate domain models.
- Content alert matching is deterministic and testable without network access.
- Pipeline alert evaluation is non-critical and cannot fail an entire scrape run.
- Dedupe prevents repeated alerts for the same rule/item.
- Notifications include a distinct `watchlist_content_alert` type.
- Existing run-stat alert-rule tests still pass.
- Topic Monitoring tests still pass if any helper is extracted or reused.

## Risks And Mitigations

- Risk: `Watchlists_DB.py` grows larger.
  - Mitigation: keep alert matching logic in `core/Watchlists/content_alerts.py`; only persistence methods belong in DB management.
- Risk: Regex matching can become expensive on large content.
  - Mitigation: validate regex at save time, cap evaluated text length if needed, and add tests for pathological inputs before allowing complex patterns.
- Risk: Topic Monitoring and product Watchlists drift.
  - Mitigation: reuse behavior and tests where useful, but keep product storage under Watchlists. Document the boundary.
- Risk: Users confuse health rules and content alerts.
  - Mitigation: separate labels, tabs, notification types, and docs. Add copy contract tests.
- Risk: Constrained viewport management becomes cramped.
  - Mitigation: test at `390x844` with CDP and prefer progressive disclosure for rule forms and evidence details.

## Review Note

This plan was locally self-reviewed against the current code seams. A subagent review was not used in this session because the user has not explicitly authorized subagents for this continuation.
