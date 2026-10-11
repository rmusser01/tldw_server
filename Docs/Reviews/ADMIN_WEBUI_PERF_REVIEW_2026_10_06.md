# Admin WebUI Efficiency Review (2026-10-06)

Static review of the admin webui (pages in `apps/tldw-frontend/pages/admin/` → routes in
`apps/packages/ui/src/routes/option-admin-*.tsx` → implementation in
`apps/packages/ui/src/components/Option/Admin/`, ~14k lines) plus the backend endpoints
that serve it. Findings were verified by reading the flagged code; severity assumes a
multi-user deployment where users, events, and usage rows grow over time.

This document is the shared spec for three implementation plans:

- [2026-10-06-admin-webui-perf-A-backend-endpoints-implementation-plan.md](../Plans/2026-10-06-admin-webui-perf-A-backend-endpoints-implementation-plan.md) — F1–F7
- [2026-10-06-admin-webui-perf-B-data-architecture-implementation-plan.md](../Plans/2026-10-06-admin-webui-perf-B-data-architecture-implementation-plan.md) — F8–F13
- [2026-10-06-admin-webui-perf-C-rendering-implementation-plan.md](../Plans/2026-10-06-admin-webui-perf-C-rendering-implementation-plan.md) — F14–F23

Deconfliction: the parallel backend perf program (perf-batches 3–6, 2026-10-06) covers
ChaCha N+1s, FTS pushdown, jobs/event-loop workers, and ingestion throughput. No file
overlaps with the findings below; the only shared theme is `asyncio.to_thread` (different
files: theirs `Jobs/manager.py` et al., ours `llamacpp.py` / `admin_ops.py`).

## Backend findings

### F1 (HIGH, functional bug) — Billing admin API route mismatch
Client calls `GET /api/v1/admin/billing/overview|subscriptions|events` plus
`subscriptions/{id}` and `subscriptions/{id}/override|credits`
(`apps/packages/ui/src/services/tldw/domains/admin.ts:631-663`). Backend defines exactly
one route: `GET /api/v1/billing/subscriptions` (`tldw_Server_API/app/api/v1/endpoints/billing.py:24,110`,
mounted at `/api/v1` via `router_groups/admin.py:63-69`). Every other call 404s. The
client contract is the source of truth (see `BillingDashboardPage.test.tsx` mocks:
overview = `{mrr, active_subscriptions, canceled_subscriptions, past_due_subscriptions}`).
Also: the subscriptions query filters `status` in Python over an unbounded read, and
resolves org names by fetching whole org rows
(`tldw_Server_API/app/services/billing_repo.py:539-583`, `billing.py:110-203`).

### F2 (HIGH) — `llm_usage_v2` is a dead table; admin token stats and cost attribution silently return nothing
`llm_usage_v2` is never created and never written anywhere in the repo — only read by:
- `tldw_Server_API/app/services/admin_system_service.py:253-276` (`WHERE date(created_at) = CURRENT_DATE`)
- `tldw_Server_API/app/services/admin_usage_service.py:1240-1290` (cost attribution; SQLite path also wraps the column: `datetime(created_at) >= datetime('now', ?)`)

Both are wrapped in `except Exception`, so the admin dashboard shows zeros forever. The
real table is `llm_usage_log` (created + indexed in `AuthNZ/migrations.py:2542,2625-2631`
and `pg_migrations_extra.py:1911,1970-1977`; columns `ts`, `prompt_tokens`,
`completion_tokens`, `total_tokens`, `user_id`, `key_id`). Note: `llm_usage_log` has no
`org_id` column — org-scoped cost attribution needs the `org_members` JOIN already used
at `admin_usage_service.py:296-300`.

### F3 (MEDIUM) — Missing index `sessions(created_at)`
`GET /api/v1/admin/activity` filters `sessions.created_at >= $1`
(`admin_system_service.py:416-463`). Existing indexes: `token_hash`, `user_id`,
`expires_at`, `access_jti` only (`AuthNZ/migrations.py:127-130`,
`pg_migrations_extra.py:1091-1094`). Full scan per monitoring dashboard load; `sessions`
grows with every login/refresh.

### F4 (MEDIUM) — Missing index `org_members(org_id, user_id)`
Only `idx_org_members_user(user_id)` exists (`pg_migrations_extra.py:1250`,
`migrations.py:2716`). Admin usage/users/audit queries JOIN `org_members` on `org_id`
(`admin_usage_service.py:296-300,1255-1261`, `repos/users_repo.py:210-223`,
`admin_system_service.py:586-592`).

### F5 (HIGH) — `GET /admin/backups` walks the whole backup filesystem per request
`list_backup_items()` (`tldw_Server_API/app/services/admin_data_ops_service.py:152-191`)
scandirs every dataset dir **and every `user_<id>/` subdir**, stats every file, sorts the
full list, then paginates with a Python slice — `limit/offset` never reach the scan.
~6,000 stat calls at 50 users × 6 datasets × 20 files. Fired by the DataOps page and by
the admin landing overview (`admin-module-signals.ts` → `listBackups()` just to count).

### F6 (MEDIUM) — `/llamacpp/inventory` directory scan blocks the event loop
`tldw_Server_API/app/api/v1/endpoints/llamacpp.py:573-580` calls the sync
`scan_inventory()` (stat on multi-GB GGUFs, possibly NAS) directly in the async handler.
The sibling `/llamacpp/assets` endpoint already wraps the identical calls in
`run_in_threadpool` (`llamacpp.py:407-415`) — copy that.

### F7 (MEDIUM) — `system_ops.json` flock + read + parse per request on the event loop
`GET /admin/feature-flags` (`admin_ops.py:441-467`), `/admin/incidents` (`:555-583`),
`/admin/maintenance` (`:263-270`), `/admin/api-keys/usage/top` (`admin_api_keys.py:165-183`)
all do sync file IO + lock + full JSON parse per call
(`admin_system_ops_service.py:82-94,304-310`). Store grows over time (90-day usage
snapshots per API key, incidents, rotation runs).

## Frontend data-architecture findings

### F8 (HIGH) — Monitoring auto-refresh refetches 6 datasets, no visibility gate
`refreshAll()` (`MonitoringDashboardPage.tsx:389-397`) refetches stats, security,
sandbox diagnostics, alert **rules** (near-static), alert history (unbounded), and 7-day
activity every interval (`:428-432`); no `document.hidden` check — polls forever in
background tabs. `LlamacppAdminPage.tsx:110-132` has the correct visibility pattern.

### F9 (HIGH) — ~1.4 MB `openapi.json` per page mount as capability probe
`BillingDashboardPage.tsx:471` and `RateLimitingPage.tsx:94` fetch and parse the entire
OpenAPI spec to test one path; Billing blocks first render on it (4s failsafe
`:502-506`). Zero sharing between pages (see `openapi-guard.ts:5` for the size note).

### F10 (HIGH) — Unbounded lists and silent truncation caps
- `OrgsTeamsPage.tsx:70-84,237-251,396-410`: `listOrgMembers`/`listTeams`/`listTeamMembers`
  rendered with `pagination={false}` — a 5,000-member org renders 5,000 rows. Server
  supports LIMIT/OFFSET (`repos/orgs_teams_repo.py:232-300`).
- `MonitoringDashboardPage.tsx:332-342`: `listAlertHistory()` sends no params (backend
  default limit 50, max 500 — pass an explicit limit).
- `ApiKeyManagementPage.tsx:62,326-338`: users capped at `limit: 100`, `showSearch`
  filters locally — accounts 101+ are unreachable for API-key management. Copy the
  debounced remote-search pattern from `WatchlistsOversightPage.tsx:108-116`.
- `OrgsTeamsPage.tsx:509` orgs / `BillingDashboardPage.tsx:177,376` subscriptions+events /
  `DataOpsPage.tsx:514` DSRs: `limit: 100` first-page only, no pager.

### F11 (MEDIUM) — No response cache; react-query installed but unused in Admin
Every mount refetches everything; `bgRequest` dedups only in-flight GETs
(`background-proxy.ts:530-532`). `getSystemStats` fetched by 4 surfaces, `listAdminUsers`
by 4 pages, `listBackups`/`getGovernorCoverage` by pages + module signals. Admin overview
fires 6+2 probe requests per visit (`AdminOperationsOverviewPage.tsx:96-110`,
`admin-module-signals.ts:118-146`). `@tanstack/react-query` is already a dependency
(`apps/packages/ui/package.json:64`) and used in Flashcards/Sidepanel.

### F12 (MEDIUM) — Per-keystroke network calls
`ServerAdminPage.tsx:991-997` + effect `:295-300`: every character in the media-budget
policy-id input fires `getMediaIngestionBudgetDiagnostics`; `:285-293` auto-preloads the
first user's diagnostics unrequested.

### F13 (LOW) — UsageAnalytics date range reaches only 2 of 6 datasets
Backend endpoints already accept `start`/`end`
(`admin_usage.py:50-62,179-186,210-218,251-258`) but client methods
`getTopUsage`/`getLlmUsage`/`getLlmUsageSummary`/`getLlmTopSpenders`
(`domains/admin.ts:304-330`) don't expose or pass them
(`UsageAnalyticsPage.tsx:195-212`).

## Frontend rendering findings

### F14 (HIGH) — Monitoring page re-renders wholesale
- `:435-446`: 10s "time since refresh" tick sets page state → full re-render of the
  835-line page incl. 5 tables.
- `:482,510,546,644`: column arrays rebuilt inline every render.
- Poll replaces every state slice with new object identities → full row re-diff even
  when unchanged.
- `:640-643,826`: activity table maps `_key: idx` (index keys on dynamic data) with
  `pagination={false}`; recursive `formatStatValue` per row.

### F15 (HIGH) — LlamacppAdminPage: one `settings` state re-renders 6 panels per keystroke
`LlamacppAdminPage.tsx:535` holds `settings` at page level; `LlamacppLaunchPanel.tsx:106`
calls `onSettingsChange({...settings, [key]: value})` for ~25 controls. Zero
`React.memo`/`useCallback` in the whole Admin dir; all handler props are fresh closures
(`:1289-1327`), so memoization is currently impossible even if added.

### F16 (HIGH/MEDIUM) — RBAC matrix scale + O(n·m) lookups
- `RbacEditorPage.tsx:165-181,214-227`: `permissions × roles` Checkbox grid,
  `pagination={false}`, columns (one per role) rebuilt per render. Hundreds of
  permissions × a dozen roles = thousands of cells.
- `:591`: per-row `allPermissions.find(p => p.id === record.permission_id)` — O(overrides ×
  allPermissions) per render (the textbook Map fix; `LlamacppRuntimePanel.tsx:163-180`
  is the in-repo model).
- `:740-744`: `userRoles.some(...)` inside `allRoles.map(...)`.
- `:147-149`: `filteredPermissions` unmemoized.

### F17 (MEDIUM) — Unvirtualized unbounded lists
`LlamacppInventoryPanel.tsx:119-180` (one `<li>` per discovered model, 4-8 Tags each);
`LlamacppAssetsPanel.tsx:463-505` (asset groups). `@tanstack/react-virtual` is installed
with in-repo examples (`Review/hooks/useMediaReviewState.ts:5`,
`Sidepanel/Chat/body.tsx:7`).

### F18 (MEDIUM/LOW) — Per-render sorts/filters without memo on hot paths
`LlamacppSnapshotsPanel.tsx:282-284` (spread+sort under a 1.5s poll);
`LlamacppAssetsPanel.tsx:64-71,167` (4 filter passes per keystroke × 5 inputs);
`LlamacppProfilesPanel.tsx:481,496` (filter/map/some ×2 per modal keystroke);
`LlamacppAdminPage.tsx:743-745,1329-1334` (per-render scans).

### F19 (MEDIUM) — Unstable rowKeys
`DataOpsPage.tsx:329,479,1017` (`JSON.stringify(record)` fallback — O(rows × payload)
per render); `BillingDashboardPage.tsx:323,433` (`Math.random()`);
`RateLimitingPage.tsx:328-330` (array index); `ServerArgsEditor.tsx:160-161` (index keys
on a removable kv list — focus misalignment on delete).

### F20 (LOW) — Per-cell `new Date(...).toLocaleString()` across ~10 tables
Representative: `MonitoringDashboardPage.tsx:519,645`; `DataOpsPage.tsx:186,622,956`;
`WatchlistsOversightPage.tsx:250,269,292`; `BillingDashboardPage.tsx:274,418`;
`MaintenancePage.tsx:302,353,359`; `ApiKeyManagementPage.tsx:192`.

### F21 (MEDIUM) — MaintenancePage banner inputs re-render 3 unpinned tables per keystroke
`MaintenancePage.tsx:404-417` (page-level `maintMessage`/`maintAllowlist` state; tables at
`:427-434,480-487,510-517`).

### F22 (MEDIUM) — ServerAdminPage columns/options rebuilt per render
`:414-506` (`userColumns` + role options), `:850-911` (roles table columns), `:986`
(media-budget user options).

### F23 (LOW) — Layout shell context value recreated per render
`Layouts/Layout.tsx:789` — new object per `RootLayoutShell` render re-renders all
`useOptionLayoutShellOverrides` consumers. Small consumer count today.

## Positive patterns to propagate (no action)

- `ServerAdminPage.tsx:144-171,302-308` — true server-side pagination with filters
  pushed to the server.
- `WatchlistsOversightPage.tsx:108-116` — debounced (300ms) remote user search,
  `filterOption={false}`, out-of-order response guard, "Showing X of Y" disclosure.
- `LlamacppRuntimePanel.tsx:163-180` — Map + Set keyed joins inside `useMemo`.
- `RbacEditorPage.tsx:165-181` — permission-matrix grid lookups are O(1) keyed access.
- `LlamacppAdminPage.tsx:110-132` — the only admin page handling `document.hidden`.
- Backend: no N+1 loops in admin endpoints; usage analytics and audit log are set-based
  SQL with real pagination and indexes.
