# Admin WebUI Perf — Plan B: Data Architecture Implementation Plan (2026-10-06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan stage-by-stage. Steps use checkbox (`- [ ]`) syntax for tracking.

**Backlog task:** TBD — `backlog task create` crashes with "Maximum call stack size exceeded" (CLI 1.44.0, 2026-10-06). Create the task before execution begins and update this line + stage commit messages with the real ID.

**Spec:** [ADMIN_WEBUI_PERF_REVIEW_2026_10_06.md](../Reviews/ADMIN_WEBUI_PERF_REVIEW_2026_10_06.md) — implements findings **F8–F13**.

**Goal:** Stop the admin webui from over-fetching: one cached capability probe instead of per-mount 1.4 MB spec downloads, a polling loop that respects tab visibility and only refreshes live data, server-side pagination for unbounded lists, debounced inputs, date ranges that reach every dataset, and react-query for shared reference data.

**Architecture:** Frontend-only (`apps/packages/ui` + nothing else; `apps/tldw-frontend` pages are thin wrappers). Keep the `tldwClient`/`bgRequest` transport — react-query wraps it, nothing replaces it. Each stage is independently shippable.

**Tech Stack:** React 18 hooks, `@tanstack/react-query` (already a dependency, `apps/packages/ui/package.json:64`), antd Table server pagination, vitest + @testing-library/react.

## Global constraints

- Frontend only: never modify `tldw_Server_API/**`. Commands run from `apps/packages/ui` with bun: `bun vitest run <path>`; typecheck with `bunx tsc --noEmit -p tsconfig.json`.
- Existing tests in `src/components/Option/Admin/__tests__/` are the contract — a stage is wrong if it needs to weaken an assertion to pass; strengthen mocks instead.
- No new dependencies. Copy existing in-repo patterns, don't invent parallel ones (debounce: `WatchlistsOversightPage.tsx:108-116`; visibility: `LlamacppAdminPage.tsx:110-132`).
- One commit per stage referencing the Backlog task ID.
- Line numbers come from the 2026-10-06 review; re-locate by symbol.
- Backend dependencies already exist for everything except nothing — every fix here is client-side (verified: `/admin/users` search, org LIMIT/OFFSET, alert-history `limit` param, usage `start`/`end` params all exist server-side).

---

## Stage 1: Shared capability probe (F9, incl. Billing render waterfall)

**Goal:** `openapi.json` is fetched at most once per server URL per session to answer "does path X exist"; BillingDashboard stops blocking its first render on the probe.

**Files:**
- Create: `apps/packages/ui/src/services/tldw/capability-probe.ts`
- Modify: `apps/packages/ui/src/components/Option/Admin/BillingDashboardPage.tsx:456-526` (probe + tab gating)
- Modify: `apps/packages/ui/src/components/Option/Admin/RateLimitingPage.tsx:94` (probe call)
- Test: `apps/packages/ui/src/services/tldw/__tests__/capability-probe.test.ts` (new); update `BillingDashboardPage.test.tsx` / `RateLimitingPage` tests only if mock setup needs the new module

**Change:**
1. `capability-probe.ts`:
   ```ts
   type ProbeResult = { supported: boolean; checkedAt: number }
   const specCache = new Map<string, Promise<Set<string>>>()   // serverUrl -> paths
   const failures = new Map<string, number>()                  // serverUrl -> monotonic-ish ms
   const FAILURE_RETRY_MS = 60_000

   export async function serverSupportsPath(serverUrl: string, path: string): Promise<boolean> {
     const cached = specCache.get(serverUrl)
     if (cached) return (await cached).has(path)
     if (Date.now() - (failures.get(serverUrl) ?? 0) < FAILURE_RETRY_MS) return false
     const p = (async () => {
       const res = await fetch(`${serverUrl}/openapi.json`)
       if (!res.ok) throw new Error(`openapi probe ${res.status}`)
       const spec = await res.json()
       return new Set(Object.keys(spec?.paths ?? {}))
     })()
     specCache.set(serverUrl, p)
     try { return (await p).has(path) } catch { specCache.delete(serverUrl); failures.set(serverUrl, Date.now()); return false }
   }
   export function clearCapabilityProbeCacheForTests(): void { specCache.clear(); failures.clear() }
   ```
   In-flight dedup comes free from the stored Promise.
2. BillingDashboard: replace the local fetch (`:471`) with `serverSupportsPath(serverUrl, BILLING_OVERVIEW_PATH)`; **render the tabs immediately** and downgrade the billing tab in place when the probe resolves false (RateLimitingPage's lazy per-endpoint style). Keep the 4s failsafe only as the probe timeout, not the render gate.
3. RateLimitingPage: replace its per-mount fetch (`:94`) with the shared helper; delete the mount-local ref cache.

**Tests:**
- [ ] `test_fetches_spec_once_for_many_paths` — two `serverSupportsPath` calls on one serverUrl → one `fetch` (mock counter).
- [ ] `test_failure_is_cached_short_term` — rejected fetch → second call within window returns false with no new fetch.
- [ ] `test_concurrent_calls_share_inflight_request` — two parallel calls → one fetch.
- [ ] BillingDashboard renders tabs before the probe resolves (assert skeleton gone while probe promise pending).

**Status:** Not Started

## Stage 2: MonitoringDashboard polling discipline (F8)

**Goal:** Auto-refresh polls only live data (system stats, security status), respects tab visibility, and passes an explicit limit to alert history; rules/diagnostics/history/activity load on mount and manual refresh only.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Admin/MonitoringDashboardPage.tsx:389-446`
- Modify: `apps/packages/ui/src/services/tldw/domains/admin.ts:284` (`listAlertHistory` gains `params?: { limit?: number }`)
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.*.test.tsx` (extend/add)

**Change:**
1. Split `refreshAll` into `refreshLive()` (stats + security status) and `refreshDeep()` (sandbox diagnostics, alert rules, alert history, activity). The auto-refresh interval calls `refreshLive` only. The refresh button calls both. Initial mount keeps loading all six (existing `:401-425` effect).
2. Visibility gate, modeled on `LlamacppAdminPage.tsx:110-132`: skip the interval body when `document.hidden`; on `visibilitychange` back to visible, fire `refreshLive()` immediately if stale.
3. `listAlertHistory({ limit: 200 })` — backend accepts `limit` (default 50, max 500). Pass 200 from the page.
4. Leave the 10s time-ago tick alone here — that is Plan C Stage 1 (render isolation); do not duplicate that work in this stage.

**Tests:**
- [ ] `test_auto_refresh_polls_only_live_datasets` — fake timers, advance one interval → stats + security fetch mocks called, rules/diagnostics/history/activity mocks NOT called again.
- [ ] `test_no_polling_while_hidden` — set `document.hidden = true` (jsdom mock), advance intervals → zero fetches; restore visibility → one live refresh fires.
- [ ] `test_alert_history_requests_limit` — assert mock called with `{ limit: 200 }`.

**Status:** Not Started

## Stage 3: Unbounded lists + remote search (F10)

**Goal:** Org members/teams and API-key user selection stop depending on fetching everything; truncation-prone selectors become remote-searched.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Admin/OrgsTeamsPage.tsx:70-84,237-251,396-410,509`
- Modify: `apps/packages/ui/src/components/Option/Admin/ApiKeyManagementPage.tsx:62,326-338`
- Modify: `apps/packages/ui/src/services/tldw/domains/admin.ts` (`listOrgMembers`, `listTeams`, `listTeamMembers` accept `{ limit, offset }` and return totals — check current return shape first and keep it backward compatible by extending, not breaking)
- Test: extend `__tests__/OrgsTeamsPage.test.tsx` and `__tests__/ApiKeyManagementPage.*.test.tsx`

**Change:**
1. OrgsTeamsPage members/teams tables: switch to antd server-side pagination — page state `{page, pageSize}` per table, fetch with `limit=pageSize, offset=(page-1)*pageSize`, table `pagination={{ current, pageSize, total }}` with `onChange` refetch. Backend already LIMIT/OFFSETs (`repos/orgs_teams_repo.py:232-300`). Keep the orgs list itself at `limit: 100` but add the "Showing X of Y" disclosure used by `WatchlistsOversightPage` if the response carries a total; if not, label as "first 100".
2. ApiKeyManagementPage user selector: copy the `WatchlistsOversightPage.tsx:108-116` pattern verbatim — `showSearch`, `filterOption={false}`, `onSearch` debounced 300ms calling `listAdminUsers({ search: q, limit: 20 })`, replace options on response, keep an out-of-order guard (`loadSeqRef`) like the oversight page. Initial mount loads the first 20 users instead of 100; the 4 pages currently sharing `listAdminUsers` are unaffected (they pass their own params).
3. BillingDashboard subscriptions/events and DataOps DSRs: leave their `limit: 100` in place but surface "Showing first 100" captions next to the tables (match the oversight-page disclosure copy pattern) — full server pagination for these waits on Plan A Stage 1's truthful `total`.

**Tests:**
- [ ] `test_org_members_server_pagination` — 250-row mock; page 2 request carries `offset=20`; table footer shows total 250.
- [ ] `test_apikey_user_select_remote_search` — type "ada", advance 300ms → `listAdminUsers` called with `{ search: "ada", limit: 20 }`; options replaced; slow-first response loses to fast-second (seq guard).
- [ ] `test_no_initial_100_user_preload` — mount asserts `listAdminUsers` called with `limit: 20`, not 100.

**Status:** Not Started

## Stage 4: react-query for shared admin reference data (F11)

**Goal:** Reference data (system stats, users page 1, roles, permissions, module signals) is fetched once per staleTime window no matter how many surfaces mount; the admin overview stops re-probing everything per visit.

**Files:**
- Create: `apps/packages/ui/src/services/tldw/adminQueries.ts` (query keys + hooks)
- Create: `apps/packages/ui/src/components/Option/Admin/AdminQueryProvider.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/AdminRouteShell.tsx` (wrap `{children}` at `:93` with the provider)
- Modify: `apps/packages/ui/src/components/Option/Admin/admin-module-signals.ts:118-146` (signals go through queries)
- Modify: `MonitoringDashboardPage.tsx`, `ServerAdminPage.tsx:212-237`, `RbacEditorPage.tsx:433-478` (swap their direct loaders for hooks where the data is reference-like: stats, roles, permissions; keep page-local lists on existing loaders)
- Test: `apps/packages/ui/src/services/tldw/__tests__/adminQueries.test.tsx` (new)

**Change:**
1. `AdminQueryProvider` — creates one `QueryClient` (`staleTime: 30_000, refetchOnWindowFocus: false`) and exports it for tests; `AdminRouteShell` renders it around children so every admin page shares it. No changes to non-admin routes.
2. `adminQueries.ts` — key factory + thin hooks over `tldwClient`:
   ```ts
   export const adminKeys = {
     systemStats: ["admin", "system-stats"] as const,
     usersPage: (page: number, limit: number, search?: string) => ["admin", "users", page, limit, search ?? ""] as const,
     roles: ["admin", "roles"] as const,
     permissions: ["admin", "permissions"] as const,
   }
   export const useSystemStats = () => useQuery({ queryKey: adminKeys.systemStats, queryFn: () => tldwClient.getSystemStats() })
   // useUsersPage / useAdminRoles / useAdminPermissions analogous
   ```
3. `admin-module-signals.ts` — each signal becomes a `useQuery` with the same keys (staleTime absorbs the per-visit probe storm; the overview page mounts and gets cached data within 30s). Keep the signal aggregation signature unchanged so `AdminOperationsOverviewPage` is untouched.
4. Migrate the three pages' reference-data loaders to the hooks; their mutation invalidations call `queryClient.invalidateQueries({ queryKey: adminKeys... })` where they currently re-call loaders after edits.

**Tests:**
- [ ] `test_two_mounts_one_fetch` — render a probe component using `useSystemStats` twice (shared provider); fetch mock called once; both see data.
- [ ] `test_stale_window_refetches` — advance `staleTime` via fake timers → next mount refetches.
- [ ] `test_module_signals_use_query_cache` — mount signals hook twice; `getSystemStats`/`listBackups` mocks each called once.

**Status:** Not Started

## Stage 5: Debounced diagnostics input + usage date-range pass-through (F12, F13)

**Goal:** No admin network call fires per keystroke; the UsageAnalytics date selector reaches all six datasets.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx:285-300,991-997`
- Modify: `apps/packages/ui/src/components/Option/Admin/UsageAnalyticsPage.tsx:195-212`
- Modify: `apps/packages/ui/src/services/tldw/domains/admin.ts:304-330` (`getTopUsage`, `getLlmUsage`, `getLlmUsageSummary`, `getLlmTopSpenders` gain optional `start`/`end`)
- Test: extend `__tests__/ServerAdminPage.*.test.tsx`, `__tests__/UsageAnalyticsPage.*.test.tsx` (or create if absent — check first)

**Change:**
1. ServerAdmin media-budget policy input: debounce the effect 300ms (`useEffect` + `setTimeout`/`clearTimeout`, same shape as `WatchlistsOversightPage.tsx:110-114`), and drop the auto-preload at `:285-293` (first-user diagnostics fetch on mount) — fetch only after an explicit user selection.
2. `domains/admin.ts`: extend the four method signatures with `start?: string; end?: string` (raw passthrough via existing `buildQuery`) — backend params are `start`/`end` (`admin_usage.py:52-53,182-183,212-213,254-255`).
3. UsageAnalyticsPage: thread the selected range into all four calls (`getTopUsage({ metric, limit, start, end })` etc.); `getDailyUsage` already passes dates. Convert the page's range picker value to the `YYYY-MM-DD` (usage/top) vs ISO (llm-usage*) shapes the endpoints document.

**Tests:**
- [ ] `test_policy_input_debounced` — type 4 chars with 299ms advances → one diagnostics call; advance past 300ms → exactly one per settled value.
- [ ] `test_no_diagnostics_preload_on_mount` — mount with users loaded → zero budget-diagnostics calls.
- [ ] `test_all_datasets_receive_range` — select "Last 30 days" → all four mocks called with non-empty `start`/`end`.

**Status:** Not Started

---

## Verification (whole plan)

- [ ] `cd apps/packages/ui && bun vitest run src/components/Option/Admin src/services/tldw` green
- [ ] `cd apps/packages/ui && bunx tsc --noEmit -p tsconfig.json`
- [ ] Manual smoke with a running server: Monitoring with auto-refresh on, background the tab ≥2 intervals, return — network panel shows no polling while hidden; Billing → Rate Limiting round trip downloads `openapi.json` exactly once
- [ ] Self-review diff; update the Backlog task with touched files, verification results, final summary
