# Admin WebUI Perf — Plan C: Rendering Efficiency Implementation Plan (2026-10-06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan stage-by-stage. Steps use checkbox (`- [ ]`) syntax for tracking.

**Backlog task:** TBD — `backlog task create` crashes with "Maximum call stack size exceeded" (CLI 1.44.0, 2026-10-06). Create the task before execution begins and update this line + stage commit messages with the real ID.

**Spec:** [ADMIN_WEBUI_PERF_REVIEW_2026_10_06.md](../Reviews/ADMIN_WEBUI_PERF_REVIEW_2026_10_06.md) — implements findings **F14–F23**.

**Goal:** Kill the whole-page re-render storms (poll ticks, single keystrokes), the O(n·m) per-row lookups, the unpaginated/unvirtualized big lists, and the unstable row keys.

**Architecture:** Frontend-only. Stage 1–2 are structural (state placement, render isolation); Stage 3–5 are mechanical sweeps (memoization, keys, formatters). The model pattern for Stage 3 already exists in-repo (`LlamacppRuntimePanel.tsx:163-180`: Map + Set inside `useMemo`). No new dependencies — `@tanstack/react-virtual` is installed (`apps/packages/ui/package.json:65`) with in-repo examples (`Review/hooks/useMediaReviewState.ts:5`, `Sidepanel/Chat/body.tsx:7`).

**Tech Stack:** React 18 (`React.memo`, `useMemo`, `useCallback`), antd Table, `@tanstack/react-virtual`, vitest + @testing-library/react.

## Global constraints

- Frontend only: never modify `tldw_Server_API/**`. Commands from `apps/packages/ui`: `bun vitest run <path>`; `bunx tsc --noEmit -p tsconfig.json`.
- Behavior-preserving: same data on screen, same testids, same copy. Existing `__tests__` must stay green without weakened assertions.
- Never memoize with stale deps to "fix" a warning — if a dependency list fights you, that's a state-placement problem; restructure per the stage instead.
- One commit per stage referencing the Backlog task ID.
- Line numbers come from the 2026-10-06 review; re-locate by symbol.
- Order note: Stage 1 (Monitoring) is independent of Plan B Stage 2 (polling) but touches the same file — land Plan B Stage 2 first or rebase cleanly; do not bundle both stages in one commit.

---

## Stage 1: MonitoringDashboard render isolation (F14)

**Goal:** A 10s "time ago" label stops re-rendering five tables; poll cycles don't re-diff unchanged data; the activity table gets stable keys and pagination.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Admin/MonitoringDashboardPage.tsx:435-446,482,510,546,640-648,826`
- Create: `apps/packages/ui/src/components/Option/Admin/RefreshedAtLabel.tsx`
- Test: extend `__tests__/MonitoringDashboardPage.*.test.tsx`; new `__tests__/RefreshedAtLabel.test.tsx`

**Change:**
1. Extract `<RefreshedAtLabel at={lastRefreshedAt} />` — the component owns the 10s interval + its own `timeSinceRefresh` state, memoized with `React.memo` (props: `at: Date | null`, `t`). Delete the page-level `timeSinceRefresh` state and the page-level tick effect (`:435-446`). One tiny component re-renders per tick; the page doesn't.
2. Memoize column arrays — `ruleColumns`, `historyColumns`, `sandboxRuntimeColumns`, `activityColumns` (and `metricOptions` if not already) in `useMemo` with `[t, ...data deps]`.
3. `activityRows` (`:640-643`): build in `useMemo([activityEntries])`; replace `_key: idx` with a stable composite — prefer a real id/timestamp field on the entry (`entry.id ?? entry.timestamp ?? entry.created_at`), fallback to index **captured once at fetch time** in the loader, not per render. Add `pagination={{ pageSize: 20 }}` (matches the alert-history table) replacing `pagination={false}` at `:826`.
4. Unchanged-data guard for poll cycles: helper `setIfChanged<T>(setter: (v: T) => void, next: T)` comparing `JSON.stringify` — call it in each `load*` setter. O(payload) stringify per poll is far cheaper than full table reconciliation; note the tradeoff in a comment. (Skippable for slices Plan B Stage 2 already removed from polling.)

**Tests:**
- [ ] `test_tick_does_not_rerender_tables` — spy render counter on a table cell component; advance 10s → counter unchanged while the "Xs ago" text updates.
- [ ] `test_refreshed_at_label_formats_buckets` — 5s → "just now", 45s → "45s ago", 95s → "1m ago" (use the existing i18n strings).
- [ ] `test_identical_poll_data_skips_setstate` — resolve a poll with the same payload → no re-render of table children.
- [ ] `test_activity_table_paginated` — 45 entries → 20 rows in DOM, pagination control present.

**Status:** Not Started

## Stage 2: LlamacppAdminPage — hoist `settings` down, memoize panels (F15)

**Goal:** Typing in one launch-form field re-renders the launch form, not the other five panels.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx:535,1289-1367`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppLaunchPanel.tsx:106`
- Test: extend `__tests__/LlamacppAdminPage.test.tsx`, `__tests__/LlamacppLaunchPanel.test.tsx`

**Change:**
1. Move the `settings` state into `LlamacppLaunchPanel`: the panel initializes from an optional `initialSettings` prop (or a defaults builder passed once) and owns all ~25 controls' edits; it no longer receives `settings`/`onSettingsChange`. The page receives final args at submit time (existing `onStart`-style callback reads the panel's current value via a ref the panel sets, or the panel passes args up on submit — pick whichever matches the existing launch flow and keep the page's submit handler signature unchanged).
2. Wrap all six panels in `React.memo` (`LlamacppAssetsPanel`, `LlamacppProfilesPanel`, `LlamacppRuntimePanel`, `LlamacppInventoryPanel`, `LlamacppReadinessPanel`, `LlamacppLaunchPanel`).
3. `useCallback` every handler prop the page passes (`:1289-1327`: `handleRegisterAssetPath`, `handleImportAssetFolder`, `handleStartProfile`, and siblings — locate by symbol). Without this, step 2 is a no-op.
4. Props that are per-render derived objects (e.g. mapped option lists) must be memoized in the page or pushed into the panels — audit each memoized panel's prop list for identity stability before declaring done.

**Tests:**
- [ ] `test_keystroke_rerenders_only_launch_panel` — render counters on two memoized panels; type in a launch input → sibling counter unchanged.
- [ ] `test_launch_submits_current_args` — edit fields, submit → page handler receives the edited payload (guard the state move didn't break the flow).
- [ ] Existing `LlamacppAdminPage.test.tsx` / `LlamacppLaunchPanel.test.tsx` green.

**Status:** Not Started

## Stage 3: Map/Set + useMemo sweep — the O(n·m) lookups (F16 lookups, F18)

**Goal:** Row renders do O(1) lookups; sorts/filters of fetched lists run only when the data or input they depend on changes.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Admin/RbacEditorPage.tsx:147-149,586-592,740-744`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppSnapshotsPanel.tsx:282-284`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx:64-71,167`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppProfilesPanel.tsx:137-154,481,496`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx:743-745,1329-1334`
- Test: extend the corresponding `__tests__` files (all exist)

**Change** (model: `LlamacppRuntimePanel.tsx:163-180`):
1. RbacEditorPage:
   - `const permissionById = React.useMemo(() => new Map(allPermissions.map(p => [p.id, p])), [allPermissions])`; `overrideColumns` permission cell becomes `permissionById.get(record.permission_id)?.name` (delete the `.find` at `:591`).
   - `const assignedRoleIds = React.useMemo(() => new Set(userRoles.map(ur => ur.id ?? ur.role_id)), [userRoles])`; role options `disabled: assignedRoleIds.has(r.id)` (replaces the `.some` at `:740-744`).
   - `filteredPermissions` in `useMemo([permissions, filterState])`.
2. LlamacppSnapshotsPanel: `const sortedSnapshots = React.useMemo(() => [...(props.catalog?.snapshots ?? [])].sort((a, b) => b.commit_sequence - a.commit_sequence), [props.catalog?.snapshots])`.
3. LlamacppAssetsPanel: `const assetGroups = React.useMemo(() => toAssetGroups(assetList), [assetList])`.
4. LlamacppProfilesPanel: `const ggufOptions = React.useMemo(() => assetOptions(assetList, "gguf", form.modelId), [assetList, form.modelId])` and the mmproj twin; use them at `:481,496`.
5. LlamacppAdminPage: `:743-745` build `runtimeByProfileId` Map in `useMemo`; `:1329-1334` replace filter+map+find with a single memoized lookup `const snapshotProfile = React.useMemo(() => ({ profile: runtimeProfiles.find(p => p.profile_id === snapshotProfileId), runtime: runtimeByProfileId.get(snapshotProfileId) }), [...])`.

**Tests:**
- [ ] `test_override_lookup_is_o1` — 300 permissions, 100 overrides; spy on `Map.prototype.get` vs array `.find` (assert `.find` not invoked during render).
- [ ] `test_role_options_disabled_via_set` — assigned roles disabled, unassigned enabled (behavior parity).
- [ ] `test_snapshot_sort_runs_once_per_catalog` — two renders with same catalog reference → sort spy called once.
- [ ] `test_asset_groups_rebuilt_only_on_list_change` — keystroke in an asset input does not re-run `toAssetGroups` (spy).
- [ ] Existing panel tests green.

**Status:** Not Started

## Stage 4: Paginate the RBAC matrix; virtualize inventory lists (F16 grid, F17)

**Goal:** The permissions×roles checkbox grid and the filesystem-derived lists render bounded DOM.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Admin/RbacEditorPage.tsx:165-181,214-227`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppInventoryPanel.tsx:119-180`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx:463-505`
- Test: extend `__tests__/RbacEditorPage.*.test.tsx`, `__tests__/LlamacppInventoryPanel.test.tsx`, `__tests__/LlamacppAssetsPanel.test.tsx`

**Change:**
1. RBAC matrix: paginate the permissions table `pagination={{ pageSize: 50 }}` (server's `matrix-boolean` already pages permissions — check the page's fetch limit and raise it to match pageSize if needed); memoize the role-column array on `[roles, grid, t]` (from Stage 3 habits). Keep `scroll={{ x: 260 + roles.length * 120 }}`. A "select all on page" affordance is **not** in scope.
2. Inventory + assets lists: `useVirtualizer` from `@tanstack/react-virtual` (copy the wiring from `Review/hooks/useMediaReviewState.ts` / `Sidepanel/Chat/body.tsx`): scroll container with fixed `maxHeight` (e.g. `h-96 overflow-y-auto`), `count: models.length`, `getItemKey: i => models[i].model_id`, estimated row height from the current `<li>` layout, `overscan: 8`. Same treatment per asset group in `LlamacppAssetsPanel` (virtualize inside each group section, not across groups).
3. If a group/list has < 30 items, virtualization may be skipped via a simple length guard (`n >= 30 ? virtualized : plain map`) — keep both render paths exercised by tests.

**Tests:**
- [ ] `test_matrix_paginates_permissions` — 120 permissions → 50 rows in DOM; page 2 renders the next 50.
- [ ] `test_matrix_columns_stable` — role columns array identity stable across re-renders with unchanged roles.
- [ ] `test_inventory_renders_windowed_rows` — 500 models, container shows ≤ 30 `<li>` in DOM; scroll jumps render the right window (assert on `getItemKey` presence).
- [ ] `test_small_lists_render_plain` — 10 models render without virtualizer container.

**Status:** Not Started

## Stage 5: Stable rowKeys, shared date formatter, small-form isolation (F19–F23)

**Goal:** No `JSON.stringify`/`Math.random()`/index row keys; one `Intl.DateTimeFormat` instead of per-cell `new Date().toLocaleString()`; the maintenance banner's inputs stop re-rendering three tables.

**Files:**
- Create: `apps/packages/ui/src/components/Option/Admin/admin-format.ts` (formatter)
- Modify: `DataOpsPage.tsx:329,479,1017`; `BillingDashboardPage.tsx:323,433`; `RateLimitingPage.tsx:328-330`; `ServerArgsEditor.tsx:160-161`; `MaintenancePage.tsx:302,353,359,404-417`; `MonitoringDashboardPage.tsx:519,645`; `WatchlistsOversightPage.tsx:250,269,292`; `ApiKeyManagementPage.tsx:192`; `WatchlistsPage.tsx:228`; `ServerAdminPage.tsx:414-506,850-911,986`; `Layouts/Layout.tsx:789`
- Test: new `__tests__/admin-format.test.ts`; extend touched pages' tests

**Change:**
1. Row keys:
   - DataOps backups/schedules/bundles: `rowKey={(r) => r.id ?? r.backup_id ?? `${r.dataset ?? "ds"}|${r.created_at ?? ""}|${r.filename ?? r.name ?? ""}`}` (deterministic composite; if duplicates are possible add a fetch-time index suffix in the loader).
   - Billing subscriptions/events: replace `Math.random()` fallbacks with the same composite style (`${r.user_id ?? r.id ?? "u"}|${r.created_at ?? ""}`).
   - RateLimiting unprotected routes: `key: r.route` (or `r.path` — use the actual path field; it is unique).
   - ServerArgsEditor kv pairs: assign `id` (incrementing counter or `crypto.randomUUID()` at pair creation); key by it; keys then survive deletion.
2. `admin-format.ts`: `export const adminDateTime = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" })` and `export const formatAdminDateTime = (v: string | number | Date) => adminDateTime.format(new Date(v))`. Replace every per-cell `new Date(...).toLocaleString()` in the files listed above (F20 list in the spec). Locale-shape drift is acceptable if tests asserted exact strings — update those assertions to the formatter's output.
3. MaintenancePage: extract `MaintenanceBannerForm` (owns `maintMessage` + `maintAllowlist` state, lifts committed values up via debounce-on-save or explicit Apply) so keystrokes re-render the form only. The three tables stay mounted with stable props.
4. ServerAdminPage: `useMemo` `userRoleOptions` (`:414-421`), `userColumns` (`:423-506`), roles-table columns (`:850-911`), media-budget user options (`:986`) with `useCallback` handlers where they close over state.
5. Layouts/Layout.tsx:789: wrap the context value in `useMemo([setOverrides])`.

**Tests:**
- [ ] `test_rowkeys_stable_across_renders` — render DataOps/Billing tables twice with identical data → same key set, no `Math.random`/`JSON.stringify` in keys (assert format).
- [ ] `test_formatter_formats_iso` — known ISO input → formatter output (snapshot the `Intl` result via `Intl.DateTimeFormat` mock or `formatToParts` count).
- [ ] `test_maintenance_keystroke_isolated` — render counters on the three tables; typing in the banner → counters unchanged.
- [ ] `test_serveradmin_columns_stable` — column array identities stable across a `resetPasswordResult` state change.
- [ ] Existing suites for all touched pages green.

**Status:** Not Started

---

## Verification (whole plan)

- [ ] `cd apps/packages/ui && bun vitest run src/components/Option/Admin src/components/Layouts` green
- [ ] `cd apps/packages/ui && bunx tsc --noEmit -p tsconfig.json`
- [ ] Manual smoke (running server): Monitoring with auto-refresh — React DevTools profiler shows only `RefreshedAtLabel` re-rendering per tick; Llamacpp admin — typing in the launch form doesn't re-render the inventory panel; RBAC editor with 200+ permissions — DOM node count stays bounded while paging
- [ ] Self-review diff; update the Backlog task with touched files, verification results, final summary
