# First-Class Watchlists Stage 6 Extension Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make full Watchlist management viable in constrained WebUI/browser-extension viewports without regressing desktop `/watchlists` workflows.

**Architecture:** Keep the existing first-class Watchlist container, scoped child tabs, store, and API contracts. Add a small shared constrained-viewport foundation, then replace wide-table-only management surfaces with list/detail and drawer patterns in the tabs that still require horizontal scrolling. Preserve the existing Items tab reader pattern where it already works, and use real FastAPI plus real WebUI CDP smoke tests as the release gate.

**Tech Stack:** React, TypeScript, Ant Design, Zustand Watchlists store, existing Watchlists service layer, Vitest/Testing Library, Playwright/CDP, real FastAPI, real Next WebUI, pytest only if backend behavior changes, Bandit only if Python code changes.

---

## Scope

Stage 6 is a frontend-heavy remediation stage for `/watchlists` and the shared WebUI/extension surface. It must not redesign unrelated pages, invent new backend workflows, or add trust/credibility scoring. Backend changes should be avoided unless a constrained-flow blocker proves an existing endpoint cannot support full management.

The accepted product requirement is stricter than "read-only mobile": extension-sized viewports must support full management. Users must be able to switch Watchlists, create/edit/delete/restore where the current page supports it, run monitors, review updates, inspect alerts/evidence, manage reports/templates, and open settings without clipped primary controls or document-level horizontal overflow.

## Current Evidence

Relevant existing files:

- Route/shell: `apps/packages/ui/src/routes/option-watchlists.tsx`
- Main shell: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- Store: `apps/packages/ui/src/store/watchlists.tsx`
- Types/services: `apps/packages/ui/src/types/watchlists.ts`, `apps/packages/ui/src/services/watchlists.ts`
- Shared helpers: `apps/packages/ui/src/components/Option/Watchlists/shared/*`
- Existing constrained reader pattern: `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/ItemsTab.tsx`
- Remaining wide table surfaces:
  - `SourcesTab/SourcesTab.tsx` uses `Table` with `scroll={{ x: "max-content" }}`.
  - `JobsTab/JobsTab.tsx` uses `Table` with `scroll={{ x: 900 }}`.
  - `RunsTab/RunsTab.tsx` uses `Table` with `scroll={{ x: 800 }}`.
  - `OutputsTab/OutputsTab.tsx` uses `Table` with `scroll={{ x: 800 }}`.
  - `TemplatesTab/TemplatesTab.tsx` uses a table-only management list.
  - `SettingsTab/SettingsTab.tsx` uses a table for claim-cluster subscriptions.
  - `OutputsTab/ReportEvidencePanel.tsx` uses a table for included evidence.
  - `RunsTab/RunDetailDrawer.tsx` uses a table for run items.
- Existing constrained tests to preserve:
  - `ItemsTab/__tests__/ItemsTab.scale-responsive.test.tsx`
  - `ItemsTab/__tests__/ItemsTab.keyboard-shortcuts.test.tsx`
  - `OutputsTab/__tests__/ReportBuilderDrawer.test.tsx`
  - `WatchlistsPlaygroundPage.*.test.tsx`

## Non-Goals

- Do not add LLM novelty, source credibility, or analyst confidence scoring.
- Do not rename the module away from Watchlists.
- Do not replace the existing Items tab reader with a table.
- Do not create a parallel mobile-only route.
- Do not mock the server for browser QA.
- Do not use Computer Use for browser QA; use Playwright/CDP.

## File Structure

Shared additions:

- Create: `apps/packages/ui/src/components/Option/Watchlists/shared/useWatchlistsViewport.ts`
  - Owns constrained-viewport detection and testable breakpoint behavior.
- Create: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsMobileNavigation.tsx`
  - Renders constrained task navigation and a drawer/list of all management destinations.
- Create if duplication becomes real: `apps/packages/ui/src/components/Option/Watchlists/shared/ResponsiveEntityList.tsx`
  - Small presentational helper for entity card lists with pagination summary and stable actions.
  - Do not add this if two tabs can be improved cleanly without it.

Shell changes:

- Modify: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
  - Replace the current mobile-only tab `Select` with constrained task navigation that keeps all destinations discoverable.
  - Keep desktop tabs and existing deep-link behavior.
- Modify tests under `apps/packages/ui/src/components/Option/Watchlists/__tests__/`.

Tab changes:

- Modify: `SourcesTab/SourcesTab.tsx` and focused source tests.
- Modify: `JobsTab/JobsTab.tsx` and focused monitor tests.
- Modify: `RunsTab/RunsTab.tsx`, `RunDetailDrawer.tsx`, and focused activity tests.
- Modify: `OutputsTab/OutputsTab.tsx`, `ReportEvidencePanel.tsx`, and focused report tests.
- Modify: `TemplatesTab/TemplatesTab.tsx`, `TemplateEditor.tsx` only if needed for constrained modal controls, and focused template tests.
- Modify: `SettingsTab/SettingsTab.tsx` and focused settings tests.
- Modify locale files only when new user-facing copy is introduced:
  - `apps/packages/ui/src/assets/locale/en/watchlists.json`
  - `apps/packages/ui/src/public/_locales/en/watchlists.json`

Docs/tasks:

- Update this plan as tasks are completed.
- Update `backlog/tasks/task-349.3*`.
- Update `Docs/API-related/Watchlists_API.md` only if API behavior changes.

## Task 0: Create Stage 6 Backlog Records

**Files:**
- Modify: `backlog/tasks/task-349.3 - Plan-Stage-6-Watchlist-extension-sized-full-management.md`
- Create via Backlog: child tasks `TASK-349.3.1` through `TASK-349.3.5`

- [x] **Step 1: Create child tasks**

Create these Backlog tasks:

- `TASK-349.3.1`: Stage 6A Watchlist constrained navigation shell.
- `TASK-349.3.2`: Stage 6B Sources and Monitors constrained management.
- `TASK-349.3.3`: Stage 6C Activity Reports and Templates constrained management.
- `TASK-349.3.4`: Stage 6D CRUD modals drawers and accessibility hardening.
- `TASK-349.3.5`: Stage 6E Real-server constrained viewport QA and closeout.

- [x] **Step 2: Link records**

Each child task must reference:

- This plan.
- `Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md`
- `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md`

- [x] **Step 3: Verify task files**

Run:

```bash
rg -n "TASK-349.3|Stage 6" backlog/tasks Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md
```

Expected: parent task and all child tasks are discoverable.

Recorded on planning closeout: parent task `TASK-349.3` and child tasks `TASK-349.3.1` through `TASK-349.3.5` are discoverable, and all child records link this plan, the approved design spec, and the Stage 5 report plan.

## Task 1: Stage 6A Constrained Navigation Shell

**Files:**
- Create: `apps/packages/ui/src/components/Option/Watchlists/shared/useWatchlistsViewport.ts`
- Create: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsMobileNavigation.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/shared/index.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.extension-navigation.test.tsx`
- Test as needed: `apps/packages/ui/src/components/Option/Watchlists/shared/__tests__/useWatchlistsViewport.test.ts`

- [x] **Step 1: Write failing viewport hook tests**

Test that the hook or pure resolver classifies:

- `390x844` as constrained.
- `420x760` as constrained.
- `768px` and above as desktop.
- SSR/no-window as desktop-safe false.

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/shared/__tests__/useWatchlistsViewport.test.ts
```

Expected before implementation: fail because the helper does not exist.

- [x] **Step 2: Implement the viewport helper**

Use `window.matchMedia` and a single breakpoint constant. Keep this helper local to Watchlists unless another page already has a shared project-wide hook that cleanly fits.

- [x] **Step 3: Write failing constrained navigation tests**

In `WatchlistsPlaygroundPage.extension-navigation.test.tsx`, render the page with `innerWidth = 420` and assert:

- The selected Watchlist switcher remains reachable.
- A constrained management navigation trigger is visible.
- Opening it exposes Overview, Feeds, Monitors, Alerts, Updates, Activity, Reports, Templates, and Settings.
- Selecting Monitors, Activity, or Templates preserves existing deep-link mapping and renders the correct child content.
- Desktop width still renders the existing tab layout.

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.extension-navigation.test.tsx
```

Expected before implementation: fail because the navigation does not exist.

- [x] **Step 4: Implement constrained navigation**

Replace the current mobile `Select` tab switcher in `WatchlistsPlaygroundPage.tsx` with a `WatchlistsMobileNavigation` component. The component should use a button plus drawer/list pattern so users can scan grouped destinations and not rely on a long select menu.

Required groups:

- Overview.
- Collect: Feeds, Monitors.
- Review: Alerts, Updates, Activity.
- Reports: Reports, Templates.
- Settings.

Keep desktop `Tabs` unchanged.

- [x] **Step 5: Run focused tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.extension-navigation.test.tsx \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.orientation-guidance.test.tsx
```

Expected: all selected tests pass.

Recorded on Stage 6A closeout: the focused Stage 6A Vitest suite passed with 4 files and 12 tests, covering constrained navigation, viewport breakpoint behavior, first-class Watchlist shell regressions, and orientation guidance. The Watchlists static guard also passed with 1 file and 3 tests.

- [x] **Step 6: Commit Stage 6A**

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/shared/useWatchlistsViewport.ts \
  apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsMobileNavigation.tsx \
  apps/packages/ui/src/components/Option/Watchlists/shared/index.ts \
  apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx \
  apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.extension-navigation.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/shared/__tests__/useWatchlistsViewport.test.ts \
  backlog/tasks/task-349.3.1*
git commit -m "feat: add watchlist constrained navigation"
```

## Task 2: Stage 6B Sources And Monitors Constrained Management

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx`
- Modify if needed: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/GroupsTree.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.extension-management.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.extension-management.test.tsx`

- [ ] **Step 1: Write failing source constrained-management tests**

At `420px` width, assert:

- The sources table is replaced by a source card/list view.
- Add Source, Import OPML, Refresh, search, type filter, group filter, and advanced-column toggle remain reachable.
- Each source row exposes active toggle, check now, seen stats, edit, and delete.
- Multi-select source actions remain reachable without a wide selection table.
- No element with `aria-label="Feeds table"` is rendered in constrained mode.

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.extension-management.test.tsx
```

Expected before implementation: fail.

- [ ] **Step 2: Implement source list/detail pattern**

Add constrained rendering inside `SourcesTab.tsx` using the Stage 6A viewport helper. Keep desktop table behavior. Use cards/list rows with stable dimensions, visible labels, and icon buttons with accessible names.

Source constrained row must include:

- Name, URL, type, tags/groups summary.
- Status/health.
- Active toggle.
- Check now.
- Seen details.
- Edit.
- Delete.
- Selection checkbox for bulk actions.

- [ ] **Step 3: Write failing monitor constrained-management tests**

At `420px` width, assert:

- The monitors table is replaced by a monitor card/list view.
- Add Monitor, Refresh, advanced details toggle, run now, preview, edit, delete, and active toggle remain reachable.
- Schedule, scope summary, filters, output linkage, last run, and next run remain visible or available in an expandable details region.
- No element with `aria-label="Monitors table"` is rendered in constrained mode.

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.extension-management.test.tsx
```

Expected before implementation: fail.

- [ ] **Step 4: Implement monitor list/detail pattern**

Add constrained rendering inside `JobsTab.tsx`. Keep desktop table behavior. Use the existing summary helpers from `job-summaries.ts`; do not duplicate scope/filter summary logic.

- [ ] **Step 5: Run focused regression tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.extension-management.test.tsx \
  src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.bulk-move.test.tsx \
  src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.delete-confirm.test.tsx \
  src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.extension-management.test.tsx \
  src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.advanced-details.test.tsx \
  src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.undo-delete.test.tsx
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Stage 6B**

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.extension-management.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.extension-management.test.tsx \
  backlog/tasks/task-349.3.2*
git commit -m "feat: adapt watchlist sources and monitors for constrained management"
```

## Task 3: Stage 6C Activity Reports And Templates Constrained Management

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportEvidencePanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/TemplatesTab.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunsTab.extension-management.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.extension-management.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportEvidencePanel.extension-management.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/__tests__/TemplatesTab.extension-management.test.tsx`

- [ ] **Step 1: Write failing Activity tests**

At `420px`, assert:

- The Activity table is replaced by run cards.
- Status filter, job filter, Refresh, CSV export, cancel where available, detail open, and relationship jumps remain reachable.
- Run detail drawer shows run items in a list/card pattern instead of a horizontally scrolling table.

- [ ] **Step 2: Implement Activity constrained cards**

Keep the existing desktop table and column chooser behavior. In constrained mode, render run cards with status, job/source summary, started/finished timing, item/error counts, actions, and details.

- [ ] **Step 3: Write failing Reports tests**

At `420px`, assert:

- Reports table is replaced by report cards.
- Create report, refresh, advanced filters, preview, evidence, download, regenerate, delivery issue actions, and relationship jumps remain reachable.
- Evidence panel renders included evidence as cards/list rows in constrained mode.

- [ ] **Step 4: Implement Reports constrained cards**

Use existing output metadata helpers from `outputMetadata.ts`; do not recreate metadata parsing. Keep Stage 5 report builder and preview drawers intact, but verify their drawers remain full-width and usable at constrained width.

- [ ] **Step 5: Write failing Templates tests**

At `420px`, assert:

- Templates management is card/list based.
- Create, preview/edit, duplicate if currently available, delete, and refresh remain reachable.
- Template format/version metadata is visible without horizontal scrolling.

- [ ] **Step 6: Implement Templates constrained cards**

Keep the existing desktop table. Use a simple list/detail pattern for constrained mode and preserve `TemplateEditor` flows.

- [ ] **Step 7: Run focused regression tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/RunsTab/__tests__/RunsTab.extension-management.test.tsx \
  src/components/Option/Watchlists/RunsTab/__tests__/RunsTab.advanced-filters.test.tsx \
  src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.source-column.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.extension-management.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.defensible-reports.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/ReportEvidencePanel.extension-management.test.tsx \
  src/components/Option/Watchlists/TemplatesTab/__tests__/TemplatesTab.extension-management.test.tsx \
  src/components/Option/Watchlists/TemplatesTab/__tests__/TemplatesTab.delete-safety.test.tsx
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit Stage 6C**

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunsTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportEvidencePanel.tsx \
  apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/TemplatesTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunsTab.extension-management.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.extension-management.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportEvidencePanel.extension-management.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/__tests__/TemplatesTab.extension-management.test.tsx \
  backlog/tasks/task-349.3.3*
git commit -m "feat: adapt watchlist activity reports and templates for constrained management"
```

## Task 4: Stage 6D CRUD Modals Drawers And Accessibility Hardening

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesBulkImport.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobPreviewModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/TemplateEditor.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SettingsTab/SettingsTab.tsx`
- Modify if needed: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- Test: focused modal/drawer accessibility tests under existing tab test folders.

- [ ] **Step 1: Write failing modal width/focus tests**

Cover at least:

- Source form.
- OPML import preflight.
- Monitor form.
- Monitor preview.
- Template editor.
- Settings drawer/cluster subscription list.

At constrained width, each test should verify:

- Primary actions are visible and have accessible names.
- Footer actions do not clip.
- The modal/drawer is full-width or internally stacked.
- Escape/cancel closes without leaving focus trapped in a removed node.

- [ ] **Step 2: Implement constrained modal/drawer behavior**

Prefer existing Ant Design modal/drawer APIs and utility classes. Do not build a custom modal system. Use full-width drawers for dense editors if Ant modal footers become cramped.

- [ ] **Step 3: Add keyboard navigation checks**

Extend existing keyboard/focus tests where practical:

- Tab reaches the constrained navigation trigger.
- Tab reaches primary create/edit/delete actions in constrained lists.
- Escape closes constrained drawers.
- The active tab/content has a stable accessible label.

- [ ] **Step 4: Run focused accessibility tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.extension-navigation.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.accessibility-baseline.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.keyboard-shortcuts.test.tsx \
  src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx \
  src/components/Option/Watchlists/SourcesTab/__tests__/SourcesBulkImport.preflight-commit.test.tsx \
  src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx \
  src/components/Option/Watchlists/JobsTab/__tests__/JobPreviewModal.focus.test.tsx \
  src/components/Option/Watchlists/TemplatesTab/__tests__/TemplateEditor.mode-contract.test.tsx \
  src/components/Option/Watchlists/SettingsTab/__tests__/SettingsTab.help.test.tsx
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Stage 6D**

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx \
  apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesBulkImport.tsx \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobPreviewModal.tsx \
  apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/TemplateEditor.tsx \
  apps/packages/ui/src/components/Option/Watchlists/SettingsTab/SettingsTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/**/__tests__/* \
  backlog/tasks/task-349.3.4*
git commit -m "feat: harden watchlist constrained CRUD accessibility"
```

## Task 5: Stage 6E Real-Server CDP QA And Closeout

**Files:**
- Modify: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md`
- Modify: `backlog/tasks/task-349.3*`
- Modify docs only if behavior or user-facing copy requires it.

- [ ] **Step 1: Run focused frontend suite**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.extension-navigation.test.tsx \
  src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.extension-management.test.tsx \
  src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.extension-management.test.tsx \
  src/components/Option/Watchlists/RunsTab/__tests__/RunsTab.extension-management.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.extension-management.test.tsx \
  src/components/Option/Watchlists/TemplatesTab/__tests__/TemplatesTab.extension-management.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.scale-responsive.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx
```

Expected: all selected tests pass.

- [ ] **Step 2: Run static checks**

Run:

```bash
git diff --check
```

If Python files changed, run Bandit on touched Python scope. If only TypeScript/Markdown/Backlog files changed, record Bandit as not applicable.

- [ ] **Step 3: Start real servers**

Use the real FastAPI server and real Next WebUI. Do not mock the server.

Expected pattern:

```bash
AUTH_MODE=single_user \
SINGLE_USER_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY \
DATABASE_URL=sqlite:////private/tmp/tldw-watchlists-stage6/users.db \
USER_DB_BASE_DIR=/private/tmp/tldw-watchlists-stage6/user_dbs \
WATCHLIST_TEMPLATE_DIR=/private/tmp/tldw-watchlists-stage6/templates \
TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS=/private/tmp \
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port <api-port>
```

```bash
cd apps/packages/ui
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced \
NEXT_PUBLIC_API_URL=http://127.0.0.1:<api-port> \
NEXT_PUBLIC_X_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY \
bun run dev -- -H 127.0.0.1 -p <web-port>
```

If the sandbox blocks port binding, request escalation rather than switching to mocked services.

- [ ] **Step 4: Seed representative data**

Use real public APIs for Watchlists, sources, monitors, templates, and outputs where available. If item/run/alert setup still lacks public creation APIs, insert deterministic run/items/alerts into the server-owned Watchlists DB and document that as QA setup, not a mocked server.

Seed at least:

- One CTI Watchlist with feeds, monitors, content alert, queued updates, completed run, and report.
- One news Watchlist with feeds, monitor, queued updates, completed run, and report.
- One source group/tag.
- One template.

- [ ] **Step 5: Run constrained CDP smoke**

Open `/watchlists?view=all` at `420x760` and verify:

- Watchlist switcher and constrained navigation work.
- Feeds: create/edit/disable/check/delete or undo path is reachable.
- Monitors: create/edit/disable/run now/preview/delete or undo path is reachable.
- Alerts: content alert review path is reachable.
- Updates: filter/sort, saved view, item detail, batch review, and report queue path are reachable.
- Activity: run detail opens and items are readable.
- Reports: create/preview/evidence/download/regenerate path is reachable.
- Templates: create/edit/preview/delete path is reachable.
- Settings: lifecycle/settings drawer is reachable.
- No document-level horizontal overflow at `420x760`.
- Console messages and request failures are recorded.

Capture screenshots under `/private/tmp/tldw-watchlists-stage6/`.

- [ ] **Step 6: Close Backlog tasks**

For each Stage 6 task:

- Check acceptance criteria.
- Record verification.
- Record screenshots and real-server notes.
- Record skips/blockers.
- Mark Done.

- [ ] **Step 7: Commit Stage 6E**

```bash
git add \
  Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md \
  backlog/tasks/task-349.3*
git commit -m "docs: close watchlist constrained management stage"
```

## Stage 6 Exit Criteria

- `/watchlists` has a constrained navigation model that exposes every management destination.
- Desktop tab behavior and existing deep links continue to work.
- Feeds, Monitors, Activity, Reports, Templates, Settings cluster subscriptions, run detail items, and report evidence no longer require horizontal table scrolling at extension width.
- Existing Items tab constrained reader remains usable and is not regressed.
- Source, monitor, alert, item, report, template, and settings management flows remain reachable at `420x760`.
- Primary actions do not clip, overlap, or require document-level horizontal scrolling.
- Keyboard/focus behavior works across constrained navigation, list actions, drawers, and modals.
- Focused Vitest coverage and real-server CDP screenshots are recorded.

## Known Deferrals To Later Stages

- Trust/calibration explanations, source credibility scoring, and analyst confidence calibration remain Stage 7.
- Data model changes for new novelty/confidence fields remain out of Stage 6.
- Reusable source ownership across multiple Watchlists remains a separate data-model decision.
- A full design-system migration of all Watchlists components is outside Stage 6 unless a local shared helper prevents clear duplication.
