# First-Class Watchlists Stage 2 Setup Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace generic quick setup with Watchlist-first onboarding that creates the project-like Watchlist before sources, monitors, and briefing outputs.

**Architecture:** Reuse the Stage 1 Watchlist CRUD/scoping contract and existing quick setup/pipeline builders. Add a focused Watchlist setup wizard model and component that can create only a Watchlist or create a Watchlist plus initial feeds and an optional monitor, then select the new Watchlist and hand off to the existing scoped tabs.

**Tech Stack:** React, Ant Design, Zustand Watchlists store, existing Watchlists service layer, Vitest/Testing Library, Playwright/CDP for constrained viewport smoke, existing FastAPI Watchlists API.

---

## Source Spec

- Design spec: `Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md`
- Stage 1 plan: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md`
- Stage 2 planning task: `TASK-370`
- Stage 1 closeout task: `TASK-365`

## Product Decisions For Stage 2

- The create path should make the Watchlist the first object, not an afterthought on Quick Setup.
- Preserve the existing `/watchlists` route and existing scoped child tabs.
- Stage 2 does not implement content-match alert rules, entity extraction, source discovery, novelty scoring, or a defensible report builder.
- "Start from topic" creates a Watchlist with objective/scope metadata and routes the user to Feeds or Monitors; it must not pretend to monitor the topic without sources.
- "Start from report goal" may create a monitor only when sources are supplied or selected. Otherwise it creates the Watchlist and opens Reports/Templates guidance.
- CTI/news presets are user-facing copy and payload defaults, not domain-specific backend logic in this stage.
- The existing quick setup logic should be reused or migrated, then demoted to an internal "initial collection setup" step under the Watchlist-first wizard.
- Full management in constrained viewports remains required. The Stage 2 wizard must work in extension-sized viewports even if later Stage 6 improves all child tabs.

## Current Evidence To Preserve

- Watchlist shell/create modal: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- Watchlist store selection: `apps/packages/ui/src/store/watchlists.tsx`
- Watchlist service CRUD: `apps/packages/ui/src/services/watchlists.ts`
- Watchlist types: `apps/packages/ui/src/types/watchlists.ts`
- Existing Quick Setup helpers: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts`
- Existing Quick Setup UI: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
- Existing Pipeline Builder UI: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
- Existing first-class shell tests: `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx`
- Existing quick setup helper tests: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts`
- Existing quick setup UI tests: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx`
- Locale copy: `apps/packages/ui/src/assets/locale/en/watchlists.json`
- Extension locale mirror if still maintained: `apps/packages/ui/src/public/_locales/en/watchlists.json`

## Implementation Boundaries

- Do not rename `/watchlists`.
- Do not replace the existing Feeds, Monitors, Activity, Articles, Reports, Templates, or Settings tabs.
- Do not change backend source URL uniqueness or Watchlist source membership semantics.
- Do not add content-match alert UI in Stage 2.
- Do not add a source discovery/search provider in Stage 2.
- Do not call reports "defensible" in the wizard; report artifacts become defensible only after Stage 5 provenance/report-builder work.
- Do not introduce a new frontend state library.
- Do not use Computer Use for browser QA; use CDP/Playwright.

## Proposed File Responsibilities

Frontend:

- `apps/packages/ui/src/components/Option/Watchlists/SetupWizard/watchlist-setup-model.ts`
  - Define preset/domain/start-mode types.
  - Define CTI/news/general/blank preset defaults.
  - Build `WatchlistCreate`, `WatchlistSourceCreate`, and `WatchlistJobCreate` payloads from wizard values.
  - Normalize topic descriptors, tags, source URLs, and report-goal labels.
- `apps/packages/ui/src/components/Option/Watchlists/SetupWizard/WatchlistSetupWizard.tsx`
  - Ant Design modal/drawer component for the Watchlist-first setup flow.
  - Step 1: domain preset and start mode.
  - Step 2: objective and tracked scope.
  - Step 3: optional sources and monitor/report goal settings.
  - Step 4: review and create.
  - Own submission orchestration using existing service functions.
- `apps/packages/ui/src/components/Option/Watchlists/SetupWizard/index.ts`
  - Re-export component and helper types.
- `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
  - Replace create-Watchlist modal entry with setup wizard for create mode.
  - Keep existing edit modal for editing Watchlist metadata.
  - Select the created Watchlist and navigate to the correct child tab after wizard completion.
- `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
  - Rename user-facing Quick Setup entry points to Watchlist setup / initial collection setup.
  - Keep internal quick setup helpers only for adding sources/monitors to an existing selected Watchlist.
  - Remove or adjust first-visit auto-open if it bypasses Watchlist selection.
- `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts`
  - Either keep for selected-Watchlist collection setup or extract shared payload helpers into `watchlist-setup-model.ts`.
- `apps/packages/ui/src/assets/locale/en/watchlists.json`
  - Add preset, start-mode, review, success, and validation copy.
  - Use generic labels first, with CTI/news examples only inside preset descriptions and placeholders.
- `apps/packages/ui/src/public/_locales/en/watchlists.json`
  - Mirror locale changes if this file remains a manual extension copy.

Tests:

- `apps/packages/ui/src/components/Option/Watchlists/SetupWizard/__tests__/watchlist-setup-model.test.ts`
  - Unit coverage for preset defaults, payload builders, source URL normalization, no-source paths, and report-goal output prefs.
- `apps/packages/ui/src/components/Option/Watchlists/SetupWizard/__tests__/WatchlistSetupWizard.test.tsx`
  - Component coverage for CTI/news presets, topic-only creation, source-backed creation, report-goal creation, validation, and completion handoff.
- `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx`
  - Update create test to expect the setup wizard create flow.
- `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts`
  - Preserve selected-Watchlist quick setup helper behavior if helpers stay.
- `apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlists-stage2-copy-contract.test.ts`
  - Static copy contract for CTI/news/general/blank preset labels and Stage 3 alert boundary.
- Optional Playwright/CDP smoke:
  - Add or reuse a Watchlists route smoke that exercises the setup wizard at `390x844`.

Docs:

- `Docs/API-related/Watchlists_API.md`
  - Update only if Stage 2 needs to clarify frontend wizard behavior over existing CRUD.
- `Docs/Published/API-related/Watchlists_API.md`
  - Mirror docs only when API docs change.

## Backlog Task Map For Implementation

Create implementation tasks before code changes:

- Stage 2A: Setup wizard model, presets, payload builders, and copy contract tests.
- Stage 2B: Watchlist setup wizard component and shell integration.
- Stage 2C: Source-backed and report-goal creation orchestration, including selected-Watchlist handoff.
- Stage 2D: Overview quick setup repositioning and constrained viewport CDP smoke.
- Stage 2E: Docs, verification, and closeout.

Keep commits aligned to these task groups.

## Task 0: Baseline And Task Setup

**Files:**
- Reference: `Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md`
- Reference: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md`
- Reference: `backlog/tasks/task-370 - Plan-Stage-2-Watchlist-creation-and-setup-wizard.md`

- [ ] **Step 1: Create implementation Backlog tasks**

Use Backlog from the worktree to create the Stage 2A-2E tasks listed above. Each task must reference this plan and the design spec.

- [ ] **Step 2: Capture current setup baseline**

Run:

```bash
rg -n "quickSetup|pipelineSetup|createWatchlist|watchlistForm|Guided quick setup|Create Watchlist" \
  apps/packages/ui/src/components/Option/Watchlists \
  apps/packages/ui/src/assets/locale/en/watchlists.json
```

Expected: current shell create modal, Quick Setup, Pipeline Builder, and locale copy are identified before edits.

- [ ] **Step 3: Run current focused frontend baseline**

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx \
  src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts \
  src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: establish current pass/fail baseline before changing onboarding.

- [ ] **Step 4: Commit task records**

Run:

```bash
git add backlog/tasks/<stage-2-task-files>
git commit -m "chore: task watchlists stage 2 setup"
```

Expected: only task records are committed.

## Task 1: Setup Wizard Model And Copy Contract

**Files:**
- Create: `apps/packages/ui/src/components/Option/Watchlists/SetupWizard/watchlist-setup-model.ts`
- Create: `apps/packages/ui/src/components/Option/Watchlists/SetupWizard/__tests__/watchlist-setup-model.test.ts`
- Create: `apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlists-stage2-copy-contract.test.ts`
- Modify: `apps/packages/ui/src/assets/locale/en/watchlists.json`
- Modify if mirrored: `apps/packages/ui/src/public/_locales/en/watchlists.json`

- [ ] **Step 1: Write failing model tests**

Add tests for:

```ts
describe("watchlist setup model", () => {
  it("applies CTI preset defaults without creating alerts", () => {
    const values = applyWatchlistSetupPreset("cti_osint")
    expect(values.domain).toBe("cti_osint")
    expect(values.tags).toContain("cti")
    expect(values).not.toHaveProperty("alertRules")
  })

  it("builds topic-only Watchlist payload without sources or monitors", () => {
    const result = buildWatchlistSetupPlan({
      preset: "news",
      startMode: "topic",
      name: "Election integrity",
      objective: "Track source diversity and recency",
      trackedScopeText: "election officials, state courts",
      sourceUrlsText: "",
      monitorName: "",
      reportGoal: ""
    })
    expect(result.watchlist).toMatchObject({
      name: "Election integrity",
      domain: "news",
      objective: "Track source diversity and recency"
    })
    expect(result.sources).toEqual([])
    expect(result.job).toBeUndefined()
  })

  it("builds source-backed Watchlist, feed, and monitor payloads", () => {
    const result = buildWatchlistSetupPlan({
      preset: "cti_osint",
      startMode: "sources",
      name: "Healthcare ransomware",
      objective: "Find ransomware reports affecting hospitals",
      trackedScopeText: "hospitals, Germany",
      sourceUrlsText: "https://example.com/feed.xml",
      monitorName: "Healthcare ransomware monitor",
      reportGoal: "daily situational brief"
    })
    expect(result.sources).toHaveLength(1)
    expect(result.job?.output_prefs?.template_name).toBe("briefing_md")
  })
})
```

Expected failure: setup model helpers do not exist.

- [ ] **Step 2: Implement setup model**

Add:

```ts
export type WatchlistSetupPreset = "cti_osint" | "news" | "general" | "blank"
export type WatchlistSetupStartMode = "sources" | "topic" | "report_goal"
```

Implement:

- `WATCHLIST_SETUP_PRESETS`
- `applyWatchlistSetupPreset(preset)`
- `parseSetupSourceUrls(value)`
- `buildWatchlistSetupPlan(values)`

Expected: helper tests pass without rendering React.

- [ ] **Step 3: Add copy contract test**

Assert `watchlists.json` includes:

- CTI / OSINT
- News
- General
- Blank
- Start from sources
- Start from topic
- Start from report goal
- A boundary string equivalent to "Content-match alerts come later."

- [ ] **Step 4: Run tests**

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Watchlists/SetupWizard/__tests__/watchlist-setup-model.test.ts \
  src/components/Option/Watchlists/__tests__/watchlists-stage2-copy-contract.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/SetupWizard/watchlist-setup-model.ts \
  apps/packages/ui/src/components/Option/Watchlists/SetupWizard/__tests__/watchlist-setup-model.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlists-stage2-copy-contract.test.ts \
  apps/packages/ui/src/assets/locale/en/watchlists.json \
  apps/packages/ui/src/public/_locales/en/watchlists.json
git commit -m "feat: add watchlist setup wizard model"
```

## Task 2: Watchlist Setup Wizard Component

**Files:**
- Create: `apps/packages/ui/src/components/Option/Watchlists/SetupWizard/WatchlistSetupWizard.tsx`
- Create: `apps/packages/ui/src/components/Option/Watchlists/SetupWizard/index.ts`
- Create: `apps/packages/ui/src/components/Option/Watchlists/SetupWizard/__tests__/WatchlistSetupWizard.test.tsx`

- [ ] **Step 1: Write failing component tests**

Add tests that render the wizard with service callbacks injected as props:

- CTI preset fills example objective/scope and tags.
- Topic-only flow calls `onCreateWatchlist` and `onComplete({ destination: "sources" })`.
- Source-backed flow calls `onCreateWatchlist`, `onCreateSource` with created Watchlist ID, then `onCreateJob` with created Watchlist ID.
- Report-goal flow without sources creates only the Watchlist and routes to Reports/Templates guidance.
- Validation prevents finishing without a Watchlist name.

Expected failure: component does not exist.

- [ ] **Step 2: Implement component shell**

Use Ant Design `Modal` for desktop and responsive internal layout that works at extension width. Keep steps compact:

1. Preset and start mode.
2. Objective and tracked scope.
3. Sources/report/monitor settings.
4. Review.

Use icons only where they clarify domain modes. Avoid separate nested cards inside cards.

- [ ] **Step 3: Implement submission orchestration**

Component props should include service-like callbacks rather than importing global services directly in tests:

```ts
interface WatchlistSetupWizardProps {
  open: boolean
  submitting?: boolean
  onCancel: () => void
  onCreateWatchlist: (payload: WatchlistCreate) => Promise<WatchlistContainer>
  onCreateSources: (watchlistId: number, sources: WatchlistSourceCreate[]) => Promise<number[]>
  onCreateJob: (watchlistId: number, job: WatchlistJobCreate) => Promise<WatchlistJob>
  onComplete: (result: WatchlistSetupCompleteResult) => void
}
```

Expected: tests can exercise the component without mocking the full page.

- [ ] **Step 4: Run component tests**

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Watchlists/SetupWizard/__tests__/WatchlistSetupWizard.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: component tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/Watchlists/SetupWizard
git commit -m "feat: add watchlist setup wizard"
```

## Task 3: Shell Integration And Create Handoff

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx`
- Modify if needed: `apps/packages/ui/src/services/watchlists.ts`
- Modify if needed: `apps/packages/ui/src/types/watchlists.ts`

- [ ] **Step 1: Write failing shell integration test**

Update the existing create test so clicking the primary create control opens the setup wizard and finishing a topic-only Watchlist:

- calls `createWatchlist`
- adds the Watchlist to store
- selects the created Watchlist
- navigates to the appropriate destination tab

Expected failure: shell still opens the old create modal.

- [ ] **Step 2: Wire setup wizard into shell**

Use the existing create button as the entry point:

- `data-testid="watchlists-create-container"` remains stable.
- Create mode opens `WatchlistSetupWizard`.
- Edit mode keeps the existing metadata modal.
- On completion, call `addWatchlist`, `setSelectedWatchlistId`, and navigate based on wizard result.

- [ ] **Step 3: Implement service adapters**

Adapters should use existing functions:

- `createWatchlist`
- `createWatchlistSource`
- `bulkCreateSources` when multiple sources are entered.
- `createWatchlistJob`
- `triggerWatchlistRun` only if the wizard includes a run-now option in the final implementation.

Ensure every child payload includes `watchlist_id: created.id`.

- [ ] **Step 4: Run shell tests**

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx \
  src/components/Option/Watchlists/__tests__/watchlists-selected-scope-contract.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Expected: tests pass and scoped request expectations remain intact.

- [ ] **Step 5: Commit**

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx \
  apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx \
  apps/packages/ui/src/services/watchlists.ts \
  apps/packages/ui/src/types/watchlists.ts
git commit -m "feat: integrate watchlist setup wizard"
```

## Task 4: Overview Repositioning And Existing Quick Setup

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/watchlists.json`
- Modify if mirrored: `apps/packages/ui/src/public/_locales/en/watchlists.json`

- [ ] **Step 1: Write failing overview tests**

Assert:

- First-visit empty state does not auto-open source-first Quick Setup before the user has a selected Watchlist.
- Existing selected-Watchlist collection setup still creates source/job payloads with `watchlist_id`.
- User-facing copy frames it as adding collection to the selected Watchlist.

Expected failure: current copy and auto-open behavior are source-first.

- [ ] **Step 2: Update Overview entry points**

Reframe Quick Setup as "Add initial collection" when a Watchlist is selected.

Rules:

- If no Watchlist is selected, prompt user to create a Watchlist through the shell wizard.
- If exactly the default imported Watchlist is selected, still allow initial collection setup.
- Keep pipeline builder scoped to the selected Watchlist.

- [ ] **Step 3: Preserve existing helper contracts**

Do not break:

- source payload trimming
- schedule presets
- extra source URL parsing
- briefing output prefs

- [ ] **Step 4: Run overview tests**

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts \
  src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx \
  apps/packages/ui/src/assets/locale/en/watchlists.json \
  apps/packages/ui/src/public/_locales/en/watchlists.json
git commit -m "fix: make quick setup watchlist-first"
```

## Task 5: Constrained Viewport Smoke And Closeout

**Files:**
- Modify: relevant Backlog task files through MCP/CLI.
- Modify docs only if behavior needs user-facing documentation.

- [ ] **Step 1: Run focused frontend suite**

Run from `apps/packages/ui`:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Watchlists/SetupWizard/__tests__/watchlist-setup-model.test.ts \
  src/components/Option/Watchlists/SetupWizard/__tests__/WatchlistSetupWizard.test.tsx \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx \
  src/components/Option/Watchlists/__tests__/watchlists-selected-scope-contract.test.ts \
  src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts \
  src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: Stage 2 focused frontend suite passes.

- [ ] **Step 2: Run TypeScript check for touched package scope if practical**

Prefer the repo's existing package command. If package-wide TypeScript has unrelated baseline failures, record exact failure summary and keep focused Vitest as the gate for this slice.

- [ ] **Step 3: Run browser/CDP smoke**

Start the WebUI in the current worktree. Use CDP/Playwright, not Computer Use, to verify:

- Desktop `/watchlists` opens setup wizard from the create control.
- Extension-sized viewport `390x844` can select CTI/news preset, enter objective/scope, choose start mode, and reach review without clipped primary controls.
- Topic-only completion creates/selects a Watchlist and routes to Feeds.
- Source-backed completion sends `watchlist_id` on source and job requests.

Expected: page renders nonblank, setup wizard controls are reachable, no horizontal overflow of critical controls.

- [ ] **Step 4: Run diff hygiene**

```bash
git diff --check
```

Expected: clean.

- [ ] **Step 5: Final Backlog updates**

Record:

- Focused tests and results.
- TypeScript result or documented baseline failure.
- CDP smoke viewport and evidence.
- Any deferred Stage 3/5 boundaries.

- [ ] **Step 6: Commit closeout**

```bash
git add backlog/tasks/<stage-2-task-files> Docs/API-related/Watchlists_API.md Docs/Published/API-related/Watchlists_API.md
git commit -m "chore: close watchlists setup wizard stage"
```

## Rollout Gates

- A user can create a Watchlist as a project-like container before configuring sources.
- CTI/OSINT and news presets are visible, concrete, and not hidden in generic copy.
- Topic-only flow is honest that monitoring starts after sources are added.
- Source-backed flow creates source and monitor records scoped to the new Watchlist.
- Report-goal flow does not promise defensible artifacts before Stage 5.
- Existing selected-Watchlist quick setup remains functional for adding collection later.
- Extension-sized viewport supports the create/setup flow end to end.

## Known Risks And Follow-Up Questions

- The current page has both a shell create modal and Overview Quick Setup. Stage 2 must reduce this duplication rather than add a third competing path.
- Topic-only setup may disappoint users expecting automatic source discovery. Copy should explicitly say sources are required before automated collection.
- CTI/news presets are copy and defaults only. Domain-specific extraction, classifications, and content-match alerts remain Stage 3+.
- The current Quick Setup auto-open behavior can conflict with Watchlist-first onboarding. Tests should lock down the new behavior.
- Full extension management remains broader than this wizard. Stage 6 must still address dense child tabs and table-heavy flows.

## Full Verification Command Set

Run before declaring Stage 2 complete:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/components/Option/Watchlists/SetupWizard/__tests__/watchlist-setup-model.test.ts \
  src/components/Option/Watchlists/SetupWizard/__tests__/WatchlistSetupWizard.test.tsx \
  src/components/Option/Watchlists/__tests__/watchlists-stage2-copy-contract.test.ts \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx \
  src/components/Option/Watchlists/__tests__/watchlists-selected-scope-contract.test.ts \
  src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts \
  src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx \
  --maxWorkers=1 --no-file-parallelism

cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/watchlists-stage1a
git diff --check
```

Expected: focused tests pass, CDP smoke is recorded, and diff hygiene is clean.

## Execution Handoff

Recommended execution mode: inline execution with superpowers:executing-plans unless the user explicitly authorizes subagents. Start with Task 0, then Task 1 only. Do not wire the shell before helper/model tests are green.
