# Research Studio UX Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Research Studio the canonical user-facing workspace route and improve the Studio experience through degraded-health pass-through, route aliases, mobile deep links, work-product-first IA, no-source guidance, returning-user efficiency, and focused release verification.

**Architecture:** Keep the existing `WorkspacePlayground` implementation and shared UI package as the behavioral core. Add canonical route aliases and user-facing naming on top, while preserving persisted/internal `workspace-playground` identifiers. Use shared route constants and small route-state helpers instead of scattering path strings or query parsing across components.

**Tech Stack:** Next.js pages in `apps/tldw-frontend`, extension/shared React Router registries, React components in `apps/packages/ui`, Vitest/Testing Library, Playwright/CDP smoke checks, Backlog.md task tracking.

---

## Plan Notes

- Parent design spec: `Docs/superpowers/specs/2026-05-12-research-studio-ux-remediation-design.md`.
- Parent Backlog task: `TASK-304`.
- Before editing product code for any stage, create or claim a child Backlog.md task for that stage and keep it updated.
- Use CDP/Playwright/browser-observed checks for UI verification. Do not use Computer Use for this work.
- Preserve internal storage, export, telemetry, and event identifiers unless a later explicit migration plan approves renaming them.
- Do not edit generated docs under `Docs/site` unless that stage explicitly includes running the docs generation pipeline.
- Do not broaden into a general frontend cleanup, backend architecture review, or unrelated routes.
- Prefer one reviewable commit per stage. Keep route, IA, and docs changes easy to inspect.

## Compatibility Boundaries

The user-facing product name and route are changing, but these internal names remain compatibility contracts for this series:

- `WorkspacePlayground` component/module names.
- `workspace-playground-basics` tutorial IDs.
- `tldw.workspace-playground.bundle`.
- `tldw:workspace:playground:telemetry`.
- `tldw:workspace-playground:*` local storage keys.
- `workspace-playground:*` event names, including artifact discussion events.
- Diagnostic values such as `from=workspace-playground`, unless the value is visibly shown to end users.

## File Map

Likely files to modify or test across the series:

- `apps/tldw-frontend/components/networking/ServerReadinessGate.tsx`
- `apps/tldw-frontend/components/networking/__tests__/ServerReadinessGate.test.tsx`
- `apps/tldw-frontend/components/navigation/RouteRedirect.tsx`
- `apps/tldw-frontend/components/navigation/__tests__/RouteRedirect*` if present
- `apps/tldw-frontend/pages/research-studio.tsx`
- `apps/tldw-frontend/pages/workspace-playground.tsx`
- `apps/tldw-frontend/pages/workspace-studio.tsx`
- `apps/tldw-frontend/__tests__/navigation/route-redirect.test.ts`
- `apps/tldw-frontend/__tests__/navigation/route-redirect-component.test.tsx`
- `apps/tldw-frontend/__tests__/extension/route-registry.workspace-playground.test.ts`
- `apps/tldw-frontend/e2e/utils/page-objects/WorkspacePlaygroundPage.ts`
- `apps/tldw-frontend/e2e/page-mapping.ts`
- `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`
- `apps/tldw-frontend/e2e/smoke/stage5-release-gate.spec.ts`
- `apps/tldw-frontend/extension/routes/route-registry.tsx`
- `apps/packages/ui/src/routes/route-paths.ts`
- `apps/packages/ui/src/routes/route-registry.tsx`
- `apps/packages/ui/src/routes/__tests__/route-paths.lorebook-debug.test.ts`
- `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- `apps/packages/ui/src/components/Common/QuickChatHelper/workflow-guides.ts`
- `apps/packages/ui/src/components/Common/QuickChatHelper/docs-rag-profile.ts`
- `apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/workflow-guides.test.ts`
- `apps/packages/ui/src/components/Option/KnowledgeQA/AnswerPanel.tsx`
- `apps/packages/ui/src/components/Option/KnowledgeQA/empty/KnowledgeReadyState.tsx`
- `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.workspace-handoff.test.tsx`
- `apps/packages/ui/src/components/Option/SharedWithMe/index.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage3.test.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/WorkProductTemplateChooser.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkProductTemplateChooser.test.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/source-location-copy.ts`
- `apps/packages/ui/src/tutorials/definitions/workspace-playground.ts`
- `apps/packages/ui/src/tutorials/__tests__/registry.test.ts`
- `Docs/User_Guides/WebUI_Extension/Quick_Chat_Docs_Assistant_Guide.md`
- `Docs/Code_Documentation/Tutorial_System_Developer_Guide.md`
- `Docs/Operations/Research_Studio_Trust_Status_Telemetry_Runbook.md`

## Task 0: Stage Tracking And Baseline

**Goal:** Prepare implementation work orders and preserve baseline evidence before product behavior changes.

**Files:**

- Backlog.md task records only.
- Optional implementation notes under the stage task records.

**Steps:**

- [ ] Create child Backlog tasks for each implementation stage, or split each stage further if the file set becomes too broad.
- [ ] Attach the design spec path and relevant screenshot paths from `/private/tmp/workspace-studio-cdp-audit/` to the task notes.
- [ ] Record the current baseline for these user-visible behaviors:
  - `/workspace-studio` 404.
  - `/workspace-playground` renders Research Studio.
  - mobile direct Studio activation works only through internal tab activation, not canonical URL state.
  - degraded HTTP 206 health can delay entry behind `ServerReadinessGate`.
  - no-source Studio shows disabled generation controls before the source hint.
- [ ] Confirm no product code changes are included in the tracking-only commit.

**Verification:**

- [ ] `backlog task TASK-304 --plain` or MCP equivalent shows linked child tasks.
- [ ] `git diff --check` passes for task/spec changes.

**Commit boundary:** Tracking/task setup only.

## Task 1: Degraded Health Pass-Through

**Goal:** Let reachable degraded backend health enter the app, while preserving the blocking path for unreachable, malformed, or explicitly unhealthy states.

**Why first:** Browser-visible route alias verification can be hidden by the readiness screen when local health returns HTTP 206. This should land before alias smoke checks.

**Files:**

- `apps/tldw-frontend/components/networking/ServerReadinessGate.tsx`
- `apps/tldw-frontend/components/networking/__tests__/ServerReadinessGate.test.tsx`
- Optional: existing status components only if evidence shows no degraded state is visible after entry:
  - `apps/packages/ui/src/components/Option/WorkspacePlayground/WorkspaceStatusBar.tsx`
  - `apps/packages/ui/src/components/Option/WorkspacePlayground/WorkspaceHeader.tsx`

**Behavior:**

- [ ] Treat HTTP `200` and HTTP `206` as parseable health responses.
- [ ] Treat body statuses `ok`, `healthy`, and `degraded` as app-enterable.
- [ ] Keep non-parseable responses, network failures, and explicit unhealthy statuses in the existing wait/retry/block flow.
- [ ] Do not claim chat or generation is safe from this gate alone. This task changes app entry only.
- [ ] Rely on existing connection/status UI for visible degraded warnings when it already renders clearly. Add a narrow warning only if CDP evidence shows users enter the page with no degraded affordance.

**Suggested implementation shape:**

- [ ] Extract a tiny readiness classifier inside `ServerReadinessGate.tsx` or next to it:

```ts
type ReadinessState = "enterable" | "blocked";

function classifyReadiness(responseStatus: number, bodyStatus: unknown): ReadinessState {
  // 200/206 plus ok/healthy/degraded enters; explicit bad or malformed blocks.
}
```

- [ ] Keep timeout/retry behavior unchanged for blocked states.
- [ ] Avoid broad changes to API health fetching outside the gate.

**Tests first:**

- [ ] Add a failing test that HTTP 206 with `{ status: "degraded" }` enters the app.
- [ ] Add a failing test that HTTP 200 with `{ status: "degraded" }` enters the app.
- [ ] Add or preserve tests showing `{ status: "unhealthy" }`, malformed JSON, and network failures do not enter immediately.
- [ ] Preserve the existing bypass behavior for setup/settings routes.

**Verification commands:**

```bash
cd apps/tldw-frontend
bun run test:run components/networking/__tests__/ServerReadinessGate.test.tsx
```

**Expected outcome:**

- The focused readiness tests pass.
- The gate no longer waits 15 seconds for reachable degraded health.

**Commit boundary:** `ServerReadinessGate` behavior and its tests only.

## Task 2: Canonical Route And Aliases

**Goal:** Make `/research-studio` canonical while preserving `/workspace-playground` and `/workspace-studio` as compatibility aliases with query/hash preservation.

**Files:**

- `apps/tldw-frontend/pages/research-studio.tsx`
- `apps/tldw-frontend/pages/workspace-playground.tsx`
- `apps/tldw-frontend/pages/workspace-studio.tsx`
- `apps/tldw-frontend/components/navigation/RouteRedirect.tsx`
- `apps/tldw-frontend/__tests__/navigation/route-redirect.test.ts`
- `apps/tldw-frontend/__tests__/navigation/route-redirect-component.test.tsx`
- `apps/tldw-frontend/extension/routes/route-registry.tsx`
- `apps/tldw-frontend/__tests__/extension/route-registry.workspace-playground.test.ts`
- `apps/packages/ui/src/routes/route-paths.ts`
- `apps/packages/ui/src/routes/route-registry.tsx`
- New optional helper: `apps/packages/ui/src/routes/RouteAliasNavigate.tsx`
- New optional test: `apps/packages/ui/src/routes/__tests__/route-alias-navigate.test.tsx`

**Behavior:**

- [ ] `/research-studio` renders the existing Research Studio surface.
- [ ] `/workspace-playground` redirects or aliases to `/research-studio`.
- [ ] `/workspace-studio` redirects or aliases to `/research-studio`.
- [ ] `?tab`, `?shared`, prefill/source-transfer params, and hashes survive aliasing.
- [ ] The extension/shared route registries expose the canonical route and retain legacy aliases.
- [ ] Internal component/module names can remain `WorkspacePlayground`.

**Suggested implementation shape:**

- [ ] Move the dynamic import currently in `pages/workspace-playground.tsx` to `pages/research-studio.tsx`.
- [ ] Replace `pages/workspace-playground.tsx` with a `RouteRedirect` to `/research-studio`.
- [ ] Add `pages/workspace-studio.tsx` as the same redirect.
- [ ] Add `RESEARCH_STUDIO_PATH = "/research-studio"` in `route-paths.ts`.
- [ ] Keep `WORKSPACE_PLAYGROUND_PATH = "/workspace-playground"` as a legacy constant.
- [ ] Add the canonical path to `VIEWPORT_CONSTRAINED_PATHS` and keep the legacy path during migration.
- [ ] For React Router aliases, avoid raw `<Navigate to="/research-studio" />` if it drops state. Use a helper that reads `useLocation()` and preserves `search` and `hash`:

```tsx
function RouteAliasNavigate({ to }: { to: string }) {
  const location = useLocation();
  return <Navigate to={{ pathname: to, search: location.search, hash: location.hash }} replace />;
}
```

**Tests first:**

- [ ] Add route redirect tests for `/workspace-studio?tab=studio` preserving `?tab=studio`.
- [ ] Add route redirect tests for `/workspace-playground?shared=abc` preserving `?shared=abc`.
- [ ] Add registry tests that canonical `/research-studio` exists.
- [ ] Add registry tests that legacy aliases exist and preserve route state.

**Verification commands:**

```bash
cd apps/tldw-frontend
bun run test:run __tests__/navigation/route-redirect.test.ts __tests__/navigation/route-redirect-component.test.tsx __tests__/extension/route-registry.workspace-playground.test.ts

cd ../../apps/packages/ui
bunx vitest run src/routes/__tests__/route-paths.lorebook-debug.test.ts src/routes/__tests__/route-alias-navigate.test.tsx
```

**CDP smoke checks after tests:**

- [ ] Open `http://127.0.0.1:3000/research-studio`.
- [ ] Open `http://127.0.0.1:3000/workspace-playground?shared=alias-test`.
- [ ] Open `http://127.0.0.1:3000/workspace-studio?tab=studio`.
- [ ] Confirm aliases do not 404 and query state is still present after navigation.

**Commit boundary:** Routes, aliases, route constants, route tests.

## Task 3: Mobile `?tab=` Route-State Contract

**Goal:** Make `?tab=sources|chat|studio` the canonical mobile deep-link mechanism, with `?tab=studio` opening Studio on mobile.

**Files:**

- `apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx`
- New helper: `apps/packages/ui/src/components/Option/WorkspacePlayground/research-studio-route-state.ts`
- New tests: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/research-studio-route-state.test.ts`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage3.test.tsx`

**Behavior:**

- [ ] Allowed tab values: `sources`, `chat`, `studio`.
- [ ] Missing or invalid tab values fall back to Chat.
- [ ] URL state wins over persisted last tab or component default.
- [ ] On mobile, initial `activeTab` comes from `?tab=...`.
- [ ] On desktop, the route state can focus a pane, but desktop panes remain visible/collapsible.
- [ ] Alias redirects preserve the query string, so `/workspace-studio?tab=studio` reaches canonical Studio state.
- [ ] Avoid browser history churn. The initial version may read URL state without writing tab clicks back into the URL.

**Suggested implementation shape:**

- [ ] Implement a small pure helper:

```ts
export type ResearchStudioTab = "sources" | "chat" | "studio";

export function parseResearchStudioTab(value: unknown): ResearchStudioTab | null {
  return value === "sources" || value === "chat" || value === "studio" ? value : null;
}
```

- [ ] Add a helper that reads `URLSearchParams` without dropping unrelated params.
- [ ] Initialize `activeTab` from the parsed tab when available.
- [ ] Reuse existing `focusWorkspacePane("studio")` for desktop focus behavior where practical.
- [ ] Do not introduce route parsing in several component effects.

**Tests first:**

- [ ] Unit tests for valid tabs, invalid tabs, empty values, and duplicate params if supported.
- [ ] Responsive component test that `/research-studio?tab=studio` starts on Studio in mobile layout.
- [ ] Responsive component test that invalid `tab` starts on Chat.
- [ ] Test that `?shared=abc&tab=studio` still lets the shared state be observed by existing callers.

**Verification commands:**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/WorkspacePlayground/__tests__/research-studio-route-state.test.ts \
  src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx \
  src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage3.test.tsx
```

**CDP smoke checks after tests:**

- [ ] Mobile viewport: `http://127.0.0.1:3000/research-studio?tab=studio` opens Studio.
- [ ] Mobile viewport: `http://127.0.0.1:3000/research-studio?tab=sources` opens Sources.
- [ ] Mobile viewport: invalid `?tab=banana` opens Chat.

**Commit boundary:** Route-state helper and mobile tab behavior.

## Task 4: User-Facing Naming And Handoff Sweep

**Goal:** Standardize visible copy and handoff links on "Research Studio" and `/research-studio`, while preserving internal identifiers.

**Files:**

- `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- `apps/packages/ui/src/components/Common/QuickChatHelper/workflow-guides.ts`
- `apps/packages/ui/src/components/Common/QuickChatHelper/docs-rag-profile.ts`
- `apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/workflow-guides.test.ts`
- `apps/packages/ui/src/components/Option/KnowledgeQA/AnswerPanel.tsx`
- `apps/packages/ui/src/components/Option/KnowledgeQA/empty/KnowledgeReadyState.tsx`
- `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.workspace-handoff.test.tsx`
- `apps/packages/ui/src/components/Option/SharedWithMe/index.tsx`
- `apps/packages/ui/src/tutorials/definitions/workspace-playground.ts`
- `apps/packages/ui/src/tutorials/__tests__/registry.test.ts`
- `apps/tldw-frontend/e2e/utils/page-objects/WorkspacePlaygroundPage.ts`
- `apps/tldw-frontend/e2e/page-mapping.ts`
- `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`
- `apps/tldw-frontend/e2e/smoke/stage5-release-gate.spec.ts`
- `Docs/User_Guides/WebUI_Extension/Quick_Chat_Docs_Assistant_Guide.md`
- `Docs/Code_Documentation/Tutorial_System_Developer_Guide.md`
- `Docs/Operations/Research_Studio_Trust_Status_Telemetry_Runbook.md`

**Behavior:**

- [ ] Navigation, helper copy, docs, tutorials, and handoff surfaces say "Research Studio".
- [ ] User-facing links target `/research-studio`.
- [ ] Knowledge QA, Shared With Me, Quick Chat docs, header shortcuts, tutorials, extension routes, sidepanel/popout state, and source-transfer/prefill utilities are inventoried before editing.
- [ ] Internal IDs and storage/export/telemetry names remain stable and are documented as compatibility names where visible in developer docs.
- [ ] Source docs are updated. Generated docs are left alone unless this task explicitly runs the docs build.

**Suggested implementation shape:**

- [ ] Replace user-facing hard-coded paths with `RESEARCH_STUDIO_PATH` where the UI package can import it without creating cycles.
- [ ] Keep legacy constants available for tests and compatibility routes.
- [ ] Update Quick Chat route label expectations from "Workspace Playground" to "Research Studio".
- [ ] Update Knowledge QA and Shared With Me handoffs to canonical paths.
- [ ] Update E2E page objects to navigate to `/research-studio`, while keeping alias coverage in route tests.
- [ ] In developer docs that mention `WorkspacePlayground`, clarify code-name vs product-name rather than rewriting every implementation reference.

**Tests first:**

- [ ] Update/extend tests that assert Quick Chat route labels.
- [ ] Update Knowledge QA handoff tests.
- [ ] Update tutorial registry tests so `/research-studio` resolves, while legacy tutorial IDs remain stable.
- [ ] Add static route inventory expectations for canonical path where existing smoke tests use route lists.

**Verification commands:**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Common/QuickChatHelper/__tests__/workflow-guides.test.ts \
  src/components/Option/KnowledgeQA/__tests__/AnswerPanel.workspace-handoff.test.tsx \
  src/tutorials/__tests__/registry.test.ts

cd ../../tldw-frontend
bun run test:run __tests__/extension/route-registry.workspace-playground.test.ts
```

**Search checks:**

```bash
rg -n "Workspace Playground|workspace-studio|/workspace-playground" apps Docs --glob '!Docs/site/**'
```

**Expected outcome:**

- Search results are either intended compatibility/internal references or remaining implementation names that should not be user-visible.

**Commit boundary:** User-facing names, handoff paths, route inventory/tests, source docs.

## Task 5: Work-Product-First Studio IA

**Goal:** Make actionable work products the primary Studio path and hide planned products until usable.

**Files:**

- `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/WorkProductTemplateChooser.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkProductTemplateChooser.test.tsx`
- Existing or new `StudioPane` tests if a local test file already covers output selection.

**Behavior:**

- [ ] Planned work products are not visible to end users.
- [ ] Actionable work products lead the Studio panel.
- [ ] If only Executive Brief is actionable, it appears as the primary work product without "planned" peers.
- [ ] Raw output types remain available as secondary actions after the work-product section.
- [ ] Recent/frequent outputs may remain visible for power users, but they do not override the work-product-first hierarchy.

**Suggested implementation shape:**

- [ ] Introduce an explicit template visibility helper:

```ts
function isActionableWorkProductTemplate(template: WorkProductTemplate): boolean {
  return template.id === "executive_brief";
}
```

- [ ] Filter templates before rendering instead of rendering disabled `Planned` cards.
- [ ] Remove or hide the visible "Planned" badge in the end-user chooser path.
- [ ] Keep template metadata available internally for future actionable templates.
- [ ] In `StudioPane`, label the secondary output-type section as a secondary path, such as "Other outputs", using existing visual hierarchy.

**Tests first:**

- [ ] Replace the current test that expects planned templates to remain visible with a test that planned templates are hidden.
- [ ] Add a test that Executive Brief remains visible and selectable.
- [ ] Add a test that secondary output types remain reachable when sources are selected.
- [ ] Add a test that the rendered chooser contains no "Planned" text in the default end-user state.

**Verification commands:**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/WorkspacePlayground/__tests__/WorkProductTemplateChooser.test.tsx
```

**CDP smoke checks after tests:**

- [ ] Desktop `/research-studio?tab=studio` default Studio shows no visible "Planned" products.
- [ ] Work-product section appears before raw output-type controls.

**Commit boundary:** Template visibility and Studio IA tests.

## Task 6: No-Source Progressive Disclosure

**Goal:** Explain the source requirement before showing unavailable generation actions.

**Files:**

- `apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx`
- Optional new component: `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/StudioSourceReadiness.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/source-location-copy.ts`
- Existing or new `StudioPane` tests.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage3.test.tsx`

**Behavior:**

- [ ] When zero sources are selected, Studio begins with source readiness guidance.
- [ ] The no-source state includes selected source count and a clear source-selection CTA.
- [ ] The CTA moves users to the Sources tab on mobile and focuses the Sources pane on desktop.
- [ ] Disabled generation controls are hidden or visually subordinate until source readiness is met.
- [ ] "More outputs" does not expand into a wall of disabled actions when no sources are selected.
- [ ] Slides settings appear only when Slides is relevant.
- [ ] Audio Settings appear only when Audio Summary is relevant.

**Suggested implementation shape:**

- [ ] Add an optional `onRequestSources?: () => void` prop to `StudioPane`.
- [ ] Pass `() => focusWorkspacePane("sources")` from `WorkspacePlayground` into `StudioPane`.
- [ ] In the mobile branch, ensure the same callback activates the Sources tab.
- [ ] Extract a small `StudioSourceReadiness` component if it keeps `StudioPane` readable.
- [ ] Keep the source gate that prevents generation calls without selected sources.
- [ ] Use a button for the CTA with an accessible label, not a tooltip-only explanation.

**Tests first:**

- [ ] Add a no-source test showing the readiness card before output buttons.
- [ ] Add a test that the source CTA calls `onRequestSources`.
- [ ] Add a mobile responsive test that the CTA switches to the Sources tab.
- [ ] Add a test that Slides and Audio settings are not visible before source/output intent.
- [ ] Add a test that generation actions reappear when `selectedMediaIds.length > 0`.

**Verification commands:**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx \
  src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage3.test.tsx
```

**CDP smoke checks after tests:**

- [ ] Desktop no-source Studio shows source readiness first.
- [ ] Mobile no-source Studio CTA navigates to Sources.
- [ ] No-source "More outputs" no longer expands into disabled-only controls.

**Commit boundary:** No-source state, CTA wiring, progressive disclosure tests.

## Task 7: Returning-User Efficiency

**Goal:** Reduce repeated setup friction without hiding critical route, source, or health state.

**Files:**

- `apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx`
- Optional helper: `apps/packages/ui/src/components/Option/WorkspacePlayground/research-studio-storage.ts`
- Existing onboarding/storage helpers near `WorkspacePlayground`.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx`
- New storage helper tests if a helper is added.

**Behavior:**

- [ ] Persist last active mobile tab only when no URL `?tab=` is present.
- [ ] URL tab state always wins over persisted state.
- [ ] Onboarding dismissal remains persistent.
- [ ] Advanced expanded/collapsed state is persisted only for low-risk sections, not warnings or required source readiness.
- [ ] Compact mobile header/status behavior after onboarding dismissal only if it does not hide degraded health or source-readiness feedback.

**Suggested implementation shape:**

- [ ] Use an existing safe storage helper if one exists locally; otherwise add a tiny guarded helper for this feature.
- [ ] Use a versioned key such as `tldw:research-studio:last-mobile-tab:v1` only if new storage is needed.
- [ ] Treat storage read/write failures as no-ops.
- [ ] Keep critical warnings outside persisted collapsed sections.
- [ ] Add a small hook only if it reduces complexity; avoid adding a global state store for this stage.

**Tests first:**

- [ ] Unit test storage read fallback on exceptions.
- [ ] Test URL `?tab=studio` wins over stored `sources`.
- [ ] Test stored `studio` applies when URL has no tab.
- [ ] Test invalid stored values fall back to Chat.
- [ ] Responsive test for compacted mobile chrome after onboarding dismissal, if implemented.

**Verification commands:**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx \
  src/components/Option/WorkspacePlayground/__tests__/research-studio-route-state.test.ts
```

**CDP smoke checks after tests:**

- [ ] Mobile Research Studio reopens to the expected persisted tab when no URL tab exists.
- [ ] `?tab=chat` still overrides persisted Studio state.
- [ ] Degraded status remains visible after onboarding dismissal.

**Commit boundary:** Returning-user persistence/compactness and focused tests.

## Task 8: Capability-Aware Health Follow-Up

**Goal:** Define and implement capability-level degraded behavior only after the backend health payload semantics are understood.

**Files:**

- Start with investigation notes in a Backlog task or docs note.
- Likely frontend files after the contract is known:
  - `apps/packages/ui/src/components/Option/WorkspacePlayground/WorkspaceStatusBar.tsx`
  - `apps/packages/ui/src/components/Option/WorkspacePlayground/WorkspaceHeader.tsx`
  - `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx`
  - New helper if needed: `apps/packages/ui/src/components/Option/WorkspacePlayground/research-studio-capabilities.ts`

**Behavior to decide:**

- [ ] Which health states are safe for browsing existing sources?
- [ ] Which health states are safe for local workspace management?
- [ ] Which health states are safe for chat?
- [ ] Which health states are safe for artifact generation?
- [ ] Which health states are safe for export/download?
- [ ] Which health states are safe for sync/share?

**Implementation guardrails:**

- [ ] Do not block the whole app when only one capability is degraded.
- [ ] Disable or warn at the action boundary.
- [ ] Keep generation and expensive operations conservative when required dependencies are known unavailable.
- [ ] Do not invent backend semantics in frontend code. If the payload cannot prove a capability is unavailable, keep messaging scoped to "server degraded" and rely on request-level failures.

**Tests first after contract is known:**

- [ ] Unit tests for capability derivation from health payload examples.
- [ ] UI tests for browse allowed/generation warned or disabled.
- [ ] Error recovery tests for failed generation request under degraded state.

**Verification commands:**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/WorkspacePlayground/__tests__/*.test.tsx
```

**Commit boundary:** Capability contract docs first, then capability UI if contract is sufficient.

## Task 9: Documentation, Accessibility, And Release Verification

**Goal:** Close the series with focused docs, accessibility, CDP evidence, and route parity checks.

**Files:**

- `Docs/User_Guides/WebUI_Extension/Quick_Chat_Docs_Assistant_Guide.md`
- `Docs/Code_Documentation/Tutorial_System_Developer_Guide.md`
- `Docs/Operations/Research_Studio_Trust_Status_Telemetry_Runbook.md`
- `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`
- `apps/tldw-frontend/e2e/smoke/stage5-release-gate.spec.ts`
- Any release notes or Backlog final summaries for the stage tasks.

**Behavior:**

- [ ] Docs use `/research-studio` as canonical.
- [ ] Docs mention `/workspace-playground` and `/workspace-studio` only as legacy aliases where useful.
- [ ] Docs use `/research-studio?tab=studio` for mobile Studio deep links.
- [ ] Keyboard focus reaches source CTA, work products, secondary outputs, generated artifacts, quick notes, and status controls in a coherent order.
- [ ] Hit targets are practical on mobile.
- [ ] Skip links still land correctly after route and tab changes.

**Verification commands:**

```bash
cd apps/tldw-frontend
bun run test:run

cd ../packages/ui
bun run test

cd ../../tldw-frontend
bunx playwright test e2e/workflows/workspace-playground.parity.spec.ts --reporter=line --workers=1
```

Run broader suites only if the stage changes shared infrastructure enough to justify the cost. If the local backend is unavailable, record the exact blocker and run route/component tests plus CDP with whatever surfaces are reachable.

**CDP smoke checklist:**

- [ ] Desktop `http://127.0.0.1:3000/research-studio`.
- [ ] Desktop alias `http://127.0.0.1:3000/workspace-playground?shared=alias-test`.
- [ ] Desktop alias `http://127.0.0.1:3000/workspace-studio?tab=studio`.
- [ ] Mobile `http://127.0.0.1:3000/research-studio?tab=studio`.
- [ ] Mobile no-source source CTA.
- [ ] Degraded-health visible status, if local health can be mocked or observed as degraded.

**Search checks:**

```bash
rg -n "Workspace Playground|Workspace Studio|/workspace-playground|/workspace-studio" apps Docs --glob '!Docs/site/**'
```

Classify every remaining result as one of:

- [ ] intentional implementation/internal reference;
- [ ] intentional legacy alias documentation;
- [ ] remaining user-facing reference to fix before release.

**Bandit/security note:**

- [ ] No backend Python changes are expected in this series. If only frontend/docs are touched, record "Bandit not run: frontend/docs-only changes."
- [ ] If any backend Python path is touched, run Bandit on the touched backend scope using the project virtualenv.

**Commit boundary:** docs, final route inventory, release verification notes.

## Suggested Commit Sequence

1. Tracking child tasks and baseline notes.
2. Degraded readiness gate.
3. Canonical route and aliases.
4. Mobile `?tab=` contract.
5. Naming and handoff sweep.
6. Work-product-first Studio IA.
7. No-source progressive disclosure.
8. Returning-user efficiency.
9. Capability-aware health follow-up if backend contract is ready.
10. Docs/accessibility/CDP release verification.

## Final Acceptance Checklist

- [ ] `/research-studio` is canonical in WebUI and extension routing.
- [ ] `/workspace-playground` and `/workspace-studio` do not 404 and preserve query/hash state.
- [ ] `?tab=studio` opens Studio on mobile.
- [ ] User-facing labels and docs say "Research Studio."
- [ ] Planned work products are hidden until actionable.
- [ ] Studio defaults to work-product-first selection.
- [ ] No-source Studio explains source requirements before unavailable generation actions.
- [ ] Advanced output settings are shown only when relevant.
- [ ] Degraded-but-reachable health enters the app with visible warning affordance.
- [ ] Capability-aware health is either implemented from a real backend contract or tracked as a follow-up.
- [ ] Returning-user tab/onboarding behavior reduces repeated setup without hiding warnings.
- [ ] Focused unit/component tests pass.
- [ ] CDP desktop and mobile checks have current screenshots or recorded observations.
- [ ] Remaining legacy string/path search hits are classified.
- [ ] Backlog child tasks include final summaries and verification notes.
