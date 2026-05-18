# WebUI Extension UX Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the approved WebUI/extension UX remediation program design into reviewable implementation slices that cover every audited finding and root route without collapsing the work into one broad redesign.

**Architecture:** Treat route metadata, capability/error states, responsive landmarks, and QA governance as foundations. Then remediate route families in small PR-sized slices that preserve existing product intent and power-user density. Each child slice must list the finding IDs and route rows it closes before editing product code.

**Tech Stack:** Next.js pages in `apps/tldw-frontend`, shared React/TypeScript UI in `apps/packages/ui`, extension route wrappers under `apps/tldw-frontend/extension` and `apps/extension`, Vitest, Testing Library, Playwright, existing shared UI state primitives, Backlog.md task tracking.

---

## Source Documents

- Source spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Source audit: `Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md`
- Planning Backlog task: `TASK-418`
- Design Backlog task: `TASK-417`

## Current Task Boundary

This file is an implementation planning artifact. It does not authorize product
code changes by itself. Future code work must create or update a Backlog task
for the slice being implemented before editing files.

Do not implement all tasks from this parent plan in one branch or PR. Start with
Task 1 foundations, then create child implementation plans for each route-family
slice as needed.

## Non-Negotiable Constraints

- No broad visual redesign or new design system.
- No route renames without explicit alias, redirect, compatibility, and smoke
  coverage.
- No backend API changes unless the child plan proves the frontend cannot
  truthfully represent the UX state with existing data.
- No explanation-only fixes for structural UX issues. Prefer clearer state,
  controls, route ownership, progressive disclosure, and recovery affordances.
- Preserve dense power-user controls when they are useful; organize them instead
  of removing them.
- Browser evidence is mandatory for changed visual routes.

## File Map

### Route Contract And Navigation Foundations

- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx`
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: `apps/packages/ui/src/components/Common/CommandPalette.tsx`
- Modify: `apps/packages/ui/src/components/Common/CommandPaletteHost.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- Modify: `apps/packages/ui/src/components/Layouts/ModeSelector.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/`
- Test: `apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx`
- Test: `apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts`

### Shared State, Shell, And Layout Foundations

- Modify: `apps/packages/ui/src/components/ui/state/StatePanel.tsx`
- Modify: `apps/packages/ui/src/components/ui/state/RecoveryCallout.tsx`
- Modify: `apps/packages/ui/src/components/ui/state/DiagnosticRow.tsx`
- Modify: `apps/packages/ui/src/components/ui/state/SetupRequiredPanel.tsx`
- Modify: `apps/packages/ui/src/components/Common/FeatureEmptyState.tsx`
- Modify: `apps/packages/ui/src/components/Common/RouteErrorBoundary.tsx`
- Modify: `apps/tldw-frontend/components/layout/WebLayout.tsx`
- Modify: `apps/tldw-frontend/components/layout/Header.tsx`
- Modify: `apps/tldw-frontend/pages/_app.tsx`
- Test: `apps/packages/ui/src/components/ui/state/__tests__/state-primitives.test.tsx`
- Test: `apps/packages/ui/src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx`
- Test: `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`

### Route-Family Surfaces

- Start/auth/account: `apps/packages/ui/src/routes/option-setup.tsx`, `apps/tldw-frontend/pages/index.tsx`, `apps/tldw-frontend/pages/setup.tsx`, `apps/tldw-frontend/pages/login.tsx`, `apps/tldw-frontend/pages/signup.tsx`, `apps/tldw-frontend/pages/account/index.tsx`, `apps/tldw-frontend/pages/profile.tsx`, `apps/tldw-frontend/pages/privileges.tsx`, `apps/tldw-frontend/pages/config.tsx`, `apps/tldw-frontend/pages/billing/index.tsx`, `apps/tldw-frontend/pages/404.tsx`
- Settings/model: `apps/packages/ui/src/components/Layouts/settings-nav.ts`, `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`, `apps/packages/ui/src/components/Option/Settings/general-settings.tsx`, `apps/packages/ui/src/components/Option/Settings/model-settings.tsx`, `apps/packages/ui/src/components/Option/Models`, `apps/tldw-frontend/pages/settings/index.tsx`, `apps/tldw-frontend/pages/settings/model.tsx`
- Chat/global chrome: `apps/packages/ui/src/routes/option-chat.tsx`, `apps/packages/ui/src/routes/option-quick-chat-popout.tsx`, `apps/packages/ui/src/components/Option/Playground`, `apps/packages/ui/src/components/Layouts/Header.tsx`, `apps/packages/ui/src/components/Layouts/HeaderShortcuts.tsx`
- Persona/context/agents: `apps/packages/ui/src/routes/option-characters.tsx`, `apps/packages/ui/src/routes/option-chat-workflows.tsx`, `apps/packages/ui/src/routes/option-companion.tsx`, `apps/packages/ui/src/routes/option-dictionaries.tsx`, `apps/packages/ui/src/routes/option-world-books.tsx`, `apps/packages/ui/src/routes/sidepanel-persona.tsx`, `apps/packages/ui/src/components/Agent`
- Media/library/sharing: `apps/packages/ui/src/routes/option-media.tsx`, `apps/packages/ui/src/routes/option-media-multi.tsx`, `apps/packages/ui/src/routes/option-media-trash.tsx`, `apps/packages/ui/src/routes/option-items.tsx`, `apps/packages/ui/src/routes/option-collections.tsx`, `apps/packages/ui/src/routes/option-notes.tsx`, `apps/packages/ui/src/routes/option-shared-with-me.tsx`, `apps/packages/ui/src/routes/option-chatbooks-playground.tsx`, `apps/tldw-frontend/pages/review.tsx`, `apps/tldw-frontend/pages/reading.tsx`, `apps/tldw-frontend/pages/chatbooks.tsx`, `apps/tldw-frontend/pages/notifications.tsx`, `apps/packages/ui/src/components/Review`
- Knowledge/workspace/transform: `apps/packages/ui/src/routes/option-knowledge.tsx`, `apps/packages/ui/src/routes/option-workspace-playground.tsx`, `apps/packages/ui/src/routes/option-chat-workspace.tsx`, `apps/packages/ui/src/routes/option-document-workspace.tsx`, `apps/packages/ui/src/routes/option-repo2txt.tsx`, `apps/packages/ui/src/routes/option-model-playground.tsx`, `apps/packages/ui/src/routes/option-writing-playground.tsx`, `apps/packages/ui/src/routes/option-presentation-studio.tsx`, `apps/tldw-frontend/pages/search.tsx`, `apps/tldw-frontend/pages/research.tsx`
- Operations/integrations: `apps/packages/ui/src/routes/option-sources.tsx`, `apps/packages/ui/src/routes/option-integrations.tsx`, `apps/packages/ui/src/routes/option-scheduled-tasks.tsx`, `apps/packages/ui/src/routes/option-watchlists.tsx`, `apps/packages/ui/src/routes/option-workflow-editor.tsx`, `apps/packages/ui/src/routes/option-mcp-hub.tsx`, `apps/packages/ui/src/routes/option-skills.tsx`, `apps/packages/ui/src/routes/option-admin-server.tsx`, `apps/packages/ui/src/routes/option-admin-integrations.tsx`, `apps/tldw-frontend/pages/connectors/index.tsx`, `apps/tldw-frontend/pages/connectors/browse.tsx`, `apps/tldw-frontend/pages/connectors/jobs.tsx`, `apps/tldw-frontend/pages/connectors/sources.tsx`
- Audio/study/safety/specialized: `apps/packages/ui/src/routes/option-speech.tsx`, `apps/packages/ui/src/routes/option-stt.tsx`, `apps/packages/ui/src/routes/option-tts.tsx`, `apps/packages/ui/src/routes/option-audiobook-studio.tsx`, `apps/packages/ui/src/routes/option-evaluations.tsx`, `apps/packages/ui/src/routes/option-flashcards.tsx`, `apps/packages/ui/src/routes/option-quiz.tsx`, `apps/packages/ui/src/routes/option-moderation-playground.tsx`, `apps/packages/ui/src/routes/option-content-review.tsx`, `apps/tldw-frontend/pages/claims-review.tsx`, `apps/packages/ui/src/routes/option-data-tables.tsx`, `apps/packages/ui/src/routes/option-chunking-playground.tsx`, `apps/packages/ui/src/routes/option-kanban-playground.tsx`, `apps/tldw-frontend/pages/vn-assets.tsx`, `apps/tldw-frontend/pages/vn-play.tsx`

## Coverage Control

### Finding Owners By Slice

| Finding IDs | Slice owner | Notes |
|---|---|---|
| F1, F8, F12, F17, F18 | Task 1 and Task 12 | Route contract, command targets, sidepanel matrix, smoke inventory, route visibility. |
| F4, F9 | Task 2 and Task 10 | Shared capability/error state plus operations adoption. |
| F3 | Task 3 | Setup, home resolver, auth/account recovery. |
| F2, F13, F15 | Task 4, plus route-family adopters | Responsive shell, heading landmarks, composer/mobile behavior. |
| F5, F7, F11, F16 | Task 5 | Settings and model/provider configuration. |
| F6 | Task 6 | Global chrome and chat-context controls. |
| F10 | Task 8 | Media first-selection and mobile master-detail. |
| F14 | Task 9 | Ask/Research/Workspace/Transform product ladder. |
| F19 | Task 11A, Task 11B, and Task 12 | Track deprecated UI cleanup only where it blocks UX remediation. |

### Route Owners By Slice

| Slice | Primary route rows |
|---|---|
| Task 1 | All 74 routes through metadata; primary code focus: `/`, `/chat`, `/search`, `/reading`, `/review`, `/audio`, `/prompt-studio`, `/workspace-playground`, `/billing`, `/account`, `/signup`, `/documentation`, `/composer-variants-preview`, `/onboarding-test`. |
| Task 2 | `/sources`, `/scheduled-tasks`, `/integrations`, `/admin`, `/agents`, `/agent-tasks`, `/acp-playground`, `/settings/model`, `/evaluations`, `/mcp-hub`, `/skills`, `/tts`, `/speech`, `/data-tables`. |
| Task 3 | `/`, `/setup`, `/login`, `/signup`, `/account`, `/profile`, `/privileges`, `/config`, `/billing`, `/404`. |
| Task 4 | `/chat`, `/media`, `/settings`, `/settings/model`, `/prompts`, `/workspace-playground`, `/setup`, `/sources`, `/mcp-hub`, `/stt`, `/tts`, `/chat-workspace`. |
| Task 5 | `/settings`, `/settings/model`, `/login`, `/privileges`, `/prompts`, `/prompt-studio`, settings subroutes. |
| Task 6 | `/chat`, `/quick-chat-popout`, `/knowledge`, `/media`, `/sources`, `/settings`, `/mcp-hub`, `/stt`, `/tts`. |
| Task 7 | `/persona`, `/characters`, `/companion`, `/agents`, `/agent-tasks`, `/acp-playground`, `/chat-workflows`, `/dictionaries`, `/world-books`. |
| Task 8 | `/media`, `/media-multi`, `/review`, `/media-trash`, `/items`, `/collections`, `/reading`, `/notes`, `/shared`, `/chatbooks`, `/chatbooks-playground`, `/notifications`. |
| Task 9 | `/knowledge`, `/search`, `/research`, `/workspace-playground`, `/chat-workspace`, `/document-workspace`, `/repo2txt`, `/model-playground`, `/writing-playground`, `/presentation-studio`. |
| Task 10 | `/admin`, `/mcp-hub`, `/sources`, `/connectors`, `/integrations`, `/scheduled-tasks`, `/watchlists`, `/workflow-editor`, `/skills`. |
| Task 11A | `/speech`, `/audio`, `/stt`, `/tts`, `/audiobook-studio`. |
| Task 11B | `/evaluations`, `/flashcards`, `/quiz`, `/moderation-playground`, `/content-review`, `/claims-review`, `/data-tables`, `/chunking-playground`, `/kanban`, `/vn-assets`, `/vn-play`. |
| Task 12 | All 74 routes through regression checks and final browser QA. |

## Baseline Commands

Run these at the beginning of any child implementation session:

```bash
git branch --show-current
git status --short
```

Expected:
- The worker knows whether the branch already contains unrelated changes.
- Unrelated dirty files are left alone.

Frontend unit tests should generally run from `apps/packages/ui`:

```bash
bunx vitest run <test-files>
```

Frontend E2E tests should generally run from `apps/tldw-frontend`:

```bash
bun run e2e:smoke
bun run e2e:smoke:stage4
```

Use narrower Playwright commands in child plans when only a route family changes:

```bash
bunx playwright test e2e/workflows/<specific-file>.spec.ts --reporter=line
```

## Task 0: Planning Setup And Evidence Freeze

**Goal:** Freeze enough evidence that future implementation sessions do not repeat the audit discovery work.

**Files:**
- Reference: `Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md`
- Reference: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Modify later only if needed: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- Backlog: `TASK-418`

- [ ] **Step 1: Verify the source documents are present**

Run:

```bash
test -f Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md
test -f Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
```

Expected: both commands exit 0.

- [ ] **Step 2: Create child Backlog tasks before product code edits**

Create child tasks for:
- Route contract and visibility policy.
- Shared capability and error states.
- First-run setup and account recovery.
- Responsive shell and page landmarks.
- Settings and model/provider configuration.
- Chat and global chrome.
- Persona/context assets/agents.
- Media/library/sharing.
- Knowledge/research/workspace/transform.
- Operations/automation/integrations.
- Audio.
- Study/safety/specialized tools.
- Final QA governance.

Expected: each future code slice has an associated Backlog task before files are edited.

- [ ] **Step 3: Capture baseline browser evidence per child slice**

For each route-family child plan, capture before screenshots or DOM snapshots for
the routes it changes. Minimum widths:
- Desktop.
- 390px mobile.
- Extension sidepanel width when the route is sidepanel-reachable.

Expected: each child task links to its before evidence before remediation.

## Task 1: Route Contract And Visibility Policy

**Goal:** Establish one authoritative route metadata contract that drives or validates navigation, labels, route visibility, command palette targets, sidepanel availability, and smoke inventory coverage.

**Findings:** F1, F8, F12, F17, F18.

**Routes:** All root routes, with first-code focus on aliases, hosted-only routes, labs/debug routes, and command targets.

**Files:**
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx`
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: `apps/packages/ui/src/components/Common/CommandPalette.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/route-registry.visibility.test.ts`
- Test: `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts`
- Test: `apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx`
- Test: `apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts`

- [x] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-route-contract-implementation-plan.md`.
It must enumerate the metadata fields before code work starts:
- canonical path
- aliases
- route label
- route group
- product surface classification
- hosted/self-hosted visibility
- sidepanel availability
- smoke inventory inclusion
- command palette availability

- [ ] **Step 2: Write failing registry tests**

Add tests proving every audited root route has metadata and that `/chat` is the
target for the "Go to Chat" command unless the command is deliberately relabeled.

Expected: tests fail before metadata and command-target work.

- [ ] **Step 3: Implement route metadata with compatibility aliases**

Add or extend route metadata in the shared route registry. Keep existing aliases
working until a child plan explicitly proves a redirect/deprecation path is safe.

- [ ] **Step 4: Wire validation before generation**

Prefer smoke inventory validation against metadata first. Generate smoke
inventory from metadata only if the local route registry shape makes it
straightforward and low risk.

- [ ] **Step 5: Verify**

Run:

```bash
bunx vitest run src/routes/__tests__/route-registry.visibility.test.ts src/routes/__tests__/route-registry.sidepanel-availability.test.ts src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx
```

From `apps/tldw-frontend`, run:

```bash
bunx playwright test e2e/smoke/route-contract-stage2.spec.ts --reporter=line
```

Expected: route metadata, command target, sidepanel matrix, and smoke validation pass.

## Task 2: Shared Capability And Error State System

**Goal:** Replace raw endpoint errors and ambiguous unavailable states with reusable, user-language capability states.

**Findings:** F4, F5 support, F9, F18 support.

**Routes:** `/sources`, `/scheduled-tasks`, `/integrations`, `/admin`, `/agents`, `/agent-tasks`, `/acp-playground`, `/settings/model`, `/evaluations`, `/mcp-hub`, `/skills`, `/tts`, `/speech`, `/data-tables`.

**Files:**
- Modify: `apps/packages/ui/src/components/ui/state/StatePanel.tsx`
- Modify: `apps/packages/ui/src/components/ui/state/RecoveryCallout.tsx`
- Modify: `apps/packages/ui/src/components/ui/state/DiagnosticRow.tsx`
- Modify: `apps/packages/ui/src/components/Common/RouteErrorBoundary.tsx`
- Create only if needed: `apps/packages/ui/src/components/ui/state/capability-state.ts`
- Test: `apps/packages/ui/src/components/ui/state/__tests__/state-primitives.test.tsx`
- Test: `apps/packages/ui/src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx`

- [ ] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-capability-state-implementation-plan.md`.
The plan must reuse the existing `components/ui/state` primitives before adding
new primitives.

- [ ] **Step 2: Define the capability vocabulary**

Cover at least:
- no data
- unavailable server capability
- missing worker
- missing permission
- not configured
- degraded
- unsupported server version
- network failure

- [ ] **Step 3: Write failing component tests**

Test that each state has:
- user-language title
- next action
- diagnostics slot
- no raw endpoint text in the primary message

- [ ] **Step 4: Adopt first routes**

Adopt the shared state in `/sources`, `/scheduled-tasks`, and `/integrations`
before broader route-family adoption.

- [ ] **Step 5: Verify**

Run representative route and state tests:

```bash
bunx vitest run src/components/ui/state/__tests__/state-primitives.test.tsx src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx src/routes/__tests__/option-sources-route-guards.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx src/routes/__tests__/integrations-route.test.tsx
```

Expected: raw `Not Found (GET <endpoint>)` text is absent from primary route states.

## Task 3: First-Run Setup And Connection Flow

**Goal:** Make setup, home, login, account, hosted-only, and recovery routes explicit about current deployment and connection state.

**Findings:** F3, F15 support, F1 support.

**Routes:** `/`, `/setup`, `/login`, `/signup`, `/account`, `/profile`, `/privileges`, `/config`, `/billing`, `/404`.

**Files:**
- Modify: `apps/packages/ui/src/routes/option-setup.tsx`
- Modify: `apps/tldw-frontend/pages/index.tsx`
- Modify: `apps/tldw-frontend/pages/setup.tsx`
- Modify: `apps/tldw-frontend/pages/login.tsx`
- Modify: `apps/tldw-frontend/pages/signup.tsx`
- Modify: `apps/tldw-frontend/pages/account/index.tsx`
- Modify: `apps/tldw-frontend/pages/profile.tsx`
- Modify: `apps/tldw-frontend/pages/privileges.tsx`
- Modify: `apps/tldw-frontend/pages/config.tsx`
- Modify: `apps/tldw-frontend/pages/billing/index.tsx`
- Modify: `apps/tldw-frontend/pages/404.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/onboarding-ingestion-first.spec.ts`
- Test: `apps/tldw-frontend/e2e/hosted/account-billing.spec.ts`

- [ ] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-setup-connection-implementation-plan.md`.
Resolve whether `/` redirects, renders setup, or renders chat for each known
connection/auth state.

- [ ] **Step 2: Write route-state tests**

Test self-hosted configured, self-hosted unconfigured, degraded backend, hosted
mode, and 404 recovery paths.

- [ ] **Step 3: Implement setup shell and home resolver**

Keep setup free of chat-specific chrome until connection state is valid. Clearly
separate frontend origin, backend API URL, saved API key status, auth mode, and
server reachability.

- [ ] **Step 4: Verify**

Run:

```bash
bunx playwright test e2e/workflows/onboarding-ingestion-first.spec.ts --reporter=line
```

Expected: first-run setup has one semantic `h1`, no chat-primary chrome, and a
clear recovery path for clearing the saved connection.

## Task 4: Responsive App Shell And Page Landmarks

**Goal:** Stop page-level horizontal overflow and enforce route orientation landmarks.

**Findings:** F2, F10 support, F11 support, F13, F15.

**Routes:** `/chat`, `/media`, `/settings`, `/settings/model`, `/prompts`, `/workspace-playground`, `/setup`, `/sources`, `/mcp-hub`, `/stt`, `/tts`, `/chat-workspace`.

**Files:**
- Modify: `apps/tldw-frontend/components/layout/WebLayout.tsx`
- Modify: `apps/tldw-frontend/components/layout/Header.tsx`
- Modify: `apps/tldw-frontend/pages/_app.tsx`
- Modify route-family components only as required by failing overflow tests.
- Test: `apps/tldw-frontend/e2e/smoke/stage4-mobile-sidebar.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/stage4-accessibility-controls.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`

- [ ] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-responsive-landmarks-implementation-plan.md`.
List every route that will be changed and the expected narrow-width behavior.

- [ ] **Step 2: Add failing heading and overflow checks**

Add or extend Playwright checks for:
- one semantic `h1` or documented exception
- `document.documentElement.scrollWidth <= window.innerWidth`
- no sticky composer overlap on `/chat`

- [ ] **Step 3: Fix shared shell constraints first**

Fix shared shell and layout constraints before page-local CSS. Page-local fixes
must be limited to routes that still fail after the shell work.

- [ ] **Step 4: Verify**

Run from `apps/tldw-frontend`:

```bash
bun run e2e:smoke:stage4
```

Expected: representative core routes pass heading and 390px overflow gates.

## Task 5: Settings And Model Provider Configuration

**Goal:** Make settings task-led and make model/provider setup configured-first.

**Findings:** F5, F7, F11, F16, F15 support, F2 support.

**Routes:** `/settings`, `/settings/model`, `/login`, `/privileges`, `/prompts`, `/prompt-studio`, settings subroutes.

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav.ts`
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- Modify: `apps/packages/ui/src/components/Option/Settings/general-settings.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/model-settings.tsx`
- Modify: `apps/packages/ui/src/components/Option/Models`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-filter.test.tsx`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-active-route.test.ts`
- Test: `apps/packages/ui/src/components/Option/Models/__tests__/modelsDisplayUtils.test.ts`
- Test: `apps/tldw-frontend/e2e/workflows/settings.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts`

- [ ] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-settings-models-implementation-plan.md`.
The plan must separate routine preferences, provider setup, data management, and
destructive actions.

- [x] **Step 2: Write failing label and configured-first tests**

Test that `providerKeys.navTitle` is never visible and that configured/usable
providers appear before the full model catalog.

- [x] **Step 3: Implement settings grouping and model readiness summary**

Keep full catalog available behind search/filter. Do not remove advanced model
controls.

- [x] **Step 4: Verify**

Run:

```bash
bunx vitest run src/components/Layouts/__tests__/settings-layout-filter.test.tsx src/components/Layouts/__tests__/settings-layout-active-route.test.ts src/components/Option/Models/__tests__/modelsDisplayUtils.test.ts
```

From `apps/tldw-frontend`, run:

```bash
bunx playwright test e2e/workflows/settings.spec.ts e2e/workflows/tier-1-critical/settings-core.spec.ts --reporter=line
```

Expected: settings routes are task-led, mobile-safe, and free of dotted i18n
keys.

**Status 2026-05-18:** Completed in child branch
`codex/webui-settings-models` and tracked by `TASK-418.14`. The slice added
task-led settings navigation groups, a visible Provider Keys route label,
separate Data Management settings, configured-first model/provider orientation,
and prompt route-intent browser guards. Verification recorded on the child task:
focused Vitest settings/model tests passed, the settings Playwright workflow
pair passed, and the WP4 responsive landmarks gate passed. The repo-wide
TypeScript check still fails on existing baseline debt outside this slice.

## Task 6: Chat, Composer, And Global Chrome

**Goal:** Make `/chat` composer-first and keep global app chrome from foregrounding chat-only controls on unrelated pages.

**Findings:** F6, F8 support, F13, F2 support, F15 support.

**Routes:** `/chat`, `/quick-chat-popout`, `/knowledge`, `/media`, `/sources`, `/settings`, `/mcp-hub`, `/stt`, `/tts`.

**Files:**
- Modify: `apps/packages/ui/src/routes/option-chat.tsx`
- Modify: `apps/packages/ui/src/routes/option-quick-chat-popout.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground`
- Modify: `apps/packages/ui/src/components/Layouts/Header.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/HeaderShortcuts.tsx`
- Modify: `apps/packages/ui/src/components/Common/CommandPalette.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/HeaderShortcuts.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/chat.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/chat-sticky-composer.spec.ts`

- [ ] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-chat-global-chrome-implementation-plan.md`.
Define which header actions are global, route-specific, and chat-context-only.

- [ ] **Step 2: Write failing command/header tests**

Test command target consistency and route-specific header action filtering.

- [ ] **Step 3: Make `/chat` first action singular**

Foreground composer readiness and model readiness. Move mode cards into
progressive disclosure without deleting them.

- [ ] **Step 4: Verify**

Run:

```bash
bunx vitest run src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts src/components/Layouts/__tests__/HeaderShortcuts.test.tsx
```

From `apps/tldw-frontend`, run:

```bash
bunx playwright test e2e/workflows/chat.spec.ts e2e/smoke/chat-sticky-composer.spec.ts --reporter=line
```

Expected: `/chat` is composer-first and non-chat pages no longer foreground
chat-only controls.

## Task 7: Persona, Context Assets, Companion, And Agents

**Goal:** Clarify persona, character, companion, context asset, and agent relationships while preserving launch speed.

**Findings:** F1 support, F9 support, F15 support, F18 support.

**Routes:** `/persona`, `/characters`, `/companion`, `/agents`, `/agent-tasks`, `/acp-playground`, `/chat-workflows`, `/dictionaries`, `/world-books`.

**Files:**
- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- Modify: `apps/packages/ui/src/routes/option-characters.tsx`
- Modify: `apps/packages/ui/src/routes/option-companion.tsx`
- Modify: `apps/packages/ui/src/routes/option-chat-workflows.tsx`
- Modify: `apps/packages/ui/src/routes/option-dictionaries.tsx`
- Modify: `apps/packages/ui/src/routes/option-world-books.tsx`
- Modify: `apps/packages/ui/src/components/Agent`
- Test: `apps/packages/ui/src/routes/__tests__/route-registry.persona.test.ts`
- Test: `apps/packages/ui/src/routes/__tests__/chat-workflows-route.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/persona.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts`

- [ ] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-persona-context-agents-implementation-plan.md`.
The plan must define route jobs and launch paths for each concept.

- [ ] **Step 2: Write route-job and launch tests**

Test headings, route labels, active state, and "use/start in chat" actions.

- [ ] **Step 3: Adopt shared capability states for agents and ACP**

Use Task 2 state language for unavailable/degraded agent paths.

- [ ] **Step 4: Verify**

Run:

```bash
bunx vitest run src/routes/__tests__/route-registry.persona.test.ts src/routes/__tests__/chat-workflows-route.test.tsx src/routes/__tests__/sidepanel-persona.test.tsx
```

From `apps/tldw-frontend`, run:

```bash
bunx playwright test e2e/workflows/persona.spec.ts e2e/workflows/journeys/character-chat.spec.ts --reporter=line
```

Expected: users can distinguish persona, character, companion, context assets,
and agents, and can launch relevant assets into chat.

## Task 8: Media, Library, Review, And Sharing

**Goal:** Preserve large-library power while improving first selection, mobile browsing, recovery, and object terminology.

**Findings:** F10, F2 support, F1 support, F18 support, F15 support.

**Routes:** `/media`, `/media-multi`, `/review`, `/media-trash`, `/items`, `/collections`, `/reading`, `/notes`, `/shared`, `/chatbooks`, `/chatbooks-playground`, `/notifications`.

**Files:**
- Modify: `apps/packages/ui/src/routes/option-media.tsx`
- Modify: `apps/packages/ui/src/routes/option-media-multi.tsx`
- Modify: `apps/packages/ui/src/routes/option-media-trash.tsx`
- Modify: `apps/packages/ui/src/routes/option-collections.tsx`
- Modify: `apps/packages/ui/src/routes/option-notes.tsx`
- Modify: `apps/packages/ui/src/routes/option-shared-with-me.tsx`
- Modify: `apps/packages/ui/src/routes/option-chatbooks-playground.tsx`
- Modify: `apps/tldw-frontend/pages/notifications.tsx`
- Modify: `apps/packages/ui/src/components/Review`
- Test: `apps/packages/ui/src/routes/__tests__/option-media-route-guards.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/option-media-multi.connection-state.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/media-review.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/media-navigation-ux-verification.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks.spec.ts`

- [ ] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-media-library-implementation-plan.md`.
The plan must preserve existing filter/bulk workflows and define narrow-width
master-detail behavior.

- [ ] **Step 2: Write failing media first-selection and mobile tests**

Test empty-detail state, selected item recovery, 390px layout, and trash policy.

- [ ] **Step 3: Implement route labels and recovery affordances**

Clarify aliases for `/review` and `/reading`; do not break existing direct URLs.

- [ ] **Step 4: Verify**

Run:

```bash
bunx vitest run src/routes/__tests__/option-media-route-guards.test.tsx src/routes/__tests__/option-media-multi.connection-state.test.tsx
```

From `apps/tldw-frontend`, run:

```bash
bunx playwright test e2e/workflows/media-review.spec.ts e2e/workflows/media-navigation-ux-verification.spec.ts e2e/workflows/tier-2-features/chatbooks.spec.ts --reporter=line
```

Expected: media and library routes support first-time orientation, mobile use,
and repeat bulk/recovery workflows.

## Task 9: Knowledge, Research, Workspace, And Transform Tools

**Goal:** Define and implement the Ask, Research, Workspace, Transform product ladder.

**Findings:** F14, F1 support, F2 support, F15 support.

**Routes:** `/knowledge`, `/search`, `/research`, `/workspace-playground`, `/chat-workspace`, `/document-workspace`, `/repo2txt`, `/model-playground`, `/writing-playground`, `/presentation-studio`.

**Files:**
- Modify: `apps/packages/ui/src/routes/option-knowledge.tsx`
- Modify: `apps/packages/ui/src/routes/option-workspace-playground.tsx`
- Modify: `apps/packages/ui/src/routes/option-chat-workspace.tsx`
- Modify: `apps/packages/ui/src/routes/option-document-workspace.tsx`
- Modify: `apps/packages/ui/src/routes/option-repo2txt.tsx`
- Modify: `apps/packages/ui/src/routes/option-model-playground.tsx`
- Modify: `apps/packages/ui/src/routes/option-writing-playground.tsx`
- Modify: `apps/packages/ui/src/routes/option-presentation-studio.tsx`
- Modify: `apps/tldw-frontend/pages/search.tsx`
- Modify: `apps/tldw-frontend/pages/research.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/knowledge-qa.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/workspace-playground.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/document-workspace.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/model-playground.spec.ts`

- [ ] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-knowledge-workspace-transform-implementation-plan.md`.
The plan must keep `/knowledge` direct cited Q&A and avoid turning it into a
generic knowledge-management hub.

- [ ] **Step 2: Write route-label and empty-state tests**

Test route labels, no-workspace states, transform input/output framing, and
workspace mobile behavior.

- [ ] **Step 3: Implement route ladder and alias behavior**

Make `/search` and `/workspace-playground` behavior intentional through metadata
and page framing. Preserve current routes.

- [ ] **Step 4: Verify**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/knowledge-qa.spec.ts e2e/workflows/workspace-playground.spec.ts e2e/workflows/tier-2-features/document-workspace.spec.ts e2e/workflows/tier-5-specialized/model-playground.spec.ts --reporter=line
```

Expected: users can choose Ask, Research, Workspace, or Transform without
memorizing route history.

## Task 10: Operations, Automation, And Integrations

**Goal:** Make operator surfaces status-first and capability-aware.

**Findings:** F4, F9, F12 support, F17 support, F18 support.

**Routes:** `/admin`, `/mcp-hub`, `/sources`, `/connectors`, `/integrations`, `/scheduled-tasks`, `/watchlists`, `/workflow-editor`, `/skills`.

**Files:**
- Modify: `apps/packages/ui/src/routes/option-sources.tsx`
- Modify: `apps/packages/ui/src/routes/option-integrations.tsx`
- Modify: `apps/packages/ui/src/routes/option-scheduled-tasks.tsx`
- Modify: `apps/packages/ui/src/routes/option-watchlists.tsx`
- Modify: `apps/packages/ui/src/routes/option-workflow-editor.tsx`
- Modify: `apps/packages/ui/src/routes/option-mcp-hub.tsx`
- Modify: `apps/packages/ui/src/routes/option-skills.tsx`
- Modify: `apps/packages/ui/src/routes/option-admin-server.tsx`
- Modify: `apps/packages/ui/src/routes/option-admin-integrations.tsx`
- Modify: `apps/tldw-frontend/pages/connectors/index.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/option-sources-route-guards.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/integrations-route.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/mcp-hub-route.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/sources.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-3-automation/chat-workflows.spec.ts`

- [ ] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-operations-integrations-implementation-plan.md`.
It must distinguish frontend-only state cleanup from backend capability-map work.

- [ ] **Step 2: Write failing capability adoption tests**

Test `/sources`, `/scheduled-tasks`, `/integrations`, and `/mcp-hub` with empty,
unavailable, unauthorized, and degraded states.

- [ ] **Step 3: Implement status-first route surfaces**

Use shared capability states from Task 2. Keep diagnostics available behind
disclosure for operators.

- [ ] **Step 4: Verify**

Run:

```bash
bunx vitest run src/routes/__tests__/option-sources-route-guards.test.tsx src/routes/__tests__/integrations-route.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx src/routes/__tests__/mcp-hub-route.test.tsx
```

From `apps/tldw-frontend`, run:

```bash
bunx playwright test e2e/workflows/tier-2-features/sources.spec.ts e2e/workflows/tier-2-features/mcp-hub.spec.ts e2e/workflows/tier-3-automation/chat-workflows.spec.ts --reporter=line
```

Expected: operator routes expose availability, degraded state, and next actions
without raw endpoint errors as primary UI.

## Task 11A: Audio Routes

**Goal:** Make the audio model and provider readiness clear across speech, STT, TTS, and audiobook routes.

**Findings:** F2 support, F9 support, F15 support, F18 support, F19 support.

**Routes:** `/speech`, `/audio`, `/stt`, `/tts`, `/audiobook-studio`.

**Files:**
- Modify: `apps/packages/ui/src/routes/option-speech.tsx`
- Modify: `apps/packages/ui/src/routes/option-stt.tsx`
- Modify: `apps/packages/ui/src/routes/option-tts.tsx`
- Modify: `apps/packages/ui/src/routes/option-audiobook-studio.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/stt-transcription.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/stage7-audio-regression.spec.ts`

- [ ] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-audio-routes-implementation-plan.md`.
Define canonical route labels for `/audio`, `/speech`, `/stt`, and `/tts`.

- [ ] **Step 2: Write readiness and heading tests**

Test provider readiness, missing voice/model state, recent jobs where present,
and one semantic heading per route.

- [ ] **Step 3: Implement readiness and alias framing**

Keep advanced provider controls available behind appropriate route sections.

- [ ] **Step 4: Verify**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/tier-2-features/stt-transcription.spec.ts e2e/workflows/tier-2-features/tts-synthesis.spec.ts e2e/smoke/stage7-audio-regression.spec.ts --reporter=line
```

Expected: audio users can tell what each audio route does, whether it is ready,
and how to recover from missing provider state.

## Task 11B: Study, Safety, And Specialized Tools

**Goal:** Improve route identity, readiness, and classification for study, safety, review, data, chunking, kanban, and VN routes.

**Findings:** F2 support, F9 support, F15 support, F18 support, F19.

**Routes:** `/evaluations`, `/flashcards`, `/quiz`, `/moderation-playground`, `/content-review`, `/claims-review`, `/data-tables`, `/chunking-playground`, `/kanban`, `/vn-assets`, `/vn-play`.

**Files:**
- Modify: `apps/packages/ui/src/routes/option-evaluations.tsx`
- Modify: `apps/packages/ui/src/routes/option-flashcards.tsx`
- Modify: `apps/packages/ui/src/routes/option-quiz.tsx`
- Modify: `apps/packages/ui/src/routes/option-moderation-playground.tsx`
- Modify: `apps/packages/ui/src/routes/option-content-review.tsx`
- Modify: `apps/tldw-frontend/pages/claims-review.tsx`
- Modify: `apps/packages/ui/src/routes/option-data-tables.tsx`
- Modify: `apps/packages/ui/src/routes/option-chunking-playground.tsx`
- Modify: `apps/packages/ui/src/routes/option-kanban-playground.tsx`
- Modify: `apps/tldw-frontend/pages/vn-assets.tsx`
- Modify: `apps/tldw-frontend/pages/vn-play.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/vn-play.spec.ts`
- Test: package-local flashcards and quiz tests under `apps/packages/ui/src/components/Flashcards` and `apps/packages/ui/src/components/Quiz`

- [ ] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-study-safety-specialized-implementation-plan.md`.
Split the child plan again if the first implementation scope includes unrelated
study, safety, and VN changes in one PR.

- [ ] **Step 2: Write route identity and classification tests**

Test heading, route classification, initial mode, degraded state, and alias
behavior for claims/content review.

- [ ] **Step 3: Implement route identity and readiness**

Keep labs/specialized routes available only where the route visibility policy
allows them.

- [ ] **Step 4: Verify**

Run targeted tests for changed routes plus:

```bash
bunx playwright test e2e/workflows/tier-2-features/evaluations.spec.ts e2e/smoke/vn-assets.spec.ts e2e/smoke/vn-play.spec.ts --reporter=line
```

Expected: specialized routes are classified, have clear primary jobs, and do not
pollute default user navigation when they are labs/internal surfaces.

## Task 12: QA, Regression, And Route Governance

**Goal:** Prevent route drift, missing headings, raw capability errors, command target regressions, and mobile overflow from recurring.

**Findings:** All findings, especially F2, F15, F17, F18.

**Routes:** All audited root routes and representative child routes.

**Files:**
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`
- Modify or create: route metadata tests under `apps/packages/ui/src/routes/__tests__/`
- Modify or create: command palette route target tests under `apps/packages/ui/src/components/Common/__tests__/`

- [ ] **Step 1: Write the child implementation plan**

Create `Docs/superpowers/plans/<date>-webui-route-governance-qa-implementation-plan.md`.
It must identify which checks are CI-suitable and which remain manual/browser
QA evidence.

- [ ] **Step 2: Add final matrix checks**

Add tests or scripts that fail when:
- a new root page file has no route metadata
- a user-facing root route lacks an `h1` without approved exception
- smoke inventory omits a user-facing route
- command palette label and target disagree
- representative core routes overflow at 390px

- [ ] **Step 3: Run final route sweep**

From `apps/tldw-frontend`, run:

```bash
bun run e2e:smoke:all-pages:gate
bun run e2e:smoke:stage4
```

Expected: all route-governance checks pass or every skip has an explicit route
metadata reason.

- [ ] **Step 4: Close the coverage matrix**

Update the active Backlog task with:
- findings closed
- routes changed
- tests run
- browser evidence paths
- known skips
- any backend dependencies deferred to separate tasks

## Release/Review Gate

Before any implementation PR from this plan is considered ready:

- [ ] The PR names the findings and route rows it closes.
- [ ] The PR names route rows intentionally left open.
- [ ] Unit/component tests cover changed shared contracts.
- [ ] Browser QA covers changed visual routes.
- [ ] No broad visual redesign or new design system was introduced.
- [ ] No backend API change was introduced without explicit child-plan evidence.
- [ ] Raw endpoint text is not primary UI for changed capability states.
- [ ] Existing route aliases remain compatible unless explicitly migrated.
- [ ] Backlog task includes final summary, verification, and known skips.

## Parent Plan Verification

After editing this planning document, run:

```bash
rg -n 'T[O]DO|T[B]D|FIX[M]E|\\.\\.\\.|\\bmaybe\\b|\\bprobably\\b|\\bshould consider\\b' Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
rg -n '[[:blank:]]$|[^\\x00-\\x7F]' Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
git diff --check -- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md 'backlog/tasks/task-418 - Plan-WebUI-UX-remediation-implementation-slices.md'
```

Expected:
- Placeholder scan exits 1 with no output.
- ASCII/trailing whitespace scan exits 1 with no output.
- `git diff --check` exits 0.
