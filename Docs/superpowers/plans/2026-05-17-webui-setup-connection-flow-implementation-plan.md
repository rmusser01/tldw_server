# WebUI Setup Connection Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make first-run setup, home resolution, auth placeholders, hosted-only routes, and 404 recovery explicit about deployment and connection state.

**Architecture:** Keep the existing `/` resolver, `/setup` setup-only route, `OnboardingConnectForm`, and placeholder/redirect components as the base. Add tests and small UX patches around state resolution, route classification, and recovery actions before changing layout or copy. Coordinate with the route contract and capability-state slices instead of duplicating their metadata systems.

**Tech Stack:** Next.js pages in `apps/tldw-frontend`, shared React/TypeScript routes in `apps/packages/ui`, existing connection store/hooks, existing setup state primitives, Vitest, Testing Library, Playwright, Backlog.md task tracking.

---

## Source Documents

- Parent plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- Source spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Source audit: `Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md`
- Planning Backlog task: `TASK-421`
- Parent planning task: `TASK-418`

## Scope

### Findings Closed Or Partially Closed

- F3: setup and onboarding are mixed with chat-oriented global chrome.
- F15 support: setup, account, placeholder, and 404 routes need stable landmarks and narrow-width behavior.
- F1 support: root, auth, account, hosted-only, and placeholder route intent must match user-facing labels.

### Route Rows

This slice owns first-run and connection-flow behavior for:

`/`, `/setup`, `/login`, `/signup`, `/account`, `/profile`, `/privileges`,
`/config`, `/billing`, `/404`.

Implementation should also cover directly related placeholder routes that share
the same route components:

`/billing/success`, `/billing/cancel`, `/auth/reset-password`,
`/auth/magic-link`, `/auth/verify-email`.

### Out Of Scope

- No auth backend changes.
- No hosted private account, signup, billing, or password-reset implementation.
- No broad app shell redesign.
- No route renaming without the Task 1 route-contract compatibility policy.
- No new onboarding product flow. Preserve `OnboardingConnectForm` and its
  first-value sequence.
- No removal of diagnostics, server URL, auth mode, or API-key recovery paths.

## Current Ownership Snapshot

| Route | Current owner | Current behavior to preserve or test |
|---|---|---|
| `/` | `apps/tldw-frontend/pages/index.tsx` -> `apps/packages/ui/src/routes/option-index.tsx` | Hosted mode renders hosted home; unfinished first run renders onboarding; completed first run renders companion home. |
| `/setup` | `apps/tldw-frontend/pages/setup.tsx` -> `apps/packages/ui/src/routes/option-setup.tsx` | Uses `OptionLayout hideHeader hideSidebar`, `SetupRequiredPanel`, and `OnboardingWizard`. |
| `/login` | `apps/tldw-frontend/pages/login.tsx` | Self-host redirects to `/settings/tldw`; hosted mode renders `TldwSettings` auth body. |
| `/signup` | `apps/tldw-frontend/pages/signup.tsx` | OSS placeholder for hosted signup. |
| `/account` | `apps/tldw-frontend/pages/account/index.tsx` | OSS placeholder for hosted account pages. |
| `/profile` | `apps/tldw-frontend/pages/profile.tsx` | Placeholder with settings recovery. |
| `/privileges` | `apps/tldw-frontend/pages/privileges.tsx` | Redirects to `/settings`. |
| `/config` | `apps/tldw-frontend/pages/config.tsx` | Placeholder with settings recovery. |
| `/billing` | `apps/tldw-frontend/pages/billing/index.tsx` | OSS placeholder for hosted billing. |
| `/billing/success`, `/billing/cancel` | `apps/tldw-frontend/pages/billing/*.tsx` | OSS placeholder for hosted checkout redirects. |
| `/auth/reset-password`, `/auth/magic-link`, `/auth/verify-email` | `apps/tldw-frontend/pages/auth/*.tsx` | OSS placeholder for hosted auth routes. |
| `/404` | `apps/tldw-frontend/pages/404.tsx` | Recovery panel with route context and navigation actions. |

## State Matrix

The implementation must lock this matrix with tests before product changes:

| Surface | Deployment/state | User-facing result | Primary action | Diagnostics or route context |
|---|---|---|---|---|
| `/` | Hosted deployment | Hosted home route. | Continue through hosted workspace. | None unless hosted shell fails. |
| `/` | Self-host, first run incomplete, unconfigured | Setup-only onboarding route content. | Connect server or use demo mode from onboarding. | Frontend origin and target server URL are separate. |
| `/` | Self-host, first run incomplete, character-chat intent | Character-chat setup lane after connection. | Continue to character creation/import/model selection. | Return target preserved. |
| `/` | Self-host, first run complete, connected | Companion home shell. | Continue active workspace. | Connection health remains in normal app chrome. |
| `/` | Self-host, first run complete, degraded/unreachable | Normal shell can show recovery state. | Retry, update server settings, or restart setup. | Raw endpoint details behind diagnostics from Task 2. |
| `/setup` | Any self-host connection state | Setup-only shell, one setup `h1`, no route sidebar/header. | Focus server URL or continue connection test. | Server URL, auth mode, saved-key status, health status. |
| `/login` | Self-host OSS | Redirect or recovery to `/settings/tldw`. | Configure local server auth. | Existing query params preserved if redirect policy supports it. |
| `/login` | Hosted distribution | Hosted login form. | Sign in. | Auth failures in user language. |
| Hosted-only placeholder routes | OSS/self-host | Placeholder says route is not part of OSS surface. | Open login or settings, depending on route. | Requested route remains visible. |
| `/profile`, `/config` | OSS/self-host | Placeholder says route is not ready and routes to settings. | Open Settings. | Requested route and planned route visible. |
| `/privileges` | OSS/self-host | Alias or redirect to settings/permissions location. | Open target route. | Source and destination visible during redirect. |
| `/404` | Any deployment | Not-found recovery panel. | Go to the correct home/chat target from route contract. | Attempted route visible. |

## File Map

### Home And Setup Resolver

- Modify: `apps/packages/ui/src/routes/option-index.tsx`
  - Preserve hosted, first-run, and completed-first-run branches.
  - Add test hooks or state labels only if tests need stable assertions.

- Modify: `apps/packages/ui/src/routes/option-setup.tsx`
  - Preserve `OptionLayout hideHeader hideSidebar`.
  - Ensure one semantic setup `h1` and no chat-primary chrome.

- Modify only if needed: `apps/packages/ui/src/components/Option/Onboarding/OnboardingWizard.tsx`
- Modify only if needed: `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/option-index.setup-flow.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx`
- Test: `apps/packages/ui/src/store/__tests__/connection.test.ts`

### Next Page Wrappers And Placeholder Routes

- Modify: `apps/tldw-frontend/pages/index.tsx`
- Modify: `apps/tldw-frontend/pages/setup.tsx`
- Modify: `apps/tldw-frontend/pages/login.tsx`
- Modify: `apps/tldw-frontend/pages/signup.tsx`
- Modify: `apps/tldw-frontend/pages/account/index.tsx`
- Modify: `apps/tldw-frontend/pages/profile.tsx`
- Modify: `apps/tldw-frontend/pages/privileges.tsx`
- Modify: `apps/tldw-frontend/pages/config.tsx`
- Modify: `apps/tldw-frontend/pages/billing/index.tsx`
- Modify: `apps/tldw-frontend/pages/billing/success.tsx`
- Modify: `apps/tldw-frontend/pages/billing/cancel.tsx`
- Modify: `apps/tldw-frontend/pages/auth/reset-password.tsx`
- Modify: `apps/tldw-frontend/pages/auth/magic-link.tsx`
- Modify: `apps/tldw-frontend/pages/auth/verify-email.tsx`
- Modify: `apps/tldw-frontend/pages/404.tsx`
- Test: `apps/tldw-frontend/__tests__/navigation/route-placeholder-component.test.tsx`
- Test: `apps/tldw-frontend/__tests__/navigation/route-redirect-component.test.tsx`
- Test: `apps/tldw-frontend/__tests__/navigation/not-found-page.test.tsx`

### Shared Route And Smoke Contracts

- Modify only as needed: `apps/tldw-frontend/components/navigation/RoutePlaceholder.tsx`
- Modify only as needed: `apps/tldw-frontend/components/navigation/RouteRedirect.tsx`
- Modify only as needed: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify only as needed after Task 1 lands: `apps/packages/ui/src/routes/route-registry.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/hosted-placeholder-routes.spec.ts`
- Test: `apps/tldw-frontend/e2e/login.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/onboarding-ingestion-first.spec.ts`
- Test if added: `apps/tldw-frontend/e2e/workflows/setup-connection-flow.spec.ts`

## Implementation Tasks

### Task 0: Baseline And Backlog Setup

**Files:**
- Reference: `Docs/superpowers/plans/2026-05-17-webui-setup-connection-flow-implementation-plan.md`
- Reference: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Backlog: create or update an implementation task before product code edits.

- [ ] **Step 1: Verify branch and dirty worktree**

Run:

```bash
git branch --show-current
git status --short
```

Expected:
- You know the active branch.
- Existing unrelated dirty files are left untouched.

- [ ] **Step 2: Create implementation Backlog task**

Create a task named `Implement WebUI setup and connection flow`.

Expected:
- The task links this plan, parent plan, source spec, and audit.
- The task lists F3, F15 support, and F1 support.

- [ ] **Step 3: Capture baseline route evidence**

Use browser or Playwright evidence for:
- `/` with first run incomplete.
- `/` with first run complete.
- `/setup`.
- `/login` in self-host mode.
- `/account`, `/signup`, `/billing`.
- `/404` with a missing route.

Expected:
- Baseline evidence is linked from the Backlog task.

### Task 1: Lock Connection UX State Matrix

**Files:**
- Modify: `apps/packages/ui/src/store/__tests__/connection.test.ts`
- Modify only if needed: `apps/packages/ui/src/types/connection.ts`
- Modify only if needed: `apps/packages/ui/src/store/connection.tsx`

- [ ] **Step 1: Add failing state-matrix tests**

Test `deriveConnectionUxState` and first-run flags for:
- unconfigured first run -> `unconfigured`
- URL entry in progress -> `configuring_url`
- auth entry in progress -> `configuring_auth`
- health test running -> `testing`
- connected and knowledge ready -> `connected_ok`
- connected with partial or offline knowledge -> `connected_degraded`
- auth failure -> `error_auth`
- unreachable backend -> `error_unreachable`
- demo mode -> `demo_mode`
- `beginOnboarding()` exits demo mode
- `restartOnboarding()` clears `__tldw_first_run_complete`

- [ ] **Step 2: Run tests to verify failure or current coverage**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/store/__tests__/connection.test.ts
```

Expected:
- Fails only for missing matrix coverage.

- [ ] **Step 3: Patch state derivation only if tests prove a gap**

Keep the existing state names. Do not add new connection states unless the
current matrix cannot represent the user-facing state.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
bunx vitest run src/store/__tests__/connection.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/store/__tests__/connection.test.ts apps/packages/ui/src/types/connection.ts apps/packages/ui/src/store/connection.tsx
git commit -m "test: lock setup connection state matrix"
```

### Task 2: Test `/` Home Resolver Behavior

**Files:**
- Create: `apps/packages/ui/src/routes/__tests__/option-index.setup-flow.test.tsx`
- Modify only if needed: `apps/packages/ui/src/routes/option-index.tsx`

- [ ] **Step 1: Write failing resolver tests**

Mock connection hooks and deployment mode. Test:
- hosted deployment renders hosted home branch inside headerless layout
- first-run incomplete renders onboarding branch inside headerless layout
- character-chat onboarding intent changes the onboarding title and return path
- first-run complete renders `CompanionHomeShell` inside normal `OptionLayout`
- first-run incomplete calls `beginOnboarding()` when phase is unconfigured
- first-run complete refreshes connection through `checkOnce()`

- [ ] **Step 2: Run test to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/option-index.setup-flow.test.tsx
```

Expected:
- Fails before missing assertions or test hooks are implemented.

- [ ] **Step 3: Implement minimal resolver fixes**

Preserve existing lazy imports. Keep `/` as the resolver route. If the primary
route target from Task 1 says chat actions must go to `/chat`, change only the
affected recovery labels/actions; do not turn `/` into a hard redirect without
explicit compatibility coverage.

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
bunx vitest run src/routes/__tests__/option-index.setup-flow.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/option-index.tsx apps/packages/ui/src/routes/__tests__/option-index.setup-flow.test.tsx
git commit -m "test: cover home setup resolver states"
```

### Task 3: Harden `/setup` Setup-Only Shell

**Files:**
- Modify: `apps/packages/ui/src/routes/option-setup.tsx`
- Modify: `apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx`
- Modify only if needed: `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`

- [ ] **Step 1: Extend setup shell tests**

Test:
- `/setup` renders exactly one semantic `h1`
- setup route uses `OptionLayout hideHeader hideSidebar`
- setup route does not render chat header, chat shortcuts, or sidebar nav
- primary setup action focuses the server URL input
- progress labels use design-system states

- [ ] **Step 2: Run test to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx
```

Expected:
- Fails on any missing semantic heading or setup-only shell assertion.

- [ ] **Step 3: Patch setup route or onboarding form**

Prefer route-level changes in `option-setup.tsx`. Patch
`OnboardingConnectForm.tsx` only if the form cannot expose a semantic heading
or stable state labels.

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/option-setup.tsx apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx
git commit -m "fix: keep setup route in setup-only shell"
```

### Task 4: Make Login And Hosted Placeholder Policy Explicit

**Files:**
- Modify: `apps/tldw-frontend/pages/login.tsx`
- Modify: `apps/tldw-frontend/pages/signup.tsx`
- Modify: `apps/tldw-frontend/pages/account/index.tsx`
- Modify: `apps/tldw-frontend/pages/billing/index.tsx`
- Modify: `apps/tldw-frontend/pages/billing/success.tsx`
- Modify: `apps/tldw-frontend/pages/billing/cancel.tsx`
- Modify: `apps/tldw-frontend/pages/auth/reset-password.tsx`
- Modify: `apps/tldw-frontend/pages/auth/magic-link.tsx`
- Modify: `apps/tldw-frontend/pages/auth/verify-email.tsx`
- Modify only if needed: `apps/tldw-frontend/components/navigation/RoutePlaceholder.tsx`
- Test: `apps/tldw-frontend/e2e/login.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/hosted-placeholder-routes.spec.ts`
- Test: `apps/tldw-frontend/__tests__/navigation/route-placeholder-component.test.tsx`

- [ ] **Step 1: Write failing placeholder/login tests**

Test:
- self-host `/login` redirects to `/settings/tldw`
- hosted `/login` remains on `/login` and renders the auth form
- OSS hosted-only placeholders have one `h1`
- placeholder primary actions do not default to "Go to Chat" for account,
  billing, signup, or auth recovery routes
- placeholder requested route remains visible

- [ ] **Step 2: Run tests to verify failure**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/navigation/route-placeholder-component.test.tsx
bunx playwright test e2e/login.spec.ts e2e/workflows/hosted-placeholder-routes.spec.ts --reporter=line
```

Expected:
- Fails only where placeholders or login behavior are ambiguous.

- [ ] **Step 3: Patch page wrappers or placeholder defaults**

Keep hosted private routes as placeholders in OSS. Use route-specific primary
actions:
- setup/configuration routes -> `/settings/tldw` or `/settings`
- hosted auth/account/billing placeholders -> `/login`
- generic future profile/config placeholders -> `/settings`

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
bunx vitest run __tests__/navigation/route-placeholder-component.test.tsx
bunx playwright test e2e/login.spec.ts e2e/workflows/hosted-placeholder-routes.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/tldw-frontend/pages/login.tsx apps/tldw-frontend/pages/signup.tsx apps/tldw-frontend/pages/account/index.tsx apps/tldw-frontend/pages/billing apps/tldw-frontend/pages/auth apps/tldw-frontend/components/navigation/RoutePlaceholder.tsx apps/tldw-frontend/e2e/login.spec.ts apps/tldw-frontend/e2e/workflows/hosted-placeholder-routes.spec.ts apps/tldw-frontend/__tests__/navigation/route-placeholder-component.test.tsx
git commit -m "fix: clarify hosted auth placeholders"
```

### Task 5: Clarify Profile, Config, Privileges, And 404 Recovery

**Files:**
- Modify: `apps/tldw-frontend/pages/profile.tsx`
- Modify: `apps/tldw-frontend/pages/config.tsx`
- Modify: `apps/tldw-frontend/pages/privileges.tsx`
- Modify: `apps/tldw-frontend/pages/404.tsx`
- Modify only if needed: `apps/tldw-frontend/components/navigation/RouteRedirect.tsx`
- Test: `apps/tldw-frontend/__tests__/navigation/route-placeholder-component.test.tsx`
- Test: `apps/tldw-frontend/__tests__/navigation/route-redirect-component.test.tsx`
- Test: `apps/tldw-frontend/__tests__/navigation/not-found-page.test.tsx`

- [ ] **Step 1: Write failing recovery tests**

Test:
- `/profile` placeholder routes to settings and names the planned profile route
- `/config` placeholder routes to settings and names the planned config route
- `/privileges` redirect exposes source and destination while redirecting
- `/404` primary action label and target match the route contract
- `/404` still includes route context and keyboard-reachable recovery actions

- [ ] **Step 2: Run tests to verify failure**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/navigation/route-placeholder-component.test.tsx __tests__/navigation/route-redirect-component.test.tsx __tests__/navigation/not-found-page.test.tsx
```

Expected:
- Fails until recovery targets and labels are explicit.

- [ ] **Step 3: Patch placeholders and recovery labels**

Preserve direct URLs. Do not remove `router.back()` recovery. If Task 1 route
contract is not merged yet, use the existing root resolver as the primary 404
target and leave a Backlog note for the Task 1 alignment follow-up.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
bunx vitest run __tests__/navigation/route-placeholder-component.test.tsx __tests__/navigation/route-redirect-component.test.tsx __tests__/navigation/not-found-page.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/tldw-frontend/pages/profile.tsx apps/tldw-frontend/pages/config.tsx apps/tldw-frontend/pages/privileges.tsx apps/tldw-frontend/pages/404.tsx apps/tldw-frontend/components/navigation/RouteRedirect.tsx apps/tldw-frontend/__tests__/navigation/route-placeholder-component.test.tsx apps/tldw-frontend/__tests__/navigation/route-redirect-component.test.tsx apps/tldw-frontend/__tests__/navigation/not-found-page.test.tsx
git commit -m "fix: clarify route recovery for setup-adjacent pages"
```

### Task 6: Browser QA For Setup And Recovery Routes

**Files:**
- Modify or create: `apps/tldw-frontend/e2e/workflows/setup-connection-flow.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/onboarding-ingestion-first.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/hosted-placeholder-routes.spec.ts`
- Backlog: update active implementation task.

- [ ] **Step 1: Add route-focused browser checks**

Cover desktop and 390px mobile for:
- `/`
- `/setup`
- `/login`
- `/account`
- `/signup`
- `/billing`
- `/404` through a missing route

Assertions:
- one semantic `h1` or documented exception
- no horizontal overflow at 390px
- setup routes do not show chat header/sidebar
- placeholders show route context and correct primary action

- [ ] **Step 2: Run browser checks**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/setup-connection-flow.spec.ts e2e/workflows/onboarding-ingestion-first.spec.ts e2e/workflows/hosted-placeholder-routes.spec.ts --reporter=line
```

Expected:
- PASS or environment-specific skip documented with screenshots/logs.

- [ ] **Step 3: Record browser evidence**

Attach evidence paths to the implementation Backlog task for:
- first-run `/`
- `/setup`
- self-host `/login`
- hosted placeholders
- `/404`

### Task 7: Final Setup Flow Gate

**Files:**
- Reference all files changed in Tasks 1-6.
- Backlog: update the active implementation task.

- [ ] **Step 1: Run focused unit tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/store/__tests__/connection.test.ts src/routes/__tests__/option-index.setup-flow.test.tsx src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend tests**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/navigation/route-placeholder-component.test.tsx __tests__/navigation/route-redirect-component.test.tsx __tests__/navigation/not-found-page.test.tsx
bunx playwright test e2e/login.spec.ts e2e/workflows/onboarding-ingestion-first.spec.ts e2e/workflows/hosted-placeholder-routes.spec.ts --reporter=line
```

Expected: PASS or environment-specific skip documented with evidence.

- [ ] **Step 3: Run diff check**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Update Backlog task**

Record:
- findings closed or partially closed
- route states verified
- tests run
- browser evidence paths
- known skips
- dependency on Task 1 route-contract target if unresolved
- whether any backend auth/capability exposure is still needed

- [ ] **Step 5: Commit final task notes if needed**

```bash
git add backlog/tasks/<task-file>.md
git commit -m "docs: record setup flow verification"
```

## Review Checklist

Before opening an implementation PR:

- [ ] `/setup` has a single semantic setup heading and setup-only shell.
- [ ] `/` behavior is explicit for hosted, first-run, demo, connected, and degraded states.
- [ ] Self-host `/login` sends users to local server configuration.
- [ ] Hosted `/login` remains usable in hosted deployment mode.
- [ ] Hosted-only OSS placeholders do not imply unavailable features are broken.
- [ ] Placeholder and 404 primary actions do not default to chat unless the route contract says they should.
- [ ] Route context and diagnostics remain available.
- [ ] 390px browser checks show no horizontal overflow on changed routes.
- [ ] No backend auth API was changed unless separately justified.

## Planning Verification

After editing this plan, run:

```bash
rg -n 'T[O]DO|T[B]D|FIX[M]E|\\.\\.\\.|\\bmaybe\\b|\\bprobably\\b|\\bshould consider\\b' Docs/superpowers/plans/2026-05-17-webui-setup-connection-flow-implementation-plan.md 'backlog/tasks/task-421 - Plan-WebUI-setup-and-connection-flow-implementation.md'
rg -n '[[:blank:]]$|[^\\x00-\\x7F]' Docs/superpowers/plans/2026-05-17-webui-setup-connection-flow-implementation-plan.md 'backlog/tasks/task-421 - Plan-WebUI-setup-and-connection-flow-implementation.md'
git diff --check -- Docs/superpowers/plans/2026-05-17-webui-setup-connection-flow-implementation-plan.md 'backlog/tasks/task-421 - Plan-WebUI-setup-and-connection-flow-implementation.md'
```

Expected:
- Placeholder scan exits 1 with no output.
- ASCII/trailing whitespace scan exits 1 with no output.
- `git diff --check` exits 0.
