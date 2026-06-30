# tldw Web Design System Proof Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the version 1 design-system proof surface for tldw_server WebUI and browser extension setup, recovery, and admin health screens.

**Architecture:** Add semantic state tokens and a typed state registry first, then build small product-state primitives in `apps/packages/ui/src/components/ui`. Migrate only the approved proof surface so WebUI and extension consumers share the same labels, diagnostics, recovery actions, and token aliases while Ant Design remains available for mechanics such as tables, forms, cards, tooltips, and inputs.

**Tech Stack:** React 18, TypeScript, Next.js, WXT, Tailwind CSS custom properties, Ant Design, lucide-react, Vitest, React Testing Library, Playwright

---

## Contract Reference

Use `Docs/Design/tldw_web_design_system_contract.md` as the source of truth.

Approved version 1 boundary:

- `apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx`
- `apps/packages/ui/src/components/Common/RouteErrorBoundary.tsx`
- `apps/tldw-frontend/components/ErrorBoundary.tsx`
- `apps/tldw-frontend/components/networking/ConfigurationGuard.tsx`
- `apps/tldw-frontend/components/networking/ConfigurationErrorScreen.tsx`
- `apps/tldw-frontend/components/networking/ServerReadinessGate.tsx`
- `apps/packages/ui/src/routes/option-settings-health.tsx`
- `apps/tldw-frontend/pages/settings/health.tsx`
- `apps/packages/ui/src/components/Option/Settings/health-status.tsx`
- `apps/packages/ui/src/routes/option-setup.tsx`
- `apps/tldw-frontend/pages/setup.tsx`
- `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`
- `apps/packages/ui/src/routes/option-admin-server.tsx`
- `apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx`

Browser extension compatibility is required from day one. The extension builds from `apps/packages/ui/src` through `apps/extension/wxt.config.ts`, and its Tailwind config inherits `apps/tldw-frontend/tailwind.config.js` through `apps/extension/tailwind.config.js`.

Do not migrate unrelated admin routes in this slice unless a change is required to preserve shared route or navigation contracts.

## File Structure

- `apps/packages/ui/src/assets/tailwind-shared.css`
  Purpose: define the new `--state-*` CSS custom properties as aliases to existing semantic color variables.
- `apps/tldw-frontend/tailwind.config.js`
  Purpose: expose state tokens as Tailwind colors; extension inherits this config.
- `apps/packages/ui/src/design-system/states.ts`
  Purpose: typed registry for canonical state keys, labels, severity, default action kinds, diagnostics behavior, and token names.
- `apps/packages/ui/src/design-system/index.ts`
  Purpose: stable export surface for design-system data.
- `apps/packages/ui/src/design-system/__tests__/states.test.ts`
  Purpose: lock the state registry to the contract.
- `apps/packages/ui/src/design-system/__tests__/state-token-aliases.test.ts`
  Purpose: verify token aliases exist and Tailwind exposes readable state color names.
- `apps/packages/ui/src/components/ui/state/ActionGroup.tsx`
  Purpose: shared action layout for primary and secondary recovery/admin actions.
- `apps/packages/ui/src/components/ui/state/DiagnosticRow.tsx`
  Purpose: shared label/value row for method, path, status, URL, and raw diagnostics.
- `apps/packages/ui/src/components/ui/state/StatePanel.tsx`
  Purpose: generic product-state panel using the canonical state registry.
- `apps/packages/ui/src/components/ui/state/RecoveryCallout.tsx`
  Purpose: recovery pattern wrapper for unavailable, retrying, setup, auth, blocked, degraded, and error states.
- `apps/packages/ui/src/components/ui/state/PermissionNotice.tsx`
  Purpose: permission/admin guard pattern for `permission_denied`.
- `apps/packages/ui/src/components/ui/state/SetupRequiredPanel.tsx`
  Purpose: setup-required pattern for setup and onboarding surfaces.
- `apps/packages/ui/src/components/ui/state/index.ts`
  Purpose: export state primitives.
- `apps/packages/ui/src/components/ui/state/__tests__/state-primitives.test.tsx`
  Purpose: accessibility and behavior tests for state primitives.
- `apps/packages/ui/src/components/ui/index.ts`
  Purpose: export the new state primitives through the canonical UI package.
- `apps/packages/ui/index.ts`
  Purpose: expose design-system exports for package consumers that import from `@tldw/ui`.
- `apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx`
  Purpose: consume `RecoveryCallout`, `DiagnosticRow`, and `ActionGroup` instead of local one-off styles.
- `apps/packages/ui/src/components/Common/__tests__/BackendUnavailableRecovery.test.tsx`
  Purpose: extend recovery tests to assert canonical state labels and diagnostics behavior.
- `apps/packages/ui/src/components/Common/RouteErrorBoundary.tsx`
  Purpose: keep behavior unchanged while passing canonical recovery details to the shared recovery component.
- `apps/packages/ui/src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx`
  Purpose: verify route recovery still gates backend recovery opt-in.
- `apps/tldw-frontend/components/ErrorBoundary.tsx`
  Purpose: keep web fatal recovery behavior and generic fallback behavior while using shared state copy and actions.
- `apps/tldw-frontend/__tests__/components/ErrorBoundary.test.tsx`
  Purpose: verify top-level web recovery and generic fallback behavior.
- `apps/tldw-frontend/components/networking/ConfigurationErrorScreen.tsx`
  Purpose: replace hardcoded inline styles with `SetupRequiredPanel` or `RecoveryCallout`.
- `apps/tldw-frontend/components/networking/__tests__/ConfigurationErrorScreen.test.tsx`
  Purpose: verify configuration state copy, action guidance, and accessible structure.
- `apps/tldw-frontend/components/networking/ServerReadinessGate.tsx`
  Purpose: replace inline spinner UI with shared `LoadingState` and `StatePanel` semantics.
- `apps/tldw-frontend/components/networking/__tests__/ServerReadinessGate.test.tsx`
  Purpose: extend current readiness tests to assert canonical loading/retrying language.
- `apps/packages/ui/src/components/Option/Settings/health-status.tsx`
  Purpose: normalize health status labels, badges, callouts, and diagnostics actions to the registry.
- `apps/packages/ui/src/components/Option/Settings/__tests__/health-status.design-system.test.tsx`
  Purpose: verify health page uses canonical labels and non-color-only status output.
- `apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx`
  Purpose: normalize admin permission denial, disabled admin API, empty, loading, and error states.
- `apps/packages/ui/src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx`
  Purpose: verify admin guard and empty/error states use canonical state language.
- `apps/packages/ui/src/routes/option-setup.tsx`
  Purpose: wrap setup route framing in shared setup-state primitives without changing onboarding behavior.
- `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`
  Purpose: use canonical setup/auth/unavailable/retrying/ready labels where setup connection state is rendered.
- `apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx`
  Purpose: verify setup, auth, unavailable, retrying, and ready states use canonical state language.
- `apps/packages/ui/src/design-system/__tests__/proof-surface-static-guard.test.ts`
  Purpose: static guard against proof-surface regressions such as new inline recovery styles or local status labels.

## Implementation Rules

- Keep this slice limited to setup, recovery, health, readiness, and `/admin/server`.
- Do not move `Button` wholesale. State primitives may import `Button` from `components/Common/Button`.
- AntD stays in `HealthStatus` and `ServerAdminPage` for `Card`, `Table`, `Descriptions`, `Select`, `Switch`, `Input`, `InputNumber`, `Tooltip`, and layout mechanics.
- Product state labels, badges, callouts, diagnostics structure, and action hierarchy must come from tldw-owned state primitives.
- Tests should assert labels, roles, actions, diagnostics, and import/export contracts rather than raw color values.
- Use a clean worktree for implementation if the main checkout has merge conflicts or unrelated staged work.

## Task 1: Add State Tokens And Registry

**Files:**
- Modify: `apps/packages/ui/src/assets/tailwind-shared.css`
- Modify: `apps/tldw-frontend/tailwind.config.js`
- Create: `apps/packages/ui/src/design-system/states.ts`
- Create: `apps/packages/ui/src/design-system/index.ts`
- Create: `apps/packages/ui/src/design-system/__tests__/states.test.ts`
- Create: `apps/packages/ui/src/design-system/__tests__/state-token-aliases.test.ts`
- Modify: `apps/packages/ui/index.ts`

- [ ] **Step 1: Write the failing state-registry test**

Create `apps/packages/ui/src/design-system/__tests__/states.test.ts`:

```ts
import {
  CANONICAL_STATE_KEYS,
  getDesignSystemState,
  isDesignSystemStateKey
} from "../states"

describe("design-system state registry", () => {
  it("defines every v1 canonical state with stable labels and tokens", () => {
    expect(CANONICAL_STATE_KEYS).toEqual([
      "ready",
      "unavailable",
      "setup_required",
      "auth_required",
      "permission_denied",
      "degraded",
      "retrying",
      "blocked",
      "empty",
      "loading",
      "error"
    ])

    expect(getDesignSystemState("permission_denied")).toMatchObject({
      label: "Permission denied",
      severity: "error",
      token: "--state-permission-denied",
      primaryAction: "request_access"
    })
    expect(isDesignSystemStateKey("ready")).toBe(true)
    expect(isDesignSystemStateKey("healthy")).toBe(false)
  })
})
```

- [ ] **Step 2: Write the failing token-alias test**

Create `apps/packages/ui/src/design-system/__tests__/state-token-aliases.test.ts`:

```ts
import fs from "node:fs"
import path from "node:path"

const sharedCss = fs.readFileSync(
  path.resolve(process.cwd(), "src/assets/tailwind-shared.css"),
  "utf8"
)
const frontendTailwindConfig = fs.readFileSync(
  path.resolve(process.cwd(), "../../tldw-frontend/tailwind.config.js"),
  "utf8"
)

describe("state token aliases", () => {
  it("aliases v1 state tokens to existing semantic tokens", () => {
    expect(sharedCss).toContain("--state-ready: var(--color-success)")
    expect(sharedCss).toContain("--state-unavailable: var(--color-danger)")
    expect(sharedCss).toContain("--state-setup-required: var(--color-warn)")
    expect(sharedCss).toContain("--state-retrying: var(--color-primary)")
    expect(sharedCss).toContain("--state-empty: var(--color-muted)")
  })

  it("exposes readable state colors through the WebUI Tailwind config", () => {
    expect(frontendTailwindConfig).toContain("state:")
    expect(frontendTailwindConfig).toContain("--state-ready")
    expect(frontendTailwindConfig).toContain("setupRequired")
    expect(frontendTailwindConfig).toContain("permissionDenied")
  })
})
```

- [ ] **Step 3: Run the new tests and verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/design-system/__tests__/states.test.ts src/design-system/__tests__/state-token-aliases.test.ts --reporter=verbose
```

Expected: FAIL because the registry and aliases do not exist.

- [ ] **Step 4: Add the token aliases**

In both `:root` and `.dark` sections of `apps/packages/ui/src/assets/tailwind-shared.css`, add:

```css
--state-ready: var(--color-success);
--state-unavailable: var(--color-danger);
--state-setup-required: var(--color-warn);
--state-auth-required: var(--color-warn);
--state-permission-denied: var(--color-danger);
--state-degraded: var(--color-warn);
--state-retrying: var(--color-primary);
--state-blocked: var(--color-danger);
--state-empty: var(--color-muted);
--state-loading: var(--color-muted);
--state-error: var(--color-danger);
```

In `apps/tldw-frontend/tailwind.config.js`, add state colors under `theme.extend.colors`:

```js
state: {
  ready: "rgb(var(--state-ready) / <alpha-value>)",
  unavailable: "rgb(var(--state-unavailable) / <alpha-value>)",
  setupRequired: "rgb(var(--state-setup-required) / <alpha-value>)",
  authRequired: "rgb(var(--state-auth-required) / <alpha-value>)",
  permissionDenied: "rgb(var(--state-permission-denied) / <alpha-value>)",
  degraded: "rgb(var(--state-degraded) / <alpha-value>)",
  retrying: "rgb(var(--state-retrying) / <alpha-value>)",
  blocked: "rgb(var(--state-blocked) / <alpha-value>)",
  empty: "rgb(var(--state-empty) / <alpha-value>)",
  loading: "rgb(var(--state-loading) / <alpha-value>)",
  error: "rgb(var(--state-error) / <alpha-value>)"
}
```

Do not edit `apps/extension/tailwind.config.js`; it inherits the WebUI config.

- [ ] **Step 5: Add the typed state registry**

Create `apps/packages/ui/src/design-system/states.ts` with:

- `DesignSystemStateKey`
- `DesignSystemSeverity`
- `DesignSystemPrimaryAction`
- `DesignSystemStateDefinition`
- `CANONICAL_STATE_KEYS`
- `DESIGN_SYSTEM_STATES`
- `getDesignSystemState(key)`
- `isDesignSystemStateKey(value)`

Use the labels and severity from `Docs/Design/tldw_web_design_system_contract.md`.

- [ ] **Step 6: Add exports**

Create `apps/packages/ui/src/design-system/index.ts`:

```ts
export * from "./states"
```

Modify `apps/packages/ui/index.ts`:

```ts
export * from "./src/design-system"
export * from "./src/components/ui"
```

- [ ] **Step 7: Re-run state tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/design-system/__tests__/states.test.ts src/design-system/__tests__/state-token-aliases.test.ts --reporter=verbose
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/assets/tailwind-shared.css apps/tldw-frontend/tailwind.config.js apps/packages/ui/src/design-system apps/packages/ui/index.ts
git commit -m "feat(ui): add design system state tokens"
```

## Task 2: Add State Primitives

**Files:**
- Create: `apps/packages/ui/src/components/ui/state/ActionGroup.tsx`
- Create: `apps/packages/ui/src/components/ui/state/DiagnosticRow.tsx`
- Create: `apps/packages/ui/src/components/ui/state/StatePanel.tsx`
- Create: `apps/packages/ui/src/components/ui/state/RecoveryCallout.tsx`
- Create: `apps/packages/ui/src/components/ui/state/PermissionNotice.tsx`
- Create: `apps/packages/ui/src/components/ui/state/SetupRequiredPanel.tsx`
- Create: `apps/packages/ui/src/components/ui/state/index.ts`
- Create: `apps/packages/ui/src/components/ui/state/__tests__/state-primitives.test.tsx`
- Modify: `apps/packages/ui/src/components/ui/index.ts`

- [ ] **Step 1: Write the failing state-primitives test**

Create `apps/packages/ui/src/components/ui/state/__tests__/state-primitives.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { RecoveryCallout, StatePanel } from "../"

describe("state primitives", () => {
  it("renders canonical state labels with accessible primary actions", () => {
    render(
      <RecoveryCallout
        state="unavailable"
        title="Cannot reach the API server"
        message="Check that your server is running."
        primaryAction={{ label: "Try again", onClick: vi.fn() }}
        secondaryActions={[{ label: "Open diagnostics", onClick: vi.fn() }]}
      />
    )

    expect(screen.getByText("Unavailable")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Open diagnostics" })).toBeInTheDocument()
  })

  it("shows diagnostics only when diagnostics are provided", () => {
    render(
      <StatePanel
        state="error"
        title="Request failed"
        diagnostics={[{ label: "Request path", value: "/api/v1/health" }]}
      />
    )

    expect(screen.getByLabelText("Diagnostics")).toBeInTheDocument()
    expect(screen.getByText("/api/v1/health")).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/ui/state/__tests__/state-primitives.test.tsx --reporter=verbose
```

Expected: FAIL because the state primitives do not exist.

- [ ] **Step 3: Implement `ActionGroup`**

Use `apps/packages/ui/src/components/Common/Button.tsx` internally. Support:

- `primaryAction?: { label: React.ReactNode; onClick?: () => void; loading?: boolean; disabled?: boolean }`
- `secondaryActions?: Array<{ label: React.ReactNode; onClick?: () => void; disabled?: boolean }>`
- responsive wrapping
- deterministic `data-testid` pass-through

- [ ] **Step 4: Implement `DiagnosticRow`**

Support:

- `label: React.ReactNode`
- `value: React.ReactNode`
- optional `code?: boolean`
- optional `copyLabel?: string` for future copy affordances

Render labels as text, values as readable text or `<code>`, and never rely on color alone.

- [ ] **Step 5: Implement `StatePanel`**

Use the state registry to render:

- visible canonical state label
- title
- message
- optional diagnostics section with `aria-label="Diagnostics"`
- optional `ActionGroup`
- severity-driven classes based on semantic tokens, not hardcoded hex values

- [ ] **Step 6: Implement `RecoveryCallout`, `PermissionNotice`, and `SetupRequiredPanel`**

These should be thin semantic wrappers over `StatePanel`:

- `RecoveryCallout` allows `unavailable`, `retrying`, `blocked`, `degraded`, `error`, `auth_required`, and `setup_required`.
- `PermissionNotice` defaults to `permission_denied`.
- `SetupRequiredPanel` defaults to `setup_required`.

- [ ] **Step 7: Export state primitives**

Create `apps/packages/ui/src/components/ui/state/index.ts` and update `apps/packages/ui/src/components/ui/index.ts`:

```ts
export * from "./state"
```

- [ ] **Step 8: Re-run state primitive tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/ui/state/__tests__/state-primitives.test.tsx --reporter=verbose
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add apps/packages/ui/src/components/ui/state apps/packages/ui/src/components/ui/index.ts
git commit -m "feat(ui): add product state primitives"
```

## Task 3: Migrate Backend Recovery And Error Boundaries

**Files:**
- Modify: `apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx`
- Modify: `apps/packages/ui/src/components/Common/__tests__/BackendUnavailableRecovery.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/RouteErrorBoundary.tsx`
- Modify: `apps/packages/ui/src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx`
- Modify: `apps/tldw-frontend/components/ErrorBoundary.tsx`
- Modify: `apps/tldw-frontend/__tests__/components/ErrorBoundary.test.tsx`

- [ ] **Step 1: Extend the failing recovery test**

Update `BackendUnavailableRecovery.test.tsx` to assert:

- visible label `Unavailable`
- primary action `Try again`
- secondary actions `Reload page`, `Open Health & diagnostics`, `Open Settings`
- diagnostics section has `aria-label="Diagnostics"` when diagnostic details exist
- diagnostics section is absent when diagnostic details do not exist

- [ ] **Step 2: Run recovery and boundary tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Common/__tests__/BackendUnavailableRecovery.test.tsx ../packages/ui/src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx __tests__/components/ErrorBoundary.test.tsx --reporter=verbose
```

Expected: FAIL at least on the new canonical-state assertions.

- [ ] **Step 3: Refactor `BackendUnavailableRecovery`**

Replace local action button classes, local diagnostic row classes, and full-card one-off styling with:

- `RecoveryCallout`
- `DiagnosticRow`
- `ActionGroup`

Keep props and external behavior unchanged so existing boundary consumers do not need a broad rewrite.

- [ ] **Step 4: Check route boundary behavior**

Only adjust `RouteErrorBoundary.tsx` if needed to keep canonical recovery details intact. Do not enable backend recovery by default for extension routes.

- [ ] **Step 5: Check WebUI error boundary behavior**

Only adjust `ErrorBoundary.tsx` if needed so the top-level boundary still renders shared recovery for backend failures and generic fallback for other runtime errors.

- [ ] **Step 6: Re-run recovery and boundary tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Common/__tests__/BackendUnavailableRecovery.test.tsx ../packages/ui/src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx __tests__/components/ErrorBoundary.test.tsx --reporter=verbose
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx apps/packages/ui/src/components/Common/__tests__/BackendUnavailableRecovery.test.tsx apps/packages/ui/src/components/Common/RouteErrorBoundary.tsx apps/packages/ui/src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx apps/tldw-frontend/components/ErrorBoundary.tsx apps/tldw-frontend/__tests__/components/ErrorBoundary.test.tsx
git commit -m "feat(ui): apply state system to backend recovery"
```

## Task 4: Migrate Configuration And Readiness Gates

**Files:**
- Modify: `apps/tldw-frontend/components/networking/ConfigurationErrorScreen.tsx`
- Create: `apps/tldw-frontend/components/networking/__tests__/ConfigurationErrorScreen.test.tsx`
- Modify: `apps/tldw-frontend/components/networking/ConfigurationGuard.tsx`
- Modify: `apps/tldw-frontend/components/networking/ServerReadinessGate.tsx`
- Modify: `apps/tldw-frontend/components/networking/__tests__/ServerReadinessGate.test.tsx`

- [ ] **Step 1: Write the failing configuration error test**

Create `apps/tldw-frontend/components/networking/__tests__/ConfigurationErrorScreen.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { ConfigurationErrorScreen } from "../ConfigurationErrorScreen"

describe("ConfigurationErrorScreen", () => {
  it("uses setup-required state language for loopback API configuration errors", () => {
    render(
      <ConfigurationErrorScreen
        issue={{
          kind: "loopback_api_not_browser_reachable",
          apiOrigin: "http://127.0.0.1:8000",
          pageOrigin: "http://192.168.1.20:8080"
        }}
      />
    )

    expect(screen.getByText("Setup required")).toBeInTheDocument()
    expect(screen.getByText(/127\.0\.0\.1:8000/)).toBeInTheDocument()
    expect(screen.getByText(/192\.168\.1\.20:8080/)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Extend readiness gate tests**

Update `ServerReadinessGate.test.tsx` to assert that the waiting state uses canonical loading or retrying language and has an accessible status region.

- [ ] **Step 3: Run configuration and readiness tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run components/networking/__tests__/ConfigurationErrorScreen.test.tsx components/networking/__tests__/ServerReadinessGate.test.tsx --reporter=verbose
```

Expected: FAIL on new design-system assertions.

- [ ] **Step 4: Refactor configuration error UI**

Replace inline styles in `ConfigurationErrorScreen.tsx` with `SetupRequiredPanel` and diagnostics rows for:

- API origin
- page origin
- required next action

Keep behavior unchanged for unknown issue kinds.

- [ ] **Step 5: Refactor server readiness UI**

Replace inline spinner and style block in `ServerReadinessGate.tsx` with shared `LoadingState` or `StatePanel` using the canonical `loading` and `retrying` states.

Keep these behaviors unchanged:

- healthy response status `ok` or `healthy` unlocks children
- `bypass` skips health checks
- timeout unlocks children
- leaving a bypass route after timeout restarts checks

- [ ] **Step 6: Re-run configuration and readiness tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run components/networking/__tests__/ConfigurationErrorScreen.test.tsx components/networking/__tests__/ServerReadinessGate.test.tsx --reporter=verbose
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/tldw-frontend/components/networking/ConfigurationErrorScreen.tsx apps/tldw-frontend/components/networking/ConfigurationGuard.tsx apps/tldw-frontend/components/networking/ServerReadinessGate.tsx apps/tldw-frontend/components/networking/__tests__/ConfigurationErrorScreen.test.tsx apps/tldw-frontend/components/networking/__tests__/ServerReadinessGate.test.tsx
git commit -m "feat(web): apply state system to setup gates"
```

## Task 5: Migrate Setup, Health, And Admin Server Surfaces

**Files:**
- Modify: `apps/packages/ui/src/routes/option-setup.tsx`
- Modify: `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/health-status.tsx`
- Create: `apps/packages/ui/src/components/Option/Settings/__tests__/health-status.design-system.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx`
- Create: `apps/packages/ui/src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx`

- [ ] **Step 1: Write setup state tests**

Create `OnboardingConnectForm.design-system.test.tsx` to cover:

- setup page renders `Setup required` before a server is configured
- connection test in progress renders `Retrying` or `Loading` with non-color-only text
- auth failures render `Sign in required` or API-key guidance from the registry
- success renders `Ready`

Mock network calls and stores using existing onboarding tests as examples.

- [ ] **Step 2: Write health status tests**

Create `health-status.design-system.test.tsx` to cover:

- all passing checks render `Ready`
- one failing check renders `Degraded`
- unreachable connection callout renders `Unavailable`
- auth callout renders `Sign in required`
- each check still exposes endpoint path and raw diagnostics when details exist

- [ ] **Step 3: Write admin server tests**

Create `ServerAdminPage.design-system.test.tsx` to cover:

- forbidden admin API renders `Permission denied`
- missing admin API renders `Blocked` or `Unavailable` with explicit next action
- system stats error renders `Error` with retry action
- empty media budget selector renders `Empty` with a clear explanation

- [ ] **Step 4: Run setup, health, and admin tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/health-status.design-system.test.tsx ../packages/ui/src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx --reporter=verbose
```

Expected: FAIL because the surfaces still use local or AntD-only state language.

- [ ] **Step 5: Refactor setup route framing**

In `option-setup.tsx`, replace the local setup header card with `SetupRequiredPanel` or `StatePanel` while keeping `OnboardingWizard` behavior unchanged.

- [ ] **Step 6: Normalize onboarding connection states**

In `OnboardingConnectForm.tsx`, use registry labels for setup, auth, unavailable, retrying, and ready states where progress, errors, and success are displayed. Keep form behavior, validation, demo mode, and host permission behavior unchanged.

- [ ] **Step 7: Normalize health page states**

In `health-status.tsx`:

- keep AntD `Card`, `Space`, `Typography`, `Tooltip`, `InputNumber`, and layout mechanics
- replace status `Tag` labels with shared `Badge` or state primitives where product state is shown
- map healthy checks to `ready`
- map failing optional checks with reachable core to `degraded`
- map unreachable core to `unavailable`
- map auth errors to `auth_required`
- keep raw endpoint diagnostics and copy actions

- [ ] **Step 8: Normalize admin server states**

In `ServerAdminPage.tsx`:

- use `PermissionNotice` for forbidden admin APIs
- use `StatePanel` or `RecoveryCallout` for disabled or missing admin APIs
- use `StatePanel state="error"` for system stats errors
- use `StatePanel state="empty"` for empty media budget/user diagnostic states
- keep AntD table, select, switch, form, descriptions, and popconfirm mechanics

- [ ] **Step 9: Re-run setup, health, and admin tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/health-status.design-system.test.tsx ../packages/ui/src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx --reporter=verbose
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add apps/packages/ui/src/routes/option-setup.tsx apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx apps/packages/ui/src/components/Option/Settings/health-status.tsx apps/packages/ui/src/components/Option/Settings/__tests__/health-status.design-system.test.tsx apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx apps/packages/ui/src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx
git commit -m "feat(ui): apply state system to setup and health"
```

## Task 6: Add Proof-Surface Guards And Parity Verification

**Files:**
- Create: `apps/packages/ui/src/design-system/__tests__/proof-surface-static-guard.test.ts`
- Modify: `Docs/Design/tldw_web_design_system_contract.md` only if implementation uncovers a contract correction

- [ ] **Step 1: Write the static guard**

Create `proof-surface-static-guard.test.ts`:

```ts
import fs from "node:fs"
import path from "node:path"

const repoRoot = path.resolve(process.cwd(), "../../..")
const read = (relativePath: string) =>
  fs.readFileSync(path.join(repoRoot, relativePath), "utf8")

describe("proof-surface design system guard", () => {
  it("keeps recovery and readiness screens on shared state primitives", () => {
    expect(read("apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx")).toContain("RecoveryCallout")
    expect(read("apps/tldw-frontend/components/networking/ServerReadinessGate.tsx")).not.toContain("style={{")
    expect(read("apps/tldw-frontend/components/networking/ConfigurationErrorScreen.tsx")).not.toContain("background: \"#")
  })

  it("keeps health and admin pages on canonical state language", () => {
    expect(read("apps/packages/ui/src/components/Option/Settings/health-status.tsx")).toContain("getDesignSystemState")
    expect(read("apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx")).toContain("PermissionNotice")
  })
})
```

Keep this guard narrow. It should protect the version 1 proof surface, not police unrelated admin routes.

- [ ] **Step 2: Run the guard**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/design-system/__tests__/proof-surface-static-guard.test.ts --reporter=verbose
```

Expected: PASS.

- [ ] **Step 3: Run the focused unit suite**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/design-system/__tests__/states.test.ts ../packages/ui/src/design-system/__tests__/state-token-aliases.test.ts ../packages/ui/src/components/ui/state/__tests__/state-primitives.test.tsx ../packages/ui/src/components/Common/__tests__/BackendUnavailableRecovery.test.tsx ../packages/ui/src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx components/networking/__tests__/ConfigurationErrorScreen.test.tsx components/networking/__tests__/ServerReadinessGate.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/health-status.design-system.test.tsx ../packages/ui/src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx ../packages/ui/src/design-system/__tests__/proof-surface-static-guard.test.ts --reporter=verbose
```

Expected: PASS.

- [ ] **Step 4: Run WebUI compile and token sync**

Run:

```bash
cd apps/tldw-frontend
bun run compile
```

Expected: Next.js compile completes and `scripts/verify-shared-token-sync.mjs --dir .next` passes.

- [ ] **Step 5: Run extension compile and token sync**

Run:

```bash
cd apps/extension
bun run compile
bun run build:chrome:dev
```

Expected: TypeScript compile passes, Chrome dev extension build completes, and post-build token sync passes.

- [ ] **Step 6: Run visual smoke checks for WebUI proof routes**

Start WebUI:

```bash
cd apps/tldw-frontend
bun run dev -- -p 8080
```

In a separate terminal, run targeted Playwright or manual browser checks for:

- `http://127.0.0.1:8080/setup`
- `http://127.0.0.1:8080/settings/health`
- `http://127.0.0.1:8080/admin/server`

Expected:

- no overlapping text at desktop or mobile width
- state labels are visible as text, not color-only
- primary and secondary actions are keyboard reachable
- diagnostics remain visible where the contract requires them
- setup and health pages remain usable when the backend is missing

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/design-system/__tests__/proof-surface-static-guard.test.ts Docs/Design/tldw_web_design_system_contract.md
git commit -m "test(ui): guard design system proof surface"
```

Only include `Docs/Design/tldw_web_design_system_contract.md` in this commit if the implementation required an intentional contract correction.

## Task 7: Final Verification And Handoff

**Files:**
- Modify: Backlog task for the implementation work
- Modify: `Docs/Design/tldw_web_design_system_contract.md` only for confirmed implementation notes

- [ ] **Step 1: Run docs and diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 2: Run Bandit only if Python code changed**

If this implementation stays frontend-only, record a Bandit skip in the Backlog task:

```text
Bandit skipped: no Python code touched.
```

If Python code was touched, run Bandit on the touched Python paths from the repo root after activating the venv:

```bash
source .venv/bin/activate
python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_design_system_proof_surface.json
```

Expected: no new findings in touched code.

- [ ] **Step 3: Review the final diff**

Run:

```bash
git diff --stat
git diff -- apps/packages/ui/src/assets/tailwind-shared.css apps/tldw-frontend/tailwind.config.js apps/packages/ui/src/design-system apps/packages/ui/src/components/ui apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx apps/tldw-frontend/components/networking apps/packages/ui/src/components/Option/Settings/health-status.tsx apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx apps/packages/ui/src/routes/option-setup.tsx
```

Expected:

- no unrelated admin route migrations
- no wholesale `Button` migration
- no hardcoded new palette
- proof-surface state copy matches the contract
- WebUI and extension compatibility remains explicit

- [ ] **Step 4: Update Backlog**

Update the implementation Backlog task with:

- tests run and outcomes
- screenshots or browser-check notes
- any known skips or blockers
- final summary of what changed and why

- [ ] **Step 5: Final commit or PR handoff**

If all verification passes:

```bash
git status --short
git log --oneline -5
```

Expected: only intentional files changed or staged, and commits are grouped by the tasks above.

## Success Criteria

- `--state-*` tokens exist and alias existing semantic color variables.
- Tailwind exposes readable state colors for both WebUI and extension builds.
- A typed state registry owns canonical labels, severities, actions, and diagnostics defaults.
- `components/ui` exports state primitives for recovery, diagnostics, action groups, permission notices, and setup-required panels.
- Backend recovery, route recovery, configuration errors, server readiness, setup, health, and `/admin/server` consume shared state primitives or registry labels.
- AntD remains only as a mechanics substrate in proof-surface screens.
- Focused tests cover state registry, token aliases, primitives, recovery, readiness, setup, health, admin server, and static proof-surface drift.
- WebUI compile and extension compile/build checks pass.
- Visual smoke checks show no obvious overlap or inaccessible state-only color signaling on proof routes.
