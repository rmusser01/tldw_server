# LLM Provider Key Settings Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/settings/provider-keys` render the existing LLM provider key manager in both the hosted WebUI and extension/options shell instead of falling through to a 404.

**Architecture:** Reuse `ProviderKeysSettings` and existing settings shells. Add the missing hosted Next page shim and extension route registry entry, while preserving the shared package settings-specific route registry as the canonical shared settings deep-link path.

**Tech Stack:** Next.js pages, React, React Router route definitions, Vitest source-contract tests, existing Ant Design/settings components.

---

## File Structure

- Create `apps/tldw-frontend/pages/settings/provider-keys.tsx`: hosted WebUI page shim that wraps `ProviderKeysSettings` in `SettingsRoute`.
- Create `apps/tldw-frontend/__tests__/pages/settings-provider-keys-route.test.tsx`: source-contract test for the hosted page shim.
- Modify `apps/tldw-frontend/extension/routes/route-registry.tsx`: add `OptionProviderKeysSettings` lazy route and `/settings/provider-keys` route definition.
- Modify `apps/tldw-frontend/__tests__/extension/route-registry.stability.test.ts`: assert the extension route registry contains the provider-key settings route and component reference.
- Modify `apps/packages/ui/src/routes/__tests__/deferred-options-route.test.tsx`: assert settings deep links can resolve `/settings/provider-keys` through `option-settings-route-registry`.
- Create `apps/packages/ui/src/routes/__tests__/option-settings-provider-keys-route.test.ts`: source-contract test that guards the real shared settings registry and settings nav entry.
- Modify `backlog/tasks/task-497 - Fix-LLM-provider-key-settings-route-404.md`: record implementation notes, verification, and Bandit non-applicability for TS/TSX-only changes.

## Task 1: Hosted WebUI Provider-Key Page Shim

**Files:**
- Create: `apps/tldw-frontend/__tests__/pages/settings-provider-keys-route.test.tsx`
- Create: `apps/tldw-frontend/pages/settings/provider-keys.tsx`

- [ ] **Step 1: Write the failing hosted page-shim test**

Create `apps/tldw-frontend/__tests__/pages/settings-provider-keys-route.test.tsx`:

```ts
import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const loadSource = (...candidates: string[]) => {
  const path = candidates.find((candidate) => existsSync(candidate))
  if (!path) {
    throw new Error(`Missing settings provider-keys page shim: ${candidates.join(" | ")}`)
  }
  return readFileSync(path, "utf8")
}

describe("settings provider keys Next.js page shim", () => {
  it("loads the settings-shell provider key management route", () => {
    const source = loadSource(
      "pages/settings/provider-keys.tsx",
      "tldw-frontend/pages/settings/provider-keys.tsx",
      "apps/tldw-frontend/pages/settings/provider-keys.tsx"
    )

    expect(source).toContain('import("@/components/Option/Settings/ProviderKeysSettings")')
    expect(source).toContain("SettingsRoute")
    expect(source).toContain("ProviderKeysSettings")
    expect(source).not.toContain("TldwSettings")
  })
})
```

- [ ] **Step 2: Run the focused hosted page-shim test and verify it fails**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/pages/settings-provider-keys-route.test.tsx
```

Expected: FAIL with `Missing settings provider-keys page shim`.

- [ ] **Step 3: Add the hosted provider-key page shim**

Create `apps/tldw-frontend/pages/settings/provider-keys.tsx`:

```tsx
import dynamic from "next/dynamic"

export default dynamic(async () => {
  const { SettingsRoute } = await import("@/routes/settings-route")
  const mod = await import("@/components/Option/Settings/ProviderKeysSettings")
  const Component = mod.ProviderKeysSettings
  const Page = () => (
    <SettingsRoute>
      <Component />
    </SettingsRoute>
  )
  return { default: Page }
}, { ssr: false })
```

- [ ] **Step 4: Run the focused hosted page-shim test and verify it passes**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/pages/settings-provider-keys-route.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit the hosted page route**

```bash
git add apps/tldw-frontend/pages/settings/provider-keys.tsx apps/tldw-frontend/__tests__/pages/settings-provider-keys-route.test.tsx
git commit -m "fix: add provider key settings page route"
```

## Task 2: Extension Provider-Key Route Registry Entry

**Files:**
- Modify: `apps/tldw-frontend/__tests__/extension/route-registry.stability.test.ts`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`

- [ ] **Step 1: Write the failing extension route-registry assertion**

Add to `apps/tldw-frontend/__tests__/extension/route-registry.stability.test.ts`:

```ts
it("registers the provider key settings options route", () => {
  expect(extensionRouteRegistrySource).toMatch(/path:\s*"\/settings\/provider-keys"/)
  expect(extensionRouteRegistrySource).toContain("OptionProviderKeysSettings")
  expect(extensionRouteRegistrySource).toContain("settings:providerKeys.navTitle")
})
```

- [ ] **Step 2: Run the focused extension registry test and verify it fails**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/extension/route-registry.stability.test.ts
```

Expected: FAIL because `/settings/provider-keys` and `OptionProviderKeysSettings` are not in the extension route registry.

- [ ] **Step 3: Add the extension lazy route wrapper**

In `apps/tldw-frontend/extension/routes/route-registry.tsx`, near the other settings route wrappers, add:

```tsx
const OptionProviderKeysSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/ProviderKeysSettings"),
  "ProviderKeysSettings"
)
```

- [ ] **Step 4: Add the extension route definition**

In `ROUTE_DEFINITIONS`, add the route after `/settings/tldw` and before `/settings/model`:

```tsx
{
  kind: "options",
  path: "/settings/provider-keys",
  element: <OptionProviderKeysSettings />,
  nav: {
    group: "server",
    labelToken: "settings:providerKeys.navTitle",
    icon: ServerIcon,
    order: 1.5
  }
},
```

Use `ServerIcon` to stay consistent with `apps/packages/ui/src/components/Layouts/settings-nav-config.ts` and avoid an unnecessary icon import.

- [ ] **Step 5: Run the focused extension registry test and verify it passes**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/extension/route-registry.stability.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit the extension route**

```bash
git add apps/tldw-frontend/extension/routes/route-registry.tsx apps/tldw-frontend/__tests__/extension/route-registry.stability.test.ts
git commit -m "fix: register extension provider key settings route"
```

## Task 3: Shared Settings Deep-Link Regression Coverage

**Files:**
- Modify: `apps/packages/ui/src/routes/__tests__/deferred-options-route.test.tsx`
- Create: `apps/packages/ui/src/routes/__tests__/option-settings-provider-keys-route.test.ts`

- [ ] **Step 1: Add a failing provider-key deep-link test**

Add to `apps/packages/ui/src/routes/__tests__/deferred-options-route.test.tsx`:

```tsx
it("resolves provider key settings deep links through the smaller settings registry", async () => {
  render(
    <MemoryRouter initialEntries={["/settings/provider-keys"]}>
      <DeferredOptionsRoute
        attemptedRoute="/settings/provider-keys"
        capabilities={null}
        capabilitiesLoading={false}
        label="Loading options..."
        description="Preparing routes"
      />
    </MemoryRouter>
  )

  expect(
    await screen.findByTestId("deferred-provider-keys-route")
  ).toBeInTheDocument()
})
```

Do not add the mocked registry entry yet.

- [ ] **Step 2: Run the focused shared route test and verify it fails**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/routes/__tests__/deferred-options-route.test.tsx
```

Expected: FAIL because the mocked settings registry does not include `/settings/provider-keys`.

- [ ] **Step 3: Extend the mocked settings registry with provider keys**

In the existing `vi.mock("../option-settings-route-registry", ...)`, add:

```tsx
{
  kind: "options",
  path: "/settings/provider-keys",
  element: <div data-testid="deferred-provider-keys-route">Provider Keys</div>
}
```

- [ ] **Step 4: Run the focused shared route test and verify it passes**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/routes/__tests__/deferred-options-route.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Add a shared settings registry source-contract test**

Create `apps/packages/ui/src/routes/__tests__/option-settings-provider-keys-route.test.ts`:

```ts
import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const findSource = (...candidates: string[]) => {
  const found = candidates.find((candidate) => existsSync(candidate))
  if (!found) {
    throw new Error(`Unable to locate source file: ${candidates.join(" | ")}`)
  }
  return readFileSync(found, "utf8")
}

describe("provider key settings shared route", () => {
  it("keeps the shared settings registry and navigation aligned", () => {
    const registrySource = findSource(
      path.resolve(process.cwd(), "src/routes/option-settings-route-registry.tsx"),
      path.resolve(process.cwd(), "../packages/ui/src/routes/option-settings-route-registry.tsx"),
      path.resolve(process.cwd(), "apps/packages/ui/src/routes/option-settings-route-registry.tsx")
    )
    const navSource = findSource(
      path.resolve(process.cwd(), "src/components/Layouts/settings-nav-config.ts"),
      path.resolve(process.cwd(), "../packages/ui/src/components/Layouts/settings-nav-config.ts"),
      path.resolve(process.cwd(), "apps/packages/ui/src/components/Layouts/settings-nav-config.ts")
    )

    expect(registrySource).toMatch(/path:\s*"\/settings\/provider-keys"/)
    expect(registrySource).toContain("OptionProviderKeysSettings")
    expect(registrySource).toContain("ProviderKeysSettings")
    expect(navSource).toMatch(/path:\s*"\/settings\/provider-keys"/)
    expect(navSource).toContain("settings:providerKeys.navTitle")
  })
})
```

Expected: PASS immediately because the shared settings-specific route already exists. This guards against future removal from the real registry, while the previous test guards the deferred-loading path.

- [ ] **Step 6: Run the full shared route coverage set**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run \
  ../packages/ui/src/routes/__tests__/deferred-options-route.test.tsx \
  ../packages/ui/src/routes/__tests__/option-settings-provider-keys-route.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit the shared deep-link coverage**

```bash
git add apps/packages/ui/src/routes/__tests__/deferred-options-route.test.tsx apps/packages/ui/src/routes/__tests__/option-settings-provider-keys-route.test.ts
git commit -m "test: cover provider key settings deep link"
```

## Task 4: Final Verification And Task Closeout

**Files:**
- Modify: `backlog/tasks/task-497 - Fix-LLM-provider-key-settings-route-404.md`

- [ ] **Step 1: Run the combined focused test set**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run \
  __tests__/pages/settings-provider-keys-route.test.tsx \
  __tests__/extension/route-registry.stability.test.ts \
  ../packages/ui/src/routes/__tests__/deferred-options-route.test.tsx \
  ../packages/ui/src/routes/__tests__/option-settings-provider-keys-route.test.ts
```

Expected: all focused tests PASS.

- [ ] **Step 2: Inspect route files for accidental server-auth conflation**

Run from repo root:

```bash
rg -n "settings/provider-keys|ProviderKeysSettings|TldwSettings" apps/tldw-frontend/pages/settings apps/tldw-frontend/extension/routes/route-registry.tsx apps/packages/ui/src/routes/option-settings-route-registry.tsx
```

Expected:

- `/settings/provider-keys` points to `ProviderKeysSettings`.
- `/settings/tldw` still points to `TldwSettings`.
- No provider-key route redirects to `/settings/tldw`.

- [ ] **Step 3: Record Bandit applicability**

No Python files are touched by this implementation. Record in `TASK-497` that Bandit is not applicable for this frontend-only TS/TSX route fix.

- [ ] **Step 4: Update Backlog task status and verification notes**

Update `backlog/tasks/task-497 - Fix-LLM-provider-key-settings-route-404.md` with:

- completed acceptance criteria;
- focused test commands and results;
- Bandit non-applicability note;
- final summary.

- [ ] **Step 5: Commit final task record updates**

```bash
git add "backlog/tasks/task-497 - Fix-LLM-provider-key-settings-route-404.md"
git commit -m "docs: close provider key route task"
```
