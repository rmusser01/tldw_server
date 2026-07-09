# Settings IA Recovery Preferences UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved settings IA redesign: `/settings` becomes Setup & Recovery, `/settings/preferences` becomes personal behavior defaults, `/settings/ui` becomes interface customization, and settings navigation/layout/sidepanel behavior match the spec.

**Architecture:** Reuse the existing settings route stack. `settings-nav-config.ts` remains the settings menu source, `settings-active-route.ts` owns canonical matching, and `SettingsOptionLayout.tsx` owns the rendered hub/section layout. Do not add a generic settings route contract, diagnostics contributor framework, or new dependency.

**Tech Stack:** React 18, Next pages, React Router extension routes, Vitest, Testing Library, Playwright, existing `@plasmohq/storage` hooks, existing settings registry/hooks.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-07-06-settings-ia-recovery-preferences-ui-design.md`
- Backlog: `TASK-12166`

## Decisions Locked For Implementation

- User-facing second hub label: **Preferences & Workflow**.
- `SearchModeSettings` is web-search behavior, not RAG retrieval. Move it to `/settings/preferences` under web search defaults.
- OCR asset enablement and OCR language belong with Quick Ingest for this slice because `QuickIngestSettings` already owns ingest OCR presets.
- `SystemSettings` visual/basic controls move to `/settings/ui`; destructive data controls stay on `/settings/data` through `DataManagementSettings`.
- Splash remains `/settings/splash`, listed under the Interface Customization section.
- Setup & Recovery may show selected chat model from `useSelectedModel()` and stored embedding model from direct storage only. Do not call `defaultEmbeddingModelForRag()` on `/settings`, because it may probe provider config on cache miss.
- If a readiness row lacks cheap data, show unknown/not checked and link to the specialist settings page.

## File Map

Create:

- `apps/packages/ui/src/components/Option/Settings/preferences-settings.tsx`
- `apps/packages/ui/src/components/Option/Settings/setup-recovery-settings.tsx`
- `apps/packages/ui/src/components/Option/Settings/__tests__/PreferencesSettings.test.tsx`
- `apps/packages/ui/src/components/Option/Settings/__tests__/SetupRecoverySettings.test.tsx`
- `apps/packages/ui/src/components/Sidepanel/Settings/__tests__/body.test.tsx`
- `apps/tldw-frontend/pages/settings/preferences.tsx`
- `apps/tldw-frontend/pages/settings/data.tsx`

Modify:

- `apps/packages/ui/src/components/Option/Settings/general-settings.tsx`
- `apps/packages/ui/src/components/Option/Settings/ui-customization.tsx`
- `apps/packages/ui/src/components/Option/Settings/QuickIngestSettings.tsx`
- `apps/packages/ui/src/components/Sidepanel/Settings/body.tsx`
- `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- `apps/packages/ui/src/components/Layouts/settings-nav.ts`
- `apps/packages/ui/src/components/Layouts/settings-active-route.ts`
- `apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx`
- `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-active-route.test.ts`
- `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-filter.test.tsx`
- `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx`
- `apps/packages/ui/src/components/Layouts/__tests__/settings-nav.guardian.test.ts`
- `apps/packages/ui/src/routes/option-settings-route-registry.tsx`
- `apps/packages/ui/src/routes/option-settings-health.tsx`
- `apps/packages/ui/src/routes/option-settings-processed.tsx`
- `apps/tldw-frontend/pages/settings/index.tsx`
- `apps/tldw-frontend/e2e/page-mapping.ts`
- `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- `apps/tldw-frontend/e2e/smoke/stage6-interaction-stage2.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-4-admin/settings-full.spec.ts`
- `apps/tldw-frontend/e2e/utils/page-objects/SettingsPage.ts`
- `apps/packages/ui/src/assets/locale/en/settings.json`

Check only if tests or import paths still use the mirror registry:

- `apps/tldw-frontend/extension/routes/route-registry.tsx`

## Task 1: Route Split Scaffolding

**Files:**
- Create: `apps/packages/ui/src/components/Option/Settings/preferences-settings.tsx`
- Create: `apps/packages/ui/src/components/Option/Settings/setup-recovery-settings.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/general-settings.tsx`
- Modify: `apps/packages/ui/src/routes/option-settings-route-registry.tsx`
- Modify: `apps/tldw-frontend/pages/settings/index.tsx`
- Create: `apps/tldw-frontend/pages/settings/preferences.tsx`
- Test: `apps/packages/ui/src/components/Option/Settings/__tests__/PreferencesSettings.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Settings/__tests__/SetupRecoverySettings.test.tsx`

- [ ] **Step 1: Write failing component route ownership tests**

Add `PreferencesSettings.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"

import { PreferencesSettings } from "../preferences-settings"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, fallback: unknown) => [fallback, vi.fn()]
}))

vi.mock("@/hooks/useI18n", () => ({
  useI18n: () => ({
    changeLocale: vi.fn(),
    locale: "en",
    supportLanguage: [{ label: "English", value: "en" }]
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ success: vi.fn(), error: vi.fn() })
}))

vi.mock("@/store/tutorials", () => ({
  useTutorialCompletion: () => ({
    completedTutorials: [],
    resetProgress: vi.fn()
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({ userPersona: null }),
  useConnectionActions: () => ({ setUserPersona: vi.fn() })
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [[], vi.fn()]
}))

vi.mock("../search-mode", () => ({
  SearchModeSettings: () => <div>Web search defaults</div>
}))

describe("PreferencesSettings", () => {
  it("contains personal defaults and web search but not setup, theme, OCR, or destructive reset", () => {
    render(
      <MemoryRouter>
        <PreferencesSettings />
      </MemoryRouter>
    )

    expect(screen.getByText("General preferences")).toBeInTheDocument()
    expect(screen.getByText("Web search defaults")).toBeInTheDocument()
    expect(screen.queryByText("Connection")).not.toBeInTheDocument()
    expect(screen.queryByText("Theme picker")).not.toBeInTheDocument()
    expect(screen.queryByText(/OCR assets/i)).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /reset all/i })).not.toBeInTheDocument()
  })
})
```

Add `SetupRecoverySettings.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"

import { SetupRecoverySettings } from "../setup-recovery-settings"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    serverUrl: "http://127.0.0.1:8000",
    isConnected: false,
    knowledgeStatus: "unknown"
  }),
  useConnectionUxState: () => ({
    uxState: "error_auth",
    errorKind: "auth",
    isChecking: false
  }),
  useConnectionActions: () => ({ restartOnboarding: vi.fn() })
}))

vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => ({
    selectedModel: "openai/gpt-4o-mini",
    selectedModelIsLoading: false
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, fallback: unknown) =>
    key === "defaultEmbeddingModel" ? ["openai/text-embedding-3-small", vi.fn()] : [fallback, vi.fn()]
}))

describe("SetupRecoverySettings", () => {
  it("shows recovery rows with specialist links and no full diagnostics payload", () => {
    render(
      <MemoryRouter>
        <SetupRecoverySettings />
      </MemoryRouter>
    )

    expect(screen.getByRole("heading", { name: "Setup & Recovery" })).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /fix auth/i })).toHaveAttribute("href", "/settings/tldw")
    expect(screen.getByRole("link", { name: /model settings/i })).toHaveAttribute("href", "/settings/model")
    expect(screen.getByRole("link", { name: /embedding defaults/i })).toHaveAttribute("href", "/settings/rag")
    expect(screen.queryByText(/\{/)).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Settings/__tests__/PreferencesSettings.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/SetupRecoverySettings.test.tsx
```

Expected: fail because the new components do not exist.

- [ ] **Step 3: Create minimal split components**

Create `preferences-settings.tsx` by moving the personal preference state and markup out of `general-settings.tsx`:

```tsx
import { SearchModeSettings } from "./search-mode"
import { useTranslation } from "react-i18next"

export const PreferencesSettings = () => {
  const { t } = useTranslation("settings")

  return (
    <div className="flex flex-col space-y-6 text-sm">
      <section className="space-y-4">
        <div>
          <h2 className="text-base font-semibold leading-7 text-text">
            {t("preferencesSettings.title", "General preferences")}
          </h2>
          <div className="border-b border-border mt-3" />
        </div>
        <p className="text-xs text-text-muted">
          {t("preferencesSettings.intro", "Personal behavior defaults live here.")}
        </p>
      </section>
      <section className="space-y-4">
        <SearchModeSettings />
      </section>
    </div>
  )
}

export default PreferencesSettings
```

Create `setup-recovery-settings.tsx` as a lightweight landing page with row data derived from cheap hooks:

```tsx
import { Link, useNavigate } from "react-router-dom"
import { Button, Modal } from "antd"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import { useConnectionActions, useConnectionState, useConnectionUxState } from "@/hooks/useConnectionState"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"

type RowStatus = "ok" | "needs-action" | "unknown"

type RecoveryRow = {
  id: string
  label: string
  status: RowStatus
  description: string
  actionLabel: string
  to: string
}

const statusClassName: Record<RowStatus, string> = {
  ok: "border-success/40 bg-success/5",
  "needs-action": "border-warn/50 bg-warn/5",
  unknown: "border-border bg-surface"
}

export const SetupRecoverySettings = () => {
  const { t } = useTranslation("settings")
  const navigate = useNavigate()
  const { restartOnboarding } = useConnectionActions()
  const connection = useConnectionState()
  const { uxState, errorKind, isChecking } = useConnectionUxState()
  const { selectedModel, selectedModelIsLoading } = useSelectedModel()
  const [storedEmbeddingModel] = useStorage<string | null>("defaultEmbeddingModel", null)

  const rows: RecoveryRow[] = [
    {
      id: "server",
      label: t("setupRecovery.server.label", "Server connection"),
      status: connection.isConnected ? "ok" : "needs-action",
      description: connection.serverUrl || t("setupRecovery.server.missing", "No server URL configured."),
      actionLabel: t("setupRecovery.server.action", "Edit server"),
      to: "/settings/tldw"
    },
    {
      id: "auth",
      label: t("setupRecovery.auth.label", "Authentication"),
      status: uxState === "error_auth" || errorKind === "auth" ? "needs-action" : isChecking ? "unknown" : "ok",
      description: uxState === "error_auth" || errorKind === "auth"
        ? t("setupRecovery.auth.failed", "Authentication needs attention.")
        : t("setupRecovery.auth.ready", "No auth issue detected from the current connection state."),
      actionLabel: t("setupRecovery.auth.action", "Fix auth"),
      to: "/settings/tldw"
    },
    {
      id: "providers",
      label: t("setupRecovery.providers.label", "Provider keys"),
      status: "unknown",
      description: t("setupRecovery.providers.unknown", "Provider key status is checked on the provider keys page."),
      actionLabel: t("setupRecovery.providers.action", "Provider keys"),
      to: "/settings/provider-keys"
    },
    {
      id: "chat-model",
      label: t("setupRecovery.chatModel.label", "Default chat model"),
      status: selectedModel ? "ok" : selectedModelIsLoading ? "unknown" : "needs-action",
      description: selectedModel || t("setupRecovery.chatModel.missing", "No chat model selected."),
      actionLabel: t("setupRecovery.chatModel.action", "Model settings"),
      to: "/settings/model"
    },
    {
      id: "embedding-model",
      label: t("setupRecovery.embeddingModel.label", "Embedding model"),
      status: storedEmbeddingModel ? "ok" : "unknown",
      description: storedEmbeddingModel || t("setupRecovery.embeddingModel.unknown", "Embedding default not checked yet."),
      actionLabel: t("setupRecovery.embeddingModel.action", "Embedding defaults"),
      to: "/settings/rag"
    },
    {
      id: "health",
      label: t("setupRecovery.health.label", "Health checks"),
      status: connection.knowledgeStatus === "offline" ? "needs-action" : "unknown",
      description: t("setupRecovery.health.description", "Open diagnostics for detailed server and subsystem checks."),
      actionLabel: t("setupRecovery.health.action", "Full diagnostics"),
      to: "/settings/health"
    }
  ]

  return (
    <div className="space-y-6 text-sm">
      <div>
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("setupRecovery.title", "Setup & Recovery")}
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          {t("setupRecovery.subtitle", "Check the basics first, then jump to the page that owns the fix.")}
        </p>
        <div className="border-b border-border mt-3" />
      </div>
      <div className="space-y-3">
        {rows.map((row) => (
          <div key={row.id} className={`rounded-md border p-4 ${statusClassName[row.status]}`}>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="font-medium text-text">{row.label}</div>
                <p className="mt-1 text-xs text-text-muted">{row.description}</p>
              </div>
              <Link className="inline-flex rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2" to={row.to}>
                {row.actionLabel}
              </Link>
            </div>
          </div>
        ))}
      </div>
      <Button
        onClick={() => {
          Modal.confirm({
            title: t("setupRecovery.restartOnboarding.confirmTitle", "Restart onboarding?"),
            content: t("setupRecovery.restartOnboarding.confirmMessage", "This resets setup state and returns you to onboarding."),
            onOk: async () => {
              await restartOnboarding()
              navigate("/")
            }
          })
        }}
      >
        {t("setupRecovery.restartOnboarding.button", "Restart onboarding")}
      </Button>
    </div>
  )
}

export default SetupRecoverySettings
```

Keep `general-settings.tsx` as a compatibility alias after moving content:

```tsx
export { PreferencesSettings as GeneralSettings } from "./preferences-settings"
export { PreferencesSettings as default } from "./preferences-settings"
```

- [ ] **Step 4: Add route wrappers**

Modify `apps/packages/ui/src/routes/option-settings-route-registry.tsx`:

```tsx
const OptionSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/setup-recovery-settings"),
  "SetupRecoverySettings"
)
const OptionPreferencesSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/preferences-settings"),
  "PreferencesSettings"
)
```

Add the route immediately after `/settings`:

```tsx
{
  kind: "options",
  path: "/settings/preferences",
  element: <OptionPreferencesSettings />
},
```

Modify `apps/tldw-frontend/pages/settings/index.tsx` to load `SetupRecoverySettings`.

Create `apps/tldw-frontend/pages/settings/preferences.tsx`:

```tsx
import dynamic from "next/dynamic"

export default dynamic(async () => {
  const { SettingsRoute } = await import("@/routes/settings-route")
  const mod = await import("@/components/Option/Settings/preferences-settings")
  const Component = mod.PreferencesSettings
  const Page = () => (
    <SettingsRoute>
      <Component />
    </SettingsRoute>
  )
  return { default: Page }
}, { ssr: false })
```

- [ ] **Step 5: Run tests to verify route split passes**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Settings/__tests__/PreferencesSettings.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/SetupRecoverySettings.test.tsx
```

Expected: pass.

- [ ] **Step 6: Commit route split**

```bash
git add apps/packages/ui/src/components/Option/Settings apps/packages/ui/src/routes/option-settings-route-registry.tsx apps/tldw-frontend/pages/settings/index.tsx apps/tldw-frontend/pages/settings/preferences.tsx
git commit -m "feat: split settings recovery and preferences pages"
```

## Task 2: Move Ambiguous General Controls To Their Owners

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Settings/preferences-settings.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/ui-customization.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/QuickIngestSettings.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/general-settings.tsx`
- Test: `apps/packages/ui/src/components/Option/Settings/__tests__/PreferencesSettings.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Settings/__tests__/GeneralSettings.test.tsx`

- [ ] **Step 1: Update tests for final ownership**

In `PreferencesSettings.test.tsx`, assert:

```tsx
expect(screen.getByText("General preferences")).toBeInTheDocument()
expect(screen.getByText("Web search defaults")).toBeInTheDocument()
expect(screen.queryByText("Browser Extension Available")).not.toBeInTheDocument()
expect(screen.queryByText("Default OCR language")).not.toBeInTheDocument()
expect(screen.queryByText("Theme picker")).not.toBeInTheDocument()
```

In `GeneralSettings.test.tsx`, replace direct rendering expectations with compatibility expectations:

```tsx
import { GeneralSettings } from "../general-settings"

it("keeps the legacy export pointing at preferences", () => {
  render(
    <MemoryRouter>
      <GeneralSettings />
    </MemoryRouter>
  )

  expect(screen.getByText("General preferences")).toBeInTheDocument()
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Settings/__tests__/PreferencesSettings.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/GeneralSettings.test.tsx
```

Expected: fail until controls are moved.

- [ ] **Step 3: Finish `PreferencesSettings` markup**

Move from old `GeneralSettings` into `PreferencesSettings`:

- language selector
- send notification after indexing
- check Ollama status
- onboarding auto-finish
- reset tutorial progress
- persona feature visibility profile
- `SearchModeSettings`

Do not include:

- connection intro/card
- restart onboarding
- extension promotion
- OCR asset controls
- `ThemePicker`
- `SystemSettings`

- [ ] **Step 4: Move theme and UI basics to `UiCustomizationSettings`**

Add imports:

```tsx
import { ThemePicker } from "@/components/Common/Settings/ThemePicker"
import { SystemSettings } from "@/components/Option/Settings/system-settings"
```

Render after shortcut controls:

```tsx
<div className="space-y-4">
  <ThemePicker />
  <SystemSettings />
</div>
```

- [ ] **Step 5: Move OCR asset controls to `QuickIngestSettings`**

Add imports:

```tsx
import { getDefaultOcrLanguage, ocrLanguages } from "@/data/ocr-language"
```

Add storage near existing preset storage:

```tsx
const [defaultOCRLanguage, setDefaultOCRLanguage] = useStorage(
  "defaultOCRLanguage",
  getDefaultOcrLanguage()
)
const [enableOcrAssets, setEnableOcrAssets] = useStorage("enableOcrAssets", false)
```

Add a compact section above preset cards:

```tsx
<section className="rounded-md border border-border bg-surface p-4 space-y-3">
  <div>
    <h3 className="text-sm font-semibold text-text">
      {t("quickIngestSettings.ocrDefaults.title", "OCR defaults")}
    </h3>
    <p className="mt-1 text-xs text-text-muted">
      {t("quickIngestSettings.ocrDefaults.description", "Used by ingest flows that extract text from images or scanned documents.")}
    </p>
  </div>
  <ToggleRow
    label={t("generalSettings.settings.enableOcrAssets.label", "Enable OCR assets")}
    checked={enableOcrAssets}
    onChange={setEnableOcrAssets}
  />
  <div className="flex flex-wrap items-center justify-between gap-3">
    <span className="text-text">
      {t("generalSettings.settings.ocrLanguage.label", "Default OCR language")}
    </span>
    <Select
      aria-label={t("generalSettings.settings.ocrLanguage.label", "Default OCR language")}
      showSearch
      className="w-full sm:w-[240px]"
      options={ocrLanguages}
      value={defaultOCRLanguage}
      onChange={setDefaultOCRLanguage}
    />
  </div>
</section>
```

- [ ] **Step 6: Run tests**

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Settings/__tests__/PreferencesSettings.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/GeneralSettings.test.tsx
```

Expected: pass.

- [ ] **Step 7: Commit ownership split**

```bash
git add apps/packages/ui/src/components/Option/Settings
git commit -m "refactor: move settings controls to owned pages"
```

## Task 3: Settings Nav Hubs, Sections, And Active Route Matching

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav.ts`
- Modify: `apps/packages/ui/src/components/Layouts/settings-active-route.ts`
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-active-route.test.ts`
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/settings-nav.guardian.test.ts`

- [ ] **Step 1: Write failing active-route tests**

Add cases to `settings-layout-active-route.test.ts`:

```ts
it("does not treat /settings as active for child routes", () => {
  expect(isSettingsNavItemActive("/settings/preferences", "/settings")).toBe(false)
})

it("canonicalizes aliases before matching", () => {
  expect(isSettingsNavItemActive("/settings/image-gen", "/settings/image-generation", ["/settings/image-gen"])).toBe(true)
})

it("resolves one canonical item for an alias path", () => {
  const matched = resolveCurrentSettingsNavItem("/settings/image-gen", [
    { items: [{ to: "/settings/image-generation", aliases: ["/settings/image-gen"], labelToken: "x", labelDefault: "Image Generation", icon: (() => null) as any }] }
  ])
  expect(matched?.to).toBe("/settings/image-generation")
})
```

- [ ] **Step 2: Write failing nav grouping tests**

Update `settings-nav.guardian.test.ts` to expect:

```ts
expect(groups.map((group) => group.key)).toEqual([
  "setupRecovery",
  "preferencesWorkflow",
  "adminDiagnostics"
])
expect(pathsByGroup.setupRecovery).toEqual(
  expect.arrayContaining([
    "/settings",
    "/settings/tldw",
    "/settings/provider-keys",
    "/settings/model",
    "/settings/health"
  ])
)
expect(pathsByGroup.preferencesWorkflow).toEqual(
  expect.arrayContaining([
    "/settings/preferences",
    "/settings/ui",
    "/settings/chat",
    "/settings/rag",
    "/settings/quick-ingest"
  ])
)
expect(pathsByGroup.adminDiagnostics).toEqual(
  expect.arrayContaining([
    "/settings/data",
    "/settings/processed",
    "/settings/mcp-hub",
    "/settings/about"
  ])
)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Layouts/__tests__/settings-layout-active-route.test.ts ../packages/ui/src/components/Layouts/__tests__/settings-nav.guardian.test.ts
```

Expected: fail on `/settings` prefix matching and old groups.

- [ ] **Step 4: Update nav config with menu fields only**

Rename local types to avoid the route ownership framing:

```ts
export type SettingsHubKey =
  | "setupRecovery"
  | "preferencesWorkflow"
  | "adminDiagnostics"

export type SettingsSectionKey =
  | "setupConnection"
  | "setupModels"
  | "setupHealth"
  | "personalPreferences"
  | "interfaceCustomization"
  | "chatKnowledge"
  | "creationMedia"
  | "data"
  | "safety"
  | "advanced"

export type SettingsNavConfigItem = {
  path: string
  hub: SettingsHubKey
  section: SettingsSectionKey
  labelToken: string
  labelDefault: string
  icon: LucideIcon
  order: number
  aliases?: string[]
  beta?: boolean
}
```

Add `/settings/preferences` and `aliases: ["/settings/image-gen"]` to `/settings/image-generation`.

- [ ] **Step 5: Update `settings-nav.ts` to emit sections**

Keep `items` flattened for older tests and add `sections` for layout:

```ts
export type SettingsNavItem = {
  to: string
  icon: LucideIcon
  labelToken: string
  labelDefault: string
  aliases?: string[]
  beta?: boolean
}

export type SettingsNavSection = {
  key: string
  titleToken: string
  titleDefault: string
  items: SettingsNavItem[]
}

export type SettingsNavGroup = {
  key: SettingsHubKey
  titleToken: string
  titleDefault: string
  sections: SettingsNavSection[]
  items: SettingsNavItem[]
}
```

Do not import app route registries here.

- [ ] **Step 6: Update `settings-active-route.ts`**

Change the active helper signature:

```ts
export const isSettingsNavItemActive = (
  currentPathname: string,
  navPath: string,
  aliases: string[] = []
): boolean => {
  const current = normalizePathname(currentPathname)
  const target = normalizePathname(navPath)
  const aliasTargets = aliases.map(normalizePathname)

  if (current === target || aliasTargets.includes(current)) return true
  if (target === "/settings") return current === "/settings"
  return current.startsWith(`${target}/`)
}
```

Update `resolveCurrentSettingsNavItem()` to pass `item.aliases`.

- [ ] **Step 7: Run unit tests**

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Layouts/__tests__/settings-layout-active-route.test.ts ../packages/ui/src/components/Layouts/__tests__/settings-nav.guardian.test.ts
```

Expected: pass.

- [ ] **Step 8: Commit nav config and matching**

```bash
git add apps/packages/ui/src/components/Layouts/settings-nav-config.ts apps/packages/ui/src/components/Layouts/settings-nav.ts apps/packages/ui/src/components/Layouts/settings-active-route.ts apps/packages/ui/src/components/Layouts/__tests__
git commit -m "fix: group settings nav into hubs with exact active routes"
```

## Task 4: Settings Layout Desktop And Mobile

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-filter.test.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/settings.json`

- [ ] **Step 1: Update layout tests**

In `settings-layout-filter.test.tsx`, mock groups with sections and assert search finds items across hubs:

```tsx
expect(screen.getByTestId("settings-nav-filter")).toBeInTheDocument()
await user.type(screen.getByTestId("settings-nav-filter"), "image")
expect(screen.getByRole("link", { name: "Image Generation" })).toBeVisible()
```

In `settings-layout-focus-order.test.tsx`, replace current-section banner expectations:

```tsx
expect(screen.queryByTestId("settings-current-section")).not.toBeInTheDocument()
expect(screen.getByTestId("settings-hub-selector")).toBeInTheDocument()
expect(screen.getByTestId("settings-page-selector")).toBeInTheDocument()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Layouts/__tests__/settings-layout-filter.test.tsx ../packages/ui/src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx
```

Expected: fail because layout still renders the old sidebar groups and current-section banner.

- [ ] **Step 3: Render hub rail and sectioned page list**

In `SettingsOptionLayout.tsx`:

- derive current hub from `currentNavItem`
- keep selected hub in state, syncing to current hub on pathname changes
- render desktop hub buttons
- render sectioned links for selected hub
- render a global filter when the user searches
- remove `settings-current-section`

Use existing `Link`, `useLocation`, `useNavigate`, `BetaTag`, and icon rendering. No new component library.

- [ ] **Step 4: Add mobile selectors**

Add two native controls before the content on small screens:

```tsx
<div className="lg:hidden space-y-3 px-4 pt-4">
  <select
    data-testid="settings-hub-selector"
    aria-label={t("settings:navigation.hubSelector", "Settings hub")}
    value={selectedHubKey}
    onChange={(event) => setSelectedHubKey(event.target.value as SettingsHubKey)}
    className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
  >
    {settingsNavGroups.map((group) => (
      <option key={group.key} value={group.key}>
        {t(group.titleToken, group.titleDefault)}
      </option>
    ))}
  </select>
  <select
    data-testid="settings-page-selector"
    aria-label={t("settings:navigation.pageSelector", "Settings page")}
    value={currentNavItem?.to ?? ""}
    onChange={(event) => navigate(event.target.value)}
    className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
  >
    {selectedHub.items.map((item) => (
      <option key={item.to} value={item.to}>
        {t(item.labelToken, item.labelDefault)}
      </option>
    ))}
  </select>
</div>
```

Hide the desktop aside on mobile with `hidden lg:block`.

- [ ] **Step 5: Add English locale keys with fallbacks**

Add only keys used directly by the layout or new pages. Example:

```json
"navigation": {
  "setupRecovery": "Setup & Recovery",
  "preferencesWorkflow": "Preferences & Workflow",
  "adminDiagnostics": "Admin & Diagnostics",
  "hubSelector": "Settings hub",
  "pageSelector": "Settings page"
}
```

Keep `t(token, defaultValue)` for new labels so missing non-English translations fall back to readable text.

- [ ] **Step 6: Run layout tests**

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Layouts/__tests__/settings-layout-filter.test.tsx ../packages/ui/src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx
```

Expected: pass.

- [ ] **Step 7: Commit layout changes**

```bash
git add apps/packages/ui/src/components/Layouts apps/packages/ui/src/assets/locale/en/settings.json
git commit -m "feat: render settings hubs and mobile selectors"
```

## Task 5: Route Wrappers, Data Page, And Search/Page Inventories

**Files:**
- Modify: `apps/packages/ui/src/routes/option-settings-health.tsx`
- Modify: `apps/packages/ui/src/routes/option-settings-processed.tsx`
- Create: `apps/tldw-frontend/pages/settings/data.tsx`
- Modify: `apps/tldw-frontend/e2e/page-mapping.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: `apps/tldw-frontend/e2e/utils/page-objects/SettingsPage.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-4-admin/settings-full.spec.ts`

- [ ] **Step 1: Update route smoke tests first**

In `SettingsPage.gotoSection()`, add:

```ts
| "preferences"
| "data"
| "processed"
```

Move `health` and `processed` into the shell list in `settings-core.spec.ts`
and `settings-full.spec.ts`. Leave `/settings/mcp-hub` behavior unchanged in
this slice unless existing route tests fail for that page.

Add assertions for one active nav item:

```ts
const activeLinks = authedPage.locator('[data-testid^="settings-nav-link-"][aria-current="page"]')
await expect(activeLinks).toHaveCount(1)
```

- [ ] **Step 2: Run targeted Playwright test to verify failure**

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/tier-1-critical/settings-core.spec.ts --project=tier-1 --reporter=line
```

Expected: fail until wrappers and route lists are updated.

- [ ] **Step 3: Wrap standalone route files**

Modify `option-settings-health.tsx`:

```tsx
import HealthStatus from "@/components/Option/Settings/health-status"
import { SettingsRoute } from "./settings-route"

export default function OptionHealthStatus() {
  return (
    <SettingsRoute>
      <HealthStatus />
    </SettingsRoute>
  )
}
```

Modify `option-settings-processed.tsx` similarly:

```tsx
import OptionProcessed from "@/components/Option/Processed"
import { SettingsRoute } from "./settings-route"

export default function OptionProcessedRoute() {
  return (
    <SettingsRoute>
      <OptionProcessed />
    </SettingsRoute>
  )
}
```

- [ ] **Step 4: Add missing Next data page**

Create `apps/tldw-frontend/pages/settings/data.tsx`:

```tsx
import dynamic from "next/dynamic"

export default dynamic(async () => {
  const { SettingsRoute } = await import("@/routes/settings-route")
  const mod = await import("@/components/Option/Settings/data-management")
  const Component = mod.DataManagementSettings
  const Page = () => (
    <SettingsRoute>
      <Component />
    </SettingsRoute>
  )
  return { default: Page }
}, { ssr: false })
```

- [ ] **Step 5: Update inventories**

Add:

```ts
{ path: "/settings/preferences", name: "Preferences Settings", category: "settings" },
{ path: "/settings/ui", name: "UI Customization Settings", category: "settings" },
{ path: "/settings/data", name: "Data Management Settings", category: "settings" },
```

Update `General Settings` mapping to `Setup & Recovery`, shared component `SetupRecoverySettings`, and add a separate `Preferences Settings` mapping.

- [ ] **Step 6: Run route tests**

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/tier-1-critical/settings-core.spec.ts --project=tier-1 --reporter=line
bunx playwright test e2e/workflows/tier-4-admin/settings-full.spec.ts --project=tier-4 --reporter=line
```

Expected: pass.

- [ ] **Step 7: Commit route wrappers and inventories**

```bash
git add apps/packages/ui/src/routes/option-settings-health.tsx apps/packages/ui/src/routes/option-settings-processed.tsx apps/tldw-frontend/pages/settings/data.tsx apps/tldw-frontend/e2e/page-mapping.ts apps/tldw-frontend/e2e/smoke/page-inventory.ts apps/tldw-frontend/e2e/utils/page-objects/SettingsPage.ts apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts apps/tldw-frontend/e2e/workflows/tier-4-admin/settings-full.spec.ts
git commit -m "fix: keep settings routes in the shared shell"
```

## Task 6: Sidepanel Settings Shortcut Panel

**Files:**
- Modify: `apps/packages/ui/src/components/Sidepanel/Settings/body.tsx`
- Create: `apps/packages/ui/src/components/Sidepanel/Settings/__tests__/body.test.tsx`

- [ ] **Step 1: Write failing sidepanel test**

Create `body.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"

import { SettingsBody } from "../body"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({ isConnected: true, serverUrl: "http://127.0.0.1:8000" }),
  useConnectionUxState: () => ({ uxState: "connected_ok" })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, fallback: unknown) => [fallback, vi.fn()]
}))

describe("sidepanel SettingsBody", () => {
  it("renders shortcut settings instead of the full settings app", () => {
    render(
      <MemoryRouter>
        <SettingsBody />
      </MemoryRouter>
    )

    expect(screen.getByRole("heading", { name: "Quick settings" })).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /open full settings/i })).toHaveAttribute("href", "/settings")
    expect(screen.getByRole("link", { name: /diagnostics/i })).toHaveAttribute("href", "/settings/health")
    expect(screen.queryByText("RAG prompts")).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Sidepanel/Settings/__tests__/body.test.tsx
```

Expected: fail because sidepanel still renders the broad settings page.

- [ ] **Step 3: Replace body with shortcut panel**

Use existing storage and connection hooks only. Keep links simple:

```tsx
import { Switch } from "antd"
import { Link } from "react-router-dom"

export const SettingsBody = () => {
  const { t } = useTranslation("settings")
  const { isConnected, serverUrl } = useConnectionState()
  const [chatWithWebsiteEmbedding, setChatWithWebsiteEmbedding] = useStorage("chatWithWebsiteEmbedding", false)
  const [copilotResumeLastChat, setCopilotResumeLastChat] = useStorage("copilotResumeLastChat", false)

  return (
    <div className="mx-auto flex w-full max-w-md flex-col gap-4 p-4">
      <section className="rounded-md border border-border bg-surface p-4">
        <h1 className="text-base font-semibold text-text">
          {t("sidepanelSettings.quickTitle", "Quick settings")}
        </h1>
        <p className="mt-1 text-xs text-text-muted">
          {serverUrl || t("sidepanelSettings.noServer", "No server configured")}
        </p>
      </section>
      <div className="rounded-md border border-border bg-surface p-4 text-sm">
        <div className="flex items-center justify-between gap-3">
          <span className="text-text">{t("sidepanelSettings.connection", "Connection")}</span>
          <span className={isConnected ? "text-success" : "text-warn"}>
            {isConnected
              ? t("sidepanelSettings.connected", "Connected")
              : t("sidepanelSettings.needsSetup", "Needs setup")}
          </span>
        </div>
      </div>
      <label className="flex items-center justify-between rounded-md border border-border bg-surface p-4 text-sm">
        <span className="text-text">{t("sidepanelSettings.resumeLastChat", "Resume last chat")}</span>
        <Switch
          checked={copilotResumeLastChat}
          onChange={setCopilotResumeLastChat}
          aria-label={t("sidepanelSettings.resumeLastChat", "Resume last chat")}
        />
      </label>
      <label className="flex items-center justify-between rounded-md border border-border bg-surface p-4 text-sm">
        <span className="text-text">{t("sidepanelSettings.pageContext", "Use page context")}</span>
        <Switch
          checked={chatWithWebsiteEmbedding}
          onChange={setChatWithWebsiteEmbedding}
          aria-label={t("sidepanelSettings.pageContext", "Use page context")}
        />
      </label>
      <Link to="/settings" className="rounded-md border border-border px-3 py-2 text-sm text-text">
        {t("sidepanelSettings.openFull", "Open full settings")}
      </Link>
      <Link to="/settings/health" className="rounded-md border border-border px-3 py-2 text-sm text-text">
        {t("sidepanelSettings.diagnostics", "Diagnostics")}
      </Link>
    </div>
  )
}
```

Do not fetch embedding models, RAG prompts, totals, or provider lists in the sidepanel settings page.

- [ ] **Step 4: Run test**

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Sidepanel/Settings/__tests__/body.test.tsx
```

Expected: pass.

- [ ] **Step 5: Commit sidepanel cleanup**

```bash
git add apps/packages/ui/src/components/Sidepanel/Settings/body.tsx apps/packages/ui/src/components/Sidepanel/Settings/__tests__/body.test.tsx
git commit -m "refactor: simplify sidepanel settings"
```

## Task 7: E2E Interaction And Visual QA

**Files:**
- Modify: `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/stage6-interaction-stage2.spec.ts`
- Modify if needed: `apps/tldw-frontend/e2e/smoke/stage4-accessibility-controls.spec.ts`

- [ ] **Step 1: Update wayfinding smoke assertions**

In `all-pages.spec.ts`, stop expecting `settings-current-section`. Assert one active link:

```ts
await expect(page.getByTestId("settings-navigation")).toBeVisible()
await expect(
  page.locator('[data-testid^="settings-nav-link-"][aria-current="page"]')
).toHaveCount(1)
```

- [ ] **Step 2: Add alias and mobile checks**

In `stage6-interaction-stage2.spec.ts`, add:

```ts
await page.goto("/settings/image-gen", { waitUntil: "domcontentloaded", timeout: LOAD_TIMEOUT })
await waitForAppShell(page, LOAD_TIMEOUT)
await expect(page.getByTestId("settings-nav-link--settings-image-generation")).toHaveAttribute("aria-current", "page")
await expect(page.locator('[data-testid^="settings-nav-link-"][aria-current="page"]')).toHaveCount(1)
```

Add a mobile viewport check:

```ts
await page.setViewportSize({ width: 390, height: 844 })
await page.goto("/settings", { waitUntil: "domcontentloaded", timeout: LOAD_TIMEOUT })
await expect(page.getByRole("heading", { name: /setup & recovery/i })).toBeVisible({ timeout: LOAD_TIMEOUT })
await expect(page.getByTestId("settings-hub-selector")).toBeVisible()
```

- [ ] **Step 3: Run focused Playwright smoke**

```bash
cd apps/tldw-frontend
bunx playwright test e2e/smoke/stage6-interaction-stage2.spec.ts --project=chromium --reporter=line
```

Expected: pass.

- [ ] **Step 4: Run visual manual check with local dev server**

Start dev server if one is not already running:

```bash
cd apps/tldw-frontend
bun run dev -- -p 8080
```

Check:

- desktop `/settings`
- desktop `/settings/preferences`
- desktop `/settings/ui`
- desktop `/settings/health`
- mobile `/settings`
- extension sidepanel `/settings`

Stop the server after screenshots/checks if this task started it.

- [ ] **Step 5: Commit E2E updates**

```bash
git add apps/tldw-frontend/e2e/smoke/all-pages.spec.ts apps/tldw-frontend/e2e/smoke/stage6-interaction-stage2.spec.ts apps/tldw-frontend/e2e/smoke/stage4-accessibility-controls.spec.ts
git commit -m "test: cover settings wayfinding redesign"
```

## Task 8: Final Verification

**Files:**
- Update: `backlog/tasks/task-12166 - Plan-settings-IA-recovery-preferences-and-UI-implementation.md` only if implementation happens under this task

- [ ] **Step 1: Run focused unit tests**

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Settings/__tests__/PreferencesSettings.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/SetupRecoverySettings.test.tsx ../packages/ui/src/components/Layouts/__tests__/settings-layout-active-route.test.ts ../packages/ui/src/components/Layouts/__tests__/settings-layout-filter.test.tsx ../packages/ui/src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx ../packages/ui/src/components/Layouts/__tests__/settings-nav.guardian.test.ts ../packages/ui/src/components/Sidepanel/Settings/__tests__/body.test.tsx
```

Expected: pass.

- [ ] **Step 2: Run focused Playwright tests**

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/tier-1-critical/settings-core.spec.ts --project=tier-1 --reporter=line
bunx playwright test e2e/workflows/tier-4-admin/settings-full.spec.ts --project=tier-4 --reporter=line
bunx playwright test e2e/smoke/stage6-interaction-stage2.spec.ts --project=chromium --reporter=line
```

Expected: pass, or record server/environment skips clearly.

- [ ] **Step 3: Typecheck touched frontend**

```bash
cd apps/tldw-frontend
bun run typecheck
```

Expected: pass.

- [ ] **Step 4: Run Bandit scope check**

This is frontend-only work. If no Python files changed, record: `Bandit skipped: no Python touched.`

If Python files are touched unexpectedly:

```bash
source .venv/bin/activate
python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_settings_ia.json
```

- [ ] **Step 5: Final self-review**

Check:

- no generic settings route contract added
- no full diagnostics/provider/model refresh on `/settings`
- `/settings` exact match does not activate child routes
- `/settings/preferences` and `/settings/ui` are distinct
- health and processed render inside `SettingsLayout`
- sidepanel settings is a shortcut panel only
- no visible translation keys on new settings surfaces

- [ ] **Step 6: Update Backlog and commit final verification note**

Update `TASK-12166` with touched files and verification results. Then commit if Backlog changed:

```bash
git add backlog/tasks/task-12166\ -\ Plan-settings-IA-recovery-preferences-and-UI-implementation.md
git commit -m "docs: record settings redesign verification"
```
