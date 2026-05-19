# WebUI Settings Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make settings task-led, fix leaked settings labels, put configured provider/model setup before the full catalog, and separate data/destructive actions from routine preferences.

**Architecture:** Start with tests for visible labels, settings grouping, model ordering, and destructive-action separation. Reuse the existing settings route registry, `SettingsLayout`, `ProviderKeysSettings`, `ModelsBody`, `AvailableModelsList`, and `SystemSettings` components before adding focused utilities.

**Tech Stack:** React, TypeScript, React Router, i18next locale assets, Ant Design controls, Tailwind utility classes, Vitest component/unit tests, Playwright settings workflow tests.

---

## Source Documents

- Backlog: `TASK-418.2`
- Parent plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- Program spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Audit report: `Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md`
- Prior dependency plan: `Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md`

## Findings Closed Or Supported

- F5: Model settings overwhelm users with full catalog before usable configuration.
- F7: Settings navigation leaks internal translation key.
- F11: Settings is too broad for one flat route experience.
- F16: General settings mixes routine preferences with high-risk system actions.
- F15 support: Settings routes must keep one clear page heading after grouping changes.
- F2 support: Settings navigation and model controls must remain mobile-safe after WP4.

## Route Scope

| Route | Outcome |
| --- | --- |
| `/settings` | Routine preferences remain reachable, but destructive data management moves into an explicit Data section or a separate data-management subroute. |
| `/settings/tldw` | Server and auth setup remains the primary connection task. |
| `/settings/provider-keys` | Nav label renders as user-facing copy, not `settings:providerKeys.navTitle`. |
| `/settings/model` | Configured/usable providers, default provider, and default model appear before full catalog browsing. |
| `/login` | Settings links from auth recovery paths continue to point at the correct setup route for self-hosted mode. |
| `/privileges` | Existing redirect to `/settings` remains valid until WP1 changes route policy. |
| `/prompts` | Prompt Library and Prompt Studio route relationship stays intact. |
| `/prompt-studio` | Existing redirect to `/prompts?tab=studio` remains intact. |
| Settings subroutes | Nav grouping stays task-led and search/filter still finds every reachable settings page. |

## Out Of Scope

- No provider API redesign unless the existing model/provider status payload cannot represent configured-first ordering.
- No backend API changes in the first implementation attempt.
- No settings visual redesign or new design system.
- No route renames beyond labels and aliases already approved by WP1.
- No removal of advanced model controls or full model catalog access.
- No broad mobile shell changes beyond preserving the WP4 no-overflow contract.

## Current Code Evidence

- `apps/packages/ui/src/components/Layouts/settings-nav-config.ts` defines `/settings/provider-keys` with `labelToken: "settings:providerKeys.navTitle"`.
- `apps/packages/ui/src/assets/locale/en/settings.json` contains `navigation` and `manageModels` keys but no `providerKeys` key today.
- `apps/packages/ui/src/components/Layouts/settings-nav.ts` builds grouped nav from `SETTINGS_ROUTE_NAV_ITEMS` and capability checks.
- `apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx` renders grouped settings navigation, a filter input, current-section summary, and beta badge controls.
- `apps/packages/ui/src/components/Option/Settings/general-settings.tsx` mixes routine preferences, onboarding reset, tutorial reset, persona selection, OCR settings, theme/search settings, and `SystemSettings`.
- `apps/packages/ui/src/components/Option/Settings/system-settings.tsx` contains data import/export, Firefox sync, chat background, storage error handling, and full reset with typed confirmation.
- `apps/packages/ui/src/components/Option/Settings/ProviderKeysSettings.tsx` already owns user-managed provider key CRUD and BYOK-unavailable state.
- `apps/packages/ui/src/components/Option/Settings/model-settings.tsx` owns the saved generation parameter defaults and existing advanced controls.
- `apps/packages/ui/src/components/Option/Models/index.tsx` already has a `Set your defaults` panel and OpenAI OAuth controls before `AvailableModelsList`.
- `apps/packages/ui/src/components/Option/Models/AvailableModelsList.tsx` renders the full provider/model catalog from `tldwClient.getModelsMetadata()`.
- `apps/packages/ui/src/components/Option/Models/modelsDisplayUtils.ts` currently only formats the refresh timestamp, making it a low-risk home for pure model display ordering helpers.

## Proposed Settings Grouping Contract

Keep route paths stable. Change the grouping and labels around the existing routes:

| Group | Routes |
| --- | --- |
| Connect | `/settings/tldw`, `/settings/provider-keys`, `/settings/health` |
| AI and Models | `/settings/model`, `/settings/rag`, `/settings/speech`, `/settings/image-generation`, `/settings/mcp-hub` |
| Experience | `/settings`, `/settings/chat`, `/settings/ui`, `/settings/splash`, `/settings/quick-ingest` |
| Knowledge and Workspace | `/settings/knowledge`, `/settings/prompt`, `/settings/prompt-studio`, `/settings/chatbooks`, `/settings/characters`, `/settings/world-books`, `/settings/chat-dictionaries`, `/settings/share` |
| Safety and Admin | `/settings/guardian`, `/settings/family-guardrails`, `/settings/evaluations` |
| Data Management | Data/export/import/reset surface added by this slice, either as `/settings/data` or as an explicit section inside `/settings` with strong local landmarking. |
| About | `/settings/about` |

If route count makes a new `/settings/data` route lower risk than an in-page section, add it through the existing `createSettingsRoute` pattern. If that route is added, update route metadata, settings nav config, settings page-object navigation, and smoke tests in the same commit.

## Implementation Tasks

### Task 1: Add Settings Label And Grouping Regression Tests

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-filter.test.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-active-route.test.ts`
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/settings-nav.guardian.test.ts`
- Test target: `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- Test target: `apps/packages/ui/src/components/Layouts/settings-nav.ts`

- [ ] **Step 1: Add a nav-label contract test**

Add a test that flattens `getSettingsNavGroups(undefined)` and asserts that no item uses a label token without a user-facing locale value in `apps/packages/ui/src/assets/locale/en/settings.json`.

Expected pre-fix result: FAIL for `settings:providerKeys.navTitle`.

- [ ] **Step 2: Add a visible navigation regression test**

Extend `settings-layout-filter.test.tsx` or create `settings-layout-labels.test.tsx` so the rendered nav never contains dotted i18n keys.

```tsx
expect(screen.queryByText("settings:providerKeys.navTitle")).not.toBeInTheDocument();
expect(screen.getByRole("link", { name: /provider keys/i })).toBeVisible();
```

- [ ] **Step 3: Add a task-led group ordering test**

Add a unit test for the expected group order and representative route placement:

- Connect contains `/settings/tldw`, `/settings/provider-keys`, and `/settings/health`.
- AI and Models contains `/settings/model`.
- Experience contains `/settings` and `/settings/chat`.
- Data Management contains the data/reset surface once that route or section exists.

- [ ] **Step 4: Run the failing tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Layouts/__tests__/settings-layout-filter.test.tsx src/components/Layouts/__tests__/settings-layout-active-route.test.ts src/components/Layouts/__tests__/settings-nav.guardian.test.ts
```

Expected before implementation: FAIL on missing provider key translation and any group assertions not yet implemented.

- [ ] **Step 5: Commit the failing tests**

```bash
git add apps/packages/ui/src/components/Layouts/__tests__/settings-layout-filter.test.tsx apps/packages/ui/src/components/Layouts/__tests__/settings-layout-active-route.test.ts apps/packages/ui/src/components/Layouts/__tests__/settings-nav.guardian.test.ts
git commit -m "test: capture settings nav label and grouping contracts"
```

### Task 2: Fix Provider Keys Label And Settings Group Taxonomy

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav.ts`
- Modify: `apps/packages/ui/src/assets/locale/en/settings.json`
- Modify generated or extension locale files only if the repo-local i18n workflow requires it: `apps/packages/ui/src/public/_locales/en/settings.json`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/settings-nav.guardian.test.ts`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-filter.test.tsx`

- [ ] **Step 1: Add locale entries**

Add the missing user-facing settings copy:

```json
{
  "providerKeys": {
    "navTitle": "Provider Keys"
  }
}
```

If the build uses flattened public locale files, add the matching public locale key through the existing localization generation workflow. Do not hand-edit generated locale files unless that is the repo pattern for this package.

- [ ] **Step 2: Update group keys**

Expand `NavGroupKey` to match the task-led groups:

```ts
export type NavGroupKey =
  | "connect"
  | "aiModels"
  | "experience"
  | "knowledgeWorkspace"
  | "safetyAdmin"
  | "dataManagement"
  | "about";
```

- [ ] **Step 3: Move route metadata into the new groups**

Keep every route path stable. Move only the `group`, `order`, and label token values needed to satisfy the task-led grouping contract.

- [ ] **Step 4: Update `NAV_GROUPS` titles**

Add locale-backed titles for each group and verify that nav filtering still searches visible item labels.

- [ ] **Step 5: Verify focused settings nav tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Layouts/__tests__/settings-nav.guardian.test.ts src/components/Layouts/__tests__/settings-layout-filter.test.tsx src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit settings nav grouping**

```bash
git add apps/packages/ui/src/components/Layouts/settings-nav-config.ts apps/packages/ui/src/components/Layouts/settings-nav.ts apps/packages/ui/src/assets/locale/en/settings.json apps/packages/ui/src/components/Layouts/__tests__
git commit -m "fix: group settings by user task"
```

### Task 3: Separate Routine Preferences From Data And Destructive Actions

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Settings/general-settings.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/system-settings.tsx`
- Create if route split is chosen: `apps/packages/ui/src/components/Option/Settings/data-management.tsx`
- Modify if route split is chosen: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify if route split is chosen: `apps/packages/ui/src/routes/option-settings-route-registry.tsx`
- Modify if route split is chosen: `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- Test: `apps/packages/ui/src/components/Option/Settings/__tests__/GeneralSettings.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Settings/__tests__/SystemSettings.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/settings.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts`

- [ ] **Step 1: Add tests for routine versus destructive surfaces**

Write tests that render `/settings` and assert routine settings appear without foregrounding Reset All. Then render the data-management surface and assert import, export, and reset are present with the existing typed confirmation.

Expected pre-fix result: FAIL because `GeneralSettings` currently includes `SystemSettings`, which contains import/export/reset.

- [ ] **Step 2: Choose the smallest split**

Use a new `/settings/data` route if that keeps landmarks, navigation, and tests clearer. Use an explicit in-page section only if route registry changes create more risk than the route adds.

- [ ] **Step 3: Preserve reset safeguards**

Keep the existing typed confirmation in `system-settings.tsx`:

- The destructive button uses a danger style.
- The modal requires typing `RESET`.
- The reset clears Dexie and browser storage through the existing code path.
- The success path still reloads after notification.

- [ ] **Step 4: Preserve import/export recovery**

Keep import/export controls available, keep the cancelable reload notification, and keep storage quota error messaging.

- [ ] **Step 5: Verify focused settings tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Settings/__tests__/GeneralSettings.test.tsx src/components/Option/Settings/__tests__/SystemSettings.test.tsx src/components/Layouts/__tests__/settings-layout-filter.test.tsx
```

Expected: PASS. If `GeneralSettings.test.tsx` or `SystemSettings.test.tsx` does not exist, create focused tests next to the components and run those files.

- [ ] **Step 6: Verify browser settings workflow**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/settings.spec.ts e2e/workflows/tier-1-critical/settings-core.spec.ts --reporter=line
```

Expected: PASS. If `/settings/data` is added, include it in settings-core section coverage.

- [ ] **Step 7: Commit data-action split**

```bash
git add apps/packages/ui/src/components/Option/Settings apps/packages/ui/src/routes/route-registry.tsx apps/packages/ui/src/routes/option-settings-route-registry.tsx apps/packages/ui/src/components/Layouts/settings-nav-config.ts apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts
git commit -m "fix: separate settings data actions"
```

### Task 4: Add Configured-First Model Display Utilities

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Models/modelsDisplayUtils.ts`
- Modify: `apps/packages/ui/src/components/Option/Models/__tests__/modelsDisplayUtils.test.ts`
- Test target: `apps/packages/ui/src/components/Option/Models/index.tsx`
- Test target: `apps/packages/ui/src/components/Option/Models/AvailableModelsList.tsx`

- [ ] **Step 1: Define pure model display inputs**

Add focused types for provider/model display data instead of coupling tests to API response objects:

```ts
export type ModelDisplayEntry = {
  id: string;
  provider: string;
  nickname?: string | null;
  configured?: boolean;
  usable?: boolean;
  selected?: boolean;
};
```

- [ ] **Step 2: Add failing ordering tests**

Add tests for:

- Selected default model first.
- Configured and usable providers before unavailable providers.
- Auto option remains available before concrete model choices.
- Full catalog remains available after the configured-first section.

- [ ] **Step 3: Implement pure ordering helpers**

Keep helper output simple:

```ts
export const sortModelsConfiguredFirst = (models: ModelDisplayEntry[]) =>
  models.slice().sort(compareConfiguredFirst);

export const summarizeProviderReadiness = (models: ModelDisplayEntry[]) =>
  buildProviderReadiness(models);
```

The helper can infer `configured` from known provider key status only when that status is available. Do not pretend unavailable providers are configured.

- [ ] **Step 4: Verify utility tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Models/__tests__/modelsDisplayUtils.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit model utilities**

```bash
git add apps/packages/ui/src/components/Option/Models/modelsDisplayUtils.ts apps/packages/ui/src/components/Option/Models/__tests__/modelsDisplayUtils.test.ts
git commit -m "test: define configured-first model ordering"
```

### Task 5: Apply Configured-First Model Settings UI

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Models/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/Models/AvailableModelsList.tsx`
- Modify only if advanced parameter placement changes: `apps/packages/ui/src/components/Option/Settings/model-settings.tsx`
- Modify if shared status fetch is needed: `apps/packages/ui/src/components/Option/Settings/ProviderKeysSettings.tsx`
- Test: `apps/packages/ui/src/components/Option/Models/__tests__/AvailableModelsList.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Models/__tests__/modelsDisplayUtils.test.ts`
- Test: `apps/tldw-frontend/e2e/workflows/settings.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts`

- [ ] **Step 1: Add a component test for defaults before catalog**

Render `ModelsBody` or the smallest extracted presentational component and assert:

- `Set your defaults` appears before `Available models` or provider catalog cards.
- Default provider and default model controls are visible when models load.
- No providers state links to server/auth or provider key setup.

- [ ] **Step 2: Add provider readiness summary**

Add a compact summary above the full catalog:

- Server reachable or not.
- Number of configured or usable providers.
- Default provider and model state.
- Refresh status.
- OpenAI OAuth state when available.

Use existing `openaiOauthStatus`, `availableModels`, and provider key data when available. If provider key data requires BYOK and returns 403, show that account-managed keys are unavailable rather than showing a false error.

- [ ] **Step 3: Keep full catalog behind search/filter or a collapsed section**

Do not remove `AvailableModelsList`. Wrap it in a titled section that is visually lower priority than defaults and readiness. Add search/filter if the full list remains long.

- [ ] **Step 4: Preserve advanced controls**

Do not remove `ModelSettings` advanced defaults or existing OpenAI OAuth controls. If they are moved, keep keyboard order and labels stable.

- [ ] **Step 5: Verify focused model tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Models/__tests__/modelsDisplayUtils.test.ts src/components/Option/Models/__tests__/AvailableModelsList.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Verify browser model settings flow**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/settings.spec.ts e2e/workflows/tier-1-critical/settings-core.spec.ts --reporter=line
```

Expected: `/settings/model` loads, default provider/model controls are visible before full catalog browsing, and no critical console errors occur.

- [ ] **Step 7: Commit configured-first UI**

```bash
git add apps/packages/ui/src/components/Option/Models apps/tldw-frontend/e2e/workflows/settings.spec.ts apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts
git commit -m "fix: prioritize configured models in settings"
```

### Task 6: Preserve Prompt Settings And Prompt Studio Routing

**Files:**
- Modify only if tests require metadata consistency: `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- Modify only if tests require route consistency: `apps/packages/ui/src/routes/route-registry.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/settings.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts`

- [ ] **Step 1: Add route relationship assertions**

Add or extend browser tests to verify:

- `/prompt-studio` redirects to `/prompts?tab=studio`.
- `/settings/prompt-studio` remains settings for Prompt Studio defaults and health.
- `/settings/prompt` remains workspace links or prompt-related settings, not the primary prompt library.

- [ ] **Step 2: Verify no nav label collision**

Settings nav must distinguish Prompt Library, Prompt Studio settings, and `/prompts` route entry labels. Do not introduce a second primary Prompt Studio page.

- [ ] **Step 3: Run prompt/settings route tests**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/settings.spec.ts e2e/workflows/tier-1-critical/settings-core.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 4: Commit prompt route guard if changed**

```bash
git add apps/packages/ui/src/components/Layouts/settings-nav-config.ts apps/packages/ui/src/routes/route-registry.tsx apps/tldw-frontend/e2e/workflows/settings.spec.ts apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts
git commit -m "test: preserve prompt settings route intent"
```

### Task 7: Final Browser QA And Governance Update

**Files:**
- Modify: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- Modify: `backlog/tasks/task-418.2 - Plan-WebUI-settings-and-model-provider-implementation.md`

- [ ] **Step 1: Run focused unit tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Layouts/__tests__/settings-layout-filter.test.tsx src/components/Layouts/__tests__/settings-layout-active-route.test.ts src/components/Layouts/__tests__/settings-nav.guardian.test.ts src/components/Option/Models/__tests__/modelsDisplayUtils.test.ts src/components/Option/Models/__tests__/AvailableModelsList.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run browser settings workflows**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/settings.spec.ts e2e/workflows/tier-1-critical/settings-core.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 3: Run responsive gate from WP4**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/smoke/stage4-responsive-landmarks.spec.ts --reporter=line
```

Expected: `/settings` and `/settings/model` still pass heading and 390px overflow checks.

- [ ] **Step 4: Capture before and after observations**

Record in Backlog or PR notes:

- `/settings` nav group labels and filter behavior.
- `/settings/provider-keys` visible title and BYOK unavailable state.
- `/settings/model` configured-first defaults and full catalog access.
- Data/reset surface location and typed reset confirmation.
- `/prompt-studio` redirect and `/settings/prompt-studio` distinction.

- [ ] **Step 5: Update parent plan status**

Mark Task 5 complete only after label, grouping, model ordering, destructive-action separation, settings browser QA, and WP4 responsive checks pass.

- [ ] **Step 6: Final commit**

```bash
git add Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md "backlog/tasks/task-418.2 - Plan-WebUI-settings-and-model-provider-implementation.md"
git commit -m "docs: record settings model remediation completion"
```

## Verification Checklist For This Child Plan

Run before handing off this plan:

```bash
rg -n "T[O]D[O]|T[B]D|F[I]XME|\\.{3}|\\bm[a]ybe\\b|\\bpr[o]bably\\b|\\bshould c[o]nsider\\b" Docs/superpowers/plans/2026-05-17-webui-settings-models-implementation-plan.md "backlog/tasks/task-418.2 - Plan-WebUI-settings-and-model-provider-implementation.md"
rg -n "[[:blank:]]$|[^\\x00-\\x7F]" Docs/superpowers/plans/2026-05-17-webui-settings-models-implementation-plan.md "backlog/tasks/task-418.2 - Plan-WebUI-settings-and-model-provider-implementation.md"
git diff --check -- Docs/superpowers/plans/2026-05-17-webui-settings-models-implementation-plan.md "backlog/tasks/task-418.2 - Plan-WebUI-settings-and-model-provider-implementation.md"
```

Expected: no output from the `rg` checks and no diff-check errors.

## Review Notes

- Use @superpowers:test-driven-development for implementation. The risk is moving settings around while losing discoverability or hiding recovery controls.
- Use @superpowers:verification-before-completion before marking this implementation done. The acceptance criteria require browser-visible settings behavior, not only unit tests.
- Keep advanced controls available. This slice changes hierarchy and defaults-first orientation, not power-user capability.
- Do not mask missing provider configuration as a healthy state. Empty, unavailable, configured, and usable states must remain distinct.
- If provider readiness cannot be known from existing frontend data, show a conservative unknown state and link to provider key or server settings.
