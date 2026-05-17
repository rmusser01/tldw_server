# WebUI Responsive Landmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate root-page horizontal overflow at 390px and establish stable semantic page landmarks for the WebUI and extension root routes covered by WP4.

**Architecture:** Add route-level Playwright gates first, then fix shared shell constraints before route-local layout code. Keep page-specific changes limited to routes that still fail the overflow, heading, or sticky-composer assertions after the shared shell pass.

**Tech Stack:** Next.js WebUI, React, TypeScript, Tailwind utility classes, Ant Design Drawer, Playwright smoke tests, Vitest component tests, axe-core Playwright checks.

---

## Source Documents

- Backlog: `TASK-418.1`
- Parent plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- Program spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Audit report: `Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md`

## Findings Closed Or Supported

- F2: Mobile layouts for core root pages overflow instead of reflowing.
- F10 support: `/media` needs mobile-safe list-detail behavior before the deeper media workflow slice.
- F11 support: Settings needs mobile-safe navigation before the deeper settings information architecture slice.
- F13: `/chat` empty state and composer compete on narrow screens.
- F15: Root pages have inconsistent heading landmarks.

## Route Scope

Primary WP4 routes:

| Route | Required narrow-width outcome | Landmark outcome |
| --- | --- | --- |
| `/chat` | No page-level overflow at 390px; sticky composer stays visible and does not collide with mode controls. | One route `h1` of `Chat`, visible or visually hidden. |
| `/media` | Desktop split layout remains; 390px uses list-first or detail-with-back behavior instead of forcing both panes side by side. | One route `h1` of `Media Inspector` unless WP1 changes the route label first. |
| `/settings` | Navigation cannot create a 2716px horizontal strip; groups wrap, collapse, or move into a local drawer. | One route `h1` of `Settings`. |
| `/settings/model` | Same settings nav constraints as `/settings`; model controls fit within the viewport. | One route `h1` of `Model settings`. |
| `/prompts` | Prompt library/studio controls reflow; table overflow is local and intentional. | Existing `Prompts` `h1` remains unique. |
| `/workspace-playground` | Research workspace panels stack or use local scroll containers at 390px. | Existing `New Research` `h1` remains unique unless WP1 renames the page. |
| `/setup` | Setup shell remains focused and app chrome does not introduce horizontal overflow. | One route `h1` for the setup task. |
| `/sources` | Capability or source list state fits at 390px without raw technical state forcing width. | One route `h1` for Sources. |
| `/mcp-hub` | Status cards, tabs, and action rows reflow without page overflow. | One route `h1` for MCP Hub. |
| `/stt` | STT provider and upload controls fit within the page shell. | One route `h1` of `Speech to Text`. |
| `/tts` | Voice and generation controls fit; any long catalogs scroll inside their own region. | One route `h1` of `Text to Speech`. |
| `/chat-workspace` | Workspace rail and inspector content stack, collapse, or use a drawer at narrow widths. | One route `h1` of `Chat Workspace`. |

## Out Of Scope

- No route renames in this slice unless WP1 has already shipped the route metadata change.
- No broad visual redesign, theme replacement, or new design system.
- No settings information architecture rewrite. That belongs to Task 5.
- No deep media first-selection workflow work. That belongs to Task 8.
- No chat cockpit/global chrome restructuring beyond constraints needed for F13 and the route gates. That belongs to Task 6.
- No backend API changes.

## Current Code Evidence

- `apps/tldw-frontend/components/layout/WebLayout.tsx` owns the top-level WebUI shell, chat sidebar Drawer, header placement, and viewport-constrained route handling through `VIEWPORT_CONSTRAINED_PATHS`.
- `apps/tldw-frontend/components/layout/Header.tsx` owns top navigation and header actions used by the shell.
- `apps/tldw-frontend/pages/_app.tsx` owns app bootstrapping, auth/readiness gates, and route wrappers.
- `apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx` currently uses `min-w-max` for settings groups and horizontal nav overflow on small screens.
- Existing smoke coverage includes `apps/tldw-frontend/e2e/smoke/stage4-mobile-sidebar.spec.ts`, `apps/tldw-frontend/e2e/smoke/stage4-accessibility-controls.spec.ts`, `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`, `apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts`, and `apps/tldw-frontend/e2e/smoke/chat-sticky-composer.spec.ts`.
- Existing route/component coverage includes `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts`, `apps/packages/ui/src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts`, `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx`, `apps/packages/ui/src/components/Option/Prompt/__tests__/PromptsWorkspace.layout.test.tsx`, `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/ChatWorkspacePage.test.tsx`, `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`, `apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx`, `apps/packages/ui/src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx`, and `apps/packages/ui/src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx`.

## File Ownership Map

Shared shell and smoke gates:

- Modify: `apps/tldw-frontend/components/layout/WebLayout.tsx`
- Modify: `apps/tldw-frontend/components/layout/Header.tsx`
- Modify only if route bootstrap state needs an explicit exception: `apps/tldw-frontend/pages/_app.tsx`
- Create: `apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts`
- Modify when needed: `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`
- Keep passing: `apps/tldw-frontend/e2e/smoke/stage4-mobile-sidebar.spec.ts`
- Keep passing: `apps/tldw-frontend/e2e/smoke/stage4-accessibility-controls.spec.ts`
- Keep passing: `apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts`
- Keep passing: `apps/tldw-frontend/e2e/smoke/chat-sticky-composer.spec.ts`

Shared route shell:

- Modify if shell padding or local scroll policy needs a reusable fix: `apps/packages/ui/src/components/Common/PageShell.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-filter.test.tsx`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/settings-nav.test.ts`

Route adopters:

- Chat: `apps/packages/ui/src/routes/option-chat.tsx`
- Chat components: `apps/packages/ui/src/components/Option/Playground/`
- Chat tests: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts`
- Chat tests: `apps/packages/ui/src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts`
- Media: `apps/packages/ui/src/routes/option-media.tsx`
- Media tests: `apps/packages/ui/src/routes/__tests__/option-media-route-guards.test.tsx`
- Media E2E: `apps/tldw-frontend/e2e/workflows/media-navigation-ux-verification.spec.ts`
- Settings: `apps/packages/ui/src/routes/settings-route.tsx`
- Settings model area: `apps/packages/ui/src/components/Option/Models/`
- Settings E2E: `apps/tldw-frontend/e2e/workflows/settings.spec.ts`
- Prompts: `apps/packages/ui/src/components/Option/Prompt/PromptsWorkspace.tsx`
- Prompts tests: `apps/packages/ui/src/components/Option/Prompt/__tests__/PromptsWorkspace.layout.test.tsx`
- Workspace: `apps/packages/ui/src/components/Option/WorkspacePlayground/`
- Workspace tests: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx`
- Sources: `apps/packages/ui/src/components/Option/Sources/SourcesWorkspacePage.tsx`
- Sources tests: `apps/packages/ui/src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx`
- MCP Hub: `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`
- MCP Hub tests: `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`
- STT: `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx`
- STT tests: `apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx`
- TTS: `apps/packages/ui/src/components/Option/TTS/TtsPlaygroundPage.tsx`
- TTS tests: `apps/packages/ui/src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx`
- Chat Workspace: `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx`
- Chat Workspace tests: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/ChatWorkspacePage.test.tsx`

## Implementation Tasks

### Task 1: Add The Responsive Landmark Smoke Gate

**Files:**
- Create: `apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts`
- Modify only for helper reuse: `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.helpers.ts`

- [ ] **Step 1: Write the route matrix**

Add the WP4 route list with expected headings and the current redirect policy. Keep route labels user-facing and avoid implementation names.

```ts
const RESPONSIVE_ROUTE_MATRIX = [
  { path: "/chat", heading: /^Chat$/i },
  { path: "/media", heading: /^Media Inspector$/i },
  { path: "/settings", heading: /^Settings$/i },
  { path: "/settings/model", heading: /^Model settings$/i },
  { path: "/prompts", heading: /^Prompts$/i },
  { path: "/workspace-playground", heading: /^New Research$/i },
  { path: "/setup", heading: /setup/i, allowRedirect: true },
  { path: "/sources", heading: /sources/i },
  { path: "/mcp-hub", heading: /mcp hub/i },
  { path: "/stt", heading: /speech to text/i },
  { path: "/tts", heading: /text to speech/i },
  { path: "/chat-workspace", heading: /chat workspace/i },
] as const;
```

- [ ] **Step 2: Add the failing heading assertion**

Assert that each settled route has exactly one `h1`, unless the route matrix carries an explicit documented exception.

```ts
const headings = page.locator("h1");
await expect(headings, `${entry.path} must expose one h1`).toHaveCount(1);
await expect(headings.first()).toHaveText(entry.heading);
```

- [ ] **Step 3: Add the failing overflow assertion**

Use a 390px viewport and report the largest offenders so implementation work can target actual containers.

```ts
const overflowState = await page.evaluate(() => {
  const root = document.documentElement;
  const offenders = Array.from(document.querySelectorAll<HTMLElement>("body *"))
    .map((element) => ({
      testId: element.getAttribute("data-testid"),
      tag: element.tagName.toLowerCase(),
      className: element.className,
      scrollWidth: element.scrollWidth,
      clientWidth: element.clientWidth,
      rectWidth: element.getBoundingClientRect().width,
    }))
    .filter((entry) => entry.scrollWidth > window.innerWidth + 4 || entry.rectWidth > window.innerWidth + 4)
    .slice(0, 10);

  return {
    rootScrollWidth: root.scrollWidth,
    viewportWidth: window.innerWidth,
    offenders,
  };
});

expect(overflowState.rootScrollWidth, JSON.stringify(overflowState.offenders, null, 2))
  .toBeLessThanOrEqual(overflowState.viewportWidth + 4);
```

- [ ] **Step 4: Add the `/chat` sticky composer assertion**

Add a route-specific assertion that the composer is visible after focusing the input and that the dock bottom remains inside the viewport.

```ts
const composer = page.getByTestId("nextgen-composer-wrapper");
await expect(composer).toBeVisible();
await page.getByTestId("chat-input").focus();
const box = await composer.boundingBox();
expect(box?.y ?? 0).toBeGreaterThanOrEqual(0);
expect((box?.y ?? 0) + (box?.height ?? 0)).toBeLessThanOrEqual(844);
```

- [ ] **Step 5: Run the new smoke test and capture the failing route list**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/smoke/stage4-responsive-landmarks.spec.ts --reporter=line
```

Expected before implementation: FAIL on the routes documented in the audit, with route-specific heading or overflow evidence.

- [ ] **Step 6: Commit the failing test**

```bash
git add apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts
git commit -m "test: add responsive landmark smoke gate"
```

### Task 2: Fix Shared App Shell Constraints First

**Files:**
- Modify: `apps/tldw-frontend/components/layout/WebLayout.tsx`
- Modify if header rows are overflow offenders: `apps/tldw-frontend/components/layout/Header.tsx`
- Modify only if bootstrapping wrappers are direct offenders: `apps/tldw-frontend/pages/_app.tsx`
- Test: `apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/stage4-mobile-sidebar.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`

- [ ] **Step 1: Confirm shell-level offenders**

Run the new smoke gate and inspect the overflow offender JSON. Only edit shell wrappers when the offender list points at shared shell containers, header rows, or missing `min-w-0` on shell flex children.

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/smoke/stage4-responsive-landmarks.spec.ts --reporter=line
```

Expected: FAIL with shell or route offenders.

- [ ] **Step 2: Add shared shell containment**

Apply the smallest shared fix. Preferred fixes are `min-w-0` on flex children, local scroll containers for known horizontal toolbars, and stable viewport wrappers for constrained routes. Use `overflow-x-hidden` only at a boundary where every wider child has its own reachable local scroll or wrap behavior.

Target class areas in `WebLayout.tsx`:

```tsx
<div className={classNames("relative flex w-full min-w-0", isViewportConstrainedRoute ? "h-screen min-h-0" : "min-h-screen")}>
<main className={classNames("relative flex min-w-0 flex-1 flex-col", hideHeader ? "bg-bg" : "")}>
<div className="relative flex min-h-0 min-w-0 flex-1 flex-col">
```

- [ ] **Step 3: Preserve mobile sidebar behavior**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/smoke/stage4-mobile-sidebar.spec.ts --reporter=line
```

Expected: PASS. The mobile chat sidebar is hidden by default, opens from the header toggle, and closes by keyboard.

- [ ] **Step 4: Re-run the responsive smoke gate**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/smoke/stage4-responsive-landmarks.spec.ts --reporter=line
```

Expected: routes that were only failing due to shell wrappers now pass; remaining failures identify route-local containers.

- [ ] **Step 5: Commit the shared shell fix**

```bash
git add apps/tldw-frontend/components/layout/WebLayout.tsx apps/tldw-frontend/components/layout/Header.tsx apps/tldw-frontend/pages/_app.tsx
git commit -m "fix: constrain webui shell on narrow viewports"
```

### Task 3: Make Settings Navigation Mobile-Safe

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-filter.test.tsx`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/settings-nav.test.ts`
- Test: `apps/tldw-frontend/e2e/workflows/settings.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts`

- [ ] **Step 1: Add a component-level responsive guard**

Add or extend a Settings layout test that asserts settings groups do not require `min-w-max` below the desktop breakpoint and that navigation links remain keyboard reachable.

Expected pre-fix result: FAIL because `SettingsOptionLayout.tsx` currently includes `min-w-max lg:min-w-0`.

- [ ] **Step 2: Replace the wide strip behavior**

Keep the settings filter and groups, but change the small-screen presentation to one of these local patterns:

- Wrapping grouped link chips with no `min-w-max`.
- A local disclosure region per group.
- A settings navigation drawer with an always-visible current-section button.

Do not change the settings taxonomy in this slice. Preserve `aria-current="page"`, group labels, the filter input, and the beta badge preference.

- [ ] **Step 3: Add the page heading**

Ensure `/settings` has one `h1` of `Settings` and `/settings/model` has one `h1` of `Model settings`. If child settings pages already own visible `h1` headings, place the route heading once at the layout level and convert child headings to `h2`.

- [ ] **Step 4: Verify focused settings tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Layouts/__tests__/settings-layout-filter.test.tsx src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx src/components/Layouts/__tests__/settings-nav.test.ts
```

Expected: PASS.

- [ ] **Step 5: Verify browser settings workflow and responsive gate**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/settings.spec.ts e2e/smoke/stage4-responsive-landmarks.spec.ts --reporter=line
```

Expected: `/settings` and `/settings/model` pass heading and 390px overflow checks.

- [ ] **Step 6: Commit settings responsive fix**

```bash
git add apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx apps/packages/ui/src/components/Layouts/__tests__
git commit -m "fix: make settings navigation mobile safe"
```

### Task 4: Stabilize Chat Landmark And Composer Behavior

**Files:**
- Modify: `apps/packages/ui/src/routes/option-chat.tsx`
- Modify as needed: `apps/packages/ui/src/components/Option/Playground/`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts`
- Test: `apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/chat-sticky-composer.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts`

- [ ] **Step 1: Add route `h1` coverage**

Add a focused test that `/chat` exposes exactly one `h1` to assistive technology. If the visual design does not need a visible page title, use the existing `sr-only` pattern.

- [ ] **Step 2: Prevent composer and empty-state collision**

Fix only the containers that the smoke gate reports. Expected fix areas are the transcript wrapper, empty-state panel, composer dock, mode strip, or advanced action rows.

Use local horizontal scroll for dense control rows only when every item remains keyboard reachable and the page itself does not overflow.

- [ ] **Step 3: Preserve existing composer invariants**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts
```

Expected: PASS.

- [ ] **Step 4: Verify browser composer behavior**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/smoke/composer-mobile-viewport.spec.ts e2e/smoke/chat-sticky-composer.spec.ts e2e/smoke/stage4-responsive-landmarks.spec.ts --reporter=line
```

Expected: `/chat` passes heading, no-overflow, and sticky composer checks.

- [ ] **Step 5: Commit chat fix**

```bash
git add apps/packages/ui/src/routes/option-chat.tsx apps/packages/ui/src/components/Option/Playground apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts
git commit -m "fix: stabilize chat mobile landmarks"
```

### Task 5: Add Media Mobile Master-Detail Constraints

**Files:**
- Modify: `apps/packages/ui/src/routes/option-media.tsx`
- Modify only the routed media page components reported by tests.
- Test: `apps/packages/ui/src/routes/__tests__/option-media-route-guards.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/media-navigation-ux-verification.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts`

- [ ] **Step 1: Add a component or route guard for mobile mode**

Write the failing test for a 390px media route state. The test must prove that list and detail are not forced side by side when no item or one item is selected.

- [ ] **Step 2: Implement list-first or detail-with-back behavior**

Preserve desktop split behavior. At 390px, render the list first when nothing is selected. When an item is selected, render the detail view with an explicit Back control that returns to the list.

- [ ] **Step 3: Preserve the route heading**

Keep exactly one `h1` for `/media`. If the current `Media Inspector` heading remains visible, do not add a second route heading.

- [ ] **Step 4: Verify focused media tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/option-media-route-guards.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Verify browser media route**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/media-navigation-ux-verification.spec.ts e2e/smoke/stage4-responsive-landmarks.spec.ts --reporter=line
```

Expected: `/media` passes heading and 390px overflow checks. Deeper first-selection copy remains for Task 8 if the route is now structurally usable.

- [ ] **Step 6: Commit media responsive fix**

```bash
git add apps/packages/ui/src/routes/option-media.tsx apps/packages/ui/src/routes/__tests__/option-media-route-guards.test.tsx apps/tldw-frontend/e2e/workflows/media-navigation-ux-verification.spec.ts
git commit -m "fix: make media inspector mobile safe"
```

### Task 6: Fix Route-Local Page Shells And Missing Headings

**Files:**
- Modify if reusable shell constraints are enough: `apps/packages/ui/src/components/Common/PageShell.tsx`
- Modify: `apps/packages/ui/src/components/Option/Prompt/PromptsWorkspace.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/`
- Modify: `apps/packages/ui/src/components/Option/Sources/SourcesWorkspacePage.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/TTS/TtsPlaygroundPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx`
- Test: `apps/packages/ui/src/components/Option/Prompt/__tests__/PromptsWorkspace.layout.test.tsx`
- Test: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx`
- Test: `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`
- Test: `apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx`
- Test: `apps/packages/ui/src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/ChatWorkspacePage.test.tsx`
- Test: `apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts`

- [ ] **Step 1: Start with `PageShell` only when repeated route failures share the same container**

If the smoke gate reports repeated overflow from `PageShell` padding or width constraints, add a reusable `min-w-0` or local-scroll fix there first. Avoid changing `PageShell` max widths in a way that visually redesigns every route.

- [ ] **Step 2: Fix `/prompts` local overflow**

Keep the existing `Prompts` `h1`. If the table or toolbar is wider than the viewport, make that region locally scrollable or collapse secondary actions into the existing overflow affordance.

- [ ] **Step 3: Fix `/workspace-playground` local overflow**

Keep the existing `New Research` `h1`. Stack source, context, and output panels or convert secondary panels into a narrow-width drawer. Preserve existing research workflow controls.

- [ ] **Step 4: Add missing headings on utility routes**

Add one semantic `h1` to each route that lacks one:

- `/sources`: Sources
- `/mcp-hub`: MCP Hub
- `/stt`: Speech to Text
- `/tts`: Text to Speech
- `/chat-workspace`: Chat Workspace

Use visible headings when they help route orientation. Use visually hidden headings only when a dense tool surface already has a clear visible title that must remain compact.

- [ ] **Step 5: Fix route-local containers that still overflow**

For each remaining failed route, use the overflow offender JSON to target the exact row, tab list, card grid, table, or panel. Do not add blanket clipping that hides controls.

- [ ] **Step 6: Verify focused component tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Prompt/__tests__/PromptsWorkspace.layout.test.tsx src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx src/components/Option/ChatWorkspace/__tests__/ChatWorkspacePage.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Verify responsive route matrix**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/smoke/stage4-responsive-landmarks.spec.ts --reporter=line
```

Expected: every WP4 route passes heading and no-overflow checks, with only documented redirect exceptions for setup-state routes.

- [ ] **Step 8: Commit route-local fixes**

```bash
git add apps/packages/ui/src/components/Common/PageShell.tsx apps/packages/ui/src/components/Option/Prompt apps/packages/ui/src/components/Option/WorkspacePlayground apps/packages/ui/src/components/Option/Sources apps/packages/ui/src/components/Option/MCPHub apps/packages/ui/src/components/Option/STT apps/packages/ui/src/components/Option/TTS apps/packages/ui/src/components/Option/ChatWorkspace
git commit -m "fix: add route landmarks and responsive page shells"
```

### Task 7: Final Browser QA And Governance Update

**Files:**
- Modify: `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts`
- Modify: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- Modify: `backlog/tasks/task-418.1 - Plan-WebUI-responsive-shell-and-landmarks-implementation.md`

- [ ] **Step 1: Fold the new route gate into Stage 4 smoke governance**

Add `stage4-responsive-landmarks.spec.ts` to the Stage 4 smoke command or document the exact command that CI or reviewers must run for this slice.

- [ ] **Step 2: Run the full Stage 4 smoke group**

Run from `apps/tldw-frontend`:

```bash
bun run e2e:smoke:stage4
```

Expected: PASS. If this command does not include the new file, run the new file explicitly and update package scripts in the smallest compatible way.

- [ ] **Step 3: Capture browser before and after observations**

Record these states in the Backlog task or PR notes:

- `/chat` at 390px after input focus.
- `/settings` and `/settings/model` at 390px.
- `/media` at 390px with no selection and with a selection.
- `/prompts` and `/workspace-playground` at 390px.
- `/mcp-hub`, `/stt`, `/tts`, `/sources`, and `/chat-workspace` heading checks.

- [ ] **Step 4: Run touched frontend tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Layouts/__tests__/settings-layout-filter.test.tsx src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx src/components/Layouts/__tests__/settings-nav.test.ts src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts src/routes/__tests__/option-media-route-guards.test.tsx src/components/Option/Prompt/__tests__/PromptsWorkspace.layout.test.tsx src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx src/components/Option/ChatWorkspace/__tests__/ChatWorkspacePage.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Update parent plan status**

Mark Task 4 complete only after browser evidence and tests are recorded. Do not mark Task 5, Task 6, Task 8, Task 9, or Task 12 findings closed unless their own route-family acceptance criteria have passed.

- [ ] **Step 6: Final commit**

```bash
git add Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md "backlog/tasks/task-418.1 - Plan-WebUI-responsive-shell-and-landmarks-implementation.md" apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts
git commit -m "test: gate responsive landmarks"
```

## Verification Checklist For This Child Plan

Run before handing off this plan:

```bash
rg -n "T[O]D[O]|T[B]D|F[I]XME|\\.{3}|\\bm[a]ybe\\b|\\bpr[o]bably\\b|\\bshould c[o]nsider\\b" Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md "backlog/tasks/task-418.1 - Plan-WebUI-responsive-shell-and-landmarks-implementation.md"
rg -n "[[:blank:]]$|[^\\x00-\\x7F]" Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md "backlog/tasks/task-418.1 - Plan-WebUI-responsive-shell-and-landmarks-implementation.md"
git diff --check -- Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md "backlog/tasks/task-418.1 - Plan-WebUI-responsive-shell-and-landmarks-implementation.md"
```

Expected: no output from the `rg` checks and no diff-check errors.

## Review Notes

- Use @superpowers:test-driven-development for implementation. This slice is test-first because the main risk is hiding overflow rather than fixing it.
- Use @superpowers:verification-before-completion before marking the implementation done. The acceptance criteria depend on browser-observed scroll width and heading state.
- If a route cannot reasonably expose a visible `h1`, the implementation must document the exception in the test matrix and still expose exactly one semantic `h1` to assistive technology.
- If a route redirects because setup or capability state is unavailable, the test must record the final path and disposition rather than silently skipping the route.
- Any local scroll region introduced to solve overflow must keep keyboard focus, touch targets, and visible affordances intact.
