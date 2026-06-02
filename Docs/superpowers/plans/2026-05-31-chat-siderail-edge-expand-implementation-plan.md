# Chat Siderail Edge Expand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `/chat` desktop siderail collapse behavior so collapsed rails release layout width and remain recoverable through same-side edge-mounted expand buttons.

**Architecture:** Keep existing stores as source of truth. At `lg` and wider `/chat` viewports, `OptionLayout` omits the full left `ChatSidebar` when collapsed and renders a left-edge expand button instead. `Playground` omits the right artifact flex child when closed, renders a right-edge expand button only when an active artifact exists, and preserves existing mobile/tablet artifact behavior.

**Tech Stack:** React 18, Zustand stores, Tailwind utility classes, lucide-react icons, Vitest with Testing Library, Playwright.

---

## Source Spec

- `Docs/superpowers/specs/2026-05-31-chat-siderail-collapse-design.md`
- Backlog task: `TASK-485`

## File Structure

- Modify `apps/packages/ui/src/components/Layouts/Layout.tsx`
  - Add `/chat` + desktop-only left-edge collapse behavior.
  - Keep the existing narrow `ChatSidebar` collapsed rail for `md` widths and non-chat routes.
  - Own focus transfer between the left edge button and the restored left rail.
- Modify `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
  - Add desktop-only right-edge artifact expand behavior.
  - Add a stable chat-shell test id for measurement-based browser checks.
  - Route artifact focus events to the edge button when the panel was collapsed.
- Modify `apps/packages/ui/src/components/Sidepanel/Chat/ArtifactsPanel.tsx`
  - Add an accessible name and test id to the close button so restored focus can target a meaningful control.
- Modify `apps/packages/ui/src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts`
  - Extend the source guard to cover `/chat` desktop edge-collapse routing and preserve reset-key behavior.
- Modify `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx`
  - Add right artifact edge-button behavior coverage using existing mocks.
- Modify `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx`
  - Assert the chat shell measurement hook exists alongside the sticky composer contract.
- Create `apps/tldw-frontend/e2e/workflows/chat-rails-collapse.spec.ts`
  - Verify desktop width and vertical anchoring with browser measurements.
  - Verify medium/tablet and mobile do not show desktop edge buttons.
- Modify `backlog/tasks/task-485 - Fix-chat-rails-regression-coverage-and-sidepanel-handoff-target.md`
  - Record implementation files, verification results, and known skips.

## Task 1: Left Rail Desktop Edge Affordance

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/Layout.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts`

- [ ] **Step 1: Add failing Layout source guard coverage**

Update `Layout.chat-sidebar-reset-signal.guard.test.ts` with a new test that requires these contract markers:

```ts
it("scopes chat desktop collapsed sidebar to an edge expand affordance", () => {
  const source = readFileSync(layoutSourcePath, "utf8")

  expect(source).toContain("useDesktop")
  expect(source).toContain("useChatEdgeCollapse")
  expect(source).toContain('data-testid="chat-sidebar-edge-expand"')
  expect(source).toContain("isChatScreen && isDesktop")
  expect(source).toContain("!useChatEdgeCollapse || !chatSidebarCollapsed")
})
```

This is a source guard because `OptionLayout` pulls in global shell, migration, query, sidebar, and header dependencies. The browser test in Task 3 provides behavior-level coverage.

- [ ] **Step 2: Run the failing Layout guard**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts
```

Expected: FAIL because `useDesktop`, `useChatEdgeCollapse`, and `chat-sidebar-edge-expand` do not exist yet.

- [ ] **Step 3: Implement desktop-only left edge collapse routing**

In `Layout.tsx`, import `useDesktop` and a compact lucide icon:

```ts
import { EraserIcon, PanelLeftOpen, XIcon } from "lucide-react"
import { useDesktop, useMobile } from "@/hooks/useMediaQuery"
```

Add refs and route/breakpoint flags near the existing sidebar state:

```ts
const leftEdgeExpandRef = React.useRef<HTMLButtonElement>(null)
const isDesktop = useDesktop()
const useChatEdgeCollapse =
  isChatScreen && isDesktop && showChatSidebar && !hideHeader && !hideSidebar
const shouldRenderChatSidebar =
  showChatSidebar &&
  !hideHeader &&
  !hideSidebar &&
  !isMobile &&
  (!useChatEdgeCollapse || !chatSidebarCollapsed)
```

Replace the current persistent sidebar condition with `shouldRenderChatSidebar`.

In the `ChatSidebar` `onToggleCollapse` handler, preserve the existing reset signal and move focus to the left edge button after collapsing:

```ts
onToggleCollapse={() => {
  if (chatSidebarCollapsed) signalChatSidebarOpen()
  const collapsingFromDesktopChat =
    useChatEdgeCollapse && !chatSidebarCollapsed
  setChatSidebarCollapsed((prev) => !prev)
  if (collapsingFromDesktopChat) {
    window.requestAnimationFrame(() => {
      leftEdgeExpandRef.current?.focus()
    })
  }
}}
```

Render the edge button inside the same outer layout shell, before `<main>` or as the first child inside `<main>`:

```tsx
{useChatEdgeCollapse && chatSidebarCollapsed && (
  <button
    ref={leftEdgeExpandRef}
    type="button"
    data-testid="chat-sidebar-edge-expand"
    aria-label={t("common:chatSidebar.expandRail", "Expand chat rail") as string}
    title={t("common:chatSidebar.expandRail", "Expand chat rail") as string}
    onClick={() => {
      signalChatSidebarOpen()
      setChatSidebarCollapsed(false)
      window.requestAnimationFrame(() => {
        document
          .querySelector<HTMLButtonElement>('[data-testid="chat-sidebar-toggle"]')
          ?.focus()
      })
    }}
    className="absolute left-2 top-20 z-30 inline-flex h-9 w-9 items-center justify-center rounded-md border border-border bg-surface text-text-muted shadow-sm transition hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
  >
    <PanelLeftOpen className="h-4 w-4" aria-hidden="true" />
  </button>
)}
```

Keep `ChatSidebar`'s existing collapsed branch intact. It remains the `md` and non-chat-route behavior.

- [ ] **Step 4: Run the Layout guard**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Stage only the touched files:

```bash
git add apps/packages/ui/src/components/Layouts/Layout.tsx apps/packages/ui/src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts
git commit -m "fix: add chat left rail edge expand affordance"
```

## Task 2: Right Artifact Rail Edge Affordance

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/ArtifactsPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx`

- [ ] **Step 1: Add failing Playground tests for right-edge behavior**

In `Playground.search.integration.test.tsx`, extend the `useMediaQuery` mock so tests can control both mobile and desktop:

```ts
const mobileViewportState = vi.hoisted(() => ({
  value: false
}))

const desktopViewportState = vi.hoisted(() => ({
  value: true
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => mobileViewportState.value,
  useDesktop: () => desktopViewportState.value
}))
```

Reset `desktopViewportState.value = true` in `beforeEach`.

Add a helper artifact:

```ts
const artifactFixture = {
  id: "artifact-1",
  title: "Generated table",
  content: "a,b\n1,2",
  kind: "table" as const
}
```

Also reset artifact state in `beforeEach` so tests do not leak active artifacts:

```ts
artifactsState.value.active = null
artifactsState.value.history = []
artifactsState.value.unreadCount = 0
```

Add tests:

```ts
it("shows a desktop right-edge artifacts expand button only when an artifact is active and the rail is closed", () => {
  artifactsState.value.active = artifactFixture
  artifactsState.value.isOpen = false

  render(<Playground />)

  expect(
    screen.getByRole("button", { name: "Expand artifacts rail" })
  ).toBeInTheDocument()
  expect(screen.queryByTestId("artifacts-panel")).not.toBeInTheDocument()
})

it("does not show the right-edge artifacts expand button without an active artifact", () => {
  artifactsState.value.active = null
  artifactsState.value.isOpen = false

  render(<Playground />)

  expect(
    screen.queryByRole("button", { name: "Expand artifacts rail" })
  ).not.toBeInTheDocument()
})

it("opens artifacts from the right edge and marks them read", () => {
  artifactsState.value.active = artifactFixture
  artifactsState.value.isOpen = false

  render(<Playground />)
  fireEvent.click(screen.getByRole("button", { name: "Expand artifacts rail" }))

  expect(artifactsState.value.setOpen).toHaveBeenCalledWith(true)
  expect(artifactsState.value.markRead).toHaveBeenCalledTimes(1)
})

it("routes artifact focus events to the edge button when the rail is closed", async () => {
  artifactsState.value.active = artifactFixture
  artifactsState.value.isOpen = false

  render(<Playground />)
  const edgeButton = screen.getByRole("button", {
    name: "Expand artifacts rail"
  })

  window.dispatchEvent(new CustomEvent("tldw:focus-artifacts-trigger"))

  await waitFor(() => {
    expect(document.activeElement).toBe(edgeButton)
  })
})
```

In `Playground.sticky-composer-layout.integration.test.tsx`, assert the measurable chat shell exists:

```ts
expect(screen.getByTestId("playground-chat-shell")).toBeInTheDocument()
```

- [ ] **Step 2: Run the failing Playground tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx
```

Expected: FAIL because the edge button, `useDesktop` mock path, and chat shell test id do not exist.

- [ ] **Step 3: Implement right edge affordance and measurement hook**

In `Playground.tsx`, import `PanelRightOpen` and `useDesktop`:

```ts
import { ChevronDown, Keyboard, PanelRightOpen, Search, X } from "lucide-react";
import { useDesktop, useMobile } from "@/hooks/useMediaQuery";
```

Add refs and flags near the artifact store selectors:

```ts
const artifactsEdgeExpandRef = React.useRef<HTMLButtonElement>(null);
const pendingArtifactsPanelFocusRef = React.useRef(false);
const isDesktopViewport = useDesktop();
const shouldShowArtifactsEdgeExpand =
  isDesktopViewport && Boolean(activeArtifact) && !artifactsOpen;
```

Add an open helper:

```ts
const openArtifactsFromEdge = React.useCallback(() => {
  if (!activeArtifact) return;
  pendingArtifactsPanelFocusRef.current = true;
  setArtifactsOpen(true);
  markArtifactsRead();
}, [activeArtifact, markArtifactsRead, setArtifactsOpen]);
```

Add a focus effect that runs after the panel opens. Because `ArtifactsPanel` is lazy-loaded, retry once on the next macrotask before falling back to the existing toolbar trigger:

```ts
React.useEffect(() => {
  if (!artifactsOpen || !pendingArtifactsPanelFocusRef.current) return;
  pendingArtifactsPanelFocusRef.current = false;
  const focusPanelClose = () => {
    const closeButton = document.querySelector<HTMLButtonElement>(
      '[data-testid="artifacts-panel-close"]',
    );
    if (closeButton) {
      closeButton.focus();
      return true;
    }
    return false;
  };
  window.requestAnimationFrame(() => {
    if (focusPanelClose()) return;
    window.setTimeout(() => {
      if (!focusPanelClose()) {
        artifactsTriggerRef.current?.focus();
      }
    }, 0);
  });
}, [artifactsOpen]);
```

Update the existing `tldw:focus-artifacts-trigger` handler so collapsed desktop panels focus the edge button:

```ts
const handleFocusArtifactsTrigger = () => {
  if (artifactsEdgeExpandRef.current) {
    artifactsEdgeExpandRef.current.focus();
    return;
  }
  artifactsTriggerRef.current?.focus();
};
```

Add `data-testid="playground-chat-shell"` to the left flex child:

```tsx
<div
  data-testid="playground-chat-shell"
  className="flex h-full min-h-0 min-w-0 flex-1 flex-col"
>
```

Render the edge expand button inside the `relative z-10 flex h-full min-h-0 w-full` shell after the chat shell and before the artifact panel:

```tsx
{shouldShowArtifactsEdgeExpand && (
  <button
    ref={artifactsEdgeExpandRef}
    type="button"
    data-testid="playground-artifacts-edge-expand"
    aria-label={t("playground:regions.expandArtifactsRail", "Expand artifacts rail") as string}
    title={t("playground:regions.expandArtifactsRail", "Expand artifacts rail") as string}
    onClick={openArtifactsFromEdge}
    className="absolute right-2 top-20 z-30 inline-flex h-9 w-9 items-center justify-center rounded-md border border-border bg-surface text-text-muted shadow-sm transition hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
  >
    <PanelRightOpen className="h-4 w-4" aria-hidden="true" />
  </button>
)}
```

Keep the existing top toolbar artifact chip. It remains a secondary control and state indicator.

In `ArtifactsPanel.tsx`, make the close button explicitly targetable:

```tsx
<button
  type="button"
  data-testid="artifacts-panel-close"
  aria-label={t("artifactsClose", "Close") as string}
  ...
>
```

- [ ] **Step 4: Run the focused Playground tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Run the existing artifact panel guard**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.jump-source.guard.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Stage only the touched files:

```bash
git add apps/packages/ui/src/components/Option/Playground/Playground.tsx apps/packages/ui/src/components/Sidepanel/Chat/ArtifactsPanel.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx
git commit -m "fix: add chat artifact rail edge expand affordance"
```

## Task 3: Browser Measurement Regression

**Files:**
- Create: `apps/tldw-frontend/e2e/workflows/chat-rails-collapse.spec.ts`

- [ ] **Step 1: Add failing Playwright coverage**

Create `chat-rails-collapse.spec.ts` with desktop measurement and compact viewport checks. Reuse the local auth/setup pattern from `e2e/smoke/smoke.setup.ts` and stub provider metadata before navigation so the test reaches `/chat` reliably:

```ts
import { expect, test, type Page } from "@playwright/test"

import { seedAuth } from "../smoke/smoke.setup"
import { waitForAppShell } from "../utils/helpers"

const artifactFixture = {
  id: "artifact-rail-e2e",
  title: "Rail artifact",
  content: "value",
  kind: "code",
  language: "text"
}

const prepareChatRailPage = async (page: Page) => {
  await seedAuth(page)
  await page.route("**/api/v1/llm/models/metadata**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ models: [] })
    })
  })
  await page.route("**/api/v1/llm/providers**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ providers: [] })
    })
  })
  await page.addInitScript(() => {
    window.localStorage.setItem("stickyChatInput", "true")
    window.localStorage.setItem("playgroundComposerOptionsExpanded", "false")
  })
}

const openArtifactPanel = async (page: Page) => {
  await page.waitForFunction(() =>
    Boolean((window as any).__tldw_useArtifactsStore),
  )
  await page.evaluate((artifact) => {
    ;(window as any).__tldw_useArtifactsStore.getState().openArtifact(artifact)
  }, artifactFixture)
}

test.describe("/chat siderail collapse", () => {
  test("desktop collapsed rails expose same-side edge buttons and release width", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 960 })
    await prepareChatRailPage(page)
    await page.goto("/chat", { waitUntil: "domcontentloaded" })
    await waitForAppShell(page)

    const chatShell = page.getByTestId("playground-chat-shell")
    const composer = page.getByTestId("playground-chat-composer-dock")
    await expect(chatShell).toBeVisible()
    await expect(composer).toBeVisible()

    const leftEdge = page.getByTestId("chat-sidebar-edge-expand")
    await expect(leftEdge).toBeVisible()
    const leftCollapsedBox = await chatShell.boundingBox()
    expect(leftCollapsedBox).not.toBeNull()

    await leftEdge.click()
    await expect(page.getByTestId("chat-sidebar")).toBeVisible()
    const leftExpandedBox = await chatShell.boundingBox()
    expect(leftExpandedBox).not.toBeNull()
    expect(leftExpandedBox!.width).toBeLessThan(leftCollapsedBox!.width)

    const expandedTop = leftExpandedBox!.y
    await page.getByTestId("chat-sidebar-toggle").click()
    await expect(leftEdge).toBeVisible()
    const leftRecollapsedBox = await chatShell.boundingBox()
    expect(leftRecollapsedBox).not.toBeNull()
    expect(leftRecollapsedBox!.width).toBeGreaterThan(leftExpandedBox!.width)
    expect(Math.abs(leftRecollapsedBox!.y - expandedTop)).toBeLessThanOrEqual(2)

    await openArtifactPanel(page)
    await expect(page.getByTestId("artifacts-panel")).toBeVisible()
    const rightOpenBox = await chatShell.boundingBox()
    expect(rightOpenBox).not.toBeNull()
    await page.getByTestId("artifacts-panel-close").click()

    const rightEdge = page.getByTestId("playground-artifacts-edge-expand")
    await expect(rightEdge).toBeVisible()
    const rightClosedBox = await chatShell.boundingBox()
    expect(rightClosedBox).not.toBeNull()
    expect(rightClosedBox!.width).toBeGreaterThan(rightOpenBox!.width)

    const composerBox = await composer.boundingBox()
    expect(composerBox).not.toBeNull()
    expect(Math.abs(960 - (composerBox!.y + composerBox!.height))).toBeLessThanOrEqual(12)
  })

  test("medium and mobile viewports do not expose desktop edge buttons", async ({ page }) => {
    await page.setViewportSize({ width: 900, height: 900 })
    await prepareChatRailPage(page)
    await page.goto("/chat", { waitUntil: "domcontentloaded" })
    await waitForAppShell(page)
    await expect(page.getByTestId("chat-sidebar-edge-expand")).toHaveCount(0)
    await openArtifactPanel(page)
    await page.getByTestId("artifacts-panel-close").click()
    await expect(page.getByTestId("playground-artifacts-edge-expand")).toHaveCount(0)

    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto("/chat", { waitUntil: "domcontentloaded" })
    await waitForAppShell(page)
    await expect(page.getByTestId("chat-sidebar-edge-expand")).toHaveCount(0)
    await openArtifactPanel(page)
    await page.getByTestId("artifacts-panel-close").click()
    await expect(page.getByTestId("playground-artifacts-edge-expand")).toHaveCount(0)
  })
})
```

- [ ] **Step 2: Run the failing Playwright test before implementation if Task 1 and 2 are not done**

If Task 1 and Task 2 were not implemented yet, run from `apps/tldw-frontend`:

```bash
TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18080' TLDW_WEB_URL=http://127.0.0.1:18080 bunx playwright test e2e/workflows/chat-rails-collapse.spec.ts --project=chromium --reporter=line
```

Expected before implementation: FAIL because selectors are missing.

- [ ] **Step 3: Run the Playwright test after Task 1 and Task 2**

Run from `apps/tldw-frontend`:

```bash
TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18080' TLDW_WEB_URL=http://127.0.0.1:18080 bunx playwright test e2e/workflows/chat-rails-collapse.spec.ts --project=chromium --reporter=line
```

Expected: PASS. If port `18080` is occupied, choose another high port and update both `TLDW_WEB_CMD` and `TLDW_WEB_URL`.

- [ ] **Step 4: Commit Task 3**

Stage only the new test:

```bash
git add apps/tldw-frontend/e2e/workflows/chat-rails-collapse.spec.ts
git commit -m "test: cover chat rail collapse layout"
```

## Task 4: Focused Verification and Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-485 - Fix-chat-rails-regression-coverage-and-sidepanel-handoff-target.md`

- [ ] **Step 1: Run the focused Vitest suite**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.jump-source.guard.test.ts
```

Expected: PASS.

- [ ] **Step 2: Run the focused Playwright suite**

Run from `apps/tldw-frontend`:

```bash
TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18080' TLDW_WEB_URL=http://127.0.0.1:18080 bunx playwright test e2e/workflows/chat-rails-collapse.spec.ts --project=chromium --reporter=line
```

Expected: PASS.

- [ ] **Step 3: Run diff hygiene**

Run from repo root:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Document Bandit skip**

No backend Python files are in scope. Record in `TASK-485` that Bandit was not run because the implementation touched frontend TypeScript/TSX, E2E tests, docs, and Backlog metadata only.

- [ ] **Step 5: Update Backlog task**

In `TASK-485`, update:

- `modified_files` with every touched file.
- Implementation Notes with test commands and results.
- Final Summary with what changed and why.
- Definition of Done checkboxes that are satisfied.

- [ ] **Step 6: Commit closeout metadata**

Stage only the task file:

```bash
git add 'backlog/tasks/task-485 - Fix-chat-rails-regression-coverage-and-sidepanel-handoff-target.md'
git commit -m "docs: record chat rail collapse verification"
```

## Final Review Checklist

- [ ] `/chat` at `>=1024px`: left collapsed rail does not reserve width and left-edge expand button is visible.
- [ ] `/chat` at `>=1024px`: right closed artifact rail does not reserve width and right-edge expand button is visible only when `activeArtifact` exists.
- [ ] `/chat` with both recoverable collapsed rails: chat shell uses the freed width, transcript top is stable, composer remains docked.
- [ ] `/chat` at `768px-1023px`: no new desktop edge expand buttons appear.
- [ ] `/chat` below `768px`: existing drawer/sheet behavior still works and no desktop edge buttons appear.
- [ ] Header/sidebar toggles still work as secondary paths.
- [ ] Focus moves to edge buttons after collapse and to meaningful restored controls after expand where practical.
- [ ] Existing `ChatSidebar` collapsed narrow rail remains available for non-chat routes and medium widths.
