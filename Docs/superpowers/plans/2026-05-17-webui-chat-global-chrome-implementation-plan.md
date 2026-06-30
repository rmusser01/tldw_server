# WebUI Chat Global Chrome Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/chat` composer-first, fix chat navigation targets, and make global chrome expose chat-specific controls only in chat contexts.

**Architecture:** Keep the existing shared WebUI/extension shell, `Playground`, `ChatHeader`, `HeaderShortcuts`, and `CommandPalette` contracts. Add small policy helpers and tests around route context instead of renaming routes or replacing the chat runtime.

**Tech Stack:** React, TypeScript, React Router, Ant Design, lucide-react, Tailwind utility classes, Vitest, Testing Library, Playwright.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Parent plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- Backlog task: `TASK-418.3`

## Findings Closed Or Supported

- F6: `/chat` presents too many equally weighted starting modes before the main user action.
- F8 support: global navigation and shortcut surfaces need route-target consistency.
- F13: mobile and compact chat must keep the composer reachable and non-overlapping.
- F2 support: first-run or disconnected chat states need to surface model and server readiness in user language.
- F15 support: advanced controls need discovery without competing with the primary chat action.

## Route Scope

Primary implementation routes:

- `/chat`
- `/quick-chat-popout`

Cross-page chrome verification routes:

- `/knowledge`
- `/media`
- `/sources`
- `/settings`
- `/mcp-hub`
- `/stt`
- `/tts`

## Out Of Scope

- No backend API changes.
- No route renaming.
- No new design system.
- No broad visual redesign.
- No replacement of `Playground`, quick chat, or model selection runtime.
- No changes to unrelated content, media, RAG, MCP, STT, or TTS behavior.

## Current Code Evidence

- `/chat` route ownership is `apps/packages/ui/src/routes/option-chat.tsx`, which renders `Playground` inside `RouteErrorBoundary`.
- `apps/packages/ui/src/components/Common/CommandPalette.tsx:187-191` labels `nav-chat` as "Go to Chat" but navigates to `/` and records target path `/`.
- `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts:79-85` already routes the header launcher `chat` shortcut to `/chat`.
- `apps/packages/ui/src/components/Layouts/Header.tsx:345-357` gates share action, chat title, and session badge on `isChatRoute`.
- `apps/packages/ui/src/components/Layouts/ChatHeader.tsx:265-298` always renders new saved chat, temporary chat, and character chat controls when callbacks are provided.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx:70-131` builds five starter cards for general chat, compare, character, RAG, and research.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx:165-204` renders the empty-state shell and visible mode deck on the first screen.
- `apps/packages/ui/src/routes/option-quick-chat-popout.tsx` owns quick helper model selection, assistant mode switching, docs guide browsing, and compact input flow.

## UX Policy

### Primary User Action

`/chat` must lead with the composer and model readiness. The empty state can provide mode discovery, but those choices must not compete with "write a message" as the first action.

Use this hierarchy:

- Primary: message composer focus and model readiness.
- Secondary: quick ingest or source attachment when available.
- Progressive disclosure: compare, character, knowledge search, and deep research starters.
- Help: keyboard shortcuts and tour access as utility actions, not primary page content.

### Header And Chrome Policy

Classify header actions by route before rendering:

| Action | Policy | Routes |
| --- | --- | --- |
| Command palette | Global | All app routes except routes that intentionally own their shortcut layer |
| Shortcuts launcher | Global navigation | All app routes |
| Settings | Global | All app routes |
| Theme toggle | Global | All app routes |
| Notifications | Global if enabled | All app routes |
| Sidebar toggle | Chat surface | `/chat` and chat-owned shells |
| New saved chat | Chat context | `/chat` only |
| Temporary chat | Chat context | `/chat` only |
| Character chat | Chat context | `/chat` only |
| Conversation title edit | Chat context | `/chat` only |
| Share conversation | Chat context | `/chat` only |
| Quick chat helper | Adjacent helper | Global launcher or floating helper, not a replacement for `/chat` |

Non-chat pages can still navigate to chat through command palette and shortcuts. They must not foreground chat session controls as their primary header actions.

### Command Target Policy

- "Go to Chat" opens `/chat`.
- Header shortcuts and command palette use the same canonical chat route.
- "Go to MCP Hub" opens `/mcp-hub`; `/settings/mcp-hub` remains a redirect and settings-index target.
- `/quick-chat-popout` remains a helper surface and is not the command target for main chat.
- Settings search results remain setting-specific when a query matches settings content.

### Progressive Disclosure Policy

The implementation can use a compact mode launcher, popover, details disclosure, or existing route-level controls. The visible result must satisfy:

- `/chat` empty state has one clear first action.
- Starter modes remain discoverable within one interaction.
- Existing mode events are preserved: `tldw:playground-starter-selected`, `tldw:playground-starter`, and `tldw:focus-composer`.
- The mode deck no longer appears as the dominant first-screen object.

## File Ownership Map

### Route Files

- Modify: `apps/packages/ui/src/routes/option-chat.tsx`
  - Preserve route boundary and `RouteErrorBoundary`.
  - Add route-level metadata only if needed to consume existing WP1 route/action metadata.

- Modify: `apps/packages/ui/src/routes/option-quick-chat-popout.tsx`
  - Preserve helper-specific assistant modes and model override behavior.
  - Add tests or accessible labels only if verification reveals helper ambiguity.

### Chat Surface

- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
  - Keep sticky composer and mobile safe-area behavior intact.
  - Keep shortcut event handling intact.
  - Wire empty-state starter actions to composer focus without layout regressions.

- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx`
  - Convert visible starter deck into progressive disclosure.
  - Add model readiness and disconnected readiness affordances using existing connection and model state where available.

- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx`
  - Assert composer-first empty state behavior.
  - Assert starter modes remain discoverable after opening the launcher.

- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`
  - Assert disconnected state still routes to `/settings/tldw` and does not imply model failure as user fault.

- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts`
  - Preserve mobile parity tokens and shortcut event tokens.

- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts`
  - Preserve composer dock offset behavior.

### Header And Navigation

- Modify: `apps/packages/ui/src/components/Layouts/Header.tsx`
  - Derive route chrome policy from normalized pathname.
  - Pass chat-only callbacks to `ChatHeader` only for chat contexts.

- Modify: `apps/packages/ui/src/components/Layouts/ChatHeader.tsx`
  - Accept explicit visibility flags or omit chat-only controls when callbacks are absent.
  - Preserve icon button affordances and keyboard focus rings.

- Modify or create: `apps/packages/ui/src/components/Layouts/header-action-policy.ts`
  - Own route classification for header actions if the policy grows beyond simple `isChatRoute` checks.

- Modify: `apps/packages/ui/src/components/Layouts/HeaderShortcuts.tsx`
  - Preserve existing launcher behavior.
  - Use the route policy only if inline shortcuts need contextual filtering.

- Modify: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
  - Keep `chat` target at `/chat`.
  - Do not add duplicate chat routes.

- Test: `apps/packages/ui/src/components/Layouts/__tests__/ChatHeader.test.tsx`
  - Assert chat-only controls render on chat when callbacks exist.
  - Assert chat-only controls are absent when callbacks are omitted.

- Test: `apps/packages/ui/src/components/Layouts/__tests__/Header.character-mode.test.tsx`
  - Preserve character chat start callbacks on `/chat`.

- Test: `apps/packages/ui/src/components/Layouts/__tests__/HeaderShortcuts.test.tsx`
  - Preserve `chat` shortcut route.
  - Assert launcher search still finds core routes.

### Command Palette

- Modify: `apps/packages/ui/src/components/Common/CommandPalette.tsx`
  - Change `nav-chat` action and `targetPath` from `/` to `/chat`.
  - Change `nav-mcp-hub` action and `targetPath` from `/settings/mcp-hub` to `/mcp-hub`.
  - Keep settings-index results able to route through `/settings/mcp-hub` so existing settings search redirects continue to work.
  - Keep sidepanel scope restrictions intact.

- Test: `apps/packages/ui/src/components/Common/__tests__/CommandPalette.mcp-hub.test.tsx`
  - Preserve MCP Hub dedupe and settings search behavior.

- Test: `apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx`
  - Preserve configured shortcut hints.

- Create if needed: `apps/packages/ui/src/components/Common/__tests__/CommandPalette.navigation.test.tsx`
  - Isolate route-target tests for `/chat`, `/knowledge`, `/media`, `/settings`, and `/mcp-hub`.

### Browser Verification

- Test: `apps/tldw-frontend/e2e/workflows/chat.spec.ts`
  - Keep basic chat workflow green.
  - Add route target smoke if command palette target is not covered by unit tests.

- Test: `apps/tldw-frontend/e2e/smoke/chat-sticky-composer.spec.ts`
  - Preserve desktop and 390px sticky composer behavior.

- Test: `apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts`
  - Re-run if empty state or composer dock spacing changes.

- Test: `apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts`
  - Re-run if route-level landmark or safe-area policy changes.

## Implementation Tasks

### Task 1: Add Command Palette Chat Target Guard

**Files:**

- Modify: `apps/packages/ui/src/components/Common/CommandPalette.tsx`
- Create: `apps/packages/ui/src/components/Common/__tests__/CommandPalette.navigation.test.tsx`
- Verify: `apps/packages/ui/src/components/Common/__tests__/CommandPalette.mcp-hub.test.tsx`
- Verify: `apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx`

- [ ] **Step 1: Write the failing navigation test**

Create `apps/packages/ui/src/components/Common/__tests__/CommandPalette.navigation.test.tsx`.

Use this structure:

```tsx
import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter, useLocation } from "react-router-dom"
import { CommandPalette } from "../CommandPalette"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue ?? key
  })
}))

const RouteProbe = () => {
  const location = useLocation()
  return <div data-testid="route-probe">{location.pathname}</div>
}

describe("CommandPalette navigation targets", () => {
  it("opens the canonical chat route from Go to Chat", async () => {
    render(
      <MemoryRouter initialEntries={["/media"]}>
        <CommandPalette />
        <RouteProbe />
      </MemoryRouter>
    )

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("option", { name: /Go to Chat/i }))

    expect(screen.getByTestId("route-probe")).toHaveTextContent("/chat")
  })

  it("opens MCP Hub on its canonical top-level route", async () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <CommandPalette />
        <RouteProbe />
      </MemoryRouter>
    )

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("option", { name: /Go to MCP Hub/i }))

    expect(screen.getByTestId("route-probe")).toHaveTextContent("/mcp-hub")
  })
})
```

- [ ] **Step 2: Run the failing test**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Common/__tests__/CommandPalette.navigation.test.tsx
```

Expected: the chat target test fails because `nav-chat` navigates to `/`. The MCP Hub target test fails while `nav-mcp-hub` navigates to `/settings/mcp-hub`.

- [ ] **Step 3: Fix the chat target**

In `apps/packages/ui/src/components/Common/CommandPalette.tsx`, change the `nav-chat` target:

```tsx
{
  id: "nav-chat",
  label: t("common:commandPalette.goToChat", "Go to Chat"),
  icon: <MessageSquare className="size-4" />,
  action: () => {
    navigate("/chat")
    setOpen(false)
  },
  targetPath: "/chat",
  category: "navigation",
  keywords: ["chat", "playground", "conversation"]
}
```

Also change the `nav-mcp-hub` navigation command while leaving setting search result routes unchanged:

```tsx
{
  id: "nav-mcp-hub",
  label: t("common:commandPalette.goToMcpHub", "Go to MCP Hub"),
  icon: <Settings className="size-4" />,
  action: () => {
    navigate("/mcp-hub")
    setOpen(false)
  },
  targetPath: "/mcp-hub",
  category: "navigation",
  keywords: ["mcp", "hub", "acp", "policy", "server"]
}
```

- [ ] **Step 4: Run command palette tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Common/__tests__/CommandPalette.navigation.test.tsx src/components/Common/__tests__/CommandPalette.mcp-hub.test.tsx src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx
```

Expected: all command palette tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Common/CommandPalette.tsx apps/packages/ui/src/components/Common/__tests__/CommandPalette.navigation.test.tsx
git commit -m "fix: route command palette chat action to chat"
```

### Task 2: Add Header Action Policy

**Files:**

- Create: `apps/packages/ui/src/components/Layouts/header-action-policy.ts`
- Modify: `apps/packages/ui/src/components/Layouts/Header.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/ChatHeader.tsx`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/ChatHeader.test.tsx`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/Header.character-mode.test.tsx`

- [ ] **Step 1: Write the policy helper tests**

Create `apps/packages/ui/src/components/Layouts/__tests__/header-action-policy.test.ts`.

```ts
import { describe, expect, it } from "vitest"
import { getHeaderActionPolicy } from "../header-action-policy"

describe("getHeaderActionPolicy", () => {
  it("enables chat session actions on the main chat route", () => {
    expect(getHeaderActionPolicy("/chat")).toMatchObject({
      showChatSessionActions: true,
      showChatTitle: true,
      showSessionModeBadge: true,
      showShareConversation: true
    })
  })

  it.each([
    "/knowledge",
    "/media",
    "/sources",
    "/settings",
    "/mcp-hub",
    "/stt",
    "/tts"
  ])("hides chat session actions on %s", (pathname) => {
    expect(getHeaderActionPolicy(pathname)).toMatchObject({
      showChatSessionActions: false,
      showChatTitle: false,
      showSessionModeBadge: false,
      showShareConversation: false
    })
  })
})
```

- [ ] **Step 2: Run the failing policy test**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Layouts/__tests__/header-action-policy.test.ts
```

Expected: fail because `header-action-policy.ts` does not exist.

- [ ] **Step 3: Implement the policy helper**

Create `apps/packages/ui/src/components/Layouts/header-action-policy.ts`.

```ts
export type HeaderActionPolicy = {
  showChatSessionActions: boolean
  showChatTitle: boolean
  showSessionModeBadge: boolean
  showShareConversation: boolean
}

const normalizePathname = (pathname: string): string => {
  const trimmed = pathname.trim()
  if (!trimmed || trimmed === "/") return "/"
  return trimmed.endsWith("/") && trimmed.length > 1
    ? trimmed.slice(0, -1)
    : trimmed
}

export const isMainChatRoute = (pathname: string): boolean =>
  normalizePathname(pathname) === "/chat"

export const getHeaderActionPolicy = (pathname: string): HeaderActionPolicy => {
  const chatRoute = isMainChatRoute(pathname)
  return {
    showChatSessionActions: chatRoute,
    showChatTitle: chatRoute,
    showSessionModeBadge: chatRoute,
    showShareConversation: chatRoute
  }
}
```

- [ ] **Step 4: Apply policy in Header**

In `apps/packages/ui/src/components/Layouts/Header.tsx`:

- Import `getHeaderActionPolicy`.
- Replace duplicated `isChatRoute` rendering decisions with `headerActionPolicy`.
- Pass chat session callbacks only when `showChatSessionActions` is true.

Required shape:

```tsx
const headerActionPolicy = React.useMemo(
  () => getHeaderActionPolicy(normalizedPath),
  [normalizedPath]
)
const chatSessionActionsEnabled = headerActionPolicy.showChatSessionActions
```

Then pass:

```tsx
onOpenShareModal={
  headerActionPolicy.showShareConversation ? openShareModal : undefined
}
shareStatusLabel={
  headerActionPolicy.showShareConversation ? shareStatusLabel : null
}
onClearChat={clearChat}
onStartSavedChat={chatSessionActionsEnabled ? startSavedChat : undefined}
onStartTemporaryChat={chatSessionActionsEnabled ? startTemporaryChat : undefined}
onStartCharacterChat={chatSessionActionsEnabled ? startCharacterChat : undefined}
showChatTitle={headerActionPolicy.showChatTitle}
showSessionModeBadge={headerActionPolicy.showSessionModeBadge}
```

- [ ] **Step 5: Make ChatHeader omit absent chat actions**

In `apps/packages/ui/src/components/Layouts/ChatHeader.tsx`, remove fallback rendering for chat session actions.

Replace:

```tsx
const startSavedChat = onStartSavedChat ?? onClearChat
const startTemporaryChat = onStartTemporaryChat ?? onClearChat
const startCharacterChat = onStartCharacterChat ?? onClearChat
```

With explicit booleans:

```tsx
const showSavedChatAction = Boolean(onStartSavedChat)
const showTemporaryChatAction = Boolean(onStartTemporaryChat)
const showCharacterChatAction = Boolean(onStartCharacterChat)
```

Wrap the existing buttons:

```tsx
{showSavedChatAction ? (
  <Tooltip title={t("playground:header.newSavedChat", "New saved chat")}>
    <button
      type="button"
      onClick={onStartSavedChat}
      aria-label={t("playground:header.newSavedChat", "New saved chat") as string}
      className={`inline-flex items-center justify-center rounded-md p-2 text-text-muted hover:bg-surface2 hover:text-text ${focusRingClasses}`}
      title={t("playground:header.newSavedChat", "New saved chat")}
      data-testid="new-chat-button"
    >
      <SquarePen className="size-4" aria-hidden="true" />
    </button>
  </Tooltip>
) : null}
```

Apply the same pattern to temporary and character buttons.

- [ ] **Step 6: Update ChatHeader tests**

In `apps/packages/ui/src/components/Layouts/__tests__/ChatHeader.test.tsx`, add:

```tsx
it("hides chat session actions when callbacks are omitted", () => {
  renderHeader({
    onStartSavedChat: undefined,
    onStartTemporaryChat: undefined,
    onStartCharacterChat: undefined
  })

  expect(
    screen.queryByRole("button", { name: "New saved chat" })
  ).not.toBeInTheDocument()
  expect(
    screen.queryByRole("button", { name: "Temporary chat (not saved)" })
  ).not.toBeInTheDocument()
  expect(
    screen.queryByRole("button", { name: "Character chat" })
  ).not.toBeInTheDocument()
})
```

Keep the existing callback test that proves those controls still work when callbacks exist.

- [ ] **Step 7: Run header tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Layouts/__tests__/header-action-policy.test.ts src/components/Layouts/__tests__/ChatHeader.test.tsx src/components/Layouts/__tests__/Header.character-mode.test.tsx src/components/Layouts/__tests__/HeaderShortcuts.test.tsx
```

Expected: all header tests pass.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Layouts/header-action-policy.ts apps/packages/ui/src/components/Layouts/Header.tsx apps/packages/ui/src/components/Layouts/ChatHeader.tsx apps/packages/ui/src/components/Layouts/__tests__/header-action-policy.test.ts apps/packages/ui/src/components/Layouts/__tests__/ChatHeader.test.tsx
git commit -m "fix: scope chat header actions to chat route"
```

### Task 3: Make The Chat Empty State Composer-First

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`

- [ ] **Step 1: Write failing empty-state tests**

Update `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx`.

Add assertions:

```tsx
it("keeps starter modes behind a discoverable launcher on first render", () => {
  render(<PlaygroundEmpty />)

  const shell = screen.getByTestId("playground-empty-shell")

  expect(
    within(shell).getByRole("button", { name: "Start chatting" })
  ).toBeInTheDocument()
  expect(
    within(shell).getByRole("button", { name: "Explore chat modes" })
  ).toBeInTheDocument()
  expect(
    within(shell).queryByRole("button", {
      name: /Compare AI models side-by-side/i
    })
  ).not.toBeInTheDocument()

  fireEvent.click(within(shell).getByRole("button", { name: "Explore chat modes" }))

  expect(
    within(shell).getByRole("button", {
      name: /Compare AI models side-by-side/i
    })
  ).toBeInTheDocument()
})
```

Keep existing starter event assertions by opening the launcher before clicking a starter card.

- [ ] **Step 2: Run the failing empty-state test**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx
```

Expected: fail because starter cards are visible on first render and the launcher button does not exist.

- [ ] **Step 3: Implement progressive starter disclosure**

In `PlaygroundEmpty.tsx`:

- Add `const [modesExpanded, setModesExpanded] = React.useState(false)`.
- Keep `handleStartChat` dispatching `general` and `tldw:focus-composer`.
- Add a secondary utility button for starter modes.
- Render `data-testid="playground-empty-mode-deck"` only when `modesExpanded` is true.

Use this button label:

```tsx
{t("playground:empty.exploreModes", "Explore chat modes")}
```

Use existing icons and starter card data. Keep the mode deck markup narrow and stable so mobile width does not change composer layout.

- [ ] **Step 4: Preserve disconnected readiness**

Keep the disconnected copy and settings route from `PlaygroundEmpty.tsx:141-157`:

- "Connect to a tldw server to start chatting."
- `navigate("/settings/tldw")`

Do not introduce provider names, API details, or implementation errors into the empty state.

- [ ] **Step 5: Run empty-state tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx
git commit -m "fix: make chat empty state composer first"
```

### Task 4: Preserve Composer And Mobile Layout Parity

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts`
- Test: `apps/tldw-frontend/e2e/smoke/chat-sticky-composer.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts`

- [ ] **Step 1: Run existing parity tests before edits**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts
```

Expected: pass before layout edits.

- [ ] **Step 2: Verify empty-state changes do not affect composer dock state**

Inspect `Playground.tsx` after Task 3. If the change only touched `PlaygroundEmpty.tsx`, do not edit `Playground.tsx`.

If a focus bridge is required, use the existing `tldw:focus-composer` event. Do not create a second composer focus mechanism.

- [ ] **Step 3: Add guard only if a regression is found**

If composer visibility changes, add a targeted assertion to `mobile-composer-layout.test.ts` for the failing metric. Do not add broad snapshot tests.

- [ ] **Step 4: Run mobile parity tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts
```

Expected: pass.

- [ ] **Step 5: Commit if files changed**

If only verification ran and no files changed, skip this commit. If guard edits were needed:

```bash
git add apps/packages/ui/src/components/Option/Playground/Playground.tsx apps/packages/ui/src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts
git commit -m "test: guard chat composer mobile parity"
```

### Task 5: Verify Quick Chat Popout Remains A Helper Surface

**Files:**

- Modify if needed: `apps/packages/ui/src/routes/option-quick-chat-popout.tsx`
- Test if needed: `apps/packages/ui/src/routes/__tests__/option-quick-chat-popout.test.tsx`

- [ ] **Step 1: Inspect quick chat labels after command target fix**

Confirm:

- The header remains "Quick Chat Helper".
- The model selector has accessible label "Model".
- The segmented modes remain "Chat", "Docs Q&A", and "Browse Guides".
- Empty state says the helper keeps the main thread clean.

- [ ] **Step 2: Add a route test only if coverage is missing**

If no existing route test covers the helper identity, create `apps/packages/ui/src/routes/__tests__/option-quick-chat-popout.test.tsx` with mocked `useQuickChat`, `useQuickChatStore`, and `fetchChatModels`.

Assert:

```tsx
expect(
  screen.getByRole("heading", { name: "Quick Chat Helper" })
).toBeInTheDocument()
expect(screen.getByLabelText("Model")).toBeInTheDocument()
expect(screen.getByText("Docs Q&A")).toBeInTheDocument()
expect(screen.getByText("Browse Guides")).toBeInTheDocument()
```

- [ ] **Step 3: Avoid helper scope expansion**

Do not route "Go to Chat" to `/quick-chat-popout`.
Do not import `Playground` into the quick-chat popout.
Do not make quick chat own global chat session actions.

- [ ] **Step 4: Run quick chat route test if created**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/option-quick-chat-popout.test.tsx
```

Expected: pass if the test exists.

- [ ] **Step 5: Commit if files changed**

```bash
git add apps/packages/ui/src/routes/option-quick-chat-popout.tsx apps/packages/ui/src/routes/__tests__/option-quick-chat-popout.test.tsx
git commit -m "test: preserve quick chat helper identity"
```

### Task 6: Browser QA The Changed Routes

**Files:**

- Test: `apps/tldw-frontend/e2e/workflows/chat.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/chat-sticky-composer.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts`
- Test: `apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts`

- [ ] **Step 1: Start the local frontend**

Run from repo root if no frontend server is already running:

```bash
bun run dev
```

Expected: frontend is reachable at the configured local URL.

- [ ] **Step 2: Run focused Playwright checks**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/chat.spec.ts e2e/smoke/chat-sticky-composer.spec.ts --reporter=line
```

Expected: pass.

- [ ] **Step 3: Run compact viewport checks**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/smoke/composer-mobile-viewport.spec.ts e2e/smoke/stage4-responsive-landmarks.spec.ts --reporter=line
```

Expected: pass or document pre-existing environment failure with browser console evidence.

- [ ] **Step 4: Capture browser observations**

Record observations in the Backlog task:

- `/chat` at 390px: composer visible, no overlap, starter modes behind launcher.
- `/chat` desktop: composer visible, command palette "Go to Chat" target is `/chat`.
- `/knowledge`, `/media`, `/sources`, `/settings`, `/mcp-hub`, `/stt`, `/tts`: no foregrounded chat session controls in header.
- `/quick-chat-popout`: helper title and segmented modes remain visible.

- [ ] **Step 5: Commit any E2E test updates**

If E2E tests changed:

```bash
git add apps/tldw-frontend/e2e/workflows/chat.spec.ts apps/tldw-frontend/e2e/smoke/chat-sticky-composer.spec.ts apps/tldw-frontend/e2e/smoke/composer-mobile-viewport.spec.ts apps/tldw-frontend/e2e/smoke/stage4-responsive-landmarks.spec.ts
git commit -m "test: verify chat chrome browser behavior"
```

## Full Verification Command Set

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Common/__tests__/CommandPalette.navigation.test.tsx src/components/Common/__tests__/CommandPalette.mcp-hub.test.tsx src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx src/components/Layouts/__tests__/header-action-policy.test.ts src/components/Layouts/__tests__/ChatHeader.test.tsx src/components/Layouts/__tests__/Header.character-mode.test.tsx src/components/Layouts/__tests__/HeaderShortcuts.test.tsx src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts
```

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/chat.spec.ts e2e/smoke/chat-sticky-composer.spec.ts e2e/smoke/composer-mobile-viewport.spec.ts e2e/smoke/stage4-responsive-landmarks.spec.ts --reporter=line
```

## Acceptance Criteria

- `/chat` foregrounds composer readiness and model readiness over starter modes.
- Starter modes remain available through one obvious launcher.
- Command palette "Go to Chat" opens `/chat`.
- Header shortcut `chat` still opens `/chat`.
- Non-chat routes do not foreground new saved chat, temporary chat, character chat, conversation title, or share conversation controls.
- `/quick-chat-popout` remains labeled and tested as a helper surface.
- Existing sticky composer desktop and 390px mobile checks pass.
- Verification notes record browser-observed behavior or state why browser verification was unavailable.

## Rollback Plan

- Revert the command palette target change if `/chat` route loading fails.
- Revert `header-action-policy.ts` and related Header wiring if any non-chat route loses required global settings, theme, notification, or shortcut access.
- Revert only the empty-state progressive disclosure change if starter mode launch events stop firing.
- Keep tests that expose the regression, then patch the smallest failing surface.

## Handoff Notes

- This plan intentionally makes small scoped UI changes after tests prove the current failure.
- The plan preserves the existing shared WebUI/extension structure.
- The implementation owner must update `TASK-418.3` or the next child implementation task with verification output.
- Do not begin unrelated WebUI cleanup from this slice.
