# Chat Workspace Terminal-Literal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a visible `/chat-workspace` web page in `apps/` that recreates the approved Terminal-Literal Chatbook-style workspace inside the existing web UI shell.

**Architecture:** Add a thin Next.js page and shared UI route, then implement a focused `ChatWorkspace` component family under `apps/packages/ui/src/components/Option/ChatWorkspace/`. The page reuses the existing web header/sidebar shell, existing chat action hook, existing workspace source store, and a new local staged-context model so browsing selection never silently attaches sources to a chat request.

**Tech Stack:** Next.js, React 18, TypeScript, Zustand stores, Tailwind utility classes, lucide-react icons, Vitest, React Testing Library, Playwright.

---

## Source References

- Spec: `Docs/superpowers/specs/2026-05-03-chat-workspace-terminal-literal-design.md`
- Web shell wrapper: `apps/tldw-frontend/pages/_app.tsx`, `apps/tldw-frontend/components/layout/WebLayout.tsx`
- Next page convention: `apps/tldw-frontend/pages/workspace-playground.tsx`
- Shared route convention: `apps/packages/ui/src/routes/option-workspace-playground.tsx`
- Shared route registry: `apps/packages/ui/src/routes/route-registry.tsx`
- Extension route mirror: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Route constants: `apps/packages/ui/src/routes/route-paths.ts`
- Shortcut/navigation config: `apps/packages/ui/src/services/settings/ui-settings.ts`, `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- Workspace source type/store: `apps/packages/ui/src/types/workspace.ts`, `apps/packages/ui/src/store/workspace.ts`
- Chat action hook: `apps/packages/ui/src/hooks/useMessageOption.tsx`, `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- Connection state hook: `apps/packages/ui/src/hooks/useConnectionState.ts`, `apps/packages/ui/src/types/connection.ts`
- Shared message renderer: `apps/packages/ui/src/components/Common/Playground/Message.tsx`
- Existing workspace chat implementation for reference only: `apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx`

## Guardrails

- Do not touch `tldw_chatbook` or any Chatbook TUI files.
- Do not change `/chat` or `/workspace-playground` behavior except for small shared chat helper changes covered by tests.
- Do not commit unrelated local changes such as `Docs/Design/Agents.md`.
- Do not commit `.superpowers/` brainstorming scratch output.
- Use existing theme tokens and utility patterns; do not create a separate global app shell.
- Inactive v1 controls must say `Not configured`, `Unavailable`, or `No active task`, not `Ready`.

## File Structure

Create:

- `apps/tldw-frontend/pages/chat-workspace.tsx` - thin Next wrapper with SSR disabled.
- `apps/packages/ui/src/routes/option-chat-workspace.tsx` - shared options route loaded by Next and extension registries.
- `apps/tldw-frontend/extension/routes/option-chat-workspace.tsx` - extension route wrapper for the mirrored extension registry.
- `apps/packages/ui/src/components/Option/ChatWorkspace/index.ts` - barrel exports for the new component family.
- `apps/packages/ui/src/components/Option/ChatWorkspace/types.ts` - local staged-source and panel state types.
- `apps/packages/ui/src/components/Option/ChatWorkspace/staging.ts` - pure staged-source builders and formatting helpers.
- `apps/packages/ui/src/components/Option/ChatWorkspace/ContextStagingCard.tsx` - staged context card with Clear, Insert, Send.
- `apps/packages/ui/src/components/Option/ChatWorkspace/InspectorRail.tsx` - scope/source/model/runtime inspector.
- `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceStatusStrip.tsx` - bottom route/status/hint strip.
- `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceRail.tsx` - left workspace/source/study rail.
- `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx` - chat transcript/composer adapter using the shared chat path.
- `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspaceConsole.tsx` - responsive three-region console layout.
- `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx` - page orchestration and store bridge.
- `apps/packages/ui/src/hooks/chat/__tests__/chat-submit-result.guard.test.ts` - source guard for explicit submit success/failure contract.

Modify:

- `apps/packages/ui/src/routes/route-paths.ts` - add `CHAT_WORKSPACE_PATH` and viewport-constrained handling.
- `apps/packages/ui/src/routes/route-registry.tsx` - add shared options route.
- `apps/tldw-frontend/extension/routes/route-registry.tsx` - add extension route mirror and nav metadata.
- `apps/packages/ui/src/services/settings/ui-settings.ts` - add shortcut id and visible sidebar default.
- `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts` - add visible `Chat Workspace` shortcut near Chat.
- `apps/packages/ui/src/hooks/chat/chat-action-utils.ts` - add pure helper for turn-level RAG media override.
- `apps/packages/ui/src/hooks/chat/useChatActions.ts` - use the helper so per-submit staged media ids can trigger RAG without relying on stale global state.
- `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts` - return explicit submitted/failed results instead of silently resolving saved errors.
- `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts` - return pipeline submit result.
- `apps/packages/ui/src/hooks/chat-modes/ragMode.ts` - return pipeline submit result.
- `apps/packages/ui/src/hooks/chat-modes/documentChatMode.ts` - return pipeline submit result.
- `apps/packages/ui/src/hooks/chat-modes/tabChatMode.ts` - return pipeline submit result.
- `apps/packages/ui/src/hooks/chat-modes/continueChatMode.ts` - return pipeline submit result.
- `apps/tldw-frontend/e2e/smoke/page-inventory.ts` - add page to smoke inventory.
- `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts` - include the dense new route in a11y scan.
- `apps/tldw-frontend/e2e/smoke/stage5-release-gate.spec.ts` - include the visible new route in console/error budget.

Test:

- `apps/tldw-frontend/__tests__/extension/route-registry.chat-workspace.test.ts`
- `apps/packages/ui/src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts`
- `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/staging.test.ts`
- `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/ContextStagingCard.test.tsx`
- `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/InspectorRail.test.tsx`
- `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceStatusStrip.test.tsx`
- `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceRail.test.tsx`
- `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx`
- `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/ChatWorkspacePage.test.tsx`

## Data Contracts

Use these local types unless implementation finds an already-equivalent type in the shared UI:

```ts
export type StagedSourceAvailability =
  | "ready"
  | "processing"
  | "error"
  | "unavailable"

export type StagedWorkspaceSource = {
  sourceId: string
  mediaId: number | null
  title: string
  type: WorkspaceSourceType
  scopeLabel: string
  availability: StagedSourceAvailability
  statusMessage?: string
}

export type ChatWorkspaceRuntimeState = {
  backendAvailable: boolean
  streaming: boolean
  selectedModelLabel: string
  selectedPersonaLabel: string | null
}
```

`selectedSourceIds` in `useWorkspaceStore` must not be treated as browsing focus in this page. The new page should keep browsing focus local to `WorkspaceRail`; only explicit `Stage for Chat`, `Use in Chat`, or handoff actions produce `StagedWorkspaceSource[]`.

`WorkspaceChatPanel` owns the `useMessageOption({ scope })` call and reports live runtime state upward through `onRuntimeStateChange`. `ChatWorkspacePage` then feeds that real model/persona/streaming state into `InspectorRail` and `WorkspaceStatusStrip`.

The chat submit path must return an explicit result because existing mode pipelines can save an error message and resolve the promise. The new page may clear draft/staged context only when `isChatSubmitSuccess(result)` returns true.

```ts
export type ChatSubmitResult =
  | { status: "submitted" }
  | { status: "failed"; errorMessage: string }
  | { status: "skipped"; reason: string }
```

## Task 1: Route, Shell, And Visible Navigation

**Files:**

- Create: `apps/tldw-frontend/pages/chat-workspace.tsx`
- Create: `apps/packages/ui/src/routes/option-chat-workspace.tsx`
- Create: `apps/tldw-frontend/extension/routes/option-chat-workspace.tsx`
- Modify: `apps/packages/ui/src/routes/route-paths.ts`
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Modify: `apps/packages/ui/src/services/settings/ui-settings.ts`
- Modify: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- Test: `apps/tldw-frontend/__tests__/extension/route-registry.chat-workspace.test.ts`

- [ ] **Step 1: Write the failing route/nav parity test**

Create `apps/tldw-frontend/__tests__/extension/route-registry.chat-workspace.test.ts`:

```ts
import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const readSource = (path: string) => readFileSync(path, "utf8")

const extensionRouteRegistryPathCandidates = [
  "extension/routes/route-registry.tsx",
  "apps/tldw-frontend/extension/routes/route-registry.tsx"
]

const extensionRouteRegistryPath = extensionRouteRegistryPathCandidates.find(
  (candidate) => existsSync(candidate)
)

if (!extensionRouteRegistryPath) {
  throw new Error("Unable to locate extension route-registry.tsx")
}

describe("chat-workspace route parity", () => {
  const extensionRegistry = readSource(extensionRouteRegistryPath)
  const sharedRegistry = readSource("../packages/ui/src/routes/route-registry.tsx")
  const routePaths = readSource("../packages/ui/src/routes/route-paths.ts")
  const uiSettings = readSource("../packages/ui/src/services/settings/ui-settings.ts")
  const shortcuts = readSource(
    "../packages/ui/src/components/Layouts/header-shortcut-items.ts"
  )

  it("registers /chat-workspace in shared and extension route registries", () => {
    expect(sharedRegistry).toMatch(/path:\s*CHAT_WORKSPACE_PATH/)
    expect(extensionRegistry).toMatch(/path:\s*"\/chat-workspace"/)
  })

  it("loads the Next route through the shared option route with SSR disabled", () => {
    const nextPage = readSource("pages/chat-workspace.tsx")
    expect(nextPage).toContain('import("@/routes/option-chat-workspace")')
    expect(nextPage).toContain("ssr: false")
  })

  it("provides an extension route wrapper for the mirrored registry", () => {
    const extensionWrapper = readSource("extension/routes/option-chat-workspace.tsx")
    expect(extensionWrapper).toContain("@/routes/option-chat-workspace")
  })

  it("marks chat workspace as viewport constrained", () => {
    expect(routePaths).toContain('export const CHAT_WORKSPACE_PATH = "/chat-workspace"')
    expect(routePaths).toMatch(/VIEWPORT_CONSTRAINED_PATHS[\s\S]*CHAT_WORKSPACE_PATH/)
  })

  it("exposes visible navigation metadata", () => {
    expect(uiSettings).toMatch(/"chat-workspace"/)
    expect(shortcuts).toMatch(/id:\s*"chat-workspace"/)
    expect(shortcuts).toMatch(/labelDefault:\s*"Chat Workspace"/)
    expect(extensionRegistry).toMatch(/labelToken:\s*"option:header\.chatWorkspace"/)
    expect(extensionRegistry).toMatch(/group:\s*"workspace"/)
  })
})
```

- [ ] **Step 2: Run the test and verify it fails**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/extension/route-registry.chat-workspace.test.ts
```

Expected: FAIL because `/chat-workspace`, route constants, and shortcut ids do not exist.

- [ ] **Step 3: Implement the route and nav stub**

Create `apps/tldw-frontend/pages/chat-workspace.tsx`:

```tsx
import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-chat-workspace"), {
  ssr: false
})
```

Create `apps/packages/ui/src/routes/option-chat-workspace.tsx`:

```tsx
import OptionLayout from "~/components/Layouts/Layout"
import { ChatWorkspacePage } from "@/components/Option/ChatWorkspace"

const OptionChatWorkspace = () => {
  return (
    <OptionLayout>
      <ChatWorkspacePage />
    </OptionLayout>
  )
}

export default OptionChatWorkspace
```

Create a temporary page component so the route compiles until later tasks replace it:

```tsx
// apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx
export const ChatWorkspacePage = () => (
  <main data-testid="chat-workspace-page" className="h-full w-full bg-background text-foreground">
    Chat Workspace
  </main>
)
```

Create the barrel without JSX:

```ts
// apps/packages/ui/src/components/Option/ChatWorkspace/index.ts
export { ChatWorkspacePage } from "./ChatWorkspacePage"
```

Modify `apps/packages/ui/src/routes/route-paths.ts`:

```ts
export const CHAT_WORKSPACE_PATH = "/chat-workspace"

export const VIEWPORT_CONSTRAINED_PATHS = [
  DOCUMENT_WORKSPACE_PATH,
  WORKSPACE_PLAYGROUND_PATH,
  CHAT_WORKSPACE_PATH,
  "/media-multi",
] as const
```

Modify `apps/packages/ui/src/routes/route-registry.tsx`:

```tsx
const OptionChatWorkspace = lazy(() => import("./option-chat-workspace"))

{
  kind: "options",
  path: CHAT_WORKSPACE_PATH,
  element: <OptionChatWorkspace />,
},
```

Modify `apps/tldw-frontend/extension/routes/route-registry.tsx` following the existing `/workspace-playground` pattern:

```tsx
const OptionChatWorkspace = lazy(() => import("./option-chat-workspace"))

{
  kind: "options",
  path: "/chat-workspace",
  element: <OptionChatWorkspace />,
  nav: {
    group: "workspace",
    labelToken: "option:header.chatWorkspace",
    icon: SquareTerminal,
    order: 1,
  },
},
```

`NavGroupKey` in `apps/tldw-frontend/extension/routes/route-registry.tsx` currently allows only `"server"`, `"knowledge"`, `"workspace"`, and `"about"`. Use `"workspace"` for this route; do not add a new nav group as part of this feature.

Create `apps/tldw-frontend/extension/routes/option-chat-workspace.tsx`:

```tsx
export { default } from "@/routes/option-chat-workspace"
```

Modify `apps/packages/ui/src/services/settings/ui-settings.ts`:

```ts
export const HEADER_SHORTCUT_IDS = [
  "chat",
  "chat-workspace",
  "prompts",
  // ...
] as const

export const DEFAULT_SIDEBAR_SHORTCUT_SELECTION: SidebarShortcutId[] = [
  "quick-ingest",
  "chat",
  "chat-workspace",
  // keep the existing defaults after this entry
]
```

Modify `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`:

```tsx
import {
  MessageSquare,
  SquareTerminal,
  // existing icons
} from "lucide-react"

{
  id: "chat-workspace",
  to: CHAT_WORKSPACE_PATH,
  icon: SquareTerminal,
  labelKey: "option:header.chatWorkspace",
  labelDefault: "Chat Workspace",
  shortcutIndex: 2,
  descriptionKey: "option:header.chatWorkspaceDesc",
  descriptionDefault: "Chat-first workspace with staged sources and runtime context"
}
```

Import `CHAT_WORKSPACE_PATH` from `@/routes/route-paths` next to the existing route constants.

- [ ] **Step 4: Run route/nav test**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/extension/route-registry.chat-workspace.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit route/nav slice**

Run from repo root:

```bash
git add apps/tldw-frontend/pages/chat-workspace.tsx apps/packages/ui/src/routes/option-chat-workspace.tsx apps/tldw-frontend/extension/routes/option-chat-workspace.tsx apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx apps/packages/ui/src/components/Option/ChatWorkspace/index.ts apps/packages/ui/src/routes/route-paths.ts apps/packages/ui/src/routes/route-registry.tsx apps/tldw-frontend/extension/routes/route-registry.tsx apps/packages/ui/src/services/settings/ui-settings.ts apps/packages/ui/src/components/Layouts/header-shortcut-items.ts apps/tldw-frontend/__tests__/extension/route-registry.chat-workspace.test.ts
git commit -m "Add chat workspace route and navigation"
```

## Task 2: Turn-Level RAG Media Overrides And Submit Result

**Files:**

- Modify: `apps/packages/ui/src/hooks/chat/chat-action-utils.ts`
- Modify: `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/ragMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/documentChatMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/tabChatMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/continueChatMode.ts`
- Test: `apps/packages/ui/src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts`
- Test: `apps/packages/ui/src/hooks/chat/__tests__/chat-submit-result.guard.test.ts`

This keeps the new page from depending on a stale global `ragMediaIds` render and gives it a reliable signal before clearing staged context. The staged send can pass media ids for the current turn through `requestOverrides`.

- [ ] **Step 1: Write failing pure helper tests**

Create `apps/packages/ui/src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import {
  chatSubmitFailed,
  chatSubmitSkipped,
  chatSubmitSubmitted,
  isChatSubmitSuccess,
  resolveTurnRagMediaIds,
  shouldUseRagForTurn
} from "../chat-action-utils"

describe("turn-level RAG media overrides", () => {
  it("prefers override media ids over captured store media ids", () => {
    expect(resolveTurnRagMediaIds([7, 9], [1])).toEqual([7, 9])
  })

  it("falls back to captured store media ids when override is absent", () => {
    expect(resolveTurnRagMediaIds(undefined, [1, 2])).toEqual([1, 2])
  })

  it("uses RAG when file retrieval is enabled and turn media ids exist", () => {
    expect(
      shouldUseRagForTurn({
        selectedKnowledge: null,
        fileRetrievalEnabled: true,
        ragMediaIds: [42]
      })
    ).toBe(true)
  })

  it("does not use RAG for an empty override", () => {
    expect(
      shouldUseRagForTurn({
        selectedKnowledge: null,
        fileRetrievalEnabled: true,
        ragMediaIds: []
      })
    ).toBe(false)
  })

  it("classifies only submitted results as successful", () => {
    expect(isChatSubmitSuccess(chatSubmitSubmitted())).toBe(true)
    expect(isChatSubmitSuccess(chatSubmitFailed("network"))).toBe(false)
    expect(isChatSubmitSuccess(chatSubmitSkipped("validation"))).toBe(false)
    expect(isChatSubmitSuccess(undefined)).toBe(false)
  })
})
```

Create `apps/packages/ui/src/hooks/chat/__tests__/chat-submit-result.guard.test.ts`:

```ts
import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readSource = (relativePath: string) =>
  fs.readFileSync(path.resolve(__dirname, "..", relativePath), "utf8")

describe("chat submit result contract", () => {
  it("does not silently resolve saved pipeline errors as successful submits", () => {
    const pipeline = readSource("../chat-modes/chatModePipeline.ts")

    expect(pipeline).toContain("chatSubmitSubmitted")
    expect(pipeline).toContain("chatSubmitFailed")
    expect(pipeline).toMatch(/return\s+chatSubmitFailed/)
  })

  it("returns submit results from chat actions and mode wrappers", () => {
    const actions = readSource("useChatActions.ts")
    const normal = readSource("../chat-modes/normalChatMode.ts")
    const rag = readSource("../chat-modes/ragMode.ts")

    expect(actions).toContain("chatSubmitSkipped")
    expect(actions).toContain("chatSubmitFailed")
    expect(actions).toContain("chatSubmitSubmitted")
    expect(normal).toMatch(/return\s+(await\s+)?runChatPipeline/)
    expect(rag).toMatch(/return\s+(await\s+)?runChatPipeline/)
  })
})
```

- [ ] **Step 2: Run the test and verify it fails**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts
```

Expected: FAIL because the helpers do not exist.

- [ ] **Step 3: Implement helpers**

Add to `apps/packages/ui/src/hooks/chat/chat-action-utils.ts`:

```ts
export const resolveTurnRagMediaIds = (
  overrideValue: unknown,
  fallbackValue: number[] | null
): number[] | null => {
  if (Array.isArray(overrideValue)) {
    const ids = overrideValue.filter(
      (value): value is number => Number.isInteger(value) && value > 0
    )
    return ids.length > 0 ? Array.from(new Set(ids)) : null
  }
  return Array.isArray(fallbackValue) && fallbackValue.length > 0
    ? Array.from(new Set(fallbackValue))
    : null
}

export const shouldUseRagForTurn = ({
  selectedKnowledge,
  fileRetrievalEnabled,
  ragMediaIds
}: {
  selectedKnowledge: unknown
  fileRetrievalEnabled: boolean
  ragMediaIds: number[] | null
}): boolean =>
  Boolean(selectedKnowledge) ||
  (fileRetrievalEnabled && Array.isArray(ragMediaIds) && ragMediaIds.length > 0)

export type ChatSubmitResult =
  | { status: "submitted" }
  | { status: "failed"; errorMessage: string }
  | { status: "skipped"; reason: string }

export const chatSubmitSubmitted = (): ChatSubmitResult => ({
  status: "submitted"
})

export const chatSubmitFailed = (error: unknown): ChatSubmitResult => ({
  status: "failed",
  errorMessage: error instanceof Error ? error.message : String(error || "Something went wrong.")
})

export const chatSubmitSkipped = (reason: string): ChatSubmitResult => ({
  status: "skipped",
  reason
})

export const isChatSubmitSuccess = (
  result: ChatSubmitResult | void | undefined
): result is { status: "submitted" } => result?.status === "submitted"
```

- [ ] **Step 4: Wire `useChatActions` to the helpers**

Import `chatSubmitFailed`, `chatSubmitSkipped`, `chatSubmitSubmitted`, and the RAG helpers in `apps/packages/ui/src/hooks/chat/useChatActions.ts`.

Replace each local `hasScopedRagMediaIds` / `shouldUseRag` block with turn-aware values:

```ts
const turnRagMediaIds = resolveTurnRagMediaIds(
  chatModeParamsWithRegen.ragMediaIds,
  ragMediaIds
)
const turnChatModeParams = {
  ...chatModeParamsWithRegen,
  ragMediaIds: turnRagMediaIds
}
const shouldUseRag = shouldUseRagForTurn({
  selectedKnowledge: turnChatModeParams.selectedKnowledge,
  fileRetrievalEnabled: Boolean(turnChatModeParams.fileRetrievalEnabled),
  ragMediaIds: turnRagMediaIds
})
if (shouldUseRag) {
  markSteeringApplied()
  return await ragMode(
    message,
    image,
    isRegenerate,
    chatHistory || messages,
    memory || history,
    signal,
    turnChatModeParams
  )
} else {
  // keep existing normal-mode branch
}
```

Apply the same idea in the per-model/compare branch lower in the file, using the local `chatModeParams` object in that scope.

Then make every `submitChat` terminal path return a `ChatSubmitResult`:

- validation/model unavailable paths return `chatSubmitSkipped("validation")` or a more specific reason
- successful mode calls return the mode wrapper result
- successful compare/all-settled branch returns `chatSubmitSubmitted()`
- outer `catch` returns `chatSubmitFailed(e)` after preserving the existing notification behavior

- [ ] **Step 5: Return submit result from chat mode pipeline wrappers**

Modify `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts`:

- import `chatSubmitFailed`, `chatSubmitSubmitted`, and `type ChatSubmitResult`
- annotate `runChatPipeline` as returning `Promise<ChatSubmitResult>`
- after successful `saveMessageOnSuccess`, return `chatSubmitSubmitted()`
- in the `catch` block, after `saveMessageOnError(...)` returns a truthy value, return `chatSubmitFailed(e)`
- keep the existing `throw e` when error persistence fails

Modify each wrapper that currently does `await runChatPipeline(...)`:

```ts
return await runChatPipeline(
  modeDefinition,
  message,
  image,
  isRegenerate,
  messages,
  history,
  signal,
  params
)
```

Apply this in:

- `normalChatMode.ts`
- `ragMode.ts`
- `documentChatMode.ts`
- `tabChatMode.ts`
- `continueChatMode.ts`

- [ ] **Step 6: Run helper and focused existing chat tests**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts ../packages/ui/src/hooks/chat/__tests__/chat-submit-result.guard.test.ts ../packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit chat helper slice**

Run from repo root:

```bash
git add apps/packages/ui/src/hooks/chat/chat-action-utils.ts apps/packages/ui/src/hooks/chat/useChatActions.ts apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts apps/packages/ui/src/hooks/chat-modes/ragMode.ts apps/packages/ui/src/hooks/chat-modes/documentChatMode.ts apps/packages/ui/src/hooks/chat-modes/tabChatMode.ts apps/packages/ui/src/hooks/chat-modes/continueChatMode.ts apps/packages/ui/src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts apps/packages/ui/src/hooks/chat/__tests__/chat-submit-result.guard.test.ts
git commit -m "Support chat workspace submit results"
```

## Task 3: Staged Source Model

**Files:**

- Create: `apps/packages/ui/src/components/Option/ChatWorkspace/types.ts`
- Create: `apps/packages/ui/src/components/Option/ChatWorkspace/staging.ts`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/staging.test.ts`

- [ ] **Step 1: Write failing staging tests**

Create `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/staging.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import type { WorkspaceSource } from "@/types/workspace"
import {
  buildStagedSourceFromWorkspaceSource,
  formatStagedSourceInsertText,
  getReadyStagedMediaIds,
  stageWorkspaceSources
} from "../staging"

const source = (overrides: Partial<WorkspaceSource> = {}): WorkspaceSource => ({
  id: "source-1",
  mediaId: 101,
  title: "Operator Notes",
  type: "document",
  status: "ready",
  addedAt: new Date("2026-05-03T00:00:00Z"),
  ...overrides
})

describe("chat workspace staging", () => {
  it("builds explicit staged source metadata from a workspace source", () => {
    expect(buildStagedSourceFromWorkspaceSource(source(), "Default workspace")).toMatchObject({
      sourceId: "source-1",
      mediaId: 101,
      title: "Operator Notes",
      type: "document",
      scopeLabel: "Default workspace",
      availability: "ready"
    })
  })

  it("deduplicates staged sources by source id", () => {
    const staged = stageWorkspaceSources(
      [buildStagedSourceFromWorkspaceSource(source(), "A")],
      [source({ title: "Renamed" })],
      "A"
    )

    expect(staged).toHaveLength(1)
    expect(staged[0].title).toBe("Renamed")
  })

  it("formats insert text and leaves sending to the user", () => {
    const staged = [buildStagedSourceFromWorkspaceSource(source(), "Default workspace")]
    expect(formatStagedSourceInsertText(staged)).toContain("Context sources")
    expect(formatStagedSourceInsertText(staged)).toContain("Operator Notes")
  })

  it("returns only ready positive media ids for structured RAG", () => {
    const ready = buildStagedSourceFromWorkspaceSource(source(), "A")
    const error = buildStagedSourceFromWorkspaceSource(
      source({ id: "source-2", mediaId: 202, status: "error" }),
      "A"
    )
    expect(getReadyStagedMediaIds([ready, error])).toEqual([101])
  })
})
```

- [ ] **Step 2: Run the test and verify it fails**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__/staging.test.ts
```

Expected: FAIL because `types.ts` and `staging.ts` do not exist.

- [ ] **Step 3: Implement types and helpers**

Implement `types.ts`:

```ts
import type { WorkspaceSourceType } from "@/types/workspace"

export type StagedSourceAvailability =
  | "ready"
  | "processing"
  | "error"
  | "unavailable"

export type StagedWorkspaceSource = {
  sourceId: string
  mediaId: number | null
  title: string
  type: WorkspaceSourceType
  scopeLabel: string
  availability: StagedSourceAvailability
  statusMessage?: string
}

export type ChatWorkspaceRuntimeState = {
  backendAvailable: boolean
  streaming: boolean
  selectedModelLabel: string
  selectedPersonaLabel: string | null
}
```

Implement `staging.ts`:

```ts
import type { WorkspaceSource } from "@/types/workspace"
import type { StagedSourceAvailability, StagedWorkspaceSource } from "./types"

const toAvailability = (source: WorkspaceSource): StagedSourceAvailability => {
  if (source.status === "processing") return "processing"
  if (source.status === "error") return "error"
  if (source.status === "ready" || !source.status) return "ready"
  return "unavailable"
}

export const buildStagedSourceFromWorkspaceSource = (
  source: WorkspaceSource,
  scopeLabel: string
): StagedWorkspaceSource => ({
  sourceId: source.id,
  mediaId: Number.isInteger(source.mediaId) && source.mediaId > 0 ? source.mediaId : null,
  title: source.title,
  type: source.type,
  scopeLabel,
  availability: toAvailability(source),
  statusMessage: source.statusMessage
})

export const stageWorkspaceSources = (
  existing: StagedWorkspaceSource[],
  sources: WorkspaceSource[],
  scopeLabel: string
): StagedWorkspaceSource[] => {
  const byId = new Map(existing.map((item) => [item.sourceId, item]))
  for (const source of sources) {
    byId.set(source.id, buildStagedSourceFromWorkspaceSource(source, scopeLabel))
  }
  return Array.from(byId.values())
}

export const formatStagedSourceInsertText = (
  sources: StagedWorkspaceSource[]
): string => {
  if (sources.length === 0) return ""
  const lines = sources.map((source, index) => {
    const state = source.availability === "ready" ? "" : ` (${source.availability})`
    return `${index + 1}. ${source.title} [${source.type}]${state}`
  })
  return `Context sources:\n${lines.join("\n")}\n\n`
}

export const getReadyStagedMediaIds = (
  sources: StagedWorkspaceSource[]
): number[] =>
  Array.from(
    new Set(
      sources
        .filter((source) => source.availability === "ready")
        .map((source) => source.mediaId)
        .filter((mediaId): mediaId is number => Number.isInteger(mediaId) && mediaId > 0)
    )
  )
```

- [ ] **Step 4: Run staging tests**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__/staging.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit staging slice**

Run from repo root:

```bash
git add apps/packages/ui/src/components/Option/ChatWorkspace/types.ts apps/packages/ui/src/components/Option/ChatWorkspace/staging.ts apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/staging.test.ts
git commit -m "Add chat workspace staged source model"
```

## Task 4: Staged Context Card

**Files:**

- Create: `apps/packages/ui/src/components/Option/ChatWorkspace/ContextStagingCard.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/index.ts`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/ContextStagingCard.test.tsx`

- [ ] **Step 1: Write failing component tests**

Create `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/ContextStagingCard.test.tsx`:

```tsx
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ContextStagingCard } from "../ContextStagingCard"
import type { StagedWorkspaceSource } from "../types"

const staged: StagedWorkspaceSource[] = [
  {
    sourceId: "s1",
    mediaId: 1,
    title: "Operator Notes",
    type: "document",
    scopeLabel: "Default workspace",
    availability: "ready"
  }
]

describe("ContextStagingCard", () => {
  it("renders staged sources as not sent", () => {
    render(<ContextStagingCard sources={staged} onClear={vi.fn()} onInsert={vi.fn()} onSend={vi.fn()} />)

    expect(screen.getByText("Context staged - not sent")).toBeInTheDocument()
    expect(screen.getByText("Operator Notes")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Clear staged context" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Insert context summary" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Send with staged context" })).toBeInTheDocument()
  })

  it("shows unavailable warnings", () => {
    render(
      <ContextStagingCard
        sources={[{ ...staged[0], availability: "error", statusMessage: "Source failed" }]}
        onClear={vi.fn()}
        onInsert={vi.fn()}
        onSend={vi.fn()}
      />
    )

    expect(screen.getByText("Source failed")).toBeInTheDocument()
    expect(screen.getByText("error")).toBeInTheDocument()
  })

  it("calls clear, insert, and send actions", () => {
    const onClear = vi.fn()
    const onInsert = vi.fn()
    const onSend = vi.fn()

    render(<ContextStagingCard sources={staged} onClear={onClear} onInsert={onInsert} onSend={onSend} />)

    fireEvent.click(screen.getByRole("button", { name: "Clear staged context" }))
    fireEvent.click(screen.getByRole("button", { name: "Insert context summary" }))
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    expect(onClear).toHaveBeenCalledTimes(1)
    expect(onInsert).toHaveBeenCalledTimes(1)
    expect(onSend).toHaveBeenCalledTimes(1)
  })
})
```

- [ ] **Step 2: Run test and verify it fails**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__/ContextStagingCard.test.tsx
```

Expected: FAIL because `ContextStagingCard` does not exist.

- [ ] **Step 3: Implement the card**

Implement with:

- `section` or `aside` with `aria-label="Staged context"`
- compact border/background using existing theme classes like `border-border`, `bg-surface`, `text-muted-foreground`
- button labels exactly as tested
- disabled Send state when `isSending` is true
- per-source status text that does not rely only on color

Export from `index.ts`:

```ts
export { ContextStagingCard } from "./ContextStagingCard"
export type { StagedWorkspaceSource } from "./types"
```

- [ ] **Step 4: Run component test**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__/ContextStagingCard.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit card slice**

Run from repo root:

```bash
git add apps/packages/ui/src/components/Option/ChatWorkspace/ContextStagingCard.tsx apps/packages/ui/src/components/Option/ChatWorkspace/index.ts apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/ContextStagingCard.test.tsx
git commit -m "Add chat workspace staged context card"
```

## Task 5: Inspector Rail And Status Strip

**Files:**

- Create: `apps/packages/ui/src/components/Option/ChatWorkspace/InspectorRail.tsx`
- Create: `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceStatusStrip.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/index.ts`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/InspectorRail.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceStatusStrip.test.tsx`

- [ ] **Step 1: Write failing tests**

Create `InspectorRail.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { InspectorRail } from "../InspectorRail"

describe("InspectorRail", () => {
  it("shows real scope and staged source state", () => {
    render(
      <InspectorRail
        scopeLabel="Default workspace"
        stagedSourceCount={2}
        stagedSourceTitles={["Operator Notes", "Research Clip"]}
        selectedModelLabel="gpt-test"
        selectedPersonaLabel="Analyst"
        backendAvailable
        streaming={false}
      />
    )

    expect(screen.getByText("Default workspace")).toBeInTheDocument()
    expect(screen.getByText("Operator Notes")).toBeInTheDocument()
    expect(screen.getByText("gpt-test")).toBeInTheDocument()
    expect(screen.getByText("Analyst")).toBeInTheDocument()
  })

  it("labels inactive v1 panels honestly", () => {
    render(
      <InspectorRail
        scopeLabel="No workspace"
        stagedSourceCount={0}
        stagedSourceTitles={[]}
        selectedModelLabel="No model selected"
        selectedPersonaLabel={null}
        backendAvailable={false}
        streaming={false}
      />
    )

    expect(screen.getByText("Not configured")).toBeInTheDocument()
    expect(screen.getByText("No active task")).toBeInTheDocument()
    expect(screen.getByText("Server unavailable")).toBeInTheDocument()
  })
})
```

Create `WorkspaceStatusStrip.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { WorkspaceStatusStrip } from "../WorkspaceStatusStrip"

describe("WorkspaceStatusStrip", () => {
  it("renders ready and keyboard hint state", () => {
    render(<WorkspaceStatusStrip backendAvailable streaming={false} stagedSourceCount={0} />)

    expect(screen.getByText("Ready")).toBeInTheDocument()
    expect(screen.getByText("Ctrl+K command")).toBeInTheDocument()
    expect(screen.getByText("Ctrl+Enter send")).toBeInTheDocument()
  })

  it("renders streaming, staged context, and backend unavailable states", () => {
    render(<WorkspaceStatusStrip backendAvailable={false} streaming stagedSourceCount={3} />)

    expect(screen.getByText("Streaming")).toBeInTheDocument()
    expect(screen.getByText("Context staged")).toBeInTheDocument()
    expect(screen.getByText("Server unavailable")).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run tests and verify they fail**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__/InspectorRail.test.tsx ../packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceStatusStrip.test.tsx
```

Expected: FAIL because components do not exist.

- [ ] **Step 3: Implement components**

Implementation requirements:

- `InspectorRail` uses headings `Scope`, `Sources`, `Model / Persona`, `Approvals`, `Task Progress`, `Runtime`.
- `Approvals` shows `Not configured`.
- `Task Progress` shows `No active task`.
- Runtime shows `Server unavailable` when `backendAvailable` is false, otherwise `Ready` or `Streaming`.
- `WorkspaceStatusStrip` is a compact `footer` with `aria-label="Chat workspace status"`.
- Both components avoid icon-only content unless there is accessible text.

- [ ] **Step 4: Run tests**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__/InspectorRail.test.tsx ../packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceStatusStrip.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit inspector/status slice**

Run from repo root:

```bash
git add apps/packages/ui/src/components/Option/ChatWorkspace/InspectorRail.tsx apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceStatusStrip.tsx apps/packages/ui/src/components/Option/ChatWorkspace/index.ts apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/InspectorRail.test.tsx apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceStatusStrip.test.tsx
git commit -m "Add chat workspace inspector and status strip"
```

## Task 6: Workspace Rail With Separate Browse And Stage States

**Files:**

- Create: `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceRail.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/index.ts`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceRail.test.tsx`

- [ ] **Step 1: Write failing rail tests**

Create `WorkspaceRail.test.tsx`:

```tsx
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { WorkspaceRail } from "../WorkspaceRail"
import type { WorkspaceSource } from "@/types/workspace"

const sources: WorkspaceSource[] = [
  {
    id: "source-1",
    mediaId: 101,
    title: "Operator Notes",
    type: "document",
    status: "ready",
    addedAt: new Date("2026-05-03T00:00:00Z")
  },
  {
    id: "source-2",
    mediaId: 202,
    title: "Research Clip",
    type: "video",
    status: "ready",
    addedAt: new Date("2026-05-03T00:00:00Z")
  }
]

describe("WorkspaceRail", () => {
  it("selecting a source for browsing does not stage it", () => {
    const onBrowseSource = vi.fn()
    const onStageSources = vi.fn()

    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={sources}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={onBrowseSource}
        onStageSources={onStageSources}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Browse Operator Notes" }))

    expect(onBrowseSource).toHaveBeenCalledWith("source-1")
    expect(onStageSources).not.toHaveBeenCalled()
    expect(screen.queryByText("Context staged")).not.toBeInTheDocument()
  })

  it("stages only through the explicit Stage for Chat action", () => {
    const onStageSources = vi.fn()

    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={sources}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={onStageSources}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Stage Operator Notes for chat" }))

    expect(onStageSources).toHaveBeenCalledWith(["source-1"])
  })

  it("filters sources without changing browse or staged state", () => {
    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={sources}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={vi.fn()}
      />
    )

    fireEvent.change(screen.getByRole("searchbox", { name: "Filter sources" }), {
      target: { value: "clip" }
    })

    expect(screen.queryByText("Operator Notes")).not.toBeInTheDocument()
    expect(screen.getByText("Research Clip")).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test and verify it fails**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceRail.test.tsx
```

Expected: FAIL because `WorkspaceRail` does not exist.

- [ ] **Step 3: Implement the rail**

Implementation requirements:

- Left rail header shows active workspace name.
- Include a source filter/search input with accessible name `Filter sources`; filtering only changes visible rows and must not stage or browse a source.
- Source rows are compact and keyboard reachable.
- `Browse <title>` button updates only local focus.
- `Stage <title> for chat` button is the only source-row action that stages.
- Disabled or non-ready sources show their status in text.
- Study section exists but uses honest v1 labels such as `No generated study set`.
- Library shortcuts may render as non-primary buttons but must not claim unavailable behavior works.

- [ ] **Step 4: Run rail test**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceRail.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit rail slice**

Run from repo root:

```bash
git add apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceRail.tsx apps/packages/ui/src/components/Option/ChatWorkspace/index.ts apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceRail.test.tsx
git commit -m "Add chat workspace source rail"
```

## Task 7: Chat Panel Adapter

**Files:**

- Create: `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/index.ts`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx`

- [ ] **Step 1: Write failing chat panel tests**

Create `WorkspaceChatPanel.test.tsx`:

```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { WorkspaceChatPanel } from "../WorkspaceChatPanel"
import type { StagedWorkspaceSource } from "../types"

const chatHookState = vi.hoisted(() => {
  const onSubmit = vi.fn(async (): Promise<any> => ({ status: "submitted" }))
  const stopStreamingRequest = vi.fn()
  const value: any = {
    messages: [],
    onSubmit,
    streaming: false,
    isLoading: false,
    isProcessing: false,
    stopStreamingRequest,
    selectedModel: "gpt-test",
    selectedAssistant: { kind: "persona", id: "p1", name: "Analyst" }
  }
  const useMessageOption = vi.fn(() => value)

  return { onSubmit, stopStreamingRequest, useMessageOption, value }
})

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: (...args: unknown[]) => chatHookState.useMessageOption(...args)
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: { message: string }) => <article>{props.message}</article>
}))

const staged: StagedWorkspaceSource[] = [
  {
    sourceId: "source-1",
    mediaId: 101,
    title: "Operator Notes",
    type: "document",
    scopeLabel: "Default workspace",
    availability: "ready"
  }
]

describe("WorkspaceChatPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    chatHookState.value.streaming = false
    chatHookState.value.isLoading = false
    chatHookState.value.isProcessing = false
    chatHookState.onSubmit.mockResolvedValue({ status: "submitted" })
  })

  it("inserts staged source summary into the composer without sending and clears structured staging", () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Insert context summary" }))

    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue(
      expect.stringContaining("Operator Notes")
    )
    expect(chatHookState.onSubmit).not.toHaveBeenCalled()
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
  })

  it("sends with staged context through the shared chat path", async () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Summarize this" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(chatHookState.onSubmit.mock.calls[0][0]).toMatchObject({
      message: expect.stringContaining("Summarize this"),
      image: "",
      requestOverrides: expect.objectContaining({
        ragMediaIds: [101],
        fileRetrievalEnabled: true
      })
    })
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
  })

  it("preserves draft and staged context when submit returns a failed result", async () => {
    chatHookState.onSubmit.mockResolvedValueOnce({
      status: "failed",
      errorMessage: "network"
    })
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Keep this draft" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await screen.findByText("Send failed")
    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("Keep this draft")
    expect(onClearStagedSources).not.toHaveBeenCalled()
  })

  it("also preserves draft and staged context when submit rejects unexpectedly", async () => {
    chatHookState.onSubmit.mockRejectedValueOnce(new Error("network"))
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Keep this draft" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await screen.findByText("Send failed")
    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("Keep this draft")
    expect(onClearStagedSources).not.toHaveBeenCalled()
  })

  it("shows loading state and wires stop streaming to the shared abort handler", () => {
    chatHookState.value.streaming = true
    chatHookState.value.isProcessing = true

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    expect(screen.getByText("Streaming")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Stop generating" }))
    expect(chatHookState.stopStreamingRequest).toHaveBeenCalledTimes(1)
  })

  it("uses workspace chat scope and reports runtime state", () => {
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    expect(chatHookState.useMessageOption).toHaveBeenCalledWith({
      scope: { type: "workspace", workspaceId: "workspace-1" }
    })
    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({
        streaming: false,
        selectedModelLabel: "gpt-test",
        selectedPersonaLabel: "Analyst"
      })
    )
  })
})
```

- [ ] **Step 2: Run test and verify it fails**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx
```

Expected: FAIL because `WorkspaceChatPanel` does not exist.

- [ ] **Step 3: Implement chat panel**

Implementation requirements:

- Use `useMessageOption({ scope })`, where `scope` is `{ type: "workspace", workspaceId }` when the page has a workspace id, otherwise `{ type: "global" }`.
- Props include `workspaceId?: string | null`, `workspaceName?: string | null`, and `onRuntimeStateChange?: (state: ChatWorkspaceRuntimeState) => void`.
- Render existing messages using `PlaygroundMessage` with required no-op handlers for edit/regenerate where this v1 page does not expose those actions.
- Own a local `draft` string for the compact Terminal-Literal composer.
- Render `ContextStagingCard` above the composer when `stagedSources.length > 0`.
- `Insert` prepends or appends `formatStagedSourceInsertText(stagedSources)` to the draft and then calls `onClearStagedSources()`.
- `Clear` calls `onClearStagedSources()` and leaves the draft unchanged.
- `Send` calls:

```ts
await onSubmit({
  message: sendMessage,
  image: "",
  requestOverrides: {
    ragMediaIds: getReadyStagedMediaIds(stagedSources),
    fileRetrievalEnabled: getReadyStagedMediaIds(stagedSources).length > 0,
    chatMode: getReadyStagedMediaIds(stagedSources).length > 0 ? "rag" : "normal"
  }
})
```

- Import `isChatSubmitSuccess` from `@/hooks/chat/chat-action-utils`.
- Only clear draft and staged sources when `isChatSubmitSuccess(result)` is true.
- If `onSubmit` returns `{ status: "failed" }`, `{ status: "skipped" }`, `undefined`, or rejects, show `Send failed`, preserve the draft, and preserve staged sources. This protects the staged context from hidden failures because chat modes can save an error assistant message and resolve the promise.
- Render loading/abort state from existing chat state:
  - when `streaming || isProcessing || isLoading`, show a visible `Streaming` or `Sending` status
  - while streaming, show a `Stop generating` button that calls `stopStreamingRequest()`
  - disable normal send buttons while `isLoading || isProcessing`
- Report real runtime state after reading `useMessageOption`. Run the callback from a `useEffect` with stable dependencies and only the fields below so parent state updates do not create a render loop:

```ts
React.useEffect(() => {
  onRuntimeStateChange?.({
    backendAvailable,
    streaming,
    selectedModelLabel: selectedModel || "No model selected",
    selectedPersonaLabel: selectedAssistant?.name ?? null
  })
}, [backendAvailable, onRuntimeStateChange, selectedAssistant?.name, selectedModel, streaming])
```

- Disable send when there is no draft and no staged context.
- Use accessible labels:
  - textarea: `Chat workspace message`
  - main send button: `Send message`
  - staged send button comes from `ContextStagingCard`: `Send with staged context`

- [ ] **Step 4: Run chat panel test**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit chat adapter slice**

Run from repo root:

```bash
git add apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx apps/packages/ui/src/components/Option/ChatWorkspace/index.ts apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx
git commit -m "Add chat workspace chat adapter"
```

## Task 8: Page Orchestration And Console Layout

**Files:**

- Create: `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspaceConsole.tsx`
- Create: `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/index.ts`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/ChatWorkspacePage.test.tsx`

- [ ] **Step 1: Write failing page orchestration test**

Create `ChatWorkspacePage.test.tsx`:

```tsx
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ChatWorkspacePage } from "../ChatWorkspacePage"

const setRouteContext = vi.fn()

vi.mock("@/store/chat-surface-coordinator", () => ({
  useChatSurfaceCoordinatorStore: (selector: any) =>
    selector({ setRouteContext })
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (selector: any) =>
    selector({
      workspaceId: "workspace-1",
      workspaceName: "Default workspace",
      sources: [
        {
          id: "source-1",
          mediaId: 101,
          title: "Operator Notes",
          type: "document",
          status: "ready",
          addedAt: new Date("2026-05-03T00:00:00Z")
        }
      ]
    })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    phase: "connected",
    isConnected: true,
    serverUrl: "http://127.0.0.1:8000"
  })
}))

vi.mock("../WorkspaceChatPanel", () => ({
  WorkspaceChatPanel: ({
    stagedSources,
    workspaceId,
    onRuntimeStateChange
  }: {
    stagedSources: unknown[]
    workspaceId?: string | null
    onRuntimeStateChange?: (state: unknown) => void
  }) => {
    React.useEffect(() => {
      onRuntimeStateChange?.({
        backendAvailable: true,
        streaming: true,
        selectedModelLabel: "gpt-test",
        selectedPersonaLabel: "Analyst"
      })
    }, [onRuntimeStateChange])
    return (
      <section data-testid="workspace-chat-panel">
        staged:{stagedSources.length}; workspace:{workspaceId}
      </section>
    )
  }
}))

describe("ChatWorkspacePage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("sets chat surface route context and renders the console regions", () => {
    render(<ChatWorkspacePage />)

    expect(setRouteContext).toHaveBeenCalledWith({
      routeId: "chat-workspace",
      surface: "webui"
    })
    expect(screen.getByRole("complementary", { name: "Workspace sources" })).toBeInTheDocument()
    expect(screen.getByTestId("workspace-chat-panel")).toBeInTheDocument()
    expect(screen.getByRole("complementary", { name: "Workspace inspector" })).toBeInTheDocument()
  })

  it("stages sources only through the explicit rail action", () => {
    render(<ChatWorkspacePage />)

    fireEvent.click(screen.getByRole("button", { name: "Browse Operator Notes" }))
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent("staged:0")

    fireEvent.click(screen.getByRole("button", { name: "Stage Operator Notes for chat" }))
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent("staged:1")
  })

  it("passes workspace scope and real runtime state into the visible rails", async () => {
    render(<ChatWorkspacePage />)

    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent("workspace:workspace-1")
    expect(await screen.findByText("gpt-test")).toBeInTheDocument()
    expect(screen.getByText("Analyst")).toBeInTheDocument()
    expect(screen.getByText("Streaming")).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test and verify it fails**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__/ChatWorkspacePage.test.tsx
```

Expected: FAIL because page orchestration and console layout do not exist.

- [ ] **Step 3: Implement `ChatWorkspaceConsole`**

Implementation requirements:

- Root element: `data-testid="chat-workspace-console"`.
- Desktop: three-column grid such as `lg:grid-cols-[minmax(260px,320px)_minmax(0,1fr)_minmax(280px,340px)]`.
- Tablet/mobile: chat-first stacking, rails below or collapsible via simple tabs if necessary.
- Stable dimensions: `min-h-0`, `overflow-hidden`, and internal scroll containers so text does not push the shell.
- No nested decorative cards. Panels are direct console regions.
- Terminal-Literal visual treatment uses compact borders, small labels, and existing semantic colors.

- [ ] **Step 4: Implement `ChatWorkspacePage`**

Implementation requirements:

- Set route context on mount:

```ts
setRouteContext({ routeId: "chat-workspace", surface: "webui" })
```

- Read workspace state:
  - `workspaceId`
  - `workspaceName`
  - `sources`
- Read backend connection state with `useConnectionState()` from `@/hooks/useConnectionState` and derive:

```ts
const backendAvailable =
  connectionState.isConnected && connectionState.phase === ConnectionPhase.CONNECTED
```

Import `ConnectionPhase` from `@/types/connection`.
- Maintain local:
  - `browsedSourceId`
  - `stagedSources`
  - `runtimeState`, initialized to:

```ts
{
  backendAvailable,
  streaming: false,
  selectedModelLabel: "No model selected",
  selectedPersonaLabel: null
}
```
- Stage sources with:

```ts
const handleStageSources = (sourceIds: string[]) => {
  const selected = sources.filter((source) => sourceIds.includes(source.id))
  setStagedSources((current) =>
    stageWorkspaceSources(current, selected, workspaceName || "Workspace")
  )
}
```

- Pass `workspaceId` and `workspaceName` into `WorkspaceChatPanel` so it can call `useMessageOption({ scope: { type: "workspace", workspaceId } })`.
- Pass `onRuntimeStateChange` into `WorkspaceChatPanel` and merge the callback into `runtimeState`.
- Pass staged state plus `runtimeState.selectedModelLabel`, `runtimeState.selectedPersonaLabel`, `runtimeState.streaming`, and `backendAvailable` to `InspectorRail`.
- Pass `runtimeState.streaming`, staged count, and `backendAvailable` to `WorkspaceStatusStrip`.
- Replace the temporary `ChatWorkspacePage` stub export in `index.ts`.

- [ ] **Step 5: Run page test**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__/ChatWorkspacePage.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Run all ChatWorkspace component tests**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/ChatWorkspace/__tests__
```

Expected: PASS.

- [ ] **Step 7: Commit page orchestration slice**

Run from repo root:

```bash
git add apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspaceConsole.tsx apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx apps/packages/ui/src/components/Option/ChatWorkspace/index.ts apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/ChatWorkspacePage.test.tsx
git commit -m "Assemble chat workspace console page"
```

## Task 9: Smoke, A11y, And Release Gate Coverage

**Files:**

- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/stage5-release-gate.spec.ts`
- Test: existing e2e smoke specs

- [ ] **Step 1: Add `/chat-workspace` to inventories**

Add near `/chat` or workspace routes as appropriate:

```ts
{ path: "/chat-workspace", name: "Chat Workspace", category: "workspace" },
```

Add to `HIGH_RISK_ROUTES` in `stage4-axe-high-risk-routes.spec.ts`:

```ts
{ path: "/chat-workspace", name: "Chat Workspace" },
```

Add to `CRITICAL_ROUTES` in `stage5-release-gate.spec.ts`:

```ts
{ path: "/chat-workspace", name: "Chat Workspace" },
```

- [ ] **Step 2: Run focused route/nav and component tests**

Run from `apps/tldw-frontend`:

```bash
bun run compile
```

Expected: PASS. If the full Next compile has known unrelated baseline failures, capture the first unrelated failure and still run the focused Vitest and Playwright checks below.

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/extension/route-registry.chat-workspace.test.ts ../packages/ui/src/components/Option/ChatWorkspace/__tests__ ../packages/ui/src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts
```

Expected: PASS.

- [ ] **Step 3: Run existing route regression tests**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/extension/route-registry.workspace-playground.test.ts ../packages/ui/src/components/Layouts/__tests__/HeaderShortcuts.test.tsx ../packages/ui/src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts
```

Expected: PASS. If shortcut count expectations fail, update expected lists to include `chat-workspace` only when the test is asserting the complete shortcut inventory.

- [ ] **Step 4: Run focused smoke/a11y checks**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/smoke/stage4-axe-high-risk-routes.spec.ts --reporter=line --grep "Chat Workspace"
```

Expected: PASS with no serious axe violations.

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/smoke/stage5-release-gate.spec.ts --reporter=line --grep "Chat Workspace"
```

Expected: PASS with no global error boundary or console budget failure.

- [ ] **Step 5: Run lint on touched frontend files**

Run from `apps/tldw-frontend`:

```bash
bunx eslint pages/chat-workspace.tsx components/layout/WebLayout.tsx ../packages/ui/src/routes/route-paths.ts ../packages/ui/src/routes/route-registry.tsx ../packages/ui/src/components/Option/ChatWorkspace ../packages/ui/src/components/Layouts/header-shortcut-items.ts ../packages/ui/src/services/settings/ui-settings.ts
```

Expected: PASS. If `components/layout/WebLayout.tsx` was not touched, omit it from the command.

- [ ] **Step 6: Commit smoke/a11y slice**

Run from repo root:

```bash
git add apps/tldw-frontend/e2e/smoke/page-inventory.ts apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts apps/tldw-frontend/e2e/smoke/stage5-release-gate.spec.ts
git commit -m "Cover chat workspace in smoke checks"
```

## Task 10: Final Verification And Manual Browser Check

**Files:**

- No required file changes unless verification exposes defects.

- [ ] **Step 1: Check worktree hygiene**

Run from repo root:

```bash
git status --short
```

Expected: only intended changes before final commit, with unrelated `Docs/Design/Agents.md` and `.superpowers/` not staged.

- [ ] **Step 2: Run focused Vitest suite**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/extension/route-registry.chat-workspace.test.ts ../packages/ui/src/components/Option/ChatWorkspace/__tests__ ../packages/ui/src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts
```

Expected: PASS.

- [ ] **Step 3: Run broader route and nav regression suite**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/extension/route-registry.workspace-playground.test.ts ../packages/ui/src/components/Layouts/__tests__/HeaderShortcuts.test.tsx ../packages/ui/src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts ../packages/ui/src/components/Layouts/__tests__/header-shortcut-descriptions.test.ts
```

Expected: PASS.

- [ ] **Step 4: Start local web UI for manual check**

Run from `apps/tldw-frontend`:

```bash
bun run dev -- -p 18001
```

Expected: dev server starts and serves `http://127.0.0.1:18001`.

- [ ] **Step 5: Browser-check desktop and mobile**

Open:

- `http://127.0.0.1:18001/chat-workspace`
- `http://127.0.0.1:18001/chat`
- `http://127.0.0.1:18001/workspace-playground`

Verify:

- existing web header/titlebar and sidebar are visible
- `/chat-workspace` fills the content viewport
- desktop shows left rail, center chat, right inspector, and bottom status strip
- mobile keeps chat primary and does not overlap controls/text
- capture one desktop screenshot and one mobile screenshot for review notes
- keyboard tab order reaches source filter, browse/stage actions, staged context controls, composer, stop/send controls, and inspector/status regions
- browsing a source does not stage it
- `Stage for Chat` creates the staged card
- `Insert` writes source summary into composer and clears staged metadata
- failed send preserves draft and staged sources
- inactive inspector sections use honest labels

- [ ] **Step 6: Stop local dev server**

Stop the server from the terminal session with `Ctrl+C`.

- [ ] **Step 7: Final diff check**

Run from repo root:

```bash
git diff --check
```

Expected: no whitespace errors.

Run from repo root:

```bash
git status --short
```

Expected: clean except unrelated pre-existing files that were intentionally not staged or committed.

## Completion Criteria

- `/chat-workspace` exists as a Next page and shared UI route.
- Route is visible through header/sidebar shortcut configuration.
- Route is viewport constrained and uses the existing web shell.
- Terminal-Literal layout renders as a dense operator console, not a landing page.
- Source browsing and source staging are separate states.
- Staged context card supports Clear, Insert, and Send.
- Insert clears structured staged metadata after writing text into the composer.
- Send uses the existing chat action path and passes ready staged media ids as turn-level RAG overrides.
- Send failures preserve draft text and staged source metadata.
- Inspector and status strip expose system status and honest v1 inactive states.
- `/chat` and `/workspace-playground` focused route tests still pass.
- Focused Vitest suite, route/nav tests, smoke/a11y checks, `eslint`, and `git diff --check` pass or any environment-only failures are documented with exact output.
