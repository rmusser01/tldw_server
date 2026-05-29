# Sidepanel Chat WebUI Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the explicit sidepanel **Continue in WebUI** handoff so `/chat` opens with the current sidepanel draft and visible page context, without changing the existing route-only open action.

**Architecture:** Add a focused extension-local handoff service that creates bounded, one-time packages and fails closed on storage errors. Wire a new ControlRow quick action to create a package and open `/chat?handoff=<id>`. In WebUI `/chat`, read the hash-route handoff, prefill the composer, render an imported-context banner, include that context in the next chat request, and consume the package only after a terminal import outcome.

**Tech Stack:** React 18, TypeScript, WXT `browser`, `@plasmohq/storage` via `createSafeStorage`, React Router hash routes, Vitest/jsdom, Testing Library.

---

## Preconditions

- Work from `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-post-regression-next`.
- Before code edits, create or claim a Backlog task for implementation work, separate from design task `TASK-546` and planning task `TASK-547`.
- Keep the existing route-only full-screen/open behavior and tests intact.
- Use `apps/packages/ui` as the test working directory.

## File Structure

- Create: `apps/packages/ui/src/services/sidepanel-chat-handoff.ts`
  - Owns handoff types, constants, validation, payload bounding, storage create/read/consume/cleanup, route building, and model-message composition.
  - Uses `createSafeStorage({ area: "local" })`, not `createLocalRegistryBucket`, because creation must fail closed.
- Create: `apps/packages/ui/src/services/__tests__/sidepanel-chat-handoff.test.ts`
  - Covers package creation, validation, read-back failure, expiry, consume, route merging, payload bounds, and message-for-model composition.
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx`
  - Adds the **Continue in WebUI** quick action.
  - Keeps `chat-open-full-app` route-only.
  - Calls the handoff service, then opens the returned full app URL only after storage succeeds.
  - Uses `useAntdNotification` or the nearest existing sidepanel notification helper for storage/create failure feedback.
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
  - Passes `draftMessage={form.values.message}` to `ControlRow`.
  - Passes a `getVisiblePageContextForHandoff` callback that returns selected tab mentions and active-tab title/URL when the existing Current page chip is active.
- Create: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ControlRow.chat-handoff.test.tsx`
  - Covers the new action, disabled state, route-only action preservation, role-play query merge, storage failure, and no draft/context-in-URL leakage.
- Create: `apps/packages/ui/src/components/Option/Playground/SidepanelImportedContextBanner.tsx`
  - Renders imported page context above the composer with remove action and accessible labels.
- Create: `apps/packages/ui/src/components/Option/Playground/hooks/useSidepanelChatHandoffImport.ts`
  - Reads `location.search`, imports or prompts on existing drafts, consumes packages after terminal outcomes, and returns imported context state plus conflict actions.
- Create: `apps/packages/ui/src/components/Option/Playground/__tests__/sidepanel-chat-handoff-import.test.tsx`
  - Covers valid import, existing draft insert/replace/cancel, stale handoff feedback, hash-route cleanup, and request inclusion/removal through a small test harness.
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
  - Calls the import hook.
  - Renders `SidepanelImportedContextBanner` above `AttachmentsSummary`.
  - Passes imported context into submit and clears it after send or dismissal.
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts`
  - Accepts optional imported sidepanel context.
  - Builds `requestOverrides.messageForModel` for the next send while keeping the visible user draft unchanged.
  - Clears imported context only after dispatch starts.

## Task 1: Handoff Storage Service

**Files:**
- Create: `apps/packages/ui/src/services/sidepanel-chat-handoff.ts`
- Test: `apps/packages/ui/src/services/__tests__/sidepanel-chat-handoff.test.ts`

- [ ] **Step 1: Write failing service tests**

Cover these named scenarios:

```ts
it("creates bounded packages and verifies read-back before returning the id", async () => {})
it("throws and does not return a handoff id when storage set fails", async () => {})
it("throws when read-back verification cannot read the saved package", async () => {})
it("returns null and removes expired or malformed packages", async () => {})
it("consumes a package exactly once", async () => {})
it("merges handoff into normal and character /chat hash routes", () => {})
it("never puts draft text or snippet text into the route", () => {})
it("builds messageForModel with visible sidepanel context", () => {})
```

- [ ] **Step 2: Run tests to verify RED**

Run from `apps/packages/ui`:

```bash
bun run test src/services/__tests__/sidepanel-chat-handoff.test.ts
```

Expected: fail because the module does not exist.

- [ ] **Step 3: Implement service types and constants**

Use this shape as the starting point:

```ts
export const SIDEPANEL_CHAT_HANDOFF_TTL_MS = 10 * 60 * 1000
export const SIDEPANEL_CHAT_HANDOFF_STORAGE_PREFIX =
  "tldw:sidepanel-chat-handoff:"
export const SIDEPANEL_CHAT_HANDOFF_MAX_SNIPPETS = 4
export const SIDEPANEL_CHAT_HANDOFF_MAX_SNIPPET_CHARS = 4_000
export const SIDEPANEL_CHAT_HANDOFF_MAX_TOTAL_SNIPPET_CHARS = 16_000
export const SIDEPANEL_CHAT_HANDOFF_MAX_DRAFT_CHARS = 32_000

export type SidepanelChatHandoffSnippet = {
  kind: "selection" | "visible-context" | "captured-snippet"
  text: string
  label?: string
  truncated?: boolean
}

export type SidepanelChatHandoffPageContext = {
  title?: string
  url?: string
  snippets: SidepanelChatHandoffSnippet[]
  truncated?: boolean
}

export type SidepanelChatHandoffPackage = {
  id: string
  source: "sidepanel-chat"
  createdAt: string
  expiresAt: string
  consumedAt?: string
  draft: { text: string; truncated?: boolean }
  pageContext?: SidepanelChatHandoffPageContext
  routeIntent?: {
    path: string
    mode?: "character"
    characterId?: string
  }
}
```

- [ ] **Step 4: Implement create/read/consume**

Requirements:

- Generate ids with `crypto.randomUUID()`, with a fallback only for test environments that lack it.
- Bound draft/snippet lengths before writing.
- Write through `createSafeStorage({ area: "local" })`.
- After write, read the same key and validate the package. If validation fails, remove the key and throw.
- `readSidepanelChatHandoff(id)` returns `null` for missing, expired, consumed, or malformed records and removes bad/expired records.
- `consumeSidepanelChatHandoff(id)` marks consumed or removes the record. Prefer removal unless preserving consumed metadata is needed for tests.

Minimal create flow:

```ts
export const createSidepanelChatHandoff = async (
  input: CreateSidepanelChatHandoffInput,
): Promise<SidepanelChatHandoffPackage> => {
  await cleanupExpiredSidepanelChatHandoffs()
  const now = Date.now()
  const pkg = buildPackage(input, now)
  await storage.set(storageKey(pkg.id), pkg)
  const saved = await readRawPackage(pkg.id)
  if (!saved || saved.id !== pkg.id) {
    await storage.remove(storageKey(pkg.id))
    throw new Error("Sidepanel chat handoff could not be saved.")
  }
  return saved
}
```

- [ ] **Step 5: Implement route and request helpers**

Add helpers:

```ts
export const buildSidepanelChatHandoffRoute = (
  baseChatPath: string,
  handoffId: string,
): string => {
  const [path, rawQuery = ""] = baseChatPath.split("?")
  const params = new URLSearchParams(rawQuery)
  params.set("handoff", handoffId)
  const query = params.toString()
  return query ? `${path}?${query}` : path
}

export const buildSidepanelHandoffMessageForModel = (
  visibleDraft: string,
  pageContext?: SidepanelChatHandoffPageContext,
): string => {
  if (!pageContext) return visibleDraft
  const lines = [
    "Sidepanel page context:",
    pageContext.title ? `Title: ${pageContext.title}` : null,
    pageContext.url ? `URL: ${pageContext.url}` : null,
    ...pageContext.snippets.map((snippet, index) =>
      `Snippet ${index + 1}${snippet.label ? ` (${snippet.label})` : ""}: ${snippet.text}`,
    ),
    "",
    "User draft:",
    visibleDraft,
  ].filter(Boolean)
  return lines.join("\n")
}
```

- [ ] **Step 6: Run service tests to verify GREEN**

Run:

```bash
bun run test src/services/__tests__/sidepanel-chat-handoff.test.ts
```

Expected: pass.

- [x] **Step 7: Commit**

```bash
git add apps/packages/ui/src/services/sidepanel-chat-handoff.ts apps/packages/ui/src/services/__tests__/sidepanel-chat-handoff.test.ts
git commit -m "feat: add sidepanel chat handoff storage"
```

## Task 2: Sidepanel Continue In WebUI Action

**Files:**
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx`
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- Create: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ControlRow.chat-handoff.test.tsx`
- Existing regression: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ControlRow.role-play-handoff.test.tsx`
- Existing regression: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx`

- [ ] **Step 1: Write failing ControlRow tests**

Test cases:

```ts
it("keeps Open full app route-only with no handoff parameter", async () => {})
it("creates a handoff package and opens /chat?handoff=<id>", async () => {})
it("merges handoff into active character route params", async () => {})
it("does not serialize draft or snippets into the URL", async () => {})
it("shows a disabled Continue in WebUI action when no draft or context exists", () => {})
it("shows an error and does not open a tab when handoff creation fails", async () => {})
```

Mock `createSidepanelChatHandoff` and `buildSidepanelChatHandoffRoute` from the service.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
bun run test src/components/Sidepanel/Chat/__tests__/ControlRow.chat-handoff.test.tsx
```

Expected: fail because props/action do not exist.

- [ ] **Step 3: Add ControlRow props**

Add props:

```ts
draftMessage?: string
hasVisiblePageContextForHandoff?: boolean
getVisiblePageContextForHandoff?: () =>
  | Promise<SidepanelChatHandoffPageContext | undefined>
  | SidepanelChatHandoffPageContext
  | undefined
```

Use `draftMessage.trim().length > 0 || hasVisiblePageContextForHandoff` to determine whether **Continue in WebUI** is enabled. The async context callback still validates at click time; if it resolves no context and the draft is empty, show a non-blocking warning and do not open a tab.

- [ ] **Step 4: Implement the new quick action**

In `ControlRow`, add a second quick action near `chat-open-full-app`:

```tsx
<button
  type="button"
  onClick={() => void continueInWebUI()}
  data-testid="chat-continue-in-webui"
  className="w-full text-left text-sm px-3 py-2 rounded flex items-center gap-2 hover:bg-surface2 disabled:opacity-60"
  title={continueInWebuiDescription}
>
  <ExternalLink className="size-4 text-text-subtle" />
  {t("sidepanel:controlRow.continueInWebUI", "Continue in WebUI")}
</button>
```

Click flow:

1. Resolve visible page context from the callback.
2. If draft/context are both empty, show a non-blocking warning and do not open a tab.
3. Call `createSidepanelChatHandoff({ draftText, pageContext, routeIntent })`.
4. Build route with `buildSidepanelChatHandoffRoute(fullAppChatPath, pkg.id)`.
5. Open `/options.html#${handoffPath}` via `browser.runtime.getURL`, matching existing `openFullApp`.
6. If storage creation fails, show error notification and do not open.

- [ ] **Step 5: Pass sidepanel draft and visible page context from form**

In `form.tsx`, pass:

```tsx
<ControlRow
  draftMessage={form.values.message}
  hasVisiblePageContextForHandoff={
    pageContextActive || selectedDocuments.length > 0
  }
  getVisiblePageContextForHandoff={getVisiblePageContextForHandoff}
  ...
/>
```

Build context from:

- selected tab mentions in `selectedDocuments`,
- active current page title/URL when `pageContextActive` is true, using `browser.tabs.query({ active: true, currentWindow: true })`.

Do not extract page body text in this callback.

- [ ] **Step 6: Run focused sidepanel tests**

Run:

```bash
bun run test src/components/Sidepanel/Chat/__tests__/ControlRow.chat-handoff.test.tsx src/components/Sidepanel/Chat/__tests__/ControlRow.role-play-handoff.test.tsx src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx apps/packages/ui/src/components/Sidepanel/Chat/form.tsx apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ControlRow.chat-handoff.test.tsx
git commit -m "feat: add sidepanel continue in webui action"
```

## Task 3: WebUI Handoff Import And Request Inclusion

**Files:**
- Create: `apps/packages/ui/src/components/Option/Playground/SidepanelImportedContextBanner.tsx`
- Create: `apps/packages/ui/src/components/Option/Playground/hooks/useSidepanelChatHandoffImport.ts`
- Create: `apps/packages/ui/src/components/Option/Playground/__tests__/sidepanel-chat-handoff-import.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts`

- [x] **Step 1: Write failing import tests**

Use a small test harness around the hook plus submit behavior. Required cases:

```ts
it("imports a valid handoff, pre-fills the composer, renders context, then consumes the package", async () => {})
it("does not consume before successful import", async () => {})
it("offers insert, replace, and cancel when a local draft exists", async () => {})
it("cleans only handoff from the hash-route query and preserves character params", async () => {})
it("includes imported page context in requestOverrides.messageForModel", async () => {})
it("omits imported page context after the user removes the banner", async () => {})
it("shows non-blocking feedback for expired or malformed handoffs", async () => {})
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
bun run test src/components/Option/Playground/__tests__/sidepanel-chat-handoff-import.test.tsx
```

Expected: fail because hook/banner do not exist.

- [x] **Step 3: Implement banner component**

Render a compact source-aware product banner above `AttachmentsSummary`, not inside a modal:

```tsx
export function SidepanelImportedContextBanner({
  context,
  onRemove,
}: {
  context: SidepanelChatHandoffPageContext
  onRemove: () => void
}) {
  const title = context.title || "Imported sidepanel context"
  const count = context.snippets.length
  return (
    <section
      aria-label="Imported sidepanel context"
      className="mb-2 rounded-lg border border-border bg-surface2/70 px-3 py-2 text-sm text-text"
    >
      <div className="flex min-w-0 items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate font-medium">{title}</div>
          <div className="truncate text-xs text-text-muted">
            {context.url || `${count} snippet${count === 1 ? "" : "s"}`}
          </div>
        </div>
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove imported context from ${title}`}
          className="rounded p-1 text-text-subtle hover:bg-surface hover:text-text"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>
    </section>
  )
}
```

Use existing token classes and avoid new visual styling systems.

- [x] **Step 4: Implement import hook**

Hook responsibilities:

- Use React Router `location.search`, not `window.location.search`.
- Read `handoff` param.
- Call `readSidepanelChatHandoff(id)`.
- If invalid, show feedback and remove only `handoff` from route.
- If current composer is empty, import immediately and consume.
- If current composer has draft, expose conflict state with `insert`, `replace`, and `cancel`.
- Consume only after successful import or cancel.

The hook should return:

```ts
{
  importedContext,
  removeImportedContext,
  conflict,
  insertHandoffDraft,
  replaceWithHandoffDraft,
  cancelHandoffImport,
}
```

- [x] **Step 5: Wire PlaygroundForm**

In `PlaygroundForm.tsx`:

- Import the handoff import hook.
- Call the hook after `setMessageValue` is defined.
- Render `SidepanelImportedContextBanner` above `AttachmentsSummary` when `importedContext` is present.
- Render the existing-draft conflict inline near the composer, or use an existing non-blocking inline state primitive. Avoid a modal unless no suitable inline pattern exists.
- Pass `importedSidepanelContext` and `clearImportedSidepanelContext` into `usePlaygroundSubmit`.

- [x] **Step 6: Wire request inclusion in usePlaygroundSubmit**

Extend deps:

```ts
importedSidepanelContext?: SidepanelChatHandoffPageContext | null
clearImportedSidepanelContext?: () => void
```

Before dispatch:

```ts
const messageForModel = importedSidepanelContext
  ? buildSidepanelHandoffMessageForModel(trimmed, importedSidepanelContext)
  : undefined

await dispatch({
  ...
  ...(messageForModel
    ? { requestOverrides: { messageForModel } }
    : {}),
}, {
  afterSend: () => {
    if (messageForModel) {
      clearImportedSidepanelContext?.()
    }
  },
})
```

If future code adds request overrides here, merge with them rather than overwriting.

- [x] **Step 7: Run focused WebUI import tests**

Run:

```bash
bun run test src/components/Option/Playground/__tests__/sidepanel-chat-handoff-import.test.tsx
```

Expected: pass.

- [x] **Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Option/Playground/SidepanelImportedContextBanner.tsx apps/packages/ui/src/components/Option/Playground/hooks/useSidepanelChatHandoffImport.ts apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts apps/packages/ui/src/components/Option/Playground/__tests__/sidepanel-chat-handoff-import.test.tsx
git commit -m "feat: import sidepanel chat handoff in webui"
```

## Task 4: Focused Regression And Packaged Smoke

**Files:**
- Possibly modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx` only if route cleanup needs an existing integration assertion.
- Possibly modify: `tests/e2e/...` only if there is already a sidepanel packaged smoke test harness for chat handoff.

- [x] **Step 1: Run focused unit regression set**

Run from `apps/packages/ui`:

```bash
bun run test src/services/__tests__/sidepanel-chat-handoff.test.ts src/components/Sidepanel/Chat/__tests__/ControlRow.chat-handoff.test.tsx src/components/Sidepanel/Chat/__tests__/ControlRow.role-play-handoff.test.tsx src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx src/components/Option/Playground/__tests__/sidepanel-chat-handoff-import.test.tsx
```

Expected: pass.

Result: Passed with `--maxWorkers=1 --no-file-parallelism` (5 files, 36 tests). Only the known Node localStorage ExperimentalWarning was emitted.

- [x] **Step 2: Run existing relevant playground/sidepanel tests**

Run from `apps/packages/ui`:

```bash
bun run test src/utils/__tests__/sidepanel-full-app-route.test.ts src/utils/__tests__/character-chat-mode-intent.test.ts src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx
```

Expected: pass, or document unrelated baseline failures with exact failing tests.

Result: Passed with `--maxWorkers=1 --no-file-parallelism` (4 files, 46 tests). Existing provider-status mock warnings were emitted but no tests failed.

- [x] **Step 3: Type/build sanity**

Run the narrowest repo-supported compile check available for `apps/packages/ui`. If no focused typecheck script exists, run the existing focused Vitest suite and document that full package typecheck is not available in `package.json`.

Result: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed from `apps/packages/ui`.

- [x] **Step 4: Packaged extension smoke**

If the local packaged extension harness used by `TASK-534` is available, verify:

- sidepanel route-only open still opens `/options.html#/chat`,
- **Continue in WebUI** opens `/options.html#/chat?handoff=<id>`,
- draft appears in WebUI composer,
- imported context banner appears,
- send includes imported context and then clears the banner,
- role-play route preserves `mode=character&characterId=...`,
- stale handoff shows non-blocking feedback and normal `/chat` remains usable.

Record screenshot or terminal evidence in the implementation task.

Result: Existing smoke harness was found and `.output/chrome-mv3` built successfully, but packaged browser assertions could not run on this host. Full smoke command skipped all 3 tests because extension launch was unavailable; JSON follow-up reported `browserType.launchPersistentContext: Timeout 30000ms exceeded`. Longer headful launch timed out before route assertions; CI-style headless launch skipped with `Could not determine extension id from [no extension targets]`.

- [x] **Step 5: Bandit**

No Python is expected. Record Bandit skipped for UI-only TypeScript changes. If the implementation unexpectedly touches Python, run Bandit on the touched Python paths from the repo root:

```bash
source .venv/bin/activate && python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_sidepanel_chat_handoff.json
```

Result: Skipped because the verification/implementation scope touched TypeScript, TSX, and markdown only; no Python paths were changed.

- [x] **Step 6: Final implementation task update**

Update the implementation Backlog task with:

- touched files,
- focused test commands and results,
- packaged smoke evidence or explicit skip reason,
- Bandit skip reason,
- final summary.

- [ ] **Step 7: Commit**

```bash
git status --short
git add <changed files and implementation task>
git commit -m "test: verify sidepanel chat handoff"
```

## Plan Review Notes

- The plan intentionally does not change the existing `SidepanelHeaderSimple` full-chat route-only action.
- The plan uses request-scoped imported context, not durable server-backed draft persistence.
- The plan does not perform fresh readable-page/body extraction at handoff time.
- A plan-document-reviewer subagent was not dispatched from this session because the available subagent tool policy requires an explicit user request for delegation. Do a local review before execution, or ask for explicit subagent review if that gate should be used.
