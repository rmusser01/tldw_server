# Mermaid Chat Card Browser QA Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stable Next WebUI debug route and Playwright smoke test for assistant-facing Mermaid chat/card rendering.

**Architecture:** A debug-only Next page renders static fixtures through the real shared `Markdown` and `MermaidDiagramBlock` components. `_app.tsx` treats `/__debug__` routes as gate-bypassed so the harness is not blocked by backend readiness or first-run setup. Route metadata classifies the page as internal QA/debug and keeps it out of the broad smoke page inventory.

**Tech Stack:** Next.js pages router, React, shared UI package components, Playwright smoke tests, Backlog.md task tracking.

---

## File Map

- Create: `apps/tldw-frontend/e2e/smoke/mermaid-chat-cards.spec.ts`
  - Owns the browser-level regression test for the harness.
- Create: `apps/tldw-frontend/pages/__debug__/mermaid-chat-cards.tsx`
  - Owns the debug-only fixture page.
- Modify: `apps/tldw-frontend/pages/_app.tsx`
  - Adds `/__debug__` gate bypass logic.
- Modify: `apps/packages/ui/src/routes/route-metadata.ts`
  - Registers the debug route as `internal_qa_debug` with `smoke: "exclude"`.
- Modify: `backlog/tasks/task-2313 - Add-Mermaid-chat-card-browser-QA-harness.md`
  - Records implementation progress, verification, and final summary.

Do not modify `apps/tldw-frontend/e2e/smoke/page-inventory.ts`. Smoke-excluded route metadata entries should stay out of that broad inventory.

## Task 1: Write Failing Playwright Smoke Test

**Files:**
- Create: `apps/tldw-frontend/e2e/smoke/mermaid-chat-cards.spec.ts`

- [ ] **Step 1: Write the failing smoke test**

Create `apps/tldw-frontend/e2e/smoke/mermaid-chat-cards.spec.ts`:

```ts
import type { Locator, Page } from "@playwright/test"
import { expect, seedAuth, test } from "./smoke.setup"

const section = (page: Page, testId: string): Locator => page.getByTestId(testId)

const expectNoGateBlockers = async (page: Page) => {
  await expect(page.getByTestId("server-readiness-recovery")).toHaveCount(0)
  await expect(page.getByTestId("first-run-gate-overlay")).toHaveCount(0)
}

test.describe("Mermaid chat-card browser QA harness", () => {
  test("renders assistant Mermaid and fallback fixtures without readiness gates", async ({
    page
  }) => {
    test.setTimeout(90_000)
    await seedAuth(page, {
      authMode: "single-user",
      apiKey: "test-key-not-placeholder",
      allowOffline: true
    })

    await page.goto("/__debug__/mermaid-chat-cards")

    await expect(page.getByTestId("mermaid-chat-card-harness")).toBeVisible({
      timeout: 30_000
    })
    await expectNoGateBlockers(page)

    const assistant = section(page, "mermaid-harness-assistant")
    await expect(assistant.getByText("Assistant Mermaid render")).toBeVisible()
    await expect(
      assistant.getByRole("button", { name: "Open Mermaid preview" })
    ).toBeVisible()
    await expect(
      assistant.getByRole("button", { name: "Copy Mermaid source" })
    ).toBeVisible()
    await expect(assistant.getByRole("img", { name: "Mermaid diagram" })).toBeVisible()

    const user = section(page, "mermaid-harness-user")
    await expect(user.getByText("```mermaid")).toBeVisible()
    await expect(user.getByText("flowchart TD")).toBeVisible()
    await expect(user.getByRole("button", { name: "Open Mermaid preview" })).toHaveCount(0)

    const disabled = section(page, "mermaid-harness-disabled")
    await expect(disabled.getByText("flowchart TD")).toBeVisible()
    await expect(disabled.getByRole("button", { name: "Open Mermaid preview" })).toHaveCount(0)

    const invalid = section(page, "mermaid-harness-invalid")
    await expect(invalid.getByText("Unable to render Mermaid diagram.")).toBeVisible({
      timeout: 30_000
    })
    await expect(invalid.getByText("not a valid mermaid diagram")).toBeVisible()

    const graphviz = section(page, "mermaid-harness-graphviz")
    await expect(graphviz.getByText("digraph G")).toBeVisible()
    await expect(graphviz.getByRole("button", { name: "Open Mermaid preview" })).toHaveCount(0)

    const artifact = section(page, "mermaid-harness-artifact")
    await expect(artifact.getByRole("img", { name: "Mermaid diagram" })).toBeVisible()
    await expect(
      artifact.getByRole("button", { name: "Open Mermaid preview" })
    ).toBeVisible()
    await expect(
      artifact.getByRole("button", { name: "Copy Mermaid source" })
    ).toBeVisible()
  })
})
```

- [ ] **Step 2: Run the test to verify RED**

Run from `apps/tldw-frontend`:

```bash
npx playwright test e2e/smoke/mermaid-chat-cards.spec.ts --reporter=line
```

Expected: FAIL because `/__debug__/mermaid-chat-cards` does not exist and/or `mermaid-chat-card-harness` is not visible.

- [ ] **Step 3: Keep the red test uncommitted**

Do not commit the red test by itself. Repo policy requires commits to contain working code. Leave the new test file unstaged or staged locally until Task 2 makes it pass, then commit the passing test with the implementation.

## Task 2: Add Debug Route And Gate Bypass

**Files:**
- Create: `apps/tldw-frontend/pages/__debug__/mermaid-chat-cards.tsx`
- Modify: `apps/tldw-frontend/pages/_app.tsx`

- [ ] **Step 1: Implement the debug route**

Create `apps/tldw-frontend/pages/__debug__/mermaid-chat-cards.tsx`:

```tsx
import dynamic from "next/dynamic"
import React from "react"

import { Markdown } from "@/components/Common/Markdown"
import { MermaidDiagramBlock } from "@/components/Common/MermaidDiagramBlock"

const validMermaidSource = `flowchart TD
  A["Assistant response"] --> B["MermaidDiagramBlock"]
  B --> C["Browser QA"]
`

const assistantMarkdown = `Here is the assistant-facing Mermaid fixture.

\`\`\`mermaid
${validMermaidSource}
\`\`\`
`

const userMessageSource = `\`\`\`mermaid
${validMermaidSource}
\`\`\``

const disabledMarkdown = `Mermaid disabled fallback:

\`\`\`mermaid
${validMermaidSource}
\`\`\`
`

const invalidMermaidMarkdown = `Invalid Mermaid fallback:

\`\`\`mermaid
not a valid mermaid diagram @@@
\`\`\`
`

const graphvizMarkdown = `Graphviz should remain code:

\`\`\`dot
digraph G {
  A -> B;
}
\`\`\`
`

type HarnessSectionProps = {
  testId: string
  title: string
  children: React.ReactNode
}

const HarnessSection = ({ testId, title, children }: HarnessSectionProps) => (
  <section
    data-testid={testId}
    className="rounded-lg border border-border bg-surface p-4 shadow-sm"
  >
    <h2 className="mb-3 text-sm font-semibold text-text">{title}</h2>
    {children}
  </section>
)

const MermaidChatCardsHarness = () => (
  <main
    data-testid="mermaid-chat-card-harness"
    className="min-h-screen bg-bg px-6 py-8 text-text"
  >
    <div className="mx-auto flex max-w-5xl flex-col gap-4">
      <header>
        <h1 className="text-xl font-semibold">Mermaid Chat Card QA</h1>
      </header>

      <HarnessSection testId="mermaid-harness-assistant" title="Assistant Mermaid render">
        <Markdown message={assistantMarkdown} enableMermaidDiagrams />
      </HarnessSection>

      <HarnessSection testId="mermaid-harness-user" title="User message unchanged">
        <pre className="overflow-auto whitespace-pre-wrap rounded-md bg-surface2 p-3 text-xs text-text">
          {userMessageSource}
        </pre>
      </HarnessSection>

      <HarnessSection testId="mermaid-harness-disabled" title="Setting-off fallback">
        <Markdown message={disabledMarkdown} />
      </HarnessSection>

      <HarnessSection testId="mermaid-harness-invalid" title="Invalid Mermaid fallback">
        <Markdown message={invalidMermaidMarkdown} enableMermaidDiagrams />
      </HarnessSection>

      <HarnessSection testId="mermaid-harness-graphviz" title="Graphviz/DOT fallback">
        <Markdown message={graphvizMarkdown} enableMermaidDiagrams />
      </HarnessSection>

      <HarnessSection testId="mermaid-harness-artifact" title="Artifact-style Mermaid card">
        <MermaidDiagramBlock
          artifactContextId="debug-mermaid-chat-card"
          blockIndex={0}
          enableArtifactAction
          source={validMermaidSource}
        />
      </HarnessSection>
    </div>
  </main>
)

export default dynamic(() => Promise.resolve(MermaidChatCardsHarness), {
  ssr: false
})
```

- [ ] **Step 2: Add debug gate bypass**

Modify `apps/tldw-frontend/pages/_app.tsx` near the existing route booleans:

```ts
const isDebugRoute = routePath === "/__debug__" || routePath.startsWith("/__debug__/")
```

Then include it in the gate bypass logic:

```ts
const shouldBypassGates =
  isPublicAuthRoute || isSettingsRoute || isSetupRoute || isDebugRoute
```

Keep `hideHeader` / `hideSidebar` behavior unchanged unless the page renders incorrectly. The debug page can be wrapped in `OptionLayout` as long as the route content is visible and unblocked.

- [ ] **Step 3: Run Playwright to verify GREEN for route and gates**

Run from `apps/tldw-frontend`:

```bash
npx playwright test e2e/smoke/mermaid-chat-cards.spec.ts --reporter=line
```

Expected: PASS for the new smoke test.

- [ ] **Step 4: Commit route and gate implementation**

```bash
git add apps/tldw-frontend/e2e/smoke/mermaid-chat-cards.spec.ts apps/tldw-frontend/pages/__debug__/mermaid-chat-cards.tsx apps/tldw-frontend/pages/_app.tsx
git commit -m "feat: add mermaid chat card debug harness"
```

## Task 3: Register Route Metadata

**Files:**
- Modify: `apps/packages/ui/src/routes/route-metadata.ts`

- [ ] **Step 1: Add metadata entry**

Modify `apps/packages/ui/src/routes/route-metadata.ts` near the existing `/__debug__` routes:

```ts
defineRoute({
  path: "/__debug__/mermaid-chat-cards",
  label: "Debug Mermaid Chat Cards",
  group: "extension",
  surface: "internal_qa_debug",
  smoke: "exclude",
  rationale: "Web debug route for Mermaid chat-card browser QA."
}),
```

Do not add this path to `apps/tldw-frontend/e2e/smoke/page-inventory.ts`.

- [ ] **Step 2: Run route governance tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/route-governance.metadata-coverage.test.ts
```

Run from `apps/tldw-frontend` if time allows:

```bash
npx playwright test e2e/smoke/route-contract-stage2.spec.ts --grep "Route metadata smoke inventory contract"
```

Expected: PASS. If route-contract expectations conflict with existing skipped debug entries, do not broaden the scope casually; document the existing inconsistency and keep this route out of `page-inventory.ts`.

- [ ] **Step 3: Commit route metadata**

```bash
git add apps/packages/ui/src/routes/route-metadata.ts
git commit -m "chore: classify mermaid chat card debug route"
```

## Task 4: Final Verification And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-2313 - Add-Mermaid-chat-card-browser-QA-harness.md`

- [ ] **Step 1: Install dependencies without lockfile changes**

Run from `apps`:

```bash
bun install --frozen-lockfile
```

Expected: PASS. If `apps/packages/ui/node_modules/antd` symlink changes, restore it before final staging:

```bash
git restore -- apps/packages/ui/node_modules/antd
```

- [ ] **Step 2: Run focused Mermaid tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/components/Common/__tests__/Mermaid.test.tsx \
  src/components/Common/__tests__/MermaidDiagramBlock.test.tsx \
  src/components/Common/__tests__/MermaidPreviewDialog.test.tsx \
  src/components/Common/__tests__/Markdown.mermaid.test.tsx \
  src/components/Common/__tests__/Markdown.github-code-blocks.test.tsx \
  src/components/Common/__tests__/CodeBlock.artifacts.test.tsx \
  src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx \
  src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx \
  src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx \
  src/components/Option/Settings/__tests__/ChatSettings.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run new browser smoke test**

Run from `apps/tldw-frontend`:

```bash
npx playwright test e2e/smoke/mermaid-chat-cards.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 4: Run frontend compile**

Run from `apps/tldw-frontend`:

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile
```

Expected: PASS with token sync OK.

- [ ] **Step 5: Browser-plugin manual check**

If the dev server is not already running, start it from `apps/tldw-frontend`:

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev -- --hostname 127.0.0.1 -p 18002
```

Use the Browser plugin to navigate to:

```text
http://127.0.0.1:18002/__debug__/mermaid-chat-cards
```

Expected: harness root visible, no readiness or first-run blocker, valid Mermaid sections render, and fallback sections show source. Stop the dev server before final response.

- [ ] **Step 6: Update Backlog final summary**

Update `TASK-2313` with:

- implemented files;
- verification commands and results;
- Bandit skipped because no Python source changed;
- any known environment skips, especially if Browser plugin is blocked.

- [ ] **Step 7: Final status check**

Run:

```bash
git status --short
```

Expected: only intended files changed.

- [ ] **Step 8: Commit closeout**

```bash
git add \
  "backlog/tasks/task-2313 - Add-Mermaid-chat-card-browser-QA-harness.md"
git commit -m "docs: close mermaid chat card qa harness task"
```
