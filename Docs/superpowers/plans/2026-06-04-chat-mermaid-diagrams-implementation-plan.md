# Chat Mermaid Diagrams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render fenced Mermaid diagrams inline for assistant-facing chat markdown, while preserving user-message behavior and raw-code fallbacks.

**Architecture:** Add a disabled-by-default Mermaid capability to the shared `Markdown` renderer, then explicitly enable it from assistant message renderers based on a chat setting. Reuse the existing client-side `Mermaid` renderer as the only Mermaid execution boundary, expose generated SVG state to a new inline block, and keep zoom/download behavior in a separate preview dialog.

**Tech Stack:** React 18, TypeScript, `react-markdown`, existing `mermaid` dependency, `@plasmohq/storage`, Ant Design, lucide-react, Vitest/JSDOM, Next.js frontend build.

---

## Source Documents

- PRD: `Docs/superpowers/specs/2026-06-04-chat-mermaid-diagrams-design.md`
- Backlog: `TASK-511`
- Prior PRD task: `TASK-510`

## File Structure

### Settings

- Modify: `apps/packages/ui/src/types/chat-settings.ts`
  - Add `renderMermaidDiagrams: boolean` to `ChatSettingsConfig`.
  - Add default `renderMermaidDiagrams: true`.
- Modify: `apps/packages/ui/src/hooks/useChatSettings.ts`
  - Add storage binding for `renderMermaidDiagrams`.
  - Return `renderMermaidDiagrams` and `setRenderMermaidDiagrams`.
- Modify: `apps/packages/ui/src/components/Option/Settings/ChatSettings.tsx`
  - Surface a `Render Mermaid diagrams` switch near markdown/rich-text/external-image controls.
- Modify: `apps/packages/ui/src/components/Option/Settings/__tests__/ChatSettings.test.tsx`
  - Add mocked state fields and coverage for the new switch.
- Optional modify: `apps/packages/ui/src/public/_locales/en/settings.json`
  - Add an English locale key only if the implementation chooses not to rely on `t(..., fallback)` strings.

### Rendering Core

- Modify: `apps/packages/ui/src/components/Common/Mermaid.tsx`
  - Keep strict security and dynamic import.
  - Expose render state and generated SVG to parents.
  - Guard stale async writes.
- Test: `apps/packages/ui/src/components/Common/__tests__/Mermaid.test.tsx`
  - Mock `mermaid` and verify success, error, and stale-render behavior.

### Inline Chat UI

- Create: `apps/packages/ui/src/components/Common/MermaidDiagramBlock.tsx`
  - Own inline block chrome, copy source, preview dialog trigger, SVG download, and raw fallback.
- Create: `apps/packages/ui/src/components/Common/MermaidPreviewDialog.tsx`
  - Own zoom, pan, reset, copy source, SVG download, and generated-SVG display.
- Test: `apps/packages/ui/src/components/Common/__tests__/MermaidDiagramBlock.test.tsx`
- Test: `apps/packages/ui/src/components/Common/__tests__/MermaidPreviewDialog.test.tsx`

### Markdown Routing

- Modify: `apps/packages/ui/src/components/Common/Markdown.tsx`
  - Add `enableMermaidDiagrams?: boolean`, default `false`.
  - Detect normalized language exactly `mermaid`.
  - Route closed Mermaid fences to `MermaidDiagramBlock` only when enabled.
  - Keep `plain`, `compact`, `github`, and default code rendering behavior unchanged when disabled.
  - Bypass ST-compatible `dangerouslySetInnerHTML` path when Mermaid component rendering is enabled and the message contains a Mermaid fence.
- Test: `apps/packages/ui/src/components/Common/__tests__/Markdown.mermaid.test.tsx`
  - Cover enabled, disabled, user/default unchanged, variant fallback, and ST-compatible bypass.

### Assistant-Facing Call Sites

- Modify: `apps/packages/ui/src/components/Common/Playground/Message.tsx`
  - Read `renderMermaidDiagrams` from storage.
  - Pass `enableMermaidDiagrams={renderMermaidDiagrams && !props.isStreaming}` to assistant `Markdown` calls only.
  - Keep user-message paths unchanged, including `useMarkdownForUserMessage`.
- Modify: `apps/packages/ui/src/components/Common/Playground/MessageContent.tsx`
  - Apply the same assistant-only Mermaid prop to the extracted message content path.
- Modify: `apps/packages/ui/src/components/Common/Playground/ReasoningBlock.tsx`
  - Accept `enableMermaidDiagrams?: boolean`.
  - Pass it to `Markdown` only when the reasoning block is not actively streaming.
- Modify: `apps/packages/ui/src/components/Common/Playground/CompactMessage.tsx`
  - Apply the assistant-only Mermaid prop to compact assistant markdown rendering.
- Modify: `apps/packages/ui/src/components/Common/QuickChatHelper/QuickChatMessage.tsx`
  - Read `renderMermaidDiagrams`.
  - Pass `enableMermaidDiagrams={renderMermaidDiagrams && !isStreaming}` for assistant messages.
- Tests:
  - `apps/packages/ui/src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx`
  - `apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx`

### Verification

- Run focused Vitest tests from `apps/packages/ui`.
- Run frontend build from `apps/tldw-frontend` to confirm `mermaid` resolves.
- Run frontend lint or type/build checks if the implementation touches typed call-site contracts broadly.
- Bandit is not applicable unless backend Python code is touched.

---

### Task 1: Add The Chat Setting Contract

**Files:**
- Modify: `apps/packages/ui/src/types/chat-settings.ts`
- Modify: `apps/packages/ui/src/hooks/useChatSettings.ts`
- Modify: `apps/packages/ui/src/components/Option/Settings/ChatSettings.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/__tests__/ChatSettings.test.tsx`

- [ ] **Step 1: Write the failing settings tests**

Add coverage to `ChatSettings.test.tsx`:

```tsx
it("renders and updates the Mermaid diagram setting", () => {
  const setRenderMermaidDiagrams = vi.fn()
  useChatSettingsMock.mockReturnValue({
    ...buildChatSettingsState(),
    renderMermaidDiagrams: true,
    setRenderMermaidDiagrams
  })

  render(<ChatSettings />)

  const toggle = screen.getByRole("switch", {
    name: "Render Mermaid diagrams"
  })
  expect(toggle).toBeChecked()

  fireEvent.click(toggle)
  expect(setRenderMermaidDiagrams).toHaveBeenCalledWith(false)
})
```

Also update `buildChatSettingsState()` with:

```ts
renderMermaidDiagrams: true,
setRenderMermaidDiagrams: vi.fn(),
```

- [ ] **Step 2: Run the failing settings test**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Settings/__tests__/ChatSettings.test.tsx
```

Expected: FAIL because `renderMermaidDiagrams` is not in the hook return and the switch is not rendered.

- [ ] **Step 3: Add the setting type and default**

In `chat-settings.ts`, add the interface field near existing display/rendering settings:

```ts
renderMermaidDiagrams: boolean
```

Add the default near `useMarkdownForUserMessage` / `chatRichTextMode`:

```ts
renderMermaidDiagrams: true,
```

- [ ] **Step 4: Add the storage hook**

In `useChatSettings.ts`, add:

```ts
const [renderMermaidDiagrams, setRenderMermaidDiagrams] = useStorage(
  "renderMermaidDiagrams",
  DEFAULT_CHAT_SETTINGS.renderMermaidDiagrams
)
```

Return both values:

```ts
renderMermaidDiagrams,
setRenderMermaidDiagrams,
```

- [ ] **Step 5: Add the settings switch**

In `ChatSettings.tsx`, destructure the new hook values and insert a `SettingRow` near `useMarkdownForUserMessage`, `chatRichTextMode`, and `allowExternalImages`:

```tsx
<SettingRow
  label={t(
    "generalSettings.settings.renderMermaidDiagrams.label",
    "Render Mermaid diagrams"
  )}
  {...getResetProps(
    renderMermaidDiagrams,
    DEFAULT_CHAT_SETTINGS.renderMermaidDiagrams,
    setRenderMermaidDiagrams
  )}
  control={
    <Switch
      checked={renderMermaidDiagrams}
      onChange={setRenderMermaidDiagrams}
      aria-label={t(
        "generalSettings.settings.renderMermaidDiagrams.label",
        "Render Mermaid diagrams"
      )}
    />
  }
/>
```

- [ ] **Step 6: Re-run the settings test**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Settings/__tests__/ChatSettings.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit the setting slice**

```bash
git add apps/packages/ui/src/types/chat-settings.ts \
  apps/packages/ui/src/hooks/useChatSettings.ts \
  apps/packages/ui/src/components/Option/Settings/ChatSettings.tsx \
  apps/packages/ui/src/components/Option/Settings/__tests__/ChatSettings.test.tsx
git commit -m "feat: add Mermaid diagram chat setting"
```

---

### Task 2: Harden The Existing Mermaid Renderer API

**Files:**
- Modify: `apps/packages/ui/src/components/Common/Mermaid.tsx`
- Create: `apps/packages/ui/src/components/Common/__tests__/Mermaid.test.tsx`

- [ ] **Step 1: Write renderer tests with a mocked Mermaid module**

Create `Mermaid.test.tsx` with tests for successful SVG reporting and render failure.

```tsx
import React from "react"
import { render, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import Mermaid from "../Mermaid"

const { initializeMock, renderMock } = vi.hoisted(() => ({
  initializeMock: vi.fn(),
  renderMock: vi.fn()
}))

vi.mock("mermaid", () => ({
  default: {
    initialize: initializeMock,
    render: renderMock
  }
}))

describe("Mermaid", () => {
  it("reports generated SVG after a successful strict render", async () => {
    renderMock.mockResolvedValueOnce({
      svg: "<svg role=\"img\"><text>ok</text></svg>",
      bindFunctions: vi.fn()
    })
    const onRenderStateChange = vi.fn()

    render(
      <Mermaid
        code="graph TD\nA-->B"
        onRenderStateChange={onRenderStateChange}
      />
    )

    await waitFor(() => {
      expect(onRenderStateChange).toHaveBeenCalledWith(
        expect.objectContaining({
          status: "success",
          svg: expect.stringContaining("<svg")
        })
      )
    })
    expect(initializeMock).toHaveBeenCalledWith(
      expect.objectContaining({ securityLevel: "strict" })
    )
  })

  it("reports render errors without throwing", async () => {
    renderMock.mockRejectedValueOnce(new Error("bad diagram"))
    const onRenderStateChange = vi.fn()

    render(
      <Mermaid
        code="not valid"
        onRenderStateChange={onRenderStateChange}
      />
    )

    await waitFor(() => {
      expect(onRenderStateChange).toHaveBeenCalledWith(
        expect.objectContaining({
          status: "error",
          error: "bad diagram"
        })
      )
    })
  })
})
```

- [ ] **Step 2: Run the failing renderer test**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Common/__tests__/Mermaid.test.tsx
```

Expected: FAIL because `onRenderStateChange` does not exist.

- [ ] **Step 3: Add render state types and callback**

In `Mermaid.tsx`, export the theme type because the render state references it:

```ts
export type MermaidTheme =
  | "default"
  | "base"
  | "dark"
  | "forest"
  | "neutral"
  | "null"
```

Then add:

```ts
export type MermaidRenderStatus = "idle" | "rendering" | "success" | "error"

export type MermaidRenderState = {
  status: MermaidRenderStatus
  svg?: string
  error?: string
  theme?: MermaidTheme
}
```

Extend props:

```ts
onRenderStateChange?: (state: MermaidRenderState) => void
```

- [ ] **Step 4: Report state and guard stale renders**

Add a separate sequence ref so stale-render detection does not reuse the SVG id counter:

```ts
const renderSequenceRef = useRef(0)
```

Inside the render effect:

```ts
const renderToken = ++renderSequenceRef.current
onRenderStateChange?.({ status: "rendering", theme })
```

After `mermaid.render`, only write state if current:

```ts
if (
  !active ||
  renderToken !== renderSequenceRef.current ||
  !containerRef.current
) {
  return
}
containerRef.current.innerHTML = svg
bindFunctions?.(containerRef.current)
setError(null)
onRenderStateChange?.({ status: "success", svg, theme })
```

On catch:

```ts
const message = err instanceof Error ? err.message : "Unable to render diagram."
setError(message)
onRenderStateChange?.({ status: "error", error: message, theme })
```

Keep `securityLevel: "strict"` and `import("mermaid")`.

- [ ] **Step 5: Re-run renderer tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Common/__tests__/Mermaid.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit the renderer slice**

```bash
git add apps/packages/ui/src/components/Common/Mermaid.tsx \
  apps/packages/ui/src/components/Common/__tests__/Mermaid.test.tsx
git commit -m "feat: expose Mermaid render state"
```

---

### Task 3: Add Inline Mermaid Block And Preview Dialog

**Files:**
- Create: `apps/packages/ui/src/components/Common/MermaidDiagramBlock.tsx`
- Create: `apps/packages/ui/src/components/Common/MermaidPreviewDialog.tsx`
- Create: `apps/packages/ui/src/components/Common/__tests__/MermaidDiagramBlock.test.tsx`
- Create: `apps/packages/ui/src/components/Common/__tests__/MermaidPreviewDialog.test.tsx`

- [ ] **Step 1: Write inline block tests**

Mock the renderer so tests do not load Mermaid:

```tsx
vi.mock("../Mermaid", () => ({
  default: ({
    code,
    onRenderStateChange
  }: {
    code: string
    onRenderStateChange?: (state: { status: string; svg?: string }) => void
  }) => {
    React.useEffect(() => {
      onRenderStateChange?.({
        status: "success",
        svg: "<svg><text>diagram</text></svg>"
      })
    }, [onRenderStateChange])
    return <div role="img" aria-label="Mermaid diagram">{code}</div>
  }
}))
```

Cover:

- header label `mermaid`;
- copy action writes source;
- download action exists only after SVG is available;
- preview opens a dialog;
- render error shows raw source fallback.

- [ ] **Step 2: Write preview dialog tests**

Cover:

- generated SVG is inserted into the dialog;
- zoom in/out/reset controls update zoom label or transform;
- copy source uses Mermaid source, not SVG;
- download serializes generated SVG;
- close button closes the dialog.

- [ ] **Step 3: Run the failing component tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/components/Common/__tests__/MermaidDiagramBlock.test.tsx \
  src/components/Common/__tests__/MermaidPreviewDialog.test.tsx
```

Expected: FAIL because the components do not exist.

- [ ] **Step 4: Implement `MermaidPreviewDialog`**

Use Ant Design `Modal` and lucide icons. Keep the generated SVG boundary explicit:

```tsx
type MermaidPreviewDialogProps = {
  open: boolean
  source: string
  generatedSvg: string | null
  onClose: () => void
}
```

Render generated SVG only:

```tsx
{generatedSvg ? (
  <div
    role="img"
    aria-label="Mermaid diagram preview"
    style={{ transform: `scale(${zoom})`, transformOrigin: "0 0" }}
    dangerouslySetInnerHTML={{ __html: generatedSvg }}
  />
) : (
  <pre className="whitespace-pre-wrap text-xs">{source}</pre>
)}
```

Do not accept raw SVG/XML model content as a separate prop.

- [ ] **Step 5: Implement `MermaidDiagramBlock`**

Use the existing code-block visual language:

```tsx
type MermaidDiagramBlockProps = {
  source: string
  blockIndex?: number
}
```

Core behavior:

- render header label `mermaid`;
- render `<Mermaid code={source} onRenderStateChange={...} />`;
- keep `generatedSvg` state;
- copy source via `navigator.clipboard.writeText(source)`;
- download SVG only when `generatedSvg` exists;
- show `Unable to render Mermaid diagram.` plus raw source when renderer status is `error`;
- open `MermaidPreviewDialog` with source and generated SVG.

- [ ] **Step 6: Re-run component tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/components/Common/__tests__/MermaidDiagramBlock.test.tsx \
  src/components/Common/__tests__/MermaidPreviewDialog.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit the inline UI slice**

```bash
git add apps/packages/ui/src/components/Common/MermaidDiagramBlock.tsx \
  apps/packages/ui/src/components/Common/MermaidPreviewDialog.tsx \
  apps/packages/ui/src/components/Common/__tests__/MermaidDiagramBlock.test.tsx \
  apps/packages/ui/src/components/Common/__tests__/MermaidPreviewDialog.test.tsx
git commit -m "feat: add Mermaid diagram chat block"
```

---

### Task 4: Route Mermaid Fences Through Shared Markdown

**Files:**
- Modify: `apps/packages/ui/src/components/Common/Markdown.tsx`
- Create: `apps/packages/ui/src/components/Common/__tests__/Markdown.mermaid.test.tsx`
- Modify as needed: `apps/packages/ui/src/components/Common/__tests__/Markdown.github-code-blocks.test.tsx`

- [ ] **Step 1: Write Markdown routing tests**

Mock the Mermaid block:

```tsx
vi.mock("../MermaidDiagramBlock", () => ({
  MermaidDiagramBlock: ({ source }: { source: string }) => (
    <div data-testid="mermaid-diagram-block">{source}</div>
  ),
  default: ({ source }: { source: string }) => (
    <div data-testid="mermaid-diagram-block">{source}</div>
  )
}))
```

Cover:

```tsx
it("renders mermaid fences with the diagram block when enabled", () => {
  render(
    <Markdown
      enableMermaidDiagrams
      message={"```mermaid\ngraph TD\nA-->B\n```"}
    />
  )
  expect(screen.getByTestId("mermaid-diagram-block")).toHaveTextContent("graph TD")
})

it("renders mermaid fences as code when disabled", () => {
  const { container } = render(
    <Markdown message={"```mermaid\ngraph TD\nA-->B\n```"} />
  )
  expect(screen.queryByTestId("mermaid-diagram-block")).not.toBeInTheDocument()
  expect(container.querySelector("pre")).toBeInTheDocument()
})
```

Add tests for:

- language `mmd` does not render as Mermaid;
- `richTextModeOverride="st_compat"` plus `enableMermaidDiagrams` uses ReactMarkdown component routing;
- non-Mermaid markdown does not import/render `MermaidDiagramBlock`.

- [ ] **Step 2: Run the failing Markdown tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Common/__tests__/Markdown.mermaid.test.tsx
```

Expected: FAIL because `enableMermaidDiagrams` and routing do not exist.

- [ ] **Step 3: Add the Markdown prop**

In `Markdown.tsx`, add:

```ts
enableMermaidDiagrams = false,
```

and the prop type:

```ts
enableMermaidDiagrams?: boolean
```

- [ ] **Step 4: Add exact Mermaid fence detection helpers**

Add local helpers near the URL helpers:

```ts
const MERMAID_FENCE_START = /^\s*```\s*mermaid\s*$/im

const containsMermaidFence = (source: string): boolean =>
  MERMAID_FENCE_START.test(source)
```

Use this only to decide whether ST-compatible HTML must be bypassed.

- [ ] **Step 5: Guard the ST-compatible HTML path**

Change:

```ts
if (richTextMode === "st_compat" && !hasManagedAssetImages) {
```

to:

```ts
const shouldUseComponentMermaid =
  enableMermaidDiagrams && containsMermaidFence(processedMessage)

if (
  richTextMode === "st_compat" &&
  !hasManagedAssetImages &&
  !shouldUseComponentMermaid
) {
```

- [ ] **Step 6: Route Mermaid pre blocks before code variants**

After computing `rawLanguage`, `normalizedLanguage`, `blockIndex`, and `value`, route exact Mermaid:

```tsx
const rawLanguage = match ? match[1] : ""
const normalizedLanguage = normalizeLanguage(rawLanguage)

if (enableMermaidDiagrams && normalizedLanguage === "mermaid") {
  return (
    <MermaidDiagramBlock source={value} blockIndex={blockIndex} />
  )
}
```

Keep `plain`, `compact`, and `github` code-block paths unchanged when disabled.

- [ ] **Step 7: Re-run Markdown tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/components/Common/__tests__/Markdown.mermaid.test.tsx \
  src/components/Common/__tests__/Markdown.github-code-blocks.test.tsx \
  src/components/Common/__tests__/Markdown.flashcard-asset-image.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit the Markdown routing slice**

```bash
git add apps/packages/ui/src/components/Common/Markdown.tsx \
  apps/packages/ui/src/components/Common/__tests__/Markdown.mermaid.test.tsx \
  apps/packages/ui/src/components/Common/__tests__/Markdown.github-code-blocks.test.tsx \
  apps/packages/ui/src/components/Common/__tests__/Markdown.flashcard-asset-image.test.tsx
git commit -m "feat: route Mermaid fences through markdown"
```

---

### Task 5: Enable Mermaid Only On Assistant-Facing Chat Surfaces

**Files:**
- Modify: `apps/packages/ui/src/components/Common/Playground/Message.tsx`
- Modify: `apps/packages/ui/src/components/Common/Playground/MessageContent.tsx`
- Modify: `apps/packages/ui/src/components/Common/Playground/ReasoningBlock.tsx`
- Modify: `apps/packages/ui/src/components/Common/Playground/CompactMessage.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickChatHelper/QuickChatMessage.tsx`
- Create: `apps/packages/ui/src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx`
- Create: `apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx`

- [ ] **Step 1: Write assistant/user call-site tests**

For `QuickChatMessage`, mock `Markdown` and assert props:

```tsx
const markdownMock = vi.fn(() => <div data-testid="markdown" />)

vi.mock("@/components/Common/Markdown", () => ({
  default: (props: unknown) => markdownMock(props)
}))
```

Cover:

- assistant message passes `enableMermaidDiagrams: true` when setting defaults on and not streaming;
- assistant streaming passes `enableMermaidDiagrams: false`;
- user message does not render `Markdown`.

For `PlaygroundMessage` or `MessageContent`, prefer a focused test that mocks `Markdown` and checks:

- assistant completed message passes `enableMermaidDiagrams: true`;
- assistant streaming branch renders `data-testid="playground-streaming-plain-text"` and does not render Markdown;
- user message path does not pass Mermaid even when `useMarkdownForUserMessage` is true.

- [ ] **Step 2: Run failing call-site tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx \
  src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx
```

Expected: FAIL because assistant Markdown call sites do not pass `enableMermaidDiagrams`.

- [ ] **Step 3: Add setting reads to assistant renderers**

In renderers that already import `useStorage`, add:

```ts
const [renderMermaidDiagrams] = useStorage(
  "renderMermaidDiagrams",
  DEFAULT_CHAT_SETTINGS.renderMermaidDiagrams
)
const enableAssistantMermaidDiagrams =
  renderMermaidDiagrams !== false && !props.isStreaming
```

Import `DEFAULT_CHAT_SETTINGS` where needed.

For `QuickChatMessage`, use `isStreaming` instead of `props.isStreaming`.

- [ ] **Step 4: Pass the prop only to assistant Markdown**

For completed assistant message Markdown calls:

```tsx
<Markdown
  message={e.content}
  className={`${MARKDOWN_BASE_CLASSES} ${assistantTextClass}`}
  searchQuery={props.searchQuery}
  codeBlockVariant="github"
  enableMermaidDiagrams={enableAssistantMermaidDiagrams}
/>
```

For greeting Markdown:

```tsx
enableMermaidDiagrams={renderMermaidDiagrams !== false && !props.isStreaming}
```

For `ReasoningBlock`, add a prop:

```ts
enableMermaidDiagrams?: boolean
```

and pass:

```tsx
enableMermaidDiagrams={Boolean(enableMermaidDiagrams) && !isReasoningStreaming}
```

Do not pass this prop to `HumanMessge.tsx`, `ShareModal.tsx`, documentation, notes, review, flashcards, or other non-chat markdown consumers.

- [ ] **Step 5: Re-run call-site tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx \
  src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit assistant wiring**

```bash
git add apps/packages/ui/src/components/Common/Playground/Message.tsx \
  apps/packages/ui/src/components/Common/Playground/MessageContent.tsx \
  apps/packages/ui/src/components/Common/Playground/ReasoningBlock.tsx \
  apps/packages/ui/src/components/Common/Playground/CompactMessage.tsx \
  apps/packages/ui/src/components/Common/QuickChatHelper/QuickChatMessage.tsx \
  apps/packages/ui/src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx \
  apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx
git commit -m "feat: enable Mermaid diagrams for assistant chat"
```

---

### Task 6: Final Verification And Build Resolution

**Files:**
- Modify only if necessary: implementation files from prior tasks.
- Update: `backlog/tasks/task-511 - Write-implementation-plan-for-Mermaid-diagram-rendering-in-assistant-chat-markdown.md` during planning only, or the implementation task created later.

- [ ] **Step 1: Run focused shared UI tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/components/Common/__tests__/Mermaid.test.tsx \
  src/components/Common/__tests__/MermaidDiagramBlock.test.tsx \
  src/components/Common/__tests__/MermaidPreviewDialog.test.tsx \
  src/components/Common/__tests__/Markdown.mermaid.test.tsx \
  src/components/Common/__tests__/Markdown.github-code-blocks.test.tsx \
  src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx \
  src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx \
  src/components/Option/Settings/__tests__/ChatSettings.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run a frontend build or compile check**

Run from `apps/tldw-frontend`:

```bash
bun run build
```

Expected: PASS and no `mermaid` resolution failure.

If build time is too high during local iteration, run at least:

```bash
bun run compile
```

before finalizing the implementation PR.

- [ ] **Step 3: Check dynamic import remains lazy**

Inspect the final `Mermaid.tsx` and `Markdown.tsx` diff:

```bash
git diff -- apps/packages/ui/src/components/Common/Mermaid.tsx \
  apps/packages/ui/src/components/Common/Markdown.tsx
```

Expected:

- `import("mermaid")` remains inside the Mermaid render effect.
- `Markdown.tsx` imports the lightweight block component only.
- No top-level static `import mermaid from "mermaid"` exists.

- [ ] **Step 4: Run Bandit only if backend Python was touched**

If no backend Python files were touched, record:

```text
Bandit not applicable: implementation touched frontend TypeScript/React only.
```

If backend Python was touched unexpectedly, run from repo root after activating the venv:

```bash
source .venv/bin/activate
python -m bandit -r <touched_backend_paths> -f json -o /tmp/bandit_mermaid_chat.json
```

- [ ] **Step 5: Final self-review checklist**

Confirm:

- assistant Mermaid fences render inline when enabled;
- disabled setting routes Mermaid fences through existing code blocks;
- user messages remain plain text or existing user-markdown behavior without Mermaid rendering;
- `mmd`, `mermaid-js`, unfenced prose, raw SVG/XML, and Graphviz/DOT are not rendered by this feature;
- streaming assistant messages do not render diagrams until complete;
- invalid Mermaid shows local fallback and copyable source;
- preview dialog zoom/pan/reset/copy/download are keyboard accessible;
- no unrelated markdown consumers opt in.

- [ ] **Step 6: Commit final verification fixes if any**

If final verification required small fixes:

```bash
git add <fixed files>
git commit -m "fix: harden Mermaid chat rendering"
```

If no fixes were needed, do not create an empty commit.

---

## Implementation Notes

- Do not unify existing artifact Mermaid viewers in this PR. `ArtifactModalContent.tsx` and `ArtifactsPanel.tsx` can continue using `Common/Mermaid`.
- Do not render raw SVG/XML blocks. SVG insertion is only for Mermaid-generated SVG returned by strict-mode Mermaid rendering.
- Do not add aliases such as `mmd` or `mermaid-js` in v1.
- Keep user messages unchanged. `HumanMessge.tsx` should not receive `enableMermaidDiagrams`, even when `useMarkdownForUserMessage` is enabled.
- Prefer lucide icons for new icon buttons.
- Keep new block/dialog styling aligned with existing code-block and artifact chrome; avoid introducing a large card-heavy visual language.
- If a test mock exposes `React.lazy` timing issues, wrap assertions with `await screen.findBy...` or `waitFor`.

## Handoff

Plan complete when this file is reviewed, `TASK-511` is marked done, and the plan commit is created. For implementation, choose one:

1. **Subagent-Driven (recommended when explicitly authorized)** - dispatch a fresh worker per task, review between tasks, and keep task ownership disjoint.
2. **Inline Execution** - execute tasks in this session using `superpowers:executing-plans`, with checkpoints after each task.
