# Research Workspace NotebookLM-Core Parity WP1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make existing Research Workspace basics obvious to NotebookLM migrants without adding new ingestion backends, media generation, or agent orchestration.

**Architecture:** Keep this as UI/copy and per-turn prompt behavior over existing Research Workspace, ChatPane, StudioPane, and clipper flows. Reuse existing source ingestion utilities, capability gates, ChatPane submit-message shaping, Studio output grouping, and web clipper navigation helpers. No new stores, backend endpoints, dependencies, or sidepanel routes.

**Tech Stack:** React 18, TypeScript, Ant Design, lucide-react, Zustand stores, Vitest and Testing Library.

---

## Source Spec

- Spec: `Docs/superpowers/specs/2026-07-04-research-workspace-notebooklm-pro-ultra-review-design.md`
- Backlog: `TASK-12149`

## Non-Goals

- Do not add Google Drive autosync, Google auth, Gemini import, quota UI, video rendering, infographic generation, or Ultra agent execution.
- Do not add CSV/PPTX/image ingestion unless existing upload validation already accepts those types. It currently does not.
- Do not create a full Research Workspace extension sidepanel route.
- Do not add a global settings store for chat style/length. Per-turn UI state is enough.
- Do not build new save-to-note plumbing. Existing coverage in
  `ChatPane.stage3.test.tsx` and `StudioPane.stage2.test.tsx` already proves
  chat messages and artifacts can be saved to notes.

## File Structure

- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx`
  - Owns Add Source tabs and user-facing source capability copy.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/source-ingestion-utils.ts`
  - Only if copy needs a reusable accepted-type label derived from current accept list. Prefer not to change validation.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx`
  - Covers Add Source visible expectations.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-ingestion-utils.test.ts`
  - Only if `source-ingestion-utils.ts` changes.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx`
  - Owns chat toolbar and per-turn message text.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage3.test.tsx`
  - Covers chat style/length controls and per-turn message instructions.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx`
  - Owns output button labels, descriptions, groups, and primary output set.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx`
  - Covers visible output grouping.
- Modify `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx`
  - Owns the existing "Save and open" and "Analyze now" handoffs.
- Modify `apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`
  - Covers extension handoff behavior.

## Task 1: Make Add Source Expectations Explicit

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx`
- Optional Modify/Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/source-ingestion-utils.ts`, `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-ingestion-utils.test.ts`

- [ ] **Step 1: Write the failing Add Source copy test**

Add one test to `AddSourceModal.stage2.intake.test.tsx`:

```tsx
it("explains supported source imports and Google-specific skips", () => {
  render(<AddSourceModal />)

  expect(screen.getByText("Supported now")).toBeInTheDocument()
  expect(screen.getByText(/PDF, DOCX, TXT\/Markdown, ePub, HTML, XML, JSON/i)).toBeInTheDocument()
  expect(screen.getByText(/audio and video files/i)).toBeInTheDocument()
  expect(screen.getByText(/URL imports use server extraction/i)).toBeInTheDocument()
  expect(screen.getByText(/Not included here: Google Drive sync/i)).toBeInTheDocument()
})
```

- [ ] **Step 2: Run the failing test**

Working directory: `apps/tldw-frontend`

Run:

```bash
bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx
```

Expected: FAIL because the source expectations copy is not rendered.

- [ ] **Step 3: Add the smallest visible copy block**

In `AddSourceModal.tsx`, add a local helper near `UploadTab`:

```tsx
const SourceImportExpectations: React.FC = () => {
  const { t } = useTranslation(["playground", "common"])
  return (
    <div className="rounded-md border border-border bg-surface2/40 p-3 text-xs text-text-muted">
      <p className="font-semibold text-text">
        {t("playground:sources.importExpectationsTitle", "Supported now")}
      </p>
      <ul className="mt-1 list-disc space-y-1 pl-4">
        <li>
          {t(
            "playground:sources.importExpectationsFiles",
            "Files: PDF, DOCX, TXT/Markdown, ePub, HTML, XML, JSON, audio and video files."
          )}
        </li>
        <li>
          {t(
            "playground:sources.importExpectationsUrls",
            "URLs: URL imports use server extraction; YouTube/direct media links depend on server support and may import extracted text or transcripts."
          )}
        </li>
        <li>
          {t(
            "playground:sources.importExpectationsSkips",
            "Not included here: Google Drive sync, Google Docs/Slides/Sheets autosync, images, CSV, and PPTX."
          )}
        </li>
      </ul>
    </div>
  )
}
```

Render `<SourceImportExpectations />` under the upload dropzone hint, before upload progress. Also tighten the unsupported-file copy in `beforeUpload` and `mapSourceIngestionError` only if tests require it, using the same accepted type list.

- [ ] **Step 4: Run Add Source tests**

Working directory: `apps/tldw-frontend`

Run:

```bash
bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage1.ingestion.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-ingestion-utils.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/source-ingestion-utils.ts apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-ingestion-utils.test.ts
git commit -m "feat: clarify research workspace source intake"
```

## Task 2: Add Chat Style And Length Presets Without Prompt Pipeline Changes

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage3.test.tsx`

- [ ] **Step 1: Write the failing chat controls test**

Add one test to `ChatPane.stage3.test.tsx`:

```tsx
it("prepends response style and length instructions only when presets are active", async () => {
  renderChatPane()

  fireEvent.change(screen.getByLabelText("Response style"), {
    target: { value: "explain" }
  })
  fireEvent.change(screen.getByLabelText("Answer length"), {
    target: { value: "brief" }
  })
  fireEvent.change(screen.getByLabelText("Chat message"), {
    target: { value: "What should I know?" }
  })
  fireEvent.keyDown(screen.getByLabelText("Chat message"), {
    key: "Enter"
  })

  await waitFor(() => expect(mockOnSubmit).toHaveBeenCalled())
  expect(mockOnSubmit).toHaveBeenCalledWith(
    expect.objectContaining({
      message: expect.stringContaining("Response preference:")
    })
  )
  expect(mockOnSubmit).toHaveBeenCalledWith(
    expect.objectContaining({
      message: expect.stringContaining("explain the answer")
    })
  )
  expect(mockOnSubmit).toHaveBeenCalledWith(
    expect.objectContaining({
      message: expect.stringContaining("Keep the answer brief")
    })
  )
  expect(mockOnSubmit).toHaveBeenCalledWith(
    expect.objectContaining({
      message: expect.stringContaining("User question: What should I know?")
    })
  )
})
```

Keep the existing exact default-submit test unchanged; it should continue to
assert `{ message: prompt, image: "" }` when Balanced/Standard are selected.

- [ ] **Step 2: Run the failing ChatPane test**

Working directory: `apps/tldw-frontend`

Run:

```bash
bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage3.test.tsx
```

Expected: FAIL because the controls do not exist.

- [ ] **Step 3: Add local preset state and per-turn message instructions**

In `ChatPane/index.tsx`, add local types/constants near `ChatModePreference`:

```ts
type ChatResponseStyle = "balanced" | "explain" | "source_first"
type ChatResponseLength = "standard" | "brief" | "detailed"

const CHAT_RESPONSE_STYLE_OPTIONS: Array<{
  value: ChatResponseStyle
  label: string
  instruction: string | null
}> = [
  { value: "balanced", label: "Balanced", instruction: null },
  {
    value: "explain",
    label: "Explain",
    instruction: "When answering, explain the answer clearly for a reader who is learning the topic."
  },
  {
    value: "source_first",
    label: "Source-first",
    instruction: "When answering, lead with what the selected sources support and call out uncertainty."
  }
]

const CHAT_RESPONSE_LENGTH_OPTIONS: Array<{
  value: ChatResponseLength
  label: string
  instruction: string | null
}> = [
  { value: "standard", label: "Standard", instruction: null },
  { value: "brief", label: "Brief", instruction: "Keep the answer brief." },
  { value: "detailed", label: "Detailed", instruction: "Give a detailed answer with useful structure." }
]
```

Inside `ChatPane`, add:

```tsx
const [responseStyle, setResponseStyle] =
  React.useState<ChatResponseStyle>("balanced")
const [responseLength, setResponseLength] =
  React.useState<ChatResponseLength>("standard")
```

Build the per-turn instruction. It should return `null` when no preset is active
so existing exact submit-payload tests keep passing:

```ts
const buildResponsePresetInstruction = React.useCallback(() => {
  const instructions = [
    CHAT_RESPONSE_STYLE_OPTIONS.find((option) => option.value === responseStyle)?.instruction,
    CHAT_RESPONSE_LENGTH_OPTIONS.find((option) => option.value === responseLength)?.instruction
  ].filter((value): value is string => Boolean(value))

  if (instructions.length === 0) return null
  return `Response preference: ${instructions.join(" ")}`
}, [responseLength, responseStyle])
```

Change the submit path before `buildFullSourceContextPrompt(...)` runs:

```ts
const responsePresetInstruction = buildResponsePresetInstruction()
const messageWithResponsePreset = responsePresetInstruction
  ? `${responsePresetInstruction}\n\nUser question: ${message}`
  : message
const preparedMessage = await buildFullSourceContextPrompt(messageWithResponsePreset)
const submitResult = await onSubmit({ message: preparedMessage, image: "" })
```

Do not send `requestOverrides` or `selectedSystemPrompt`. In the real hook path,
`selectedSystemPrompt` is a stored prompt/template id, not literal instruction
text. This task deliberately uses the same user-message shaping pattern already
used by full-source-context prompts.

Existing tests assert the default plain payload shape:

```ts
expect(mockOnSubmit).toHaveBeenCalledWith({
  message: preparedMessage,
  image: ""
})
```

- [ ] **Step 4: Render compact native selects in the existing toolbar**

Add two labeled native selects near the General/RAG mode control:

```tsx
<label className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-surface/80 px-2 py-1 text-[11px] text-text-muted">
  <span>{t("playground:chat.responseStyleLabel", "Style")}</span>
  <select
    aria-label={t("playground:chat.responseStyleAria", "Response style")}
    className="bg-transparent text-text outline-none"
    value={responseStyle}
    onChange={(event) => setResponseStyle(event.target.value as ChatResponseStyle)}
  >
    {CHAT_RESPONSE_STYLE_OPTIONS.map((option) => (
      <option key={option.value} value={option.value}>
        {option.label}
      </option>
    ))}
  </select>
</label>
<label className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-surface/80 px-2 py-1 text-[11px] text-text-muted">
  <span>{t("playground:chat.responseLengthLabel", "Length")}</span>
  <select
    aria-label={t("playground:chat.responseLengthAria", "Answer length")}
    className="bg-transparent text-text outline-none"
    value={responseLength}
    onChange={(event) => setResponseLength(event.target.value as ChatResponseLength)}
  >
    {CHAT_RESPONSE_LENGTH_OPTIONS.map((option) => (
      <option key={option.value} value={option.value}>
        {option.label}
      </option>
    ))}
  </select>
</label>
```

- [ ] **Step 5: Run ChatPane tests**

Working directory: `apps/tldw-frontend`

Run:

```bash
bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage3.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage3.test.tsx
git commit -m "feat: add research workspace chat response presets"
```

## Task 3: Make Studio Output Groups Match NotebookLM Expectations

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx`

- [ ] **Step 1: Write the failing Studio grouping test**

Add one test to `StudioPane.stage3.test.tsx`:

```tsx
it("groups Notebook-style outputs before advanced analysis outputs", async () => {
  renderExpandedStudioPane()

  expect(await screen.findByText("Notebook basics")).toBeInTheDocument()
  expect(screen.getByRole("button", { name: "Summary" })).toBeInTheDocument()
  expect(screen.getByRole("button", { name: "Audio Summary" })).toBeInTheDocument()
  expect(screen.getByRole("button", { name: "Mind Map" })).toBeInTheDocument()
  expect(screen.getByRole("button", { name: "Flashcards" })).toBeInTheDocument()
  expect(screen.getByRole("button", { name: "Quiz" })).toBeInTheDocument()
})
```

- [ ] **Step 2: Run the failing Studio test**

Working directory: `apps/tldw-frontend`

Run:

```bash
bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx
```

Expected: FAIL because the group label is still `Study Aids`, `Analysis`, or `Creative`.

- [ ] **Step 3: Update only labels, descriptions, grouping, and primary set**

In `StudioPane/index.tsx`, change `OUTPUT_GROUPS` to:

```ts
const OUTPUT_GROUPS: Array<{
  id: string
  label: string
  types: ArtifactType[]
}> = [
  {
    id: "notebook-basics",
    label: "Notebook basics",
    types: ["summary", "audio_overview", "mindmap", "flashcards", "quiz"]
  },
  {
    id: "reports-and-tables",
    label: "Reports and tables",
    types: ["report", "slides", "data_table"]
  },
  {
    id: "evidence-analysis",
    label: "Evidence analysis",
    types: ["compare_sources", "timeline"]
  }
]
```

Change `PRIMARY_OUTPUT_TYPES` to:

```ts
const PRIMARY_OUTPUT_TYPES = new Set<ArtifactType>([
  "summary",
  "audio_overview",
  "mindmap",
  "flashcards",
  "quiz"
])
```

Update the existing `shows primary and secondary output actions with description
tooltips` test in `StudioPane.stage3.test.tsx` so it no longer expects `Report`
or `Compare Sources` in the primary button row. The primary row should show
Summary, Audio Summary, Mind Map, Flashcards, and Quiz before expanding "More
outputs". After expanding, assert that Report, Slides, Data Table, Compare
Sources, and Timeline are available:

```tsx
expect(screen.getByRole("button", { name: "Audio Summary" })).toBeInTheDocument()
expect(screen.getByRole("button", { name: "Mind Map" })).toBeInTheDocument()
expect(screen.queryByRole("button", { name: "Report" })).not.toBeInTheDocument()

fireEvent.click(screen.getByRole("button", { name: /More outputs/ }))

expect(screen.getByRole("button", { name: "Report" })).toBeInTheDocument()
expect(screen.getByRole("button", { name: "Compare Sources" })).toBeInTheDocument()
```

Keep literature templates where they already are. Do not add video overview or infographic in this task.

- [ ] **Step 4: Run Studio tests**

Working directory: `apps/tldw-frontend`

Run:

```bash
bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx
git commit -m "feat: clarify research workspace output groups"
```

## Task 4: Tighten Existing Extension Handoff, No Sidepanel Clone

**Files:**
- Modify: `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx`
- Test: `apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`

- [ ] **Step 1: Update the existing workspace-open routing matrix**

In `WebClipperPanel.save-flow.test.tsx`, update the existing
`save and open routes %s clips to %s` matrix so `Both` expects
`#/research-workspace` when the save response includes a workspace placement:

```tsx
it.each([
  ["Note", "#/notes"],
  ["Both", "#/research-workspace"],
  ["Workspace", "#/research-workspace"]
])(
  "save and open routes %s clips to %s",
  async (destinationLabel, expectedPath) => {
    const user = userEvent.setup()
    const hasWorkspacePlacement = destinationLabel !== "Note"
    apiMocks.saveWebClip.mockResolvedValueOnce({
      clip_id: "clip-123",
      note_id: "note-123",
      note: { id: "note-123", title: "Example Story", version: 1 },
      workspace_placement: hasWorkspacePlacement
        ? {
            workspace_id: "workspace-alpha",
            workspace_note_id: 42,
            source_note_id: "note-123"
          }
        : null,
      attachments: [],
      status: "saved",
      warnings: [],
      workspace_placement_saved: hasWorkspacePlacement,
      workspace_placement_count: hasWorkspacePlacement ? 1 : 0
    })

    render(<WebClipperPanel draft={createDraft()} onCancel={vi.fn()} />)

    if (destinationLabel !== "Note") {
      await chooseWorkspaceDestination(
        user,
        destinationLabel as "Workspace" | "Both"
      )
    }

    await user.click(screen.getByRole("button", { name: "Save and open" }))

    await waitFor(() => {
      expect(apiMocks.saveWebClip).toHaveBeenCalledTimes(1)
    })

    expect(openTabMock).toHaveBeenCalledWith(
      expect.objectContaining({
        url: expect.stringContaining(expectedPath)
      })
    )
  })
)
```

- [ ] **Step 2: Run the failing clipper test**

Working directory: `apps/tldw-frontend`

Run:

```bash
bunx vitest run ../packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
```

Expected: FAIL if current `createOpenTargetUrl` returns notes for `both` destination mode.

- [ ] **Step 3: Prefer workspace route whenever placement exists**

Change `createOpenTargetUrl` in `WebClipperPanel.tsx`:

```ts
if (response.workspace_placement) {
  return chromeApi.runtime.getURL("options.html#/research-workspace")
}
```

Keep the existing notes fallback for note-only saves:

```ts
return chromeApi.runtime.getURL("options.html#/notes")
```

Do not add a Research Workspace sidepanel route.

- [ ] **Step 4: Run clipper tests**

Working directory: `apps/tldw-frontend`

Run:

```bash
bunx vitest run ../packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
git commit -m "fix: open clipped workspace sources in research workspace"
```

## Final Verification

Working directory: `apps/tldw-frontend`

Run:

```bash
bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage1.ingestion.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/source-ingestion-utils.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage3.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx ../packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
```

Run typecheck if the focused tests pass:

```bash
bun run typecheck
```

Run Bandit only if backend Python files are touched. This plan should not touch Python, so record Bandit as skipped for frontend-only changes.

## Expected Outcome

- Add Source tells users what is supported now and what is intentionally not Google-synced.
- Chat exposes style/length presets without global settings or new APIs.
- Studio's first output group matches NotebookLM-style expectations while preserving advanced literature work products.
- The existing extension clipper opens Research Workspace when a saved clip was placed in a workspace.
- WP2 video/infographic and WP4 agentic tasks remain deferred.
