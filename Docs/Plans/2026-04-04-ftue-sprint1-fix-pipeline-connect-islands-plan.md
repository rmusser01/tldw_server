# FTUE Sprint 1: Fix Broken Pipeline + Connect the Islands

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make documents ingested via QuickIngest openable in Document Workspace, then connect ingest → knowledge → workspace with cross-page CTAs so first-time users have a guided path to value.

**Architecture:** Three isolated changes that layer on each other: (1) fix `keep_original_file` in the QuickIngest batch pipeline so document files are preserved, (2) add "Open in Workspace" and "Search in Knowledge" CTAs to the wizard results step, (3) add a "Recently ingested" section to the Document Workspace picker and an "Open in Workspace" action to Knowledge QA source cards. No new endpoints, no new stores, no backend schema changes — all changes are frontend except one field addition to the ingest payload.

**Tech Stack:** TypeScript/React (Vitest for tests), existing Zustand stores, existing `useIngestResults` navigation patterns

**Design doc:** `/Users/macbook-dev/.claude/plans/golden-chasing-bumblebee.md`

---

### Task 1: Add `keep_original_file` to QuickIngest batch payload (ING-011 fix)

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts:396-401`
- Test: `apps/packages/ui/src/services/tldw/__tests__/quick-ingest-batch-keep-file.test.ts` (create)

**Step 1: Write the failing test**

Create test file `apps/packages/ui/src/services/tldw/__tests__/quick-ingest-batch-keep-file.test.ts`:

```typescript
// @vitest-environment jsdom
import { describe, expect, it } from "vitest"

/**
 * The buildFields() function is not exported directly, so we test the behavior
 * indirectly by importing the module and checking the built payload includes
 * keep_original_file for document types but not for audio/video.
 *
 * Since buildFields is a module-private function, we test through the
 * public submitQuickIngestBatch or by extracting buildFields for test.
 * For now we test the intent: document types (pdf, ebook, document) should
 * include keep_original_file=true in the payload.
 */

// We'll directly test the logic we're adding by extracting it as a small helper.
import { shouldKeepOriginalFile } from "../quick-ingest-batch"

describe("shouldKeepOriginalFile", () => {
  it("returns true for pdf media type", () => {
    expect(shouldKeepOriginalFile("pdf")).toBe(true)
  })

  it("returns true for ebook media type", () => {
    expect(shouldKeepOriginalFile("ebook")).toBe(true)
  })

  it("returns true for document media type", () => {
    expect(shouldKeepOriginalFile("document")).toBe(true)
  })

  it("returns false for audio media type", () => {
    expect(shouldKeepOriginalFile("audio")).toBe(false)
  })

  it("returns false for video media type", () => {
    expect(shouldKeepOriginalFile("video")).toBe(false)
  })

  it("returns false for html media type", () => {
    expect(shouldKeepOriginalFile("html")).toBe(false)
  })

  it("returns false for unknown types", () => {
    expect(shouldKeepOriginalFile("auto")).toBe(false)
  })
})
```

**Step 2: Run test to verify it fails**

```bash
cd apps/packages/ui && npx vitest run src/services/tldw/__tests__/quick-ingest-batch-keep-file.test.ts
```

Expected: FAIL — `shouldKeepOriginalFile` is not exported from `quick-ingest-batch`.

**Step 3: Add helper and wire into buildFields**

In `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts`, add the exported helper BEFORE the `buildFields` function (before line 378):

```typescript
/** Document types whose original file should be preserved for Document Workspace. */
const KEEP_FILE_TYPES = new Set(["pdf", "ebook", "document"])

/** Returns true if this media type's original file should be stored on the server. */
export const shouldKeepOriginalFile = (mediaType: string): boolean =>
  KEEP_FILE_TYPES.has(mediaType)
```

Then in the `buildFields` function, add `keep_original_file` to the fields object. Change lines 396-401 from:

```typescript
  const fields: Record<string, any> = {
    media_type: mediaType,
    perform_analysis: Boolean(common?.perform_analysis),
    perform_chunking: resolvePerformChunking(common?.perform_chunking),
    overwrite_existing: Boolean(common?.overwrite_existing)
  }
```

To:

```typescript
  const fields: Record<string, any> = {
    media_type: mediaType,
    perform_analysis: Boolean(common?.perform_analysis),
    perform_chunking: resolvePerformChunking(common?.perform_chunking),
    overwrite_existing: Boolean(common?.overwrite_existing),
    keep_original_file: shouldKeepOriginalFile(mediaType)
  }
```

**Step 4: Run test to verify it passes**

```bash
cd apps/packages/ui && npx vitest run src/services/tldw/__tests__/quick-ingest-batch-keep-file.test.ts
```

Expected: PASS — all 7 assertions pass.

**Step 5: Run existing QuickIngest tests to check for regressions**

```bash
cd apps/packages/ui && npx vitest run src/components/Common/QuickIngest/__tests__/
```

Expected: All existing tests pass. The new field is additive and doesn't change any existing behavior.

**Step 6: Commit**

```
fix(ingest): send keep_original_file=true for document types in QuickIngest

QuickIngest wizard never sent keep_original_file to the backend, causing
all ingested documents (PDF, EPUB, DOCX) to have their original files
discarded. This made Document Workspace unable to display any document
ingested through the standard path (ING-011 / DOC-002 P0).

The backend defaults keep_original_file to False, so we now explicitly
set it to true for document types (pdf, ebook, document) in the batch
payload builder.
```

---

### Task 2: Wire navigation callbacks into WizardResultsStep (ING-004)

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx:1354`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx:20-25, 56-109`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx` (create)

**Step 1: Write the failing test for new navigation buttons**

Create `apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx`:

```typescript
// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"
import React from "react"

// Mock react-i18next
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, opts?: any) => opts?.defaultValue ?? key,
  }),
}))

// Mock IngestWizardContext
vi.mock("../IngestWizardContext", () => ({
  useIngestWizard: () => ({
    state: {
      results: [
        {
          id: "test-1",
          status: "ok" as const,
          type: "pdf",
          title: "My Test PDF",
          mediaId: 42,
        },
      ],
      processingState: { elapsed: 10 },
    },
    reset: vi.fn(),
  }),
}))

import { WizardResultsStep } from "../WizardResultsStep"

describe("WizardResultsStep navigation buttons", () => {
  it("renders 'Search in Knowledge' button when onSearchKnowledge is provided", () => {
    const onSearchKnowledge = vi.fn()
    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onSearchKnowledge={onSearchKnowledge}
      />
    )
    const btn = screen.getByText("Search in Knowledge")
    expect(btn).toBeTruthy()
    fireEvent.click(btn)
    expect(onSearchKnowledge).toHaveBeenCalledTimes(1)
  })

  it("renders 'Open in Workspace' button when onOpenWorkspace is provided", () => {
    const onOpenWorkspace = vi.fn()
    render(
      <WizardResultsStep
        onClose={vi.fn()}
        onOpenWorkspace={onOpenWorkspace}
      />
    )
    const btn = screen.getByText("Open in Workspace")
    expect(btn).toBeTruthy()
    fireEvent.click(btn)
    expect(onOpenWorkspace).toHaveBeenCalledTimes(1)
  })

  it("does not render navigation buttons when callbacks are not provided", () => {
    render(<WizardResultsStep onClose={vi.fn()} />)
    expect(screen.queryByText("Search in Knowledge")).toBeNull()
    expect(screen.queryByText("Open in Workspace")).toBeNull()
  })
})
```

**Step 2: Run test to verify it fails**

```bash
cd apps/packages/ui && npx vitest run src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx
```

Expected: FAIL — `onSearchKnowledge` and `onOpenWorkspace` are not recognized props.

**Step 3: Add new props and CTA section to WizardResultsStep**

In `apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx`:

**3a.** Add new icon imports at line 8 (after `Trash2`):

```typescript
import { Search, BookOpen } from "lucide-react"
```

**3b.** Extend the props type (lines 20-25) — add two new optional callbacks:

```typescript
type WizardResultsStepProps = {
  onClose: () => void
  onRetryItems?: (itemIds: string[]) => void
  onOpenMedia?: (item: WizardResultItem) => void
  onDiscussInChat?: (item: WizardResultItem) => void
  onSearchKnowledge?: () => void
  onOpenWorkspace?: (item: WizardResultItem) => void
}
```

**3c.** Destructure the new props at line 189-194:

```typescript
export const WizardResultsStep: React.FC<WizardResultsStepProps> = ({
  onClose,
  onRetryItems,
  onOpenMedia,
  onDiscussInChat,
  onSearchKnowledge,
  onOpenWorkspace,
}) => {
```

**3d.** Add a "Next steps" CTA section after the success section (after line 299, before the errors section). Insert between the `</section>` closing tag of successes and the `{errors.length > 0 &&` conditional:

```tsx
        {/* Next steps CTAs */}
        {successes.length > 0 && (onSearchKnowledge || onOpenWorkspace) && (
          <div className="mt-4 rounded-lg border border-primary/20 bg-primary/5 px-4 py-3">
            <p className="mb-2 text-xs font-medium text-text-muted">
              {qi("wizard.results.nextSteps", "What's next?")}
            </p>
            <div className="flex flex-wrap gap-2">
              {onSearchKnowledge && (
                <button
                  type="button"
                  onClick={onSearchKnowledge}
                  className="flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2 transition-colors"
                  aria-label={qi("wizard.results.searchKnowledgeAria", "Search your ingested content in Knowledge QA")}
                >
                  <Search className="h-3.5 w-3.5" aria-hidden="true" />
                  {qi("wizard.results.searchKnowledge", "Search in Knowledge")}
                </button>
              )}
              {onOpenWorkspace && successes.some(s => ["pdf", "ebook"].includes(s.type)) && (
                <button
                  type="button"
                  onClick={() => {
                    const docItem = successes.find(s => ["pdf", "ebook"].includes(s.type))
                    if (docItem) onOpenWorkspace(docItem)
                  }}
                  className="flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2 transition-colors"
                  aria-label={qi("wizard.results.openWorkspaceAria", "Open document in Document Workspace")}
                >
                  <BookOpen className="h-3.5 w-3.5" aria-hidden="true" />
                  {qi("wizard.results.openWorkspace", "Open in Workspace")}
                </button>
              )}
            </div>
          </div>
        )}
```

**Step 4: Run test to verify it passes**

```bash
cd apps/packages/ui && npx vitest run src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx
```

Expected: PASS.

**Step 5: Run all QuickIngest tests for regression check**

```bash
cd apps/packages/ui && npx vitest run src/components/Common/QuickIngest/__tests__/
```

Expected: All existing tests pass. New props are optional so existing renders are unaffected.

**Step 6: Commit**

```
feat(ingest): add Knowledge and Workspace CTAs to results step

After successful ingest, users now see "Search in Knowledge" and
"Open in Workspace" buttons (ING-004). The workspace button only
appears when PDF/EPUB documents were ingested. Callbacks are optional
so existing modal usage is unaffected.
```

---

### Task 3: Wire callbacks from QuickIngestWizardModal to results step

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx:1354`

**Step 1: Add navigation imports and callbacks**

At the top of `QuickIngestWizardModal.tsx`, add/verify these imports exist:

```typescript
import { useNavigate } from "react-router-dom"
```

And add to the existing lucide imports if needed:

```typescript
import { DOCUMENT_WORKSPACE_PATH } from "@/routes/route-paths"
```

**Step 2: Add navigation callbacks inside `WizardModalContent`**

Inside the `WizardModalContent` component (which renders the step content), add navigation callbacks. Find the component body — it should be around lines 699-1388. Add before the `stepContent` useMemo (before ~line 1339):

```typescript
  const navigate = useNavigate()

  const handleSearchKnowledge = useCallback(() => {
    onClose()
    // Small delay to let modal close animation complete
    window.setTimeout(() => navigate("/knowledge"), 150)
  }, [navigate, onClose])

  const handleOpenWorkspace = useCallback(
    (item: WizardResultItem) => {
      onClose()
      const mediaId = item.mediaId
      if (mediaId != null) {
        window.setTimeout(
          () => navigate(`${DOCUMENT_WORKSPACE_PATH}?open=${mediaId}`),
          150
        )
      } else {
        window.setTimeout(() => navigate(DOCUMENT_WORKSPACE_PATH), 150)
      }
    },
    [navigate, onClose]
  )
```

**Step 3: Pass callbacks to WizardResultsStep**

Change line 1354 from:

```typescript
        return <WizardResultsStep onClose={onClose} />
```

To:

```typescript
        return (
          <WizardResultsStep
            onClose={onClose}
            onSearchKnowledge={handleSearchKnowledge}
            onOpenWorkspace={handleOpenWorkspace}
          />
        )
```

**Step 4: Add the `WizardResultItem` import if not already present**

Verify `WizardResultItem` is imported at the top of the file. If not, add:

```typescript
import type { WizardResultItem } from "./QuickIngest/types"
```

**Step 5: Run existing wizard modal tests**

```bash
cd apps/packages/ui && npx vitest run src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx
cd apps/packages/ui && npx vitest run src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx
```

Expected: Pass. If `useNavigate` mock is needed, add to test mocks following the existing pattern from `QuickIngestModal.session-cancel.test.tsx`:

```typescript
vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return { ...actual, useNavigate: () => vi.fn() }
})
```

**Step 6: Commit**

```
feat(ingest): wire Knowledge and Workspace navigation from wizard modal

QuickIngestWizardModal now passes onSearchKnowledge and onOpenWorkspace
callbacks to WizardResultsStep. After successful ingest, clicking
"Search in Knowledge" navigates to /knowledge. Clicking "Open in
Workspace" navigates to /document-workspace?open={mediaId}.
```

---

### Task 4: Add "Recently ingested" section to DocumentPickerModal (DOC-001)

**Files:**
- Modify: `apps/packages/ui/src/components/DocumentWorkspace/DocumentPickerModal.tsx`
- Modify: `apps/packages/ui/src/store/quick-ingest.tsx`

**Step 1: Add `recentlyIngestedIds` to the QuickIngest store**

In `apps/packages/ui/src/store/quick-ingest.tsx`, extend the store to track recently ingested document media IDs. Add to the store type:

```typescript
  recentlyIngestedDocIds: number[]
  addRecentlyIngestedDocId: (id: number) => void
  clearRecentlyIngestedDocIds: () => void
```

Add to the store implementation (inside `create()`):

```typescript
  recentlyIngestedDocIds: [],
  addRecentlyIngestedDocId: (id) =>
    set((s) => ({
      recentlyIngestedDocIds: [id, ...s.recentlyIngestedDocIds.filter((x) => x !== id)].slice(0, 20),
    })),
  clearRecentlyIngestedDocIds: () => set({ recentlyIngestedDocIds: [] }),
```

**Step 2: Record ingested document IDs from the results step**

In `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`, after the successful results are available, record document media IDs. Add an effect inside `WizardModalContent` that runs when step 5 (results) is reached:

```typescript
  const addRecentlyIngestedDocId = useQuickIngestStore(s => s.addRecentlyIngestedDocId)

  // Record successfully ingested document IDs for DocumentPicker "recently ingested"
  useEffect(() => {
    if (state.currentStep !== 5) return
    for (const item of state.results) {
      if (
        item.status === "ok" &&
        item.mediaId != null &&
        ["pdf", "ebook", "document"].includes(item.type)
      ) {
        addRecentlyIngestedDocId(Number(item.mediaId))
      }
    }
  }, [state.currentStep, state.results, addRecentlyIngestedDocId])
```

Add the store import at the top:

```typescript
import { useQuickIngestStore } from "@/store/quick-ingest"
```

**Step 3: Show "Recently ingested" in DocumentPickerModal**

In `apps/packages/ui/src/components/DocumentWorkspace/DocumentPickerModal.tsx`, add a section at the top of the Library tab that shows recently ingested items.

Import the store:

```typescript
import { useQuickIngestStore } from "@/store/quick-ingest"
```

Inside the component, read the IDs:

```typescript
const recentlyIngestedDocIds = useQuickIngestStore(s => s.recentlyIngestedDocIds)
```

In the Library tab rendering (the `libraryPane` section), before the existing media list, add:

```tsx
{recentlyIngestedDocIds.length > 0 && !searchQuery && (
  <div className="mb-4">
    <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
      Recently ingested
    </h4>
    <div className="space-y-1">
      {recentlyIngestedDocIds.slice(0, 3).map((id) => (
        <button
          key={id}
          type="button"
          onClick={() => handleOpenDocument(id)}
          className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm hover:bg-surface2 transition-colors"
        >
          <FileText className="h-4 w-4 flex-shrink-0 text-primary" />
          <span className="truncate">Document #{id}</span>
          <span className="ml-auto text-xs text-text-muted">Just ingested</span>
        </button>
      ))}
    </div>
  </div>
)}
```

Note: The exact rendering depends on what data is available. Since we only store IDs, we show a minimal view. If richer data is needed, we can fetch titles via `tldwClient.getMediaDetails()` in a follow-up, but this provides the immediate connection between ingest and workspace.

**Step 4: Handle `?open={mediaId}` URL parameter**

In `apps/packages/ui/src/components/DocumentWorkspace/DocumentWorkspacePage.tsx`, add URL parameter detection to auto-open a document when navigated from the ingest results:

```typescript
import { useSearchParams } from "react-router-dom"
```

Inside the component, after initial setup:

```typescript
const [searchParams, setSearchParams] = useSearchParams()
const autoOpenId = searchParams.get("open")

useEffect(() => {
  if (autoOpenId && !activeDocumentId) {
    handleOpenDocument(Number(autoOpenId))
    // Clean up URL param after use
    setSearchParams((prev) => {
      prev.delete("open")
      return prev
    }, { replace: true })
  }
}, [autoOpenId, activeDocumentId])
```

**Step 5: Run existing DocumentWorkspace tests**

```bash
cd apps/packages/ui && npx vitest run src/components/DocumentWorkspace/
```

Expected: Pass. New URL parameter handling only triggers when param is present.

**Step 6: Commit**

```
feat(workspace): add recently-ingested section and auto-open from URL

DocumentPickerModal now shows a "Recently ingested" section at the top
of the Library tab, pulling IDs from the QuickIngest store (DOC-001).

DocumentWorkspacePage accepts ?open={mediaId} URL parameter to
auto-open a specific document when navigated from QuickIngest results.

The QuickIngest store now tracks recentlyIngestedDocIds (up to 20).
```

---

### Task 5: Improve error messaging for documents without original files (DOC-002)

**Files:**
- Modify: `apps/packages/ui/src/components/DocumentWorkspace/DocumentPickerModal.tsx`

**Step 1: Improve the "file not available" error**

When a user tries to open a document that was ingested before the `keep_original_file` fix, they currently see "The original file is not available." This needs better guidance.

Find the error handling section in `DocumentPickerModal.tsx` where the 404 / missing file error is shown. Update the error message to include actionable guidance:

Change the error message from something like:

```
"The original file is not available (it may have failed to store)."
```

To:

```
"This document's original file was not preserved during ingest. To view it in the workspace, re-upload it using the Upload tab above, or re-ingest it — newer ingests automatically preserve document files."
```

**Step 2: Run existing tests**

```bash
cd apps/packages/ui && npx vitest run src/components/DocumentWorkspace/
```

Expected: Pass — this is a string-only change.

**Step 3: Commit**

```
fix(workspace): improve error for documents without original files

When a document was ingested before the keep_original_file fix, the
workspace now explains how to resolve the issue (re-upload or re-ingest)
instead of the generic "file not available" message (DOC-002).
```

---

### Task 6: Add "Open in Workspace" to Knowledge QA source cards (XC-001 partial)

**Files:**
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SourceCard.tsx`

**Step 1: Read the existing SourceCard to understand the action button pattern**

Read `SourceCard.tsx` fully to understand how existing actions (View, Cite, Pin, Ask) are implemented and what data is available from `RagResult`. Identify:
- Whether `result.metadata.media_id` exists
- Whether `result.metadata.media_type` is available
- Where to add the new "Open in Workspace" action

**Step 2: Add "Open in Workspace" button to the overflow menu**

Add a new menu item in the overflow `MoreHorizontal` dropdown. Only show it when the source is a document type (PDF or EPUB):

```tsx
{isDocumentType && (
  <button
    type="button"
    onClick={() => {
      const mediaId = result.metadata?.media_id
      if (mediaId != null) {
        window.open(`/document-workspace?open=${mediaId}`, "_blank")
      }
    }}
    className="flex w-full items-center gap-2 px-3 py-1.5 text-xs hover:bg-surface2 transition-colors"
  >
    <BookOpen className="h-3.5 w-3.5" />
    Open in Document Workspace
  </button>
)}
```

Where `isDocumentType` checks:

```typescript
const sourceType = (result.metadata?.media_type || "").toLowerCase()
const isDocumentType = sourceType.includes("pdf") || sourceType.includes("epub") || sourceType.includes("ebook")
```

**Step 3: Run KnowledgeQA tests**

```bash
cd apps/packages/ui && npx vitest run src/components/Option/KnowledgeQA/
```

**Step 4: Commit**

```
feat(knowledge): add "Open in Workspace" to source card overflow menu

PDF and EPUB source cards in Knowledge QA now have an "Open in Document
Workspace" option in the overflow menu (XC-001). Uses the ?open={mediaId}
URL parameter to auto-open the document in the workspace.
```

---

### Task 7: Final integration test and verification

**Step 1: Run all modified test suites together**

```bash
cd apps/packages/ui && npx vitest run \
  src/services/tldw/__tests__/quick-ingest-batch-keep-file.test.ts \
  src/components/Common/QuickIngest/__tests__/ \
  src/components/DocumentWorkspace/ \
  src/components/Option/KnowledgeQA/
```

Expected: All pass.

**Step 2: Manual verification checklist**

Start the dev server and verify each change:

1. **Ingest a PDF via QuickIngest** → Verify in browser DevTools Network tab that the request payload includes `keep_original_file: true`
2. **Check results step** → After ingest completes, verify "What's next?" section appears with "Search in Knowledge" and "Open in Workspace" buttons
3. **Click "Open in Workspace"** → Verify navigation to `/document-workspace?open={id}` and document auto-opens
4. **Click "Search in Knowledge"** → Verify navigation to `/knowledge`
5. **Open Document Workspace directly** → Verify "Recently ingested" section shows the document you just ingested
6. **Open Knowledge QA** → Search for content from the PDF → Verify source card overflow menu has "Open in Document Workspace"
7. **Test with audio/video** → Ingest a video → Verify `keep_original_file` is NOT set to true → Verify "Open in Workspace" button does NOT appear (audio/video not supported in workspace)

**Step 3: Commit final integration (if any fixups needed)**

```
test: verify FTUE Sprint 1 integration across ingest, knowledge, workspace
```

---

## Summary of Changes

| File | Change | Issue |
|------|--------|-------|
| `quick-ingest-batch.ts` | Add `keep_original_file: shouldKeepOriginalFile(mediaType)` to payload | ING-011 (P0) |
| `WizardResultsStep.tsx` | Add `onSearchKnowledge` and `onOpenWorkspace` props + CTA section | ING-004 (P1) |
| `QuickIngestWizardModal.tsx` | Wire navigation callbacks to WizardResultsStep | ING-004 (P1) |
| `quick-ingest.tsx` (store) | Add `recentlyIngestedDocIds` state | DOC-001 (P1) |
| `DocumentPickerModal.tsx` | Add "Recently ingested" section + improve missing-file error | DOC-001 + DOC-002 |
| `DocumentWorkspacePage.tsx` | Handle `?open={mediaId}` URL parameter | DOC-001 (P1) |
| `SourceCard.tsx` | Add "Open in Document Workspace" to overflow menu | XC-001 (P0) |
