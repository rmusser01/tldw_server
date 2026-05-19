# FTUE Sprint 2: Simplify First Use

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the default ingest path require zero configuration knowledge by promoting "Use defaults & process" as the primary action and hiding advanced options behind progressive disclosure.

**Architecture:** Two changes to existing components: (1) swap button priority and extend "Use defaults & process" to multi-item queues in AddContentStep, (2) collapse advanced options in WizardConfigureStep behind an expandable section. No new components, no new stores, no backend changes.

**Tech Stack:** TypeScript/React (Vitest), Ant Design Button component, Tailwind CSS

---

### Task 1: Make "Use defaults & process" primary and available for all queue sizes (ING-003)

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx:467-489`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.buttons.test.tsx` (create)

**Step 1: Write the failing test**

Create `apps/packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.buttons.test.tsx`:

```typescript
// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import React from "react"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, opts?: any) => opts?.defaultValue ?? key,
  }),
}))

// Mock IngestWizardContext with configurable queue items
const mockQueueItems = vi.fn()
const mockGoNext = vi.fn()
vi.mock("../IngestWizardContext", () => ({
  useIngestWizard: () => ({
    state: {
      queueItems: mockQueueItems(),
      currentStep: 1,
    },
    goNext: mockGoNext,
  }),
}))

import AddContentStep from "../AddContentStep"

const makeItem = (id: string, type = "pdf") => ({
  id,
  type,
  url: `https://example.com/${id}`,
  fileName: `${id}.pdf`,
  validation: { valid: true, errors: [], warnings: [] },
})

describe("AddContentStep button priority", () => {
  it("shows 'Use defaults & process' as primary button for single item", () => {
    mockQueueItems.mockReturnValue([makeItem("a")])
    const onQuickProcess = vi.fn()
    render(<AddContentStep onQuickProcess={onQuickProcess} />)
    
    const quickBtn = screen.getByText("Use defaults & process")
    expect(quickBtn).toBeTruthy()
    // The quick process button should have primary styling
    expect(quickBtn.closest("button")?.className).toContain("primary")
  })

  it("shows 'Use defaults & process' for multi-item queues too", () => {
    mockQueueItems.mockReturnValue([makeItem("a"), makeItem("b"), makeItem("c")])
    const onQuickProcess = vi.fn()
    render(<AddContentStep onQuickProcess={onQuickProcess} />)
    
    const quickBtn = screen.getByText("Use defaults & process")
    expect(quickBtn).toBeTruthy()
  })
})
```

**Step 2: Run test to verify it fails**

```bash
cd apps/packages/ui && npx vitest run src/components/Common/QuickIngest/__tests__/AddContentStep.buttons.test.tsx
```

Expected: FAIL — button doesn't appear for multi-item queues and doesn't have primary styling.

**Step 3: Implement the changes**

In `apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx`, modify the button row section (lines 467-489).

**3a.** Remove the `queueItems.length <= 1` condition from the "Use defaults & process" button. Change line 469 from:

```typescript
  {hasItems && queueItems.length <= 1 && onQuickProcess && (
```

To:

```typescript
  {hasItems && onQuickProcess && (
```

**3b.** Swap the button styling — make "Use defaults & process" the primary button and "Configure" the secondary:

Change the "Use defaults & process" button to add `type="primary"`:
```typescript
    <Button
      type="primary"
      onClick={onQuickProcess}
      disabled={!canProceed}
    >
      {qi("wizard.useDefaultsProcess", "Use defaults & process")}
    </Button>
```

Change the "Configure" button to remove `type="primary"` (making it secondary):
```typescript
    <Button
      onClick={goNext}
      disabled={!canProceed}
      aria-label={qi("wizard.configureItems", "Configure {{count}} items", {
        count: validItemCount,
      })}
    >
      {qi("wizard.configureItems", "Configure {{count}} items >", {
        count: validItemCount,
      })}
    </Button>
```

**Step 4: Run test to verify it passes**

```bash
cd apps/packages/ui && npx vitest run src/components/Common/QuickIngest/__tests__/AddContentStep.buttons.test.tsx
```

**Step 5: Run regression tests**

```bash
cd apps/packages/ui && npx vitest run src/components/Common/QuickIngest/__tests__/
```

Check if any integration tests break because they assert on button type/ordering. If so, update them to match the new order.

**Step 6: Commit**

```
fix(ingest): promote "Use defaults & process" to primary CTA for all queue sizes

The "Use defaults & process" button was secondary and hidden for multi-item
queues (>1 item). Now it's the primary (blue) button and available regardless
of queue size. "Configure" becomes secondary. This removes friction for
first-time users who just want to process their files (ING-003).
```

---

### Task 2: Add progressive disclosure to configure step (ING-001)

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/WizardConfigureStep.tsx:391-668`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardConfigureStep.disclosure.test.tsx` (create)

**Step 1: Write the failing test**

Create `apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardConfigureStep.disclosure.test.tsx`:

```typescript
// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"
import React from "react"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, opts?: any) => opts?.defaultValue ?? key,
  }),
}))

vi.mock("../IngestWizardContext", () => ({
  useIngestWizard: () => ({
    state: {
      queueItems: [
        { id: "1", type: "pdf", validation: { valid: true, errors: [], warnings: [] } },
      ],
      presetConfig: { name: "standard" },
      common: { perform_analysis: true, perform_chunking: true, overwrite_existing: false },
    },
    setCommon: vi.fn(),
    goNext: vi.fn(),
    goBack: vi.fn(),
  }),
}))

describe("WizardConfigureStep progressive disclosure", () => {
  it("hides advanced options by default", () => {
    // After implementation, advanced options like Chunking, Diarization,
    // OCR, Storage should be collapsed by default
    // This test will need to be adapted to match the actual implementation
  })

  it("shows advanced options when expanded", () => {
    // After clicking an "Advanced options" toggle, all options should be visible
  })
})
```

NOTE: The exact test implementation depends on how the disclosure is implemented. The test above is a placeholder that needs to be filled in during implementation.

**Step 2: Implement progressive disclosure**

In `apps/packages/ui/src/components/Common/QuickIngest/WizardConfigureStep.tsx`:

**2a.** Add a state variable for the advanced section toggle:

```typescript
const [showAdvanced, setShowAdvanced] = useState(false)
```

Import `useState` from React if not already imported.

**2b.** Add plain-language labels for common options. Where the labels currently say "Chunking" and "Diarization", add subtitles:

For the Chunking toggle, add a helper text:
```
"Split into searchable sections"
```

For the Diarization toggle:
```
"Identify different speakers"
```

For Analysis:
```
"Generate AI summary and key findings"
```

These should be added as small helper text below each toggle label.

**2c.** Wrap the advanced sections (type-specific audio/document/video options + storage configuration) in a collapsible container:

```tsx
{/* Advanced options toggle */}
<button
  type="button"
  onClick={() => setShowAdvanced(!showAdvanced)}
  className="mt-3 flex w-full items-center gap-1.5 text-xs font-medium text-text-muted hover:text-text transition-colors"
  aria-expanded={showAdvanced}
>
  <ChevronRight className={`h-3.5 w-3.5 transition-transform ${showAdvanced ? "rotate-90" : ""}`} />
  {showAdvanced
    ? qi("wizard.hideAdvanced", "Hide advanced options")
    : qi("wizard.showAdvanced", "Advanced options")}
</button>

{showAdvanced && (
  <div className="mt-3 space-y-4">
    {/* Audio Options section */}
    {/* ... existing audio options ... */}
    
    {/* Document Options section */}
    {/* ... existing document options ... */}
    
    {/* Video Options section */}
    {/* ... existing video options ... */}
    
    {/* Storage Configuration section */}
    {/* ... existing storage section ... */}
  </div>
)}
```

Import `ChevronRight` from lucide-react.

The key structural change: Move the type-specific options (Audio lines 455-532, Document lines 534-553, Video lines 555-574) and Storage configuration (lines 576-666) inside the `{showAdvanced && (...)}` block. Keep the Preset selector and the three common toggles (Analysis, Chunking, Overwrite) always visible.

**Step 3: Update the test to match implementation**

```typescript
describe("WizardConfigureStep progressive disclosure", () => {
  it("hides advanced options by default", () => {
    render(<WizardConfigureStep isStepVisible={true} />)
    // Advanced options section should not be visible
    expect(screen.queryByText("Audio options")).toBeNull()
    // The toggle should show "Advanced options"
    expect(screen.getByText("Advanced options")).toBeTruthy()
  })

  it("shows advanced options when expanded", () => {
    render(<WizardConfigureStep isStepVisible={true} />)
    fireEvent.click(screen.getByText("Advanced options"))
    // Now audio options should be visible (if audio items queued)
    // and storage section should be visible
    expect(screen.getByText("Hide advanced options")).toBeTruthy()
  })
})
```

**Step 4: Run tests**

```bash
cd apps/packages/ui && npx vitest run src/components/Common/QuickIngest/__tests__/WizardConfigureStep.disclosure.test.tsx
cd apps/packages/ui && npx vitest run src/components/Common/QuickIngest/__tests__/
```

**Step 5: Commit**

```
feat(ingest): add progressive disclosure to configure step

Advanced ingest options (audio settings, document OCR, video captions,
storage configuration) are now collapsed behind an "Advanced options"
toggle. Common options (Analysis, Chunking, Overwrite) remain visible
with plain-language descriptions added (ING-001).

First-time users see a simpler configure step by default. Power users
can expand advanced options with one click.
```

---

### Task 3: Add plain-language descriptions to common options

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/WizardConfigureStep.tsx`

If not already done in Task 2, add descriptive subtitles below each common option toggle:

For **Analysis**:
```
"Generate AI summary and key findings"
```

For **Chunking**:
```
"Split content into searchable sections for Knowledge QA"
```

For **Overwrite existing**:
```
"Replace content if previously ingested"
```

These should appear as small muted text (`text-xs text-text-muted`) below each toggle label, using the `qi()` i18n helper.

**Commit:**
```
feat(ingest): add plain-language descriptions to common ingest options

Each toggle in the configure step now has a brief description explaining
what it does in non-technical language (ING-001).
```

---

### Task 4: Run full regression and verify

**Step 1: Run all QuickIngest tests**

```bash
cd apps/packages/ui && npx vitest run src/components/Common/QuickIngest/__tests__/
```

**Step 2: Manual verification**

1. Open QuickIngest modal, add 1 PDF → verify "Use defaults & process" is the blue primary button
2. Add 2+ files → verify "Use defaults & process" still appears (was hidden before)
3. Click "Configure" → verify advanced options are collapsed by default
4. Click "Advanced options" → verify all options expand
5. Verify plain-language descriptions visible under each common toggle

**Step 3: Commit any fixups**

```
test: verify Sprint 2 ingest simplification
```

---

## Summary of Changes

| File | Change | Issue |
|------|--------|-------|
| `AddContentStep.tsx` | Swap button priority, remove queue size gate | ING-003 (P1) |
| `WizardConfigureStep.tsx` | Collapse advanced options, add descriptions | ING-001 (P1) |
