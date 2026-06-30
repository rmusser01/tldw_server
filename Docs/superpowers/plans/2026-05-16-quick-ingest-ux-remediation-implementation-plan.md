# Quick Ingest UX Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the active shared Quick Ingest wizard clearer, more recoverable, and better verified across the WebUI and browser extension.

**Architecture:** Keep the remediation in the shared quick-ingest wizard and shared services under `apps/packages/ui/src` so WebUI and extension behavior stay aligned. Start with evidence alignment, then improve copy and launch clarity, result handoff, recovery/progress states, input validation, and current-flow verification. Avoid broad WebUI redesign and avoid backend API changes unless an implementation task proves the quick-ingest UX cannot be completed without one.

**Tech Stack:** React, TypeScript, Ant Design, Zustand stores, shared `apps/packages/ui` package, WXT/browser-extension runtime messaging, Vitest, Testing Library, Playwright e2e, Backlog.md.

---

## Inputs

- Design spec: `Docs/superpowers/specs/2026-05-16-quick-ingest-ux-remediation-stages-design.md`
- Backlog task for this plan: `TASK-393`
- Completed design task: `TASK-392`

## Scope Rules

- Work only on quick-ingest launch, wizard, processing/cancel/minimize, results/recovery, shared runtime support, and tests.
- Do not redesign the full Media page, Knowledge page, Chat page, settings, or backend architecture.
- Do not preserve legacy quick-ingest test ids purely to satisfy stale tests.
- If a task discovers that the legacy `QuickIngestModal.tsx` path is still actively reachable, stop and record that in the task notes before changing behavior that could fork WebUI and extension UX.
- Keep commits focused. Each task below should produce a reviewable commit unless the implementation owner intentionally splits it further.

## File Map

### Active quick-ingest wizard

- Modify: `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`
  - Owns wizard orchestration, session lifecycle, `AddContentStep` wiring, processing start, close/cancel behavior, and results callbacks.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx`
  - Owns file/URL add state, URL validation, local queue validation, first-step copy, quick-process action availability.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/WizardConfigureStep.tsx`
  - Owns presets, common ingest options, advanced options, storage/review explanation.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx`
  - Owns run summary, estimate copy, storage summary, and final start action.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx`
  - Owns in-modal progress copy, per-item progress rows, Cancel All, and Minimize to Background.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/FloatingProgressWidget.tsx`
  - Owns minimized/background status display.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx`
  - Owns completed/skipped/error result rows, retry/remove actions, and next-step CTAs.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/constants.ts`
  - Owns quick-ingest file accept string, file-size constants, and duplicate message constants.
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/types.ts`
  - Owns queue/result types if new result action or validation metadata is needed.

### Candidate new helpers

- Create if useful: `apps/packages/ui/src/components/Common/QuickIngest/result-actions.ts`
  - Pure functions for deciding result actions from a `WizardResultItem`.
- Create if useful: `apps/packages/ui/src/components/Common/QuickIngest/queue-validation.ts`
  - Pure URL/file queue validation helpers if `AddContentStep.tsx` grows too large.

Only create these helpers if the implementation would otherwise duplicate logic or make the component harder to test.

### Shared services and stores

- Modify if needed: `apps/packages/ui/src/entries/shared/ingest-payloads.ts`
  - Existing URL normalization and context-menu ingest payload helpers.
- Modify if needed: `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts`
  - Direct and extension-runtime quick-ingest submission, fallback, and cancellation.
- Modify if needed: `apps/packages/ui/src/services/tldw/quick-ingest-session-reattach.ts`
  - Reattach behavior for direct jobs.
- Read/use: `apps/packages/ui/src/store/connection.tsx`
  - Central connection/health status source for offline wizard behavior.
- Modify if needed: `apps/packages/ui/src/store/quick-ingest-session.ts`
  - Persisted quick-ingest session record.
- Modify if needed: `apps/packages/ui/src/store/quick-ingest.tsx`
  - Header badge and last-run summary state.

### Launch surfaces

- Modify if needed: `apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx`
- Read/verify: `apps/tldw-frontend/components/layout/WebLayout.tsx`
- Read/verify: `apps/tldw-frontend/extension/routes/sidepanel-chat.tsx`
- Read/verify: `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- Read/verify: `apps/packages/ui/src/entries/background.ts`

### Tests

- Modify/add: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts`
- Modify/add: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx`
- Modify/add: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx`
- Modify/add: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`
- Modify/add: `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`
- Modify/add: `apps/packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts`
- Modify/add: `apps/tldw-frontend/e2e/utils/journey-helpers.ts`
- Modify/add: `apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts`
- Modify or retire after reachability decision: `apps/extension/tests/e2e/quick-ingest-ux-audit.spec.ts`
- Modify or retire after reachability decision: `apps/extension/tests/e2e/quick-ingest-cancel.spec.ts`

### Planning artifact for Stage 1

- Create: `Docs/superpowers/plans/2026-05-16-quick-ingest-active-path-map.md`

---

## Task 1: Active Path Map And Legacy/Test Classification

**Files:**
- Create: `Docs/superpowers/plans/2026-05-16-quick-ingest-active-path-map.md`
- Read: `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`
- Read: `apps/packages/ui/src/components/Common/QuickIngestModal.tsx`
- Read: `apps/packages/ui/src/components/Common/QuickIngest/ResultsPanel.tsx`
- Read: `apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx`
- Read: `apps/tldw-frontend/components/layout/WebLayout.tsx`
- Read: `apps/tldw-frontend/extension/routes/sidepanel-chat.tsx`
- Read: `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- Read: `apps/packages/ui/src/entries/background.ts`
- Read: `apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts`
- Read: `apps/tldw-frontend/e2e/utils/journey-helpers.ts`
- Read: `apps/extension/tests/e2e/quick-ingest-ux-audit.spec.ts`
- Read: `apps/extension/tests/e2e/quick-ingest-cancel.spec.ts`

- [ ] **Step 1: Generate launcher and import evidence**

Run:

```bash
rg -n "QuickIngestWizardModal|QuickIngestModal|tldw:open-quick-ingest|open-quick-ingest|quick-ingest-run|quick-ingest-cancel|quick-ingest-open-media-primary" apps/tldw-frontend apps/packages/ui/src apps/extension/tests/e2e
```

Expected: output shows current wizard imports and any stale legacy/test references.

- [ ] **Step 2: Write the active-path map artifact**

Create `Docs/superpowers/plans/2026-05-16-quick-ingest-active-path-map.md` with these sections:

```markdown
# Quick Ingest Active Path Map

Date: 2026-05-16
Backlog: TASK-393

## Active Launch Paths

| Surface | Trigger | Modal/Runtime | Evidence |
|---|---|---|---|

## Legacy Reachability Decision

| File/Test | Current status | Decision | Rationale |
|---|---|---|---|

## Test Classification

| Test file/helper | Current wizard | Legacy reachable | Stale selector | Missing coverage |
|---|---:|---:|---:|---|

## Follow-Up Notes
```

Use exact file paths and short evidence snippets.

- [ ] **Step 3: Verify the map names every active surface**

Run:

```bash
rg -n "Active Launch Paths|Legacy Reachability Decision|Test Classification" Docs/superpowers/plans/2026-05-16-quick-ingest-active-path-map.md
```

Expected: all three headings are present.

- [ ] **Step 4: Commit the evidence artifact**

Run:

```bash
git add Docs/superpowers/plans/2026-05-16-quick-ingest-active-path-map.md
git commit -m "docs: map quick ingest active path"
```

Expected: commit contains only the active-path map.

---

## Task 2: First-Time Clarity And Entry Consistency

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx`
- Modify if needed: `apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts`

- [ ] **Step 1: Write failing shared wizard test for first-open purpose copy**

Add a test to `QuickIngestWizardModal.integration.test.tsx` that renders the Add step and expects:

```tsx
expect(screen.getByText(/Add URLs or files/i)).toBeInTheDocument()
expect(screen.getByText(/Media/i)).toBeInTheDocument()
expect(screen.getByText(/Knowledge/i)).toBeInTheDocument()
```

Use the existing test harness in that file rather than creating a new harness.

- [ ] **Step 2: Run the failing test**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx --runInBand
```

Expected: the new assertion fails because the purpose/destination copy is not present yet.

- [ ] **Step 3: Add concise first-time copy to Add step**

Modify `AddContentStep.tsx` near the file/URL input area. Keep the copy short and specific:

```tsx
<Typography.Text className="block text-xs text-text-muted">
  {qi(
    "wizard.addPurpose",
    "Add URLs or files. Stored items appear in Media; analyzed and chunked items become searchable in Knowledge."
  )}
</Typography.Text>
```

Keep it outside Advanced options and below the main input heading/drop zone area.

- [ ] **Step 4: Align launcher terminology without removing power-user speed**

In `QuickIngestButton.tsx`, align visible text, title, and aria label around one concept. Prefer:

- visible text: `Quick Ingest`
- title: `Import URLs, documents, and media to your knowledge base`
- aria label: keep the count-aware label if present, but use "Quick Ingest" consistently.

Do not remove the existing queued badge or `Process queued items` CTA.

- [ ] **Step 5: Fix the double-tilde review estimate if still present**

In `ReviewStep.tsx`, make only one layer own approximate formatting. If `estimatedTimeLabel` already includes `~`, remove the leading `~` from the translation default:

```tsx
"{{count}} items | {{preset}} preset | {{time}} estimated"
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx apps/packages/ui/src/components/Layouts/__tests__ --runInBand
```

Expected: quick-ingest integration tests pass. If the layout test glob does not exist, record the missing path and run the exact Quick Ingest test file only.

- [ ] **Step 7: Browser-check WebUI Add step**

Run the WebUI using the repo's normal command, or reuse an existing local server:

```bash
bun run dev -- -p 18001
```

Open `/media`, launch Quick Ingest, and verify:

- launcher uses consistent naming
- first Add step explains Media and Knowledge destinations
- invalid URL and valid URL behavior still match the old flow
- "Use defaults & process" remains visible after a valid item

- [ ] **Step 8: Commit the clarity change**

Run:

```bash
git add apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx
git commit -m "fix: clarify quick ingest entry and destination"
```

Expected: commit includes only the clarity/test files touched in this task.

---

## Task 3: Results Handoff And Recovery Actions

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx`
- Modify if needed: `apps/packages/ui/src/components/Common/QuickIngest/types.ts`
- Create if useful: `apps/packages/ui/src/components/Common/QuickIngest/result-actions.ts`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts`
- Test helper: `apps/tldw-frontend/e2e/utils/journey-helpers.ts`

- [ ] **Step 1: Write failing tests for result actions**

Extend `WizardResultsStep.navigation.test.tsx` with tests that prove:

```tsx
it("renders Open in Media for persisted results when onOpenMedia is provided", () => {
  const onOpenMedia = vi.fn()
  setSinglePdfResult({ mediaId: 42, persisted: true })
  render(<WizardResultsStep onClose={vi.fn()} onOpenMedia={onOpenMedia} />)
  fireEvent.click(screen.getByRole("button", { name: /open/i }))
  expect(onOpenMedia).toHaveBeenCalledWith(expect.objectContaining({ mediaId: 42 }))
})

it("does not render Remove for errors when no remove callback exists", () => {
  wizardHarness.results = [{ id: "err-1", status: "error", error: "Network failed", url: "https://example.com" } as any]
  render(<WizardResultsStep onClose={vi.fn()} />)
  expect(screen.queryByRole("button", { name: /remove/i })).toBeNull()
})
```

Adjust to the existing harness types and helpers.

- [ ] **Step 2: Run the failing results tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx --runInBand
```

Expected: new result-action tests fail before implementation.

- [ ] **Step 3: Add or refactor result action mapping**

If `WizardResultsStep.tsx` becomes difficult to reason about, create `result-actions.ts` with pure helpers:

```ts
export type QuickIngestResultAction = "open-media" | "search-knowledge" | "open-workspace" | "chat"

export const canOpenMedia = (item: WizardResultItem): boolean =>
  item.persisted === true && item.mediaId != null
```

Keep helpers pure and unit-testable. Do not add routing side effects to the helper.

- [ ] **Step 4: Wire primary Media handoff from the modal**

In `QuickIngestWizardModal.tsx`, pass an `onOpenMedia` callback into `WizardResultsStep`. The callback should:

- close or hide the modal
- navigate to the Media route or select/open the persisted media item if a stable route/selection contract already exists
- avoid inventing a fake URL when `mediaId` is missing

If no stable direct-media route exists, use the existing Media page destination and preserve the item label in visible result text. Record the limitation in task notes.

- [ ] **Step 5: Remove or implement error Remove**

In `WizardResultsStep.tsx`, remove the unconditional no-op remove path. Acceptable outcomes:

- no Remove button is rendered until a real `onRemoveItems` callback exists, or
- a real callback removes the error from the result list and tests prove it.

Do not leave a visible action wired to a no-op.

- [ ] **Step 6: Clarify duplicate and skipped copy**

Update skipped/duplicate copy so it distinguishes:

- local queue duplicate: "Already queued"
- backend/library duplicate: "Already in library"
- overwrite recovery: use existing `Overwrite existing` setting or Deep preset

Keep this copy in existing i18n/default-string patterns.

- [ ] **Step 7: Run focused results tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --runInBand
```

Expected: result-action tests pass and session lifecycle tests still pass.

- [ ] **Step 8: Run focused WebUI e2e for result handoff**

Run:

```bash
npx playwright test apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts --grep "quick ingest ingests deterministic local URL"
```

Expected: the deterministic quick-ingest URL completion test passes and observes the updated handoff behavior.

- [ ] **Step 9: Commit the results/recovery change**

Run:

```bash
git add apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx apps/packages/ui/src/components/Common/QuickIngest/types.ts apps/packages/ui/src/components/Common/QuickIngest/result-actions.ts apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts apps/tldw-frontend/e2e/utils/journey-helpers.ts
git commit -m "fix: improve quick ingest result handoff"
```

If a listed file was not touched or does not exist, omit it from `git add`.

---

## Task 4: Offline, Cancel, Progress, And Background Status Correctness

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/FloatingProgressWidget.tsx`
- Modify if needed: `apps/packages/ui/src/store/connection.tsx`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts`

- [ ] **Step 1: Write failing test for disconnected processing guard**

In `QuickIngestWizardModal.integration.test.tsx`, mock the connection store or the direct hook chosen during implementation so the Add step receives disconnected state. Assert:

```tsx
expect(screen.getByRole("button", { name: /use defaults & process/i })).toBeDisabled()
expect(screen.getByText(/server/i)).toBeInTheDocument()
expect(screen.getByText(/retry|diagnostics|settings/i)).toBeInTheDocument()
```

Use the exact wording chosen in implementation, but keep the behavior: users can see why processing is blocked.

- [ ] **Step 2: Write failing tests for minimized terminal states**

Add or extend a FloatingProgressWidget-focused test. If no test file exists, add widget assertions to `QuickIngestWizardModal.session.test.tsx` or create:

`apps/packages/ui/src/components/Common/QuickIngest/__tests__/FloatingProgressWidget.test.tsx`

Cover:

- status `complete` shows Done
- status `error` shows Failed
- status `cancelled` shows Cancelled
- status `interrupted` or equivalent lifecycle shows Interrupted if represented in state

- [ ] **Step 3: Run failing tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --runInBand
```

Expected: new disconnected/widget assertions fail before implementation.

- [ ] **Step 4: Pass connection state into AddContentStep**

In `QuickIngestWizardModal.tsx`, read the central connection state from `useConnectionStore` or the existing quick-ingest connection abstraction if Task 1 identified one. Pass a boolean into:

```tsx
<AddContentStep
  isOnlineForIngest={isOnlineForIngest}
  onQuickProcess={handleQuickProcess}
/>
```

Do not block users from editing queued input while offline unless the existing `FileDropZone` behavior requires it. Block only processing actions or clearly route them to setup/retry.

- [ ] **Step 5: Add visible disconnected recovery in Add step**

In `AddContentStep.tsx`, when `isOnlineForIngest` is false:

- show a compact warning near the action buttons
- disable `Use defaults & process`
- disable or explain `Configure` only if configuring would lead to a dead end
- provide "Retry connection" or "Health & diagnostics" only if an existing shared handler is available

Avoid a new global notification system.

- [ ] **Step 6: Make progress copy neutral or item-aware**

In `ProcessingStep.tsx`, replace type-specific wrong copy such as "Transcribing and indexing content" with neutral copy for mixed batches:

```tsx
"Processing and indexing content"
```

If item-type-aware labels are easy from existing per-item metadata, use them in row-level stages, not in the global banner.

- [ ] **Step 7: Split minimized widget terminal states**

In `FloatingProgressWidget.tsx`, replace the `allDone` single label with terminal-state rendering:

- `complete`: Done
- `error`: Failed
- `cancelled`: Cancelled
- interrupted/reattach-required lifecycle if available: Interrupted

Use non-success colors for failed/cancelled.

- [ ] **Step 8: Run focused tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --runInBand
```

Expected: disconnected, cancel, stale completion, and widget terminal-state tests pass.

- [ ] **Step 9: Run focused e2e for dismiss/resume**

Run:

```bash
npx playwright test apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts --grep "quick ingest can be dismissed during processing"
```

Expected: dismiss/minimize/reopen flow still passes.

- [ ] **Step 10: Commit status/recovery changes**

Run:

```bash
git add apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx apps/packages/ui/src/components/Common/QuickIngest/FloatingProgressWidget.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__ apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts
git commit -m "fix: clarify quick ingest recovery states"
```

Expected: commit contains only status/recovery changes and related tests.

---

## Task 5: URL And File Input Hardening

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/constants.ts`
- Modify if needed: `apps/packages/ui/src/components/Common/QuickIngest/types.ts`
- Modify if needed: `apps/packages/ui/src/entries/shared/ingest-payloads.ts`
- Create if useful: `apps/packages/ui/src/components/Common/QuickIngest/queue-validation.ts`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx`
- Test: `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`

**Large-file decision for this implementation pass:** use the **Truthful limit fix**. Lower the displayed and enforced Quick Ingest client-buffered upload limit to a conservative value and document that preserving 500 MB requires a separate transport fix. Do not attempt the large-file transport redesign inside this remediation pass unless the human owner explicitly changes this decision.

- [ ] **Step 1: Write failing tests for normalized URL dedupe**

In `AddContentStep.url-detection.test.ts` or a new `queue-validation.test.ts`, cover:

```ts
expect(normalizeUrlForDedupe("https://EXAMPLE.com/a/?utm_source=x#frag"))
  .toBe("https://example.com/a")
expect(normalizeUrlForDedupe("https://youtu.be/abc123?t=30"))
  .toContain("watch?v=abc123")
```

If `normalizeUrlForDedupe` already has tests elsewhere, write the failing wizard queue test instead: adding two normalized-equivalent URLs should show a duplicate warning before processing.

- [ ] **Step 2: Write failing tests for mixed URL paste counts**

In `QuickIngestWizardModal.integration.test.tsx`, paste one valid and one invalid URL. Assert the UI communicates both counts and does not make users infer validity item-by-item only.

Example assertion shape:

```tsx
expect(screen.getByText(/1 valid/i)).toBeInTheDocument()
expect(screen.getByText(/1 invalid/i)).toBeInTheDocument()
```

- [ ] **Step 3: Run failing input tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx --runInBand
```

Expected: new dedupe/count assertions fail before implementation.

- [ ] **Step 4: Reuse URL normalization in wizard queue validation**

In `AddContentStep.tsx` or `queue-validation.ts`, import or call the existing `normalizeUrlForDedupe` from `apps/packages/ui/src/entries/shared/ingest-payloads.ts`.

Use normalized keys only for validation/dedupe. Preserve the original URL for display and submission unless the existing ingest path already normalizes submission URLs.

- [ ] **Step 5: Add mixed paste summary**

Add a compact queue summary near the queued-items header when both valid and invalid items exist:

```tsx
{hasItems && (
  <Typography.Text className="text-xs text-text-muted">
    {validItemCount} valid / {queueItems.length - validItemCount} invalid
  </Typography.Text>
)}
```

Use i18n default-string patterns and avoid adding a noisy alert for every normal paste.

- [ ] **Step 6: Reconcile file support copy and accept string**

Compare `detectTypeFromExtension`, `QUICK_INGEST_ACCEPT_STRING`, and backend-supported ingest types. Choose one of:

- add missing supported extensions to `QUICK_INGEST_ACCEPT_STRING` and copy, or
- stop detecting/advertising types not actually accepted by the picker and backend.

Do not advertise images, HTML, JSON, CSV, XML, or other types unless the quick-ingest upload path can process them end-to-end.

- [ ] **Step 7: Apply truthful large-file limit**

In `constants.ts`, introduce an explicitly named limit for the current buffered-client implementation:

```ts
export const QUICK_INGEST_MAX_FILE_SIZE = 50 * 1024 * 1024 // 50MB buffered client upload guard
export const QUICK_INGEST_TRANSPORT_REDESIGN_FILE_SIZE = 500 * 1024 * 1024 // future direct-upload target
```

Use only one displayed current limit. Update copy from "500 MB" to the chosen current limit. If the owner rejects `50 MB`, choose another explicit value before coding and keep tests aligned.

- [ ] **Step 8: Add a preflight warning for larger files**

For files over the current limit, show an error that explains the current quick-ingest limit and points users to the intended large-file path if one exists. Do not silently imply the file can be processed.

- [ ] **Step 9: Run input tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts --runInBand
```

Expected: queue validation and quick-ingest batch tests pass.

- [ ] **Step 10: Run constrained viewport e2e**

Run:

```bash
npx playwright test apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts --grep "quick ingest configure options stay reachable"
```

Expected: constrained viewport test passes with updated copy and file limit.

- [ ] **Step 11: Commit input hardening changes**

Run:

```bash
git add apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx apps/packages/ui/src/components/Common/QuickIngest/constants.ts apps/packages/ui/src/components/Common/QuickIngest/types.ts apps/packages/ui/src/components/Common/QuickIngest/queue-validation.ts apps/packages/ui/src/components/Common/QuickIngest/__tests__ apps/packages/ui/src/entries/shared/ingest-payloads.ts apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts
git commit -m "fix: harden quick ingest input validation"
```

Omit untouched files from `git add`.

---

## Task 6: Current-Flow Verification And Stale Selector Cleanup

**Files:**
- Modify: `apps/tldw-frontend/e2e/utils/journey-helpers.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts`
- Modify or retire after Task 1 decision: `apps/extension/tests/e2e/quick-ingest-ux-audit.spec.ts`
- Modify or retire after Task 1 decision: `apps/extension/tests/e2e/quick-ingest-cancel.spec.ts`
- Modify if needed: `apps/tldw-frontend/__tests__/e2e-harness-readiness.guard.test.ts`
- Update Backlog task notes for any deferred verification.

- [ ] **Step 1: Replace stale helper selectors with current wizard selectors**

In `journey-helpers.ts`, prefer current labels and roles:

- dialog: `/quick ingest/i`
- Add step URL input: `/Paste URLs input/i` or current aria label
- Add URLs button: `/add urls/i`
- Configure button: `/configure \d+ items/i`
- Start button: `/start processing/i`
- Results assertions: completed/skipped/errors section text from `WizardResultsStep`

Do not depend on `quick-ingest-run`, `quick-ingest-cancel`, or tabbed result ids unless Task 1 proves the legacy modal remains active.

- [ ] **Step 2: Update WebUI e2e expectations for current behavior**

In `media-ingest.spec.ts`, ensure coverage exists for:

- visible launchers opening the wizard
- deterministic URL completion and result handoff
- skipped duplicate restore
- fallback after recognized 429
- constrained viewport
- dismiss/minimize/reopen during processing
- refresh restore for queued/processing/completed states
- file refresh requiring reattach

Add small assertions for any behavior changed in Tasks 2 through 5.

- [ ] **Step 3: Classify extension e2e specs**

Using the Task 1 map, update `apps/extension/tests/e2e/quick-ingest-ux-audit.spec.ts` and `apps/extension/tests/e2e/quick-ingest-cancel.spec.ts`:

- if they are intended to cover the current wizard, migrate selectors and expected copy to the wizard
- if they cover legacy modal only and legacy is unreachable, retire or skip with a clear comment and replace coverage in `media-ingest.spec.ts`
- if extension packaging needs separate coverage, keep extension-specific assertions focused on sidepanel/runtime constraints

- [ ] **Step 4: Run focused unit coverage**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/QuickIngest/__tests__ apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts apps/packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts --runInBand
```

Expected: shared quick-ingest unit and integration tests pass.

- [ ] **Step 5: Run focused WebUI e2e coverage**

Run:

```bash
npx playwright test apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts --grep "Quick Ingest"
```

Expected: WebUI quick-ingest e2e coverage passes or only skips known server-unavailable live-ingest cases with explicit skip messages.

- [ ] **Step 6: Run focused extension e2e coverage if the extension harness is available**

Run:

```bash
npx playwright test apps/extension/tests/e2e/quick-ingest-ux-audit.spec.ts apps/extension/tests/e2e/quick-ingest-cancel.spec.ts
```

Expected: extension quick-ingest tests pass. If the local extension harness is unavailable, record the exact failure and keep this as a known verification gap in Backlog.

- [ ] **Step 7: Run final static checks for touched frontend paths**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

If the repo has a focused TypeScript check for touched frontend files, run it here and record the command/output in Backlog. Do not run a broad slow suite unless required by the PR owner.

- [ ] **Step 8: Commit verification updates**

Run:

```bash
git add apps/tldw-frontend/e2e/utils/journey-helpers.ts apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts apps/extension/tests/e2e/quick-ingest-ux-audit.spec.ts apps/extension/tests/e2e/quick-ingest-cancel.spec.ts apps/tldw-frontend/__tests__/e2e-harness-readiness.guard.test.ts
git commit -m "test: update quick ingest wizard coverage"
```

Omit untouched files from `git add`.

---

## Task 7: Final Review, Backlog, And PR-Ready Summary

**Files:**
- Modify: relevant Backlog task files for implementation tasks.
- Read: all touched files from Tasks 1 through 6.

- [ ] **Step 1: Review all commits and touched files**

Run:

```bash
git log --oneline --decorate --max-count=12
git diff --stat dev...HEAD
```

Expected: only quick-ingest/shared test/planning files are touched.

- [ ] **Step 2: Run final verification command set**

Run the final focused commands from previous tasks:

```bash
bunx vitest run apps/packages/ui/src/components/Common/QuickIngest/__tests__ apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts apps/packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts --runInBand
npx playwright test apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts --grep "Quick Ingest"
git diff --check
```

Expected: all pass or known skips are documented with exact reason.

- [ ] **Step 3: Run Bandit only if backend Python code was touched**

This plan should not require backend Python changes. If implementation expands
into backend Python anyway, first record why in Backlog, then run Bandit on the
exact touched backend paths. Example for a media endpoint change:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/media -f json -o /tmp/bandit_quick_ingest_ux.json
```

Expected: no new findings in touched backend code.

If no Python/backend files were touched, document: "Bandit not applicable; frontend/docs/tests only."

- [ ] **Step 4: Update Backlog implementation task notes**

Record:

- files changed
- tests run
- skipped checks and why
- remaining verification gaps, if any
- whether the large-file strategy used Truthful limit fix or was changed by the human owner

- [ ] **Step 5: Prepare PR summary with human-owned Change summary section**

Draft PR notes with:

```markdown
## Summary
- Clarified Quick Ingest entry and first-open destination copy.
- Improved terminal result handoff and recovery actions.
- Corrected offline, cancel, progress, and minimized status semantics.
- Hardened URL/file validation and refreshed current wizard coverage.

## Tests
- [commands and results]

## Change summary
Human requester to write: what changed and why these implementation choices were made.
```

Do not fabricate the human-owned Change summary.

- [ ] **Step 6: Commit Backlog closeout if needed**

Run:

Run `git status --short backlog` and add only the Backlog task files updated by
the implementation pass. For example, if the implementation pass uses `TASK-393`
only:

```bash
git add "backlog/tasks/task-393 - Plan-Quick-Ingest-UX-remediation-implementation.md"
git commit -m "docs: record quick ingest remediation verification"
```

Expected: commit contains only Backlog/task closeout files.

---

## Execution Handoff

Recommended execution mode: **Subagent-Driven** if the user explicitly approves subagent execution. The tasks have natural boundaries and can be implemented as separate reviewable slices.

Fallback execution mode: **Inline Execution** using `superpowers:executing-plans`, one task at a time with checkpoints after each commit.

Before implementation starts, confirm whether the large-file strategy in Task 5 should remain the conservative Truthful limit fix or be upgraded into a larger transport redesign.
