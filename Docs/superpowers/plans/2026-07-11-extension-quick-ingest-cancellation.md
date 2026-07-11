# Extension Quick Ingest Cancellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Quick Ingest cancellation terminal across extension runtime, direct submission, setup, and persisted-job reattachment, then prove PDF, URL, duplicate, and YouTube ingestion through one installed extension context.

**Architecture:** Keep the existing wizard state machine and transport contracts. Add one modal-owned synchronous Cancel All handler backed by a run-level cancellation ref, route both cancellation entry points through it, and make every asynchronous continuation consult that intent before writing state or starting work. Preserve the existing session-ID fence as defense against stale runtime messages.

**Tech Stack:** React 18, TypeScript, Vitest, Testing Library, WXT MV3 extension, Playwright, FastAPI, SQLite.

**Design:** `Docs/superpowers/specs/2026-07-11-extension-e2e-launch-cancel-race-design.md`

**Backlog:** `TASK-12947`

---

## File Map

- Modify `apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx`: accept an optional modal-owned Cancel All callback while retaining the context fallback for standalone callers.
- Modify `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`: own cancellation intent and guard setup, runtime messages, direct submission, and errors while preserving the existing persisted-reattachment cleanup fence.
- Modify `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`: add behavioral regressions for each asynchronous race.
- Verify `apps/extension/tests/e2e/quick-ingest-cancel.spec.ts`: existing packaged-extension stale-completion regression must pass unchanged.
- Update `backlog/tasks/task-12947 - Fix-browser-extension-E2E-launch-and-validate-Quick-Ingest.md`: record red/green evidence, host UAT evidence, verification, and PR.

## Stage 1: Terminal Runtime Cancellation
**Goal**: Prove Cancel All fences the run synchronously and rejects late completion and progress.
**Success Criteria**: The new tests fail as success/progress before production changes, then pass with cancelled results after the minimal callback/fence implementation.
**Tests**: Focused Vitest tests in `QuickIngestWizardModal.session.test.tsx`.
**Status**: Complete

### Task 1: Add the synchronous cancellation fence

- [x] **Step 1: Change the ProcessingStep test double to exercise the callback contract**

Update the existing mock so the button calls the supplied callback and only falls back to context cancellation when no callback is provided:

```tsx
ProcessingStep: ({ onCancelAll }: { onCancelAll?: () => void }) => {
  const { state, cancelProcessing } = actual.useIngestWizard()
  return (
    <div data-testid="wizard-processing">
      {state.processingState.status}:{state.processingState.perItemProgress.length}
      <button onClick={onCancelAll || cancelProcessing}>Cancel Processing</button>
    </div>
  )
}
```

- [x] **Step 2: Write failing late-completion and late-progress tests**

Start an extension session, click Cancel Processing, immediately emit `completed`, then emit `progress`. Assert the wizard remains `cancelled`, unresolved items have `outcome: "cancelled"`, and no successful result is merged.

- [x] **Step 3: Run the focused tests on the host and verify RED**

Run from `apps/packages/ui` outside the sandbox:

```bash
bunx vitest run src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism -t "cancellation"
```

Expected: FAIL because the current passive effect permits an immediate completion and explicitly accepts progress for cancelled sessions.

- [x] **Step 4: Implement the minimal callback and synchronous fence**

Add optional props to the real component:

```tsx
type ProcessingStepProps = { onCancelAll?: () => void }

export const ProcessingStep: React.FC<ProcessingStepProps> = ({ onCancelAll }) => {
  // existing hooks
  const handleCancelAll = useCallback(() => {
    if (onCancelAll) onCancelAll()
    else cancelProcessing()
  }, [cancelProcessing, onCancelAll])
}
```

In `WizardModalContent`, add `cancelRequestedRef`, create one idempotent `handleCancelAll` that synchronously records intent, fences an available active/persisted session ID, clears the reattach timer, sends best-effort cancellation when an ID exists, and calls `finalizeFailure("Cancelled by user.", "cancelled")`. Pass it to `<ProcessingStep onCancelAll={handleCancelAll} />` and use it from close confirmation. Remove the passive cancellation effect. Do not clear the intent inside `startRun`; `QuickIngestWizardModal` already keys the provider by session ID, so Ingest More remounts fresh refs.

Change the runtime guard to return for every message whose session ID is fenced, including progress.

- [x] **Step 5: Run the focused tests and verify GREEN**

Run the command from Step 3. Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx backlog/tasks/task-12947\ -\ Fix-browser-extension-E2E-launch-and-validate-Quick-Ingest.md
git commit -m "fix: make quick ingest cancellation terminal"
```

## Stage 2: Async Start and Reattachment Races
**Goal**: Prevent any awaited continuation from reviving cancellation or starting additional work.
**Success Criteria**: Deferred extension/direct acknowledgements, setup, errors, and persisted polls cannot mutate cancelled state; direct submission never begins after pre-ack cancellation.
**Tests**: Focused deferred-promise Vitest tests.
**Status**: Complete

### Task 2: Guard setup and start acknowledgement

- [x] **Step 1: Add a local deferred helper in the session test**

```ts
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}
```

Expose the existing API-client initialize mock through the hoisted `mocks`
object so setup can be deferred deterministically:

```ts
initialize: vi.fn(),
// ...
tldwClient: { initialize: (...args: unknown[]) => mocks.initialize(...args) }
```

- [x] **Step 2: Write failing pre-ack tests**

Add separate tests for:

1. Deferred extension acknowledgement: cancel first, resolve `qi-runtime-*` second, assert `cancelQuickIngestSession` receives the returned ID.
2. Deferred direct acknowledgement: cancel first, resolve `qi-direct-*` second, assert `submitQuickIngestBatch` is never called.
3. Deferred setup/error continuation: cancel while setup is awaiting, then resume or reject it, assert no session starts and cancelled results remain terminal.

- [x] **Step 3: Run the three tests on the host and verify RED**

```bash
bunx vitest run src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism -t "acknowledgement|setup"
```

Expected: FAIL because `startRun` currently continues after cancellation and its catch path finalizes as failed.

- [x] **Step 4: Add cancellation checkpoints to startRun**

Initialize cancellation intent once per keyed wizard-session mount. After each awaited setup boundary, return immediately when cancellation was requested. Immediately after a successful start acknowledgement, fence and cancel a returned extension session or return before direct submission. In `catch`, return without finalizing when cancellation already won.

Do not write active tracking for a session acknowledged after cancellation.

- [x] **Step 5: Run the focused tests and full session file**

```bash
bunx vitest run src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: all tests PASS with no unhandled rejections or React update-depth warnings.

### Task 3: Characterize persisted direct-job reattachment

- [x] **Step 1: Write deferred reattachment characterization tests**

Return a deferred promise from `reattachQuickIngestSession`, cancel the refreshed direct session, then resolve the in-flight poll as `processing`. Assert cancelled results remain visible, processing is not restored, and advancing timers does not schedule another call. Add a separate late-`completed` case to prove terminal results cannot be overwritten.

- [x] **Step 2: Run the focused tests on the host**

```bash
bunx vitest run src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism -t "late persisted reattach"
```

Observed before any poll implementation change: PASS. The existing effect-local
cleanup flag already prevents both late state writes and another timer.

- [x] **Step 3: Confirm no additional poll guard is needed**

The terminal session update reruns the persisted-reattach effect and invokes its
cleanup, setting the existing local `cancelled` flag before the deferred poll
continues. Keep the characterization tests and do not add redundant production
state.

- [x] **Step 4: Run the focused test and full session file**

Expected: all tests PASS.

- [x] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx backlog/tasks/task-12947\ -\ Fix-browser-extension-E2E-launch-and-validate-Quick-Ingest.md
git commit -m "test: cover quick ingest cancellation races"
```

## Stage 3: Full Installed-Extension User Acceptance
**Goal**: Validate real ingestion through the packaged browser extension before treating automated browser regressions as release evidence.
**Success Criteria**: One headed installed-extension context ingests PDF, reachable URL, duplicate URL, exact YouTube Short, and duplicate YouTube; jobs leave queued state; exactly three unique media rows exist; no update-depth, page, or console errors occur.
**Tests**: Host-side UAT against an isolated FastAPI process and Media DB.
**Status**: In Progress

### Task 4: Run the required UAT matrix

- [ ] **Step 1: Build production extension artifacts on the host**

```bash
bun run build:chrome:prod
```

Run from `apps/extension`. Expected: `.output/chrome-mv3` builds successfully.

- [ ] **Step 2: Start a fresh isolated backend on an unused host port**

Activate `.venv`, use a temporary config/database root, and start uvicorn outside the sandbox. Confirm `/api/v1/health` before opening the extension.

- [ ] **Step 3: Launch one headed persistent Chromium extension context**

Seed the isolated server URL and API key, enable Quick Ingest, and retain the profile, screenshots, browser console, page errors, and request/response log until all UAT items finish.

- [ ] **Step 4: Ingest multiple item types through the visible extension UI**

In this order:

1. Upload a real PDF through the Quick Ingest file picker.
2. Add a reachable standards-document URL and wait for terminal success.
3. Add the same URL again and verify it is visibly skipped as existing.
4. Add `https://www.youtube.com/shorts/6-rf_YXDpPg` and wait for terminal success.
5. Add the same YouTube Short again and verify it is visibly skipped as existing.

- [ ] **Step 5: Validate backend lifecycle and storage**

Capture job timestamps/status transitions and query the isolated Media DB. Require every job to leave queued state promptly and exactly three unique records for PDF, standards URL, and YouTube Short.

- [ ] **Step 6: Fail the stage on any browser or product error**

Treat `Maximum update depth exceeded`, page errors, relevant console errors, stuck 0% jobs, missing terminal results, duplicates stored as new, or an incorrect Media DB count as a failure requiring renewed investigation before browser regression work.

## Stage 4: Automated Browser and Static Verification
**Goal**: Convert the successful UAT behavior into repeatable regression evidence.
**Success Criteria**: Packaged-extension cancellation and launch-health tests pass without skips; compile, lint, and changed-scope security checks pass.
**Tests**: Playwright, TypeScript, ESLint, Bandit applicability review.
**Status**: Not Started

### Task 5: Run browser regressions after UAT

- [ ] **Step 1: Run the existing headed cancellation regression on the host**

```bash
TLDW_E2E_SKIP_EXTENSION_BUILD=1 TLDW_E2E_EXTENSION_HEADLESS=0 TLDW_E2E_EXTENSION_MINIMAL_LOCALES=1 bunx playwright test tests/e2e/quick-ingest-cancel.spec.ts --reporter=line
```

Expected: PASS with cancelled/error results still visible after late completion.

- [ ] **Step 2: Run packaged launch health in the validated matrix**

Run headed/minimal, headless/minimal, and headless/full explicitly. Expected: all PASS without skips or missing MV3 targets.

- [ ] **Step 3: Run adjacent unit tests, compile, and lint**

```bash
bunx vitest run src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism
bun run compile
bunx eslint packages/ui/src/components/Common/QuickIngestWizardModal.tsx packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx
```

Use `apps/packages/ui` for Vitest and `apps/extension` for compile. Run ESLint from `apps` with paths adjusted to the workspace root.

- [ ] **Step 4: Record security applicability**

Bandit does not scan TypeScript. Record a changed-scope Bandit skip with rationale, while ensuring no credentials, auth headers, or backend payloads are logged by new code.

- [ ] **Step 5: Commit verification/task updates**

```bash
git add backlog/tasks/task-12947\ -\ Fix-browser-extension-E2E-launch-and-validate-Quick-Ingest.md Docs/superpowers/plans/2026-07-11-extension-quick-ingest-cancellation.md
git commit -m "docs: record extension quick ingest validation"
```

## Stage 5: Review and Pull Request
**Goal**: Deliver a current, reviewed PR against `dev` with complete evidence.
**Success Criteria**: Diff review has no unresolved P1/P2 findings, branch is rebased on latest `origin/dev`, checks pass, and PR includes the required human-owned Change summary gate.
**Tests**: `git diff --check`, focused verification rerun after rebase, GitHub checks.
**Status**: Not Started

### Task 6: Review, rebase, and publish

- [ ] **Step 1: Review the complete diff against origin/dev**

Prioritize terminal-state races, stale closures, duplicate finalization, persisted tracking regressions, unhandled promises, and missing test cases.

- [ ] **Step 2: Address all valid findings and rerun affected verification**

- [ ] **Step 3: Rebase on latest origin/dev**

Resolve conflicts without discarding unrelated upstream changes, then rerun focused Vitest, compile, packaged cancellation, and launch health.

- [ ] **Step 4: Finalize TASK-12947 and remove this plan only after every stage is complete**

Record UAT artifacts, commands/results, known limitations, Bandit rationale, commit IDs, and PR URL in the Backlog task.

- [ ] **Step 5: Push and open a PR against dev**

The PR must state what changed and why. The human requester must add their own Change summary before merge, per repository policy.
