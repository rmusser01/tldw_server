# Quick Ingest Explicit Submission Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace effect-driven Quick Ingest submission with one explicit, authority-checked command that starts each WebUI or extension run at most once.

**Architecture:** A pure transition builder produces an immutable processing snapshot without side effects. Review, quick-process, and `autoProcessQueued` pass that snapshot through one `beginProcessing` command, which validates, acquires and reconciles the lease, builds the payload, renews ownership, durably commits `creating_run`, applies the UI state once, and invokes a token-fenced runner. The component that owns `IngestWizardProvider` subscribes only to the primitive external-authority revision; ordinary session, persistence-status, ownership, and tracking subscriptions live in a descendant bridge below the Provider so a persistence write cannot synchronously rerender the active reducer owner. Processing-state effects remain reattach/status consumers only and never create backend work.

**Tech Stack:** React 18, TypeScript, Zustand, Dexie, Vitest, Testing Library.

## Global Constraints

- Do not add dependencies, change the Dexie v15 schema, or change extension-runtime message contracts.
- Restored `processing` or `creating_run` state must never create a new backend session.
- Pre-submit storage, ownership, payload, or durable-handoff failure must not call `startQuickIngestSession` or `submitQuickIngestBatch`.
- Quick/auto pre-submit failure must land on Review so existing persistence and ownership alerts are visible.
- Preserve the existing `START_PROCESSING` and `SKIP_TO_PROCESSING` context APIs for compatibility during this task; production entry points stop using them directly.
- An ordinary Quick Ingest session-store write must not rerender the component that owns `IngestWizardProvider`; only a changed external-authority revision may replace Provider state.
- Full TypeScript remains skipped under the recorded Task 1 three-attempt repository-baseline cap.
- The tracked `antd` symlink must be restored to `../../../node_modules/.bun/antd@6.2.1+6dbf9a050bc9aadb/node_modules/antd` after every UI test command.
- Never read, edit, stage, or delete the two protected untracked watchlist templates.

---

## File map

- `apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx`: pure processing transition and exact transition application.
- `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`: non-subscribing Provider owner, descendant session-store bridge, explicit command, durable handoff guard, immutable runner, and removal of the Step-4 submission effect.
- `apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx`: async explicit Start callback and loading state.
- `apps/packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx`: transition-unit coverage.
- `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`: command ordering, exactly-once, restore, and stale-continuation coverage.
- `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx`: Review/quick/auto UX and visible failure coverage.
- `apps/packages/ui/src/store/quick-ingest-session.ts`: reuse the existing acquisition reconciliation and awaited `commitProcessingHandoff`; no new persisted phase.

### Task 1: Pure processing transition

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx`

**Interfaces:**
- Produces: `buildProcessingTransition(state: IngestWizardState): { ok: true; nextState: IngestWizardState } | { ok: false; nextState: IngestWizardState }`.
- Produces: `applyProcessingTransition(nextState: IngestWizardState): void` on `IngestWizardContextValue`.
- Preserves: `startProcessing()` and `skipToProcessing()` as compatibility wrappers around the same builder.

- [x] **Step 1: Write failing transition tests**

```tsx
const ready = buildProcessingTransition(seedWithValidOccurrences)
expect(ready).toMatchObject({
  ok: true,
  nextState: {
    currentStep: 4,
    processingState: { status: "running" },
  },
})

const blocked = buildProcessingTransition(seedWithExpiredMaterialization)
expect(blocked).toMatchObject({
  ok: false,
  nextState: {
    currentStep: 3,
    processingBlock: { code: "materialization_expired" },
  },
})
```

- [x] **Step 2: Verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: FAIL because `buildProcessingTransition` and `applyProcessingTransition` do not exist.

- [x] **Step 3: Extract the shared pure builder**

```tsx
export const buildProcessingTransition = (
  state: IngestWizardState
): ProcessingTransition => {
  const { request, block } = buildPlaylistIngestRunRequest(state.queueItems)
  if (!request) {
    return {
      ok: false,
      nextState: {
        ...state,
        currentStep: 3,
        highestStep: Math.max(state.highestStep, 3) as WizardStep,
        pendingRunRequest: null,
        processingBlock: block,
      },
    }
  }
  const perItemProgress = buildInitialProgress(state.queueItems)
  if (perItemProgress.length === 0) return { ok: false, nextState: state }
  return {
    ok: true,
    nextState: {
      ...state,
      pendingRunRequest: request,
      processingBlock: null,
      currentStep: 4,
      highestStep: Math.max(state.highestStep, 4) as WizardStep,
      processingState: {
        status: "running",
        perItemProgress,
        elapsed: 0,
        estimatedRemaining: 0,
      },
      results: [],
    },
  }
}
```

- [x] **Step 4: Verify GREEN**

Run the command from Step 2.

Expected: all context tests pass with zero unhandled errors.

### Task 2: Explicit authority command and immutable runner

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`

**Interfaces:**
- Consumes: `buildProcessingTransition` and `applyProcessingTransition` from Task 1.
- Consumes: existing `acquireSubmissionLease(): Promise<boolean>`, `renewSubmissionLease(): Promise<boolean>`, and `commitProcessingHandoff(state, tracking): Promise<boolean>`.
- Produces: a session-id shell whose Provider owner subscribes only to `externalAuthorityRevision`, plus a descendant store bridge for ordinary session/display updates.
- Produces: `beginProcessing(candidateState: IngestWizardState): Promise<boolean>`.
- Produces: an internal `runSubmission(snapshot, payload, attemptToken): Promise<void>` that never infers work from rendered Step 4 state.

- [x] **Step 1: Write failing command-boundary tests**

```tsx
await clickQuickProcess()
expect(commitProcessingHandoff.mock.invocationCallOrder[0]).toBeLessThan(
  startQuickIngestSession.mock.invocationCallOrder[0]
)
expect(startQuickIngestSession).toHaveBeenCalledTimes(1)

rerenderRestoredCreatingRun()
expect(startQuickIngestSession).not.toHaveBeenCalled()

resolveOlderAttemptAfterReplacement()
expect(screen.getByTestId("wizard-results")).toHaveTextContent("new-attempt")
expect(screen.getByTestId("wizard-results")).not.toHaveTextContent("old-attempt")
```

Add controls for unavailable, quota, non-owner, lease loss after payload construction, durable-handoff failure, two simultaneous commands, terminal completion, cancellation, and unmount while start acknowledgement is pending. Include row and whole-run cancellation while file bytes are still being prepared. The row case omits only the cancelled occurrence from the eventual payload; the whole-run case performs no backend start or submit mutation.

- [x] **Step 2: Verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: the existing maximum-update-depth controls fail and Step-4 state can still trigger submission.

- [x] **Step 3: Implement one explicit command**

```tsx
const activeAttemptRef = useRef<symbol | null>(null)

const showReviewRecovery = useCallback(
  (sourceState: IngestWizardState) =>
    applyProcessingTransition({
      ...sourceState,
      currentStep: 3,
      highestStep: Math.max(sourceState.highestStep, 3) as WizardStep,
    }),
  [applyProcessingTransition]
)

const beginProcessing = useCallback(async (candidateState: IngestWizardState) => {
  if (activeAttemptRef.current) return false
  if (persistenceStatus !== "ready") {
    showReviewRecovery(candidateState)
    return false
  }
  const transitionBeforeLease = buildProcessingTransition(candidateState)
  if (!transitionBeforeLease.ok) {
    applyProcessingTransition(transitionBeforeLease.nextState)
    return false
  }
  const attemptToken = Symbol("quick-ingest-submission")
  activeAttemptRef.current = attemptToken
  let runnerStarted = false
  let recoveryState = candidateState
  const attemptIsCurrent = () =>
    activeAttemptRef.current === attemptToken && isMountedRef.current
  try {
    const revisionBeforeAcquire = externalAuthorityRevisionRef.current
    const acquired = await acquireSubmissionLease()
    if (!attemptIsCurrent()) return false
    const live = useQuickIngestSessionStore.getState()
    const durableState =
      live.session?.id === session.id
        ? buildInitialWizardState(live.session)
        : null
    if (!acquired) {
      showReviewRecovery(durableState ?? candidateState)
      return false
    }
    const authoritativeState =
      live.externalAuthorityRevision > revisionBeforeAcquire && durableState
        ? durableState
        : candidateState
    recoveryState = authoritativeState
    const transition = buildProcessingTransition(authoritativeState)
    if (!transition.ok) {
      applyProcessingTransition(transition.nextState)
      return false
    }
    const selectedItems = transition.nextState.queueItems.filter(
      (item) =>
        item.validation.valid &&
        item.conferenceOverride?.selected !== false &&
        item.playlistReview?.selected !== false
    )
    const config = transition.nextState.presetConfig
    const payload = await buildQuickIngestPayload(
      selectedItems,
      transition.nextState.conferenceBatchMetadata,
      transition.nextState.pendingRunRequest,
      {
        ...config.common,
        storeRemote: config.storeRemote,
        reviewBeforeStorage: config.reviewBeforeStorage,
        advancedValues: config.advancedValues,
        typeDefaults: config.typeDefaults,
      },
      {
        isCancelled: () => activeAttemptRef.current !== attemptToken,
        isOccurrenceCancelled: (occurrenceId) =>
          preAuthorityCancelledOccurrenceIdsRef.current.has(occurrenceId),
      }
    )
    if (!attemptIsCurrent()) return false
    const renewed = await renewSubmissionLease()
    if (!attemptIsCurrent()) return false
    if (!renewed) {
      showReviewRecovery(recoveryState)
      return false
    }
    const tracking = buildCreatingRunTracking(payload, selectedItems)
    const committed = await commitProcessingHandoff(
      transition.nextState,
      tracking
    )
    if (!attemptIsCurrent()) return false
    if (!committed) {
      showReviewRecovery(recoveryState)
      return false
    }
    applyProcessingTransition(transition.nextState)
    runnerStarted = true
    void runSubmission(transition.nextState, payload, attemptToken)
    return true
  } catch (error) {
    if (!attemptIsCurrent()) return false
    setSubmissionStartError(safeSubmissionStartError(error))
    showReviewRecovery(recoveryState)
    return false
  } finally {
    if (!runnerStarted && activeAttemptRef.current === attemptToken) {
      activeAttemptRef.current = null
    }
  }
}, [
  acquireSubmissionLease,
  applyProcessingTransition,
  commitProcessingHandoff,
  persistenceStatus,
  renewSubmissionLease,
  session.id,
  showReviewRecovery,
])
```

Claim `activeAttemptRef` before the first await so simultaneous commands cannot both cross lease acquisition. A duplicate command returns without changing visible state; unavailable persistence still routes to Review. `showReviewRecovery` applies an exact local Step-3 snapshot instead of `goToStep(3)`, whose backward-only guard cannot move a quick intent forward from Step 1. When lease acquisition exposes a same-session durable record, build the recovery snapshot from that authority; the outer persistence guard prevents this local Review display from weakening a non-draft durable run. Before the claim, validation is pure and cannot mutate backend or durable state. `runnerStarted` transfers token ownership only after the durable handoff and exact UI application; every earlier exit clears only its own matching token.

`buildCreatingRunTracking` is a local helper that returns the existing bounded `PersistedQuickIngestTracking` fields: `mode: "unknown"`, `submissionState: "creating_run"`, selected occurrence IDs, and `startedAt`. `safeSubmissionStartError` returns one fixed safe message for non-`Error` values and the existing sanitized `Error.message` path otherwise. The unmount cleanup invalidates `activeAttemptRef` before any pending continuation settles.

The outer persistence adapter owns `processingHandoffGuardRef`, because `persistWizardState` and `commitSessionProcessingHandoff` live outside `WizardModalContent`. Before its awaited durable commit, `commitSessionProcessingHandoff` sets the guard to the exact `transition.nextState` object. In `persistWizardState`, ignore every state while the guard is set; when `state === processingHandoffGuardRef.current`, clear the guard and return without another write because that exact snapshot is already durable. The adapter clears the guard on throw or false commit; a successful commit leaves it until that exact UI snapshot is observed. This closes the store/UI transition window without a revision serializer. Remove `hasStartedRunRef`, `startRunRef`, the render-confirmed latch, `abortPreSubmit`, and the effect whose Step-4 predicate calls `startRun`.

Check the attempt token after every await and before each result, tracking, warning, Review-return, interruption, or terminal mutation:

```tsx
if (activeAttemptRef.current !== attemptToken || !isMountedRef.current) return
```

`runSubmission` owns the token after the durable handoff. Wrap the runner body in
`try/finally` and clear `activeAttemptRef.current` only when it still equals that
runner's token. Cancellation and every terminal path invalidate the matching token;
unmount invalidates it before any pending continuation can mutate UI or storage.

- [x] **Step 4: Historical full-suite gate (superseded by the approved resume amendment)**

Run the command from Step 2.

The original gate mixed Task 2 command controls with Task 3 entry-path wiring.
The resume amendment's Steps 7-8 replace it: Task 2 must make its focused
command/authority controls green and leave only the enumerated Task 3 entry
controls red. The full session suite becomes green in Task 3.

**STOP RECORD (2026-07-14):** Task 2 is blocked at the repository's
three-attempt limit. The initial RED was 22 failed / 81 passed with 16
unhandled maximum-update-depth errors. The explicit runner replacement made
the command-boundary assertions pass but its first full attempt ended at 32
failed / 71 passed with 29 unhandled depth errors. A render-time session-ref
sync and then a complete persisted-wizard projection equality guard each
failed the isolated durable-handoff control with the same unhandled error and
were reverted. Diagnostic instrumentation showed one
`UPDATE_ITEM_PROGRESS` action being reprocessed 52 times: the Provider mounted
once, `onStateChange` stayed referentially stable, external authority revision
stayed unchanged, and each synchronous session-store write re-entered the
Provider before React finished settling the queued reducer action. Both a
deferred test double and the faithful store-backed handoff reproduced it.
Task 2 production/tests remain uncommitted and not green; the full suite was
not rerun after the focused third-attempt failure. Further work requires an
approved architecture change that prevents ordinary wizard persistence from
synchronously rerendering the component that owns the active reducer.

**APPROVED RESUME AMENDMENT (2026-07-14):** The user approved the architecture
revision after the stop. This amendment overrides conflicting Task 2 example
signatures above; the retained implementation's actual
`buildCreatingRunTracking(payload, selectedItems, startedAt)` and
`runSubmission(snapshot, payload, selectedItems, attemptToken, authority)`
signatures remain authoritative. Do not rewrite the command or add reducer-wide
idempotence. The next change is limited to moving ordinary store subscriptions
below the Provider boundary.

- [x] **Step 5: Preserve the focused architectural RED**

Use the existing focused control `awaits durable creating-run authority before
the backend start mutation` as the regression test. Its command-order assertions
already pass, but Vitest must remain RED before the composition change because
the persisted `UPDATE_ITEM_PROGRESS` dispatch produces an unhandled
maximum-update-depth error.

Run:
`cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism -t "awaits durable creating-run authority before the backend start mutation"`

Expected: the assertion passes but the process exits nonzero with exactly the
known unhandled reducer/store feedback-loop error. Restore the tracked `antd`
symlink immediately after the command.

- [x] **Step 6: Split ordinary store subscriptions below the Provider**

Keep the public component as a session-id shell. It subscribes only to the
primitive session ID so ordinary updates to the same record do not rerender it:

```tsx
export const QuickIngestWizardModal: React.FC<QuickIngestWizardModalProps> =
  (props) => {
    const sessionId = useQuickIngestSessionStore(
      (store) => store.session?.id ?? null
    )

    useEffect(() => {
      if (!props.open || sessionId) return
      useQuickIngestSessionStore.getState().createDraftSession()
    }, [props.open, sessionId])

    const initialSession = useQuickIngestSessionStore.getState().session
    if (!sessionId || initialSession?.id !== sessionId) return null
    return (
      <QuickIngestWizardSession
        key={sessionId}
        {...props}
        initialSession={initialSession}
        sessionId={sessionId}
      />
    )
  }
```

`QuickIngestWizardSession` owns the persistence refs, handoff guards, local
request nonces, and `IngestWizardProvider`. It subscribes only to the primitive
`externalAuthorityRevision`. Capture the same-session initial record once, and
derive a new external snapshot only when that revision changes:

```tsx
const externalAuthorityRevision = useQuickIngestSessionStore(
  (store) => store.externalAuthorityRevision
)
const initialSessionRef = useRef(initialSession)
const initialStateRef = useRef(
  buildInitialWizardState(initialSessionRef.current)
)
const authoritativeSession = useMemo(() => {
  const current = useQuickIngestSessionStore.getState().session
  return current?.id === sessionId ? current : initialSessionRef.current
}, [externalAuthorityRevision, sessionId])
const externalState = useMemo(
  () => buildInitialWizardState(authoritativeSession),
  [authoritativeSession]
)
sessionRef.current = authoritativeSession
```

The shell validates the captured record's ID before mounting the session
component; a changed/cleared ID unmounts it through the primitive selector.
Persistence and handoff callbacks
must invoke the latest actions through
`useQuickIngestSessionStore.getState()` and refresh `sessionRef.current` from the
store after each synchronous mutation; they must not subscribe the Provider
owner to `session`, `persistenceStatus`, `isSubmissionOwner`, tracking, or store
action objects.

Render a `QuickIngestSessionStoreBridge` as a child of
`IngestWizardProvider`. The bridge subscribes to the same-session record,
`persistenceStatus`, and `isSubmissionOwner`, derives
`shouldAttemptPersistedReattach`, and passes them to `WizardModalContent`.
Those descendant rerenders are allowed because they cannot re-enter the
component that owns the Provider reducer:

```tsx
<IngestWizardProvider
  key={sessionId}
  initialState={initialStateRef.current}
  externalState={externalState}
  externalStateRevision={externalAuthorityRevision}
  onStateChange={persistWizardState}
  onCancelProcessing={deferAuthoritativeCancellation}
  onCancelItem={deferAuthoritativeItemCancellation}
  onCheckStatus={requestAuthoritativeStatus}
  onReconnect={reconnect}
>
  <QuickIngestSessionStoreBridge
    open={open}
    onClose={onClose}
    autoProcessQueued={autoProcessQueued}
    sessionId={sessionId}
    markProcessingTracking={markProcessingTracking}
    commitReviewHandoff={commitSessionReviewHandoff}
    commitProcessingHandoff={commitSessionProcessingHandoff}
    acquireSubmissionLease={acquireSubmissionLease}
    renewSubmissionLease={renewSubmissionLease}
    markInterrupted={markSessionInterrupted}
    setProcessingWarning={setSessionProcessingWarning}
    cancellationRequestNonce={cancellationRequestNonce}
    itemCancellationRequest={itemCancellationRequest}
    statusCheckRequestNonce={statusCheckRequestNonce}
  />
</IngestWizardProvider>
```

Do not add a queue, serializer, reducer-wide equality guards, schema fields,
dependencies, or exported APIs. Preserve the exact processing-handoff object
guard and every current command token check.

- [x] **Step 7: Verify the focused GREEN**

Run the exact focused command from Step 5.

Expected: 1/1 passes, `startQuickIngestSession` is called once after the durable
handoff, and Vitest reports zero unhandled errors. Restore the tracked `antd`
symlink immediately after the command.

- [x] **Step 8: Verify scoped Task 2 GREEN and preserve Task 3 RED**

Run the five Task 2 command/authority controls with this exact name filter:

```text
claims one attempt before awaiting acquisition when two commands race|fences stale tracking, warning, result, and token cleanup after cancellation replacement|replaces the mounted wizard with a newer durable draft after acquisition reconciliation|rechecks ownership through the real Modal callback and does not duplicate an authoritative run|forwards a pending 'whole' cancellation once an 'acknowledged' extension start reveals authority
```

Then run the complete session suite from Step 2 once as a sequencing control.

Expected focused result: 5/5 pass with zero unhandled errors. Expected complete
suite at this boundary: 97/103 pass with zero unhandled errors; the only six
failures are `keeps blocked auto-process retry available until Review resolves
the block`, `clears extension review authority and starts exactly one corrected
retry`, `omits a row cancelled while its direct payload bytes are still being
prepared`, `does not start a direct run cancelled while payload bytes are still
being prepared`, `does not pre-seed direct tracking item identities before
backend submissions are acknowledged`, and `restarts direct processing after
refresh when tracking exists without persisted job ids`. These are Task 3's
accepted RED. Restore the tracked `antd` symlink immediately after each command.

### Task 3: Route every product entry through the command

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`

**Interfaces:**
- Consumes: `beginProcessing(candidateState): Promise<boolean>` from Task 2.
- Produces: `ReviewStep.onBeginProcessing(state): Promise<boolean>`.

- [x] **Step 1: Write failing entry and UX tests**

```tsx
await startFromReview()
await startFromQuickProcess()
rerenderWithAutoProcessQueued()
expect(startQuickIngestSession).toHaveBeenCalledTimes(3)

setPersistenceStatus("unavailable")
await startFromQuickProcess()
expect(screen.getByTestId("wizard-review")).toBeVisible()
expect(screen.getByText(/local recovery is unavailable/i)).toBeVisible()
expect(startQuickIngestSession).not.toHaveBeenCalled()
```

Cover quota, other-tab ownership, payload-build failure, automatic retry only after readiness changes, and Review loading state. Replace legacy restored-Step-4 expectations that call `startQuickIngestSession` or `submitQuickIngestBatch`: restored `processing` and `creating_run` records are display/reattach-only and issue zero new backend start/submit mutations, whether tracking is absent or contains a direct session without persisted job IDs.

Use the real `ProcessingStep` in a focused preparation control: before durable
handoff, it must list the selected queue rows and expose per-row and whole-run
cancellation. After a row is cancelled, assert that the outbound payload, the
durable `pendingRunRequest`, the persisted queue/progress snapshot, and the
applied UI snapshot all omit that occurrence. Strengthen the payload-failure
control to require a visible generic Review alert and assert that the raw thrown
value is absent from both UI and persistence. Add an `autoProcessQueued` restored
Step-4/non-draft control that performs zero lease/start/submit calls and remains
display-only.

- [x] **Step 2: Verify RED**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: quick and auto paths bypass the new command or fail without visible Review recovery.

- [x] **Step 3: Wire the three entry paths**

```tsx
const stateRef = useRef(state)
stateRef.current = state

const handleQuickProcess = useCallback(() => {
  void beginProcessing(stateRef.current)
}, [beginProcessing])

const lastAutoAttemptKeyRef = useRef("")

const buildAutoProcessIntentKey = (
  candidateState: IngestWizardState
): string =>
  JSON.stringify({
    sessionId: session.id,
    persistenceStatus,
    isSubmissionOwner,
    externalAuthorityRevision,
    queueItems: buildPersistedQueueItems(candidateState.queueItems),
    selectedPreset: candidateState.selectedPreset,
    customBasePreset: candidateState.customBasePreset,
    presetConfig: candidateState.presetConfig,
    customOptions: candidateState.customOptions,
    conferenceBatchMetadata: candidateState.conferenceBatchMetadata,
  })

useEffect(() => {
  if (!autoProcessQueued || !isOnlineForIngest) return
  const attemptKey = buildAutoProcessIntentKey(state)
  if (lastAutoAttemptKeyRef.current === attemptKey) return
  lastAutoAttemptKeyRef.current = attemptKey
  void beginProcessing(state)
}, [
  autoProcessQueued,
  beginProcessing,
  externalAuthorityRevision,
  isOnlineForIngest,
  isSubmissionOwner,
  persistenceStatus,
  session.id,
  state,
])
```

The key is written before the async command starts, so rerenders cannot consume the same automatic intent twice. A readiness, ownership, durable-authority, or bounded submission-input change produces a new key and permits one retry. It intentionally excludes display-only step, progress, and result state so a running render cannot manufacture a fresh intent. Review calls the same command and keeps its button loading until the pre-submit command settles. Quick/auto storage or ownership failure uses `showReviewRecovery` so the existing Review alert is visible even when the intent started before Step 3. A local safe error alert covers payload-build failure without exposing raw transport output.

Pass Review the same command:

```tsx
<ReviewStep
  isOnlineForIngest={isOnlineForIngest}
  isCheckingConnection={isCheckingConnection}
  connectionRecoveryMessage={connectionRecoveryMessage}
  onRetryConnection={handleRetryConnection}
  persistenceStatus={persistenceStatus}
  isSubmissionOwner={isSubmissionOwner}
  onCheckSubmissionOwnership={acquireSubmissionLease}
  onBeginProcessing={beginProcessing}
/>
```

`ReviewStep` awaits `onBeginProcessing`, keeps Start loading only while the
command settles, and does not call `startProcessing()` when the callback is
present; it passes its current context state to the callback. Preserve the existing compatibility path for other `ReviewStep`
consumers. Auto-process must call `beginProcessing(state)` directly and must
not call `skipToProcessing()`.

During quick/auto payload preparation, render
`<ProcessingStep preparingSubmission onPreparingCancelItem={...} onPreparingCancelAll={...} />`. `ProcessingStep` derives transient
per-item progress with the existing `buildProcessingTransition(state)` builder;
it must not dispatch or persist Step 4 before `commitProcessingHandoff` succeeds.
After payload construction, filter the immutable transition snapshot by the
recorded pre-authority cancelled occurrence IDs and rebuild the processing
transition. Use that rebuilt snapshot consistently for tracking, the durable
handoff, UI application, and `runSubmission`; never combine a filtered payload
with the stale pre-cancel snapshot.

Preparation cancellation handlers are owned by `WizardModalContent`, not the
ordinary context cancellation callbacks. They synchronously record accepted row
or whole-run cancellation while an `acceptingPreparationCancellationRef` is
true. Immediately after payload construction, close that gate and hide the
preparation UI before lease renewal. Build one frozen remaining-ID set, then
re-filter `entries`, `files`, `pendingRunRequest.inputs`, and per-occurrence
conference metadata from that set before tracking or handoff. A cancellation
attempt after the gate closes is ignored and cannot mutate the live queue;
cancellation accepted before the gate closes is absent everywhere. Add a
two-file regression where file A finishes reading, file B remains blocked, and
file A is then cancelled; A must be absent from the final payload and every
authoritative snapshot. Add a renewal-pending control proving preparation
cancellation controls are gone and a captured stale callback cannot change the
frozen set.

Pass the persisted safe pre-submit warning to Review as a dedicated error prop
and render it in an error alert. `safeSubmissionStartError` must not include a
raw thrown value. Clear/replace the warning only through the existing session
adapter. Auto eligibility requires a draft session with a current wizard step
below Step 4; restored, processing, interrupted, terminal, and result states do
not invoke `beginProcessing`, lease acquisition, or backend mutation merely
because `autoProcessQueued` is true.

- [x] **Step 4: Verify GREEN**

Run the command from Step 2.

Expected: both Modal suites pass with zero unhandled errors and every blocked entry shows a visible Review explanation.

Task 3 completed test-first after the approved Provider/store-boundary revision. Review, Quick, and Auto now share the explicit authority command; restored non-draft Step 4 remains display-only; preparation is a transient projection; and pre-authority cancellation freezes one remaining occurrence set before lease renewal. That set filters payload entries/files, pending inputs, conference metadata, the durable snapshot, tracking, applied UI, and runner inputs. Focused race controls passed 3/3, including completed-file cancellation and a stale callback during renewal. The final combined Modal gate passed 180/180 (integration 74/74, session 106/106) with zero unhandled errors. The final specification/quality re-review approved the occurrence-freeze boundary with no critical, important, or minor findings. No schema, dependency, serializer, backend, or extension-runtime contract changed.

### Task 4: Full Task 8 verification and review

**Files:**
- Modify: `Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-shared-frontend.md`
- Modify through Backlog MCP: `TASK-12113`

- [x] **Step 1: Run focused persistence and context gates**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/store/__tests__/quick-ingest-session.test.ts ../packages/ui/src/store/__tests__/quick-ingest-indexeddb.test.ts ../packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: all tests pass with zero unhandled errors.

- [x] **Step 2: Run full Modal gates**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: all tests pass with zero maximum-update-depth or unhandled errors.

- [x] **Step 3: Run existing Dexie, scoped lint, and whitespace gates**

Run the existing Dexie suite, repository-pinned ESLint with `apps/tldw-frontend/eslint.config.mjs` across every Task 8 TypeScript/TSX file, and `git diff --check`.

Expected: all commands exit 0. Restore the tracked `antd` symlink immediately after Vitest.

Fresh root verification passed the focused persistence/context gate 129/129 (Context 61, IndexedDB 36, session store 32), the full Modal gate 180/180 (integration 74, session 106), and the unchanged Dexie helper/migration gate 10/10, all with zero unhandled errors and exit 0. Repository-pinned ESLint with the explicit frontend config passed all 13 Task 8 TypeScript/TSX files with zero findings. `git diff --check` passed for tracked changes; the two new TypeScript files emitted no whitespace diagnostics under `git diff --no-index --check` and will receive the authoritative cached-diff check after staging. The tracked `antd` symlink was restored and verified after every Vitest command. Full frontend TypeScript was not rerun under the documented Task 1 three-attempt repository-baseline cap. Bandit is not applicable to this TypeScript-only scope.

#### Final whole-diff review remediation

The first whole-diff specification and quality reviews both returned NEEDS FIXES. The following gates are required before Task 4 Step 4 can be checked:

- [x] **Remediation Step 1: Add focused RED controls**

  Cover an expired renewed lease that another tab takes before the durable processing handoff; stale Review handoff against newer durable authority; same-session external-authority replacement while the runner awaits; an external replacement render with stale reducer queue/new tracking; pre-hydration draft mutation against durable non-draft authority; and Start over through the real IndexedDB adapter.

- [x] **Remediation Step 2: Make durable handoffs transactional CAS operations**

  Replace the generic authoritative write with separate adapter operations. Processing handoff must atomically require the exact expected durable envelope, draft lifecycle, this adapter owner's unexpired lease, and a processing target before writing `creating_run`. Renewal must require draft lifecycle. Review handoff must atomically require the exact expected durable envelope before replacing it with the explicit draft Review target. CAS loss returns `false`, publishes no captured session, and never permits backend mutation.

- [x] **Remediation Step 3: Fence runner and external replacement state**

  Capture accepted post-acquisition authority, then require the attempt token, wizard session ID, and live external-authority revision after every runner await and in tracking callbacks. Invalidate the active attempt when accepted authority is replaced. While a Provider external replacement is pending, synchronously expose the external snapshot to descendants so no render combines an old reducer queue with new session tracking. Remove the unused `replaceExternalState` context escape hatch.

- [x] **Remediation Step 4: Preserve hydration and Start-over authority**

  Do not create a Modal-owned draft before Zustand persistence hydration completes, retain durable non-draft authority during hydration merge, and queue the early open until hydration settles. Route Results Start over/Ingest More through `replaceWithNewDraft` so the prior terminal/processing row is compare-and-set cleared and the new draft has a fresh session ID before the Provider restarts.

- [x] **Remediation Step 4a: Preserve persisted drafts from side-panel opens**

  Gate the side-panel `handleOpenQuickIngest` entry point on the session store's persistence hydration. Before hydration, retain the exact open detail through the existing pending-open request mechanism and replay it only after hydration, so a newly created in-memory draft cannot overwrite a durable draft's queue, preset, or Review state. Add a delayed-hydration regression with a persisted draft and remove the two unused duplicate side-panel callbacks.

  Completed test-first on 2026-07-14. RED failed in two focused files because the side-panel hydration hook and identity-preserving singleton helper did not exist (`1 failed | 7 passed`, zero unhandled errors). GREEN passed the side-panel delayed-hydration behavior, exact detail/options replay, competing-host singleton claim, unmount/rejection lifecycle, form contract, QuickIngestButton resume, and utility suites (`4 files | 22 tests`, zero unhandled errors). The persisted draft retained its queue, deep preset, and Review step; no draft was created before hydration; replay occurred once; and another mounted host can win the singleton consume race without the side panel recreating it. Relevant store/Modal hydration controls also passed (`3 files | 3 passed | 215 skipped`). Repository-pinned explicit-config ESLint passed the seven touched TypeScript/TSX files, scoped whitespace checks passed, and the Ant Design symlink was restored exactly. No schema, dependency, backend, extension-runtime, staging, commit, or full-TypeScript change was made.

- [x] **Remediation Step 5: Re-run focused and full gates, then re-review**

  Run the new focused controls, the 129-test persistence/context gate, both full Modal suites, existing Dexie tests, scoped ESLint, symlink restoration, and whitespace checks. Re-dispatch the same two final reviewers against the remediation diff and leave Step 4 open until both approve.

- [x] **Step 4: Request specification and code-quality review**

The specification reviewer must confirm all Task 8 requirements. A fresh quality reviewer must inspect command ordering, stale continuation fencing, two-tab behavior, restored-session behavior, failure UX, and unnecessary abstractions.

Final closeout review is approved. After Step 4a, the same specification and code-quality reviewers both returned APPROVE with no critical, important, or minor findings. Fresh root verification passed persistence/context `135/135`, full Modal `184/184`, Results/pending-open `26/26`, side-panel/pending-open `22/22`, and existing Dexie migration helpers `10/10`, all with exit 0 and no unhandled test failures. Repository-pinned explicit-config ESLint passed all 22 dirty TypeScript/TSX files; tracked and untracked whitespace checks were clean; and the Ant Design symlink was restored exactly. Full TypeScript remains skipped under the documented Task 1 three-attempt baseline cap, and Bandit is not applicable to this TypeScript-only scope. The internal-QA debug side-panel can co-mount two already-hydrated modal hosts, but both reviewers confirmed this pre-existing debug-only ownership issue is a non-blocking follow-up: production WebUI uses the global host and the production extension side panel uses the local host.

- [x] **Step 5: Update tracking and commit**

Use Backlog MCP to record exact verification counts, review results, and touched files. Check Task 8 Step 5 only after both reviews approve.

Task 8 implementation committed as `e97cc15b8b` (`feat: persist quick ingest runs in indexeddb (TASK-12113)`). The verified staged set contained exactly the 26 Task 8 implementation, test, design, plan, and Backlog files; cached whitespace was clean and both protected watchlist templates remained untracked and excluded.

```bash
git add Docs/superpowers/specs/2026-07-12-youtube-playlist-per-item-ingest-design.md \
  Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-shared-frontend.md \
  Docs/superpowers/plans/2026-07-14-quick-ingest-explicit-submission-command.md \
  apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx \
  apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx \
  apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx \
  apps/packages/ui/src/components/Sidepanel/Chat/form.tsx \
  apps/packages/ui/src/components/Sidepanel/Chat/useSidepanelQuickIngestOpen.ts \
  apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.queue.contract.test.tsx \
  apps/packages/ui/src/components/Sidepanel/Chat/__tests__/useSidepanelQuickIngestOpen.test.tsx \
  apps/packages/ui/src/db/dexie/schema.ts \
  apps/packages/ui/src/db/dexie/types.ts \
  apps/packages/ui/src/db/dexie/quick-ingest.ts \
  apps/packages/ui/src/store/quick-ingest-session.ts \
  apps/packages/ui/src/store/__tests__/quick-ingest-session.test.ts \
  apps/packages/ui/src/store/__tests__/quick-ingest-indexeddb.test.ts \
  apps/packages/ui/src/utils/quick-ingest-open.ts \
  apps/packages/ui/src/utils/__tests__/quick-ingest-open.test.ts \
  "backlog/tasks/task-12113 - Implement-shared-WebUI-and-extension-per-video-playlist-ingestion.md"
git commit -m "feat: persist quick ingest runs in indexeddb (TASK-12113)"
```
