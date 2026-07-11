# Browser Extension Launch and Quick Ingest Cancellation Design

## Problem

Browser-extension validation after PR #2709 initially reported missing MV3
targets, but that launch failure is not reproducible in a clean worktree with
the apps workspace dependencies installed. The packaged launch-health probe
passes headed with minimal locales, headless with minimal locales, and headless
with the full locale catalog. No launch-helper behavior change is justified by
the current evidence.

The reproducible failure is in the Quick Ingest cancellation workflow. The
test reaches the real extension UI but loses a cancellation/completion race.
`ProcessingStep` dispatches the
wizard cancellation state, while `QuickIngestWizardModal` records the active
session in `cancelledSessionIdsRef` later in a passive effect. A completion
message received before that effect runs is accepted and replaces the cancelled
result with success.

There is a second form of the same race: Cancel All is visible before
`startQuickIngestSession` returns. If cancellation occurs during that await,
there is no session ID to fence or cancel. A later extension acknowledgement can
leave background work running, and a later direct acknowledgement can still
submit the backend batch.

## Goals

- Record a repeatable extension launch command and retain current helper
  behavior unless a launch failure can be reproduced.
- Make user cancellation terminal before any asynchronous cancellation work.
- Prevent extension work or direct submission from continuing when cancellation
  precedes the session acknowledgement.
- Prevent payload preparation or error handling from reviving a cancelled run.
- Preserve current background-runtime cancellation behavior and direct-job
  cancellation metadata.
- Prove the product flow through the installed extension with PDF, web-link,
  duplicate-link, YouTube Short, and duplicate-YouTube ingestion.

## Non-Goals

- Redesign the Quick Ingest state machine.
- Change public ingestion APIs or background message schemas.
- Add arbitrary sleeps to make browser tests pass.
- Change cancellation behavior for individual items.

## Design

### Launch validation without speculative changes

The PR will not change extension launch defaults or manifest staging. The
existing launch-health spec and the final live UAT will use explicit environment
settings and report whether a run is headed, headless, minimal-locale, or full.
If a missing-target failure recurs, diagnostics and the generated profile will
be retained for a separate evidence-driven launcher change.

### Synchronous cancellation fence

`ProcessingStep` will accept an optional `onCancelAll` callback. The modal will
use one cancellation handler for the Processing step and the close-confirmation
Cancel All action.

The handler will synchronously set a run-level `cancelRequestedRef`. If an
active or persisted session ID is available, it will also insert that ID into
`cancelledSessionIdsRef` and send the background cancellation request. It will
clear any pending persisted-reattach timer, immediately finalize unresolved
items as cancelled, and move to results instead of waiting for a passive effect.

The cancellation intent is initialized once for each keyed wizard session and
is never cleared by `startRun`; choosing Ingest More creates a new keyed session
and therefore a fresh intent ref. `startRun` will inspect the intent after each
awaited setup boundary and immediately after
`startQuickIngestSession` resolves. Cancellation during payload preparation
returns before a session is started. Cancellation while awaiting the start
acknowledgement fences the returned ID, cancels extension-runtime work, and
returns. For a direct session, it returns without calling
`submitQuickIngestBatch`. The error path also returns without replacing an
already-cancelled outcome.

Persisted-job reattachment retains its existing effect-local cleanup fence.
Deferred `processing` and `completed` characterization tests prove that a
terminal session update already runs the effect cleanup before an in-flight poll
can write state or schedule another poll, so no additional production guard is
needed.

Runtime completion and failure messages continue to pass through the existing
message handler. Once fenced, every subsequent message for that session,
including progress, is ignored. A user cancellation therefore wins regardless
of whether the background acknowledgement, a stale completion, or the next
state update runs first.

## Data Flow

1. User clicks `Cancel All` in `ProcessingStep`.
2. The modal handler synchronously records cancellation intent and fences an
   available session ID.
3. The handler issues an available-session cancellation request, finalizes
   unresolved items, and moves immediately to cancelled results.
4. Any later message for a fenced session is ignored.
5. If the start acknowledgement arrives later, `startRun` fences and cancels
   extension work or suppresses direct batch submission.
6. If payload setup resumes later, it observes the cancellation intent and
   returns without updating state.
7. The results step remains cancelled even if a stale completion arrives
   immediately.

## Testing

### Red/green unit coverage

- A modal session test starts an extension-runtime run, invokes Cancel All, and
  emits completion immediately. Before the fix it resolves as success; after
  the fix it remains cancelled.
- A deferred extension-runtime acknowledgement test cancels first, resolves the
  acknowledgement second, and proves the returned session is cancelled.
- A deferred direct acknowledgement test proves cancellation prevents
  `submitQuickIngestBatch` from being called.
- A cancelled-session test proves later progress cannot mutate terminal state.
- Deferred persisted-reattachment characterization tests prove the existing
  cleanup fence prevents late running or completed snapshots from replacing
  cancelled results or scheduling another poll.
- A cancellation-during-setup test proves no session starts and no later setup
  failure replaces the cancelled outcome.

### Browser regression

- The existing `quick-ingest-cancel.spec.ts` must pass in a real headed packaged
  extension and retain the cancelled/error region after its late completion.
- The packaged extension launch-health spec must pass without skips.

### User acceptance testing

Against a fresh isolated backend and Media DB, one installed extension context
will ingest:

1. A PDF upload.
2. A reachable standards-document URL.
3. The same URL again, visibly skipped as existing.
4. `https://www.youtube.com/shorts/6-rf_YXDpPg`.
5. The same YouTube Short again, visibly skipped as existing.

The run will capture page errors, console errors, API traffic, visible progress,
job lifecycle timestamps, and final Media DB rows. Success requires no maximum
update-depth error, all jobs leaving queued state promptly, and exactly three
unique stored media records.

## Rollout and Compatibility

The production change is confined to internal React callbacks and refs. No API,
background message, persisted-session, or extension-launch contract changes.
