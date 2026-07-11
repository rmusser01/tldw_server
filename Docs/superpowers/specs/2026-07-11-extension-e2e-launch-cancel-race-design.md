# Browser Extension Launch and Quick Ingest Cancellation Design

## Problem

Browser-extension validation after PR #2709 exposed two independent failures.

First, extension helpers default to headless Chromium even for local Playwright
runs. Current Chromium does not expose the packaged MV3 service-worker target in
that mode, so extension ID discovery fails with `no extension targets`. The
headed launch path is healthy: the existing packaged-extension health probe
loads the options application in under five seconds when the helper is given
the repository's CI launch settings.

Second, the Quick Ingest cancellation regression reaches the real extension UI
but loses a cancellation/completion race. `ProcessingStep` dispatches the
wizard cancellation state, while `QuickIngestWizardModal` records the active
session in `cancelledSessionIdsRef` later in a passive effect. A completion
message received before that effect runs is accepted and replaces the cancelled
result with success.

## Goals

- Make supported local extension launches explicit and deterministic.
- Make user cancellation terminal before any asynchronous cancellation work.
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

### Explicit extension launch mode

`resolveExtensionHeadlessMode` will remain the single launch-mode resolver, but
its default will match the repository's Playwright configuration: headed for
local runs and headless when `CI` is set. `TLDW_E2E_EXTENSION_HEADLESS` remains
authoritative in either environment. Existing headed CI jobs already set `0`
and run under `xvfb`, while the nightly workflow can retain its unspecified,
headless CI behavior.

The built-extension launcher will stage a deterministic manifest key when it
creates a minimal-locale launch tree. This gives extension ID discovery a
stable fallback even if the background target is delayed. Unit tests will cover
both the launch-mode default/override contract and deterministic staging.

### Synchronous cancellation fence

`ProcessingStep` will accept an optional `onCancelAll` callback. Its Cancel All
handler will invoke that callback synchronously before dispatching
`cancelProcessing()`.

`QuickIngestWizardModal` will provide a callback that resolves the current
active or persisted session ID and inserts it into
`cancelledSessionIdsRef`. This is the cancellation fence. The existing passive
effect remains responsible for sending the background cancellation request and
finalizing unresolved items as cancelled, but it must not return merely because
the session has already been fenced. It will distinguish "already fenced" from
"cancellation side effects already issued" with a separate ref so side effects
remain exactly once.

Runtime completion and failure messages continue to pass through the existing
message handler. Once fenced, all non-progress terminal messages for that
session are ignored. A user cancellation therefore wins regardless of whether
the background acknowledgement, a stale completion, or React's passive effect
runs next.

## Data Flow

1. User clicks `Cancel All` in `ProcessingStep`.
2. `onCancelAll` synchronously fences the active session ID.
3. `cancelProcessing()` changes wizard status to `cancelled`.
4. The modal cancellation effect issues the background cancellation request
   once and finalizes unresolved items as cancelled.
5. Any completion or failure for the fenced session is ignored.
6. The results step remains cancelled even if a stale completion arrives
   immediately.

## Testing

### Red/green unit coverage

- A modal session test starts an extension-runtime run, invokes Cancel All, and
  emits completion immediately. Before the fix it resolves as success; after
  the fix it remains cancelled.
- Launch-mode tests prove the local default is headed and explicit true/false
  values remain honored.
- Built-launcher tests prove deterministic manifest staging is requested.

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

The production change is confined to internal React callbacks and refs. No API
or persisted-session schema changes. The launch helper change affects test
processes only. Environment overrides remain available for every CI workflow.
