# UAT104 / TASK13260.45 — read-only diagnosis

**Confirmed state mismatch; no production fix applied during frozen cycle4.** All7 investigated production/test paths match functional freeze7c9409fad2; source hashes in `uat104-diagnosis-manifest.json`. No browser/runtime access or product/test edits.

## Native evidence

- Multi: `output/playwright/cycle4-full-uat-2026-09-16/multi/ingest-minimize-remains-modal.txt` retains the modal after two clicks.
- Single: `/private/tmp/uat-cycle4-single-minimize-control.txt` retains the active Minimize button and still-visible Processing dialog.
- Both actual runs remain in processing; this is not job failure/cancellation.

## Exact causal chain

1. `ProcessingStep.tsx:385` handles the button with only `minimize()`.
2. `IngestWizardContext.tsx:441` changes local `isMinimized` totrue.
3. `QuickIngestWizardModal.tsx:618` persists processing/result state but does not change session visibility. The current session remains visible.
4. `QuickIngestButton.tsx:76` derives `quickIngestOpen` from that visibility. Its `closeQuickIngest` is the existing action that calls `hideSession`; the Processing button never invokes it. The sidepanel host similarly requires its supplied onClose callback to clear its local open flag and hide the session.
5. `QuickIngestWizardModal.tsx:1062` sees `open && state.isMinimized` and calls `restore()` immediately. Modal open expression at1768 becomes true again. `FloatingProgressWidget.tsx:221` also requires session visibility hidden, so local minimize alone cannot reveal it.
6. Existing close-confirmation path at1575 invokes both `minimize()` and `onClose()` and therefore does not have the mismatch. This is the closest correct pattern. Floating widget's Open action calls both showSession+restore in the reverse direction.

## Private regression reproduction

Config `/private/tmp/uat104-minimize-readonly.config.ts` transforms only the existing session test module in memory. It removes ProcessingStep/FloatingProgressWidget mocks, preserving actual wizard/context/store/step/widget behavior. It retains the existing small AntD modal facade and synthetic authority/service boundaries; no real jobs, browser or HTTP requests are used. Existing suite beforeEach establishes its normal synthetic authority. Two added cases share one session-backed owner rendering open from persisted visibility and closing through hideSession.

Result `/private/tmp/uat104-minimize-readonly.log`: **1 failing +1 passing**,43 unrelated cases filtered. Real Processing button settles as visibilityvisible,1dialog,lastModalOpentrue. Existing close-confirmation minimizes to visibilityhidden,0dialogs,lastModalOpenfalse while lifecycleprocessing remains. Positive control verifies same session ID, no cancel/start transport calls, and reopening restores the current Processing step. This is an interaction regression, not a source-text or mocked-minimize assertion. It complements the native WebUI failures; the private seed uses extension-runtime tracking to avoid invoking a backend.

Run from `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run --config /private/tmp/uat104-minimize-readonly.config.ts --maxWorkers=1 --no-file-parallelism
```

## Minimal repair scope after full matrix

- `ProcessingStep.tsx`: accept the owner's minimize/dismiss callback and invoke it with the existing local minimize action.
- `QuickIngestWizardModal.tsx`: supply its existing onClose path, or share the existing minimize+close callback between Processing and close-confirmation. Keep the restore-on-explicit-open behavior.
- Existing `QuickIngestWizardModal.session.test.tsx`: permanent real ProcessingStep interaction coverage. Its current ProcessingStep mock omits the Minimize button entirely, while isolated context tests only prove the reducer flag; neither catches this two-owner mismatch.

No session-store schema, global authority, persistence or transport changes indicated. Preserve background runtime/polling and the existing owner masks. Required controls: click Processing Minimize hides dialog/shows widget; same session keeps processing and receives terminal result while hidden; reopen active/completed session without duplicate submission or cancellation; close-confirmation remains working; direct WebUI held submission/polling and extension event paths; current owner switch masks prior session. Native single/multi rerun required after review. No task AC marked complete by this diagnosis.
