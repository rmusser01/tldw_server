# UAT013 accepted-handoff intent reassessment

## Three failure angles and stop/reassess

1. Native Home handoff set display mode/source IDs but left effective retrieval false. Actual Form/action/RAG boundary gave4 RED/1 GREEN. Enabling retrieval repaired that deterministic dispatch gate.
2. Actual asynchronous session restore could replay old mode/source IDs after a changed-value handoff, also when accepted before restore invocation. Invocation-only capture was insufficient. Initial same-target value capture plus invocation capture for later restore gave the prior GREEN controls.
3. Independent same-value control now fails: live RAG42 and old persisted RAG7; accepted new Rowan42 leaves value comparison unchanged and dispatches7. Unchanged original reviewer probe rerun:1 RED. Permanent tests add same-value RAG and ordinary full-content handoffs before/during restore, and Rowan→Cedar→Rowan during restore. Value equality does not represent user intent; no further tuple/snapshot heuristic is proposed.

## Existing patterns examined

- `hooks/useSelectedAssistant.ts` increments selectedAssistantOperationRevision for every accepted setter operation, even if normalized values match, then uses that revision to guard asynchronous persistence. This records operation order rather than value differences.
- `store/playground-session.tsx` has nonpersisted restoreRevision and cancelPendingRestore. It already coordinates asynchronous restore ownership. Using that cancellation directly would also cancel transcript restoration, violating this task's retained-history contract.
- `PlaygroundForm.tsx` assistantActionRevisionRef increments on accepted inline/template application; captureAssistantActionGuard combines that local operation revision with history/server/restore ownership. It handles same-value intent but its local ref cannot directly coordinate the parent session hook.

## Alternatives

1. Add more values to the comparison (including retrieval flag): rejected; the reproduced RAG42 case already has retrieval true, and ordinary same-value intent/ABA still cannot be inferred.
2. Call cancelPendingRestore on source handoff: rejected; it discards desired old conversation/history restoration and does not preserve independent transcript/selection responsibilities.
3. Publish persisted handoff metadata or add a cross-component event/controller: unnecessary extra lifecycle/schema surface for a single in-memory ordering problem.
4. **Selected:** a single ephemeral source-handoff revision in the existing playground-session store, plus its increment action. No new store, watcher, event bus, persisted schema or migration.

## Proposed bounded correction

Add `sourceSelectionRevision` and `markSourceSelectionIntent` to the existing session store's ephemeral state (outside persisted data/partialize). The accepted Form media handoff increments it synchronously after owner/payload validation and after a valid RAG media ID is established, for both RAG and ordinary full-content handoffs, including identical values. Rejected/foreign/delayed-invalid payloads do not increment.

Replace the value strings in the current session guard with this revision. Retain its limited initial same-scope/history/server-target baseline and existing initialRestoreSettledRef; later requested restores capture the revision at invocation, so they still intentionally replay saved selection. Skip only persisted mode/source-ID replay when a newer accepted handoff occurred. Continue transcript/metadata restoration and existing restoreRevision/authority guards. Keep the revision monotonic through clearSession; the existing cancellation revision already retires obsolete session work.

New production scope: existing `store/playground-session.tsx` in addition to the two owned files. Tests: current actual Form/action/RAG fixture and existing session-store tests. Permanent controls prove same-value RAG/ordinary/ABA, accepted-vs-rejected revision, no revision persistence, later restores/different owner/session and existing synchronous hydration. Existing source routing/empty-error fail-closed/authority controls remain.

No production correction yet; parent approval and release of its prior-candidate runner are required before edits. No browser/runtime/API/model calls.
