# UAT114 independent review — REVIEW CLEAR

Reviewed supplied `/private/tmp/uat114-review.diff`, implementation report, cycle4 design's hidden-tab transport contract, actual NotificationLifecycleProvider, web notification adapter, shared subscription cleanup, and affected consumer/tests. Only this package and necessary adjacent definitions reviewed; no other agent source included.

## Behavior / ownership

- `startWork` retains existing generation/epoch and scope reset, but exits before bootstrap, subscription and poll creation when hidden. Explicit retry while hidden also does no transport work.
- The visibility handler increments generation before cleanup. `stopWork` aborts pending unread/cursor requests, clears the poll interval, nulls/unsubscribes the stream. The actual shared subscription returns controller.abort and abort-aware retry delays; unsubscribe is effective, not merely an inert flag.
- Existing async bootstrap/poll callbacks and stream callbacks check their captured generation before publishing or proceeding. Hidden-tab late results therefore cannot reopen a stream or replace a newer snapshot.
- Activation starts the existing fresh unread/cursor bootstrap. Existing work is cleaned before another start; there is no new global transport or shared credential owner. Fresh lifecycle epoch allows consumers to refresh rather than consume stale buffered events.
- Activation deliberately preserves terminal auth-required/unavailable states: no automatic retry for401/403. Explicit recovery and existing permitted credential/scope transitions retain their prior semantics.
- Scope reset/projection remains active while hidden. Old-account data clears before new-account work, which waits for activation. Existing A→B→A, rotation, disconnected, StrictMode, one-owner and no-overlap behavior stays covered.
- Toast fixture additions provide existing authStorage exports already read by production; they preserve its prior single-user/null-session setup and do not stub the new visibility behavior.

The six new visibility cases use the actual provider with controlled API boundaries and DOM visibility events. They verify no initial hidden work, explicit hidden retry, release/poll pause/resume cursor/unread catch-up, ignored stale stream events, aborted pending bootstrap, terminal-state preservation, and hidden account replacement. Assertions exercise observable state and transport ownership rather than merely checking effect invocation.

## Independent verification

From `apps/tldw-frontend`:

```sh
bun run test:run __tests__/components/notification-lifecycle-provider.test.tsx __tests__/components/notification-rotation.integration.test.tsx __tests__/components/notification-toast-bridge.test.tsx __tests__/pages/notifications.test.tsx --maxWorkers=1 --no-file-parallelism
```

**4 files,111 tests passed,3.98s**, no skipped tests. `/private/tmp/cycle4-uat114-independent-tests.log`. Provider42, rotation32, Toast6, Notifications page31. Existing Node localStorage warning remains; no clean-console assertion. Scoped git diff --check passes. Supplied lint0errors/0warnings comparison inspected, not redundantly rerun.

No actionable defect found within this bounded diff. The implementation establishes the intended hidden-tab resource ownership in tests; it does not itself prove browser HTTP connection capacity or six-tab recovery. Required native six-tab ordinary request/Prompt-save verification remains pending. Multiple concurrently visible windows are not given a new global coordination mechanism, consistent with the design.

Read-only reviewer: no source/test edits, runtime/browser/inference, commits or subagents.
