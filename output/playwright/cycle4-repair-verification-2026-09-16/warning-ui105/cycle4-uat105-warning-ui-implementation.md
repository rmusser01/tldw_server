# UAT105 saved-source Warning projection

Task: TASK-13260.46, approved cycle4 design and Task3 plan. This is the native-discovered frontend continuation of the existing backend repair. Base: `400db3877d1ca6f2da8897d32f45d37c534a61cb`.

## Root cause and resulting behavior

The native completed-job response had `result.status=Warning`, `media_id=1`, `error=null` and duplicate analysis-failure warnings. The source was saved. The shared `extractCompletedIngestJobError` previously used `warnings[0]` as an error; both polling paths therefore classified the completed job as failed. The direct batch then threw that error, losing the saved result and media identity. Separately, the wizard did not copy warning text into normalized or reattached results, and its results screen had only success/failure outcome groups.

The shared classifier now separates explicit failures from Warning results that carry their own usable saved media identity. An absent/invalid ID still fails; an aggregate may not borrow a saved ID from its first child. Error/detail/errors and terminal error/cancellation metadata still take priority. The warning extractor deduplicates string warning details and supplies explicit generic processing-warning copy if the server provides no details.

Both wizard normalization paths retain the warning. The existing result-row presentation uses an amber warning icon and escaped plain-text details, a separate Saved with warnings group and count, and the existing Open in Media/Workspace/Knowledge actions. Warning rows are not counted as clean successes or offered automatic retry. No job scheduling, authentication, cancellation requests, persistence, backend, or transport-selection code changed.

Independent review also confirmed a pre-existing wrapper boundary gap: an outer completed job with a nested `cancelled`, `canceled`, or `unknown` status was counted as clean success. The narrowly approved correction adds these three failure tokens. They now fail closed, with existing generic failure guidance; ordinary outer cancellation retains its existing cancelled outcome. Missing status, Success and duplicate compatibility remains. This was an automated finding in the same projection boundary, not an additional native-observed incident.

## TDD and regression evidence

- Initial RED: `/private/tmp/cycle4-uat105-warning-ui-red.log` — 22 failed, 131 passed across five suites. Existing service/helper expectations and actual mounted flow reproduce the warning-as-failure defect. New warning API assertions also fail before its implementation.
- Corrected actual-file fixture RED: `/private/tmp/cycle4-uat105-warning-ui-batch-red.log` — one focused failure confirms the actual direct upload and real single-job poller return status=error and discard saved Warning data. The fixture uses the existing name/type/data file contract.
- Additional mixed aggregate RED: `/private/tmp/cycle4-uat105-warning-ui-mixed-red.log` — one failure/30 pass before narrowing saved identity to the warning payload itself; prevents promotion of mixed aggregate responses.
- Independent-review correction RED: `/private/tmp/cycle4-uat105-warning-ui-nested-red.log` — nine failures/51 pass across classifier, actual single-job poller and reattach for the three nested non-success statuses.
- Final full focused GREEN: `/private/tmp/cycle4-uat105-warning-ui-final-green.log` — 359 tests, 22 suites passed.
- Final required-callback fixture check: `/private/tmp/cycle4-uat105-warning-ui-final-fixture-green.log` — 13/13 shared orchestrator tests passed after adding its existing required `onCancel` callback to the new test fixtures.
- Actual transport boundary means mocked HTTP upload/request seams only: real `submitQuickIngestBatch`, `pollSingleIngestJob`, `reattachQuickIngestSession`, wizard/session authority and results UI execute. The mounted session test checks warning state, absence of failed-item controls, the exact job GET and working Media navigation for both direct-file submission and reattachment.
- Shared tracked-job poller control covers Warning + explicit failure + unsaved Warning + cancellation in one batch, retaining completed Warning data; this covers the orchestrator used by background batches.
- UI controls cover separate warning counts, media/workspace/Knowledge actions, mixed unknown failure and cancellation with no retry promise. Existing all-QuickIngest and private-scope/authority tests remain green.

### Final focused test command

From `apps/tldw-frontend`:

```sh
./node_modules/.bin/vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__ ../packages/ui/src/services/__tests__/ingest-job-results.test.ts ../packages/ui/src/services/__tests__/ingest-jobs-orchestrator.test.ts ../packages/ui/src/services/__tests__/quick-ingest-batch.test.ts ../packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts ../packages/ui/src/services/__tests__/quick-ingest.private-scope.test.ts ../packages/ui/src/services/__tests__/quick-ingest-authority.test.ts ../packages/ui/src/store/__tests__/quick-ingest-session.test.ts
```

## Static validation and review boundary

Scoped ESLint: 0 errors, 97 warnings, all 97 unchanged against base source passed to ESLint via stdin using the same file path/config. `/private/tmp/cycle4-uat105-warning-ui-lint-comparison.json` records exact diagnostic multiset comparison; current/baseline JSON logs are adjacent. `git diff --check` passes. Bandit does not analyze this TS/TSX-only scope; no Python or backend files changed.

Final post-review-guard TypeScript comparison: exact 90 existing diagnostics, zero added/removed after normalizing source locations and retaining complete message/multiplicity. `/private/tmp/cycle4-uat105-warning-ui-typecheck-comparison.json` compares against `/private/tmp/uat032-merged-typecheck.log`; raw current log is adjacent. The test fixture was corrected to pass the existing required `onCancel` callback before final freeze.

Independent post-guard review from account_access is clear: 173 permanent tests/six suites and two unchanged private tracked-poller probes passed; all nine code hashes match the 2026-09-16T04:29:17.129Z freeze. Root owns native browser validation. No runtime/browser/inference work, server mutation, staging or commit performed in this repair. This does not certify live warning rendering or extension-popup destruction; tests cover the actual frontend request seams and shared tracked-job orchestration, not a running MV3 worker.

Exact nine code/test paths and SHA-256 hashes: `/private/tmp/cycle4-uat105-warning-ui-code-freeze.json`. `/private/tmp/cycle4-uat105-warning-ui-owned-manifest.json` additionally includes the official task46 record. Unrelated concurrent files are excluded.
