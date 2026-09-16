# Independent review: UAT105 saved-source warning UI

## Verdict

No actionable findings remain in the frozen nine-file code/test slice against base `400db3877d1ca6f2da8897d32f45d37c534a61cb`. Freeze `2026-09-16T04:29:17.129Z`; all nine SHA-256 values matched after independent final tests. Scope is the three production files named in the author manifest plus their six test files; concurrent Chat changes were excluded. No product files were edited during review.

## Resolved review finding

The initial version accepted an outer completed job with nested `unknown` or `cancelled` status as clean success. Two private tests using the actual tracked-job poller reproduced this (2 failed), beyond the original helper test's weaker warning-undefined assertion. Root approved a narrow guard in `services/tldw/ingest-job-results.ts`: `cancelled`, `canceled`, and `unknown` now join failure tokens. Permanent tests cover the actual single-job poller and reattach path; missing status, Success and duplicate compatibility remain tested. The unchanged private tracked-poller probes now both pass. This was an automated discovery in a pre-existing classification boundary, not a newly observed native incident.

## Independent verification

- Final permanent focused suite: **173 tests / 6 files passed**, rerun after explicit freeze. Log `/private/tmp/cycle4-uat105-warning-ui-independent-final-tests.log` (start21:29:44).
- Private probes: **2 / 2 passed** after previously failing2 /2. Logs `/private/tmp/cycle4-uat105-warning-probe.log` and `/private/tmp/cycle4-uat105-warning-probe-green.log`; probe/config retained privately.
- Inspected original RED22 failed/131 passed, corrected direct-file upload RED1 failed, mixed aggregate RED, and post-review nested-status RED9 failed/51 passed. These demonstrate failures at the relevant behavior seams rather than only missing symbols.
- Inspected the committed real terminal job result: completed wrapper, own media_id1, statusWarning, error null and duplicate analysis-failure warnings. The classifier preserves that source, deduplicates warning copy, and the UI displays a distinct saved-warning group/count and existing Media navigation.
- Direct-upload and reattach mounted tests use actual batch/poller or actual reattach service, actual session state and actual results UI; only HTTP/host seams are mocked. No-warning clean success, explicit failures, no usable own ID, mixed aggregate ownership, cancellation and scoped service controls are exercised by the selected suites.
- Warning qualification checks the payload's own media identity, so the mixed aggregate's first saved sibling cannot promote an unsaved warning. Error rows do not expose successful source navigation. Warning rows are excluded from clean-success counts and receive escaped text rendering.

Command from `apps/packages/ui`:

```
./node_modules/.bin/vitest run --config vitest.config.ts src/services/__tests__/ingest-job-results.test.ts src/services/__tests__/ingest-jobs-orchestrator.test.ts src/services/__tests__/quick-ingest-batch.test.ts src/services/__tests__/quick-ingest-session-reattach.test.ts src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx
```

## Limits and baseline

No native browser, inference, full UAT, backend tests, full compiler or independent lint run in this review. Author's broader final log reports359 /22. Inspected author static comparisons show ESLint0errors/97unchanged warnings and initial TypeScript90existing diagnostics with no added/removed normalized diagnostics; final author compiler refresh was still completing at review time. Do not describe the baseline compiler as clean. Bandit is not applicable to this TS/TSX-only change. No auth, transport orchestration or permission changes occur in the production diff. Live warning rendering remains the parent's targeted verification step; this review does not provide native signoff.

## Private probe reproduction

Config: `/private/tmp/cycle4-uat105-warning-probe.config.mts`; test: `/private/tmp/cycle4-uat105-warning-probe.test.ts`. From `apps/packages/ui` run:

```
./node_modules/.bin/vitest run --config /private/tmp/cycle4-uat105-warning-probe.config.mts
```

The config imports the existing absolute UI `vitest.config.ts`, preserving aliases, setup and jsdom, and overrides only test.include to the private probe path. Author's completed post-guard compiler comparison subsequently confirmed exact90 baseline diagnostics, zero added/removed. No product hashes changed.
