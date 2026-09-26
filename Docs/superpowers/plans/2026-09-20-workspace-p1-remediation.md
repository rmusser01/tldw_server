# Workspace P1 Review Remediation

Tracking: TASK-12020.50. User approved correcting R1-R4 in
`../../Development/workspace-unmerged-review-2026-09-20.md`.
Preserve the dirty .49/.50 worktree; do not enable the owned route or rebase it.

## Stage 1: Persistence Safety
**Goal**: Prevent prehydration writes and preserve independent window drafts.
**Success Criteria**: No-op invalidation never reaches persistence; the store's
storage boundary rejects writes until hydration completes. Unchanged departures
do not overwrite another window. Writer-specific durable revisions preserve
competing edits; ambiguous recovery fails closed with all variants retained,
without an unsafe localStorage read-then-write CAS or silent winner selection.
**Tests**: Delayed hydration and reload; stale window departure; interleaved writes,
reload conflict, storage failure, scope separation and existing draft regressions.
**Status**: Complete

## Stage 2: Account-Bound Reads And Clone Requests
**Goal**: Keep every request bound to the verified account and server.
**Success Criteria**: All owned bundle reads carry expected-user, with server
enforcement before DB access. Cross-tab configuration/session changes invalidate
clone readiness and transport cannot substitute current credentials or follow
redirects outside its verified request context.
**Tests**: Cookie A-B-A reads, backend wrong-user guards, cross-tab invalidation,
transport changes between verification and dispatch, polling and retries.
**Status**: Complete

## Stage 3: Verification And Review
**Goal**: Verify all fixes together and update the evidence record truthfully.
**Success Criteria**: Targeted RED/GREEN, broad workspace regressions, scoped
backend tests/Bandit, real-backend boundary checks and independent review. Keep
PostgreSQL, latest-dev and browser acceptance limitations explicit.
**Tests**: Combined regression commands and real isolated HTTP checks.
**Status**: Complete

Final verification: 1510 frontend tests in 53 files; 141 selected backend tests;
23 isolated real-backend HTTP assertions; seven Chromium/CDP storage cases;
expanded scoped TypeScript; OpenAPI fingerprint/type generation; production
Bandit zero findings/errors. ESLint retains only 13 HEAD-baseline warnings and
the existing pages-directory notice. The full design-system check remains failing
outside the changed SharedWithMe screen, which has zero introduced findings.
Independent rereviews confirm all scoped corrections, including mixed-version
draft preservation, foreign recovery records, uncertain keys and explicit 403
credential refresh. All implementer/reviewer agents and verification sessions
finished; no route enablement, commit, push, rebase or CI cancellation.

Review corrections included in this pass: preserve legacy-tab edits after journal
adoption without any modern writes/deletes to the legacy key; preserve foreign
clone recovery envelopes before all ordinary cleanup/write operations; retain
uncertain admission keys through authentication recovery. A typed account-change
412 now uses the existing denial state and explicit re-verification.

Evidence and final disposition live in
`../../Development/workspace-unmerged-review-2026-09-20.md`. Chromium/CDP storage
probes and isolated API-key/SQLite HTTP checks are deliberately narrower than
full WebUI, PostgreSQL, or live JWT/cookie-switch acceptance.

R5/R6 are not part of these four P1 fixes. Full deletion, conflict-resolution UI,
and remaining mutation boundaries remain prerequisites to owned-route enablement.
