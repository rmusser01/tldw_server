# ACP Final Reconciliation (#2398)

## Stage 1: Evidence Inventory

**Goal**: Confirm all #2398 child issues and follow-ups are closed or explicitly accounted for.
**Success Criteria**: A final child-issue map identifies the outcome for #2404, #2403, #2401, #2400, #2402, #2408, and #2399.
**Tests**: GitHub issue state checks and local evidence-doc presence checks.
**Status**: Complete

## Stage 2: Surface Reconciliation

**Goal**: Audit ACP docs, setup/registry surfaces, and compatibility language for stale caveats or overclaims.
**Success Criteria**: Readiness, compatibility, retention/redaction, sandbox, live-agent, and setup-guide surfaces agree with final evidence state.
**Tests**: Targeted `rg` searches over ACP docs, API setup-guide code, registry config, runner config, and frontend Agent Registry copy.
**Status**: Complete

## Stage 3: Closeout

**Goal**: Apply any minimal corrections, validate, and update #2398 with a final reconciliation note.
**Success Criteria**: Any drift is fixed, verification is recorded, and #2398 can be closed or left open with a concrete blocker.
**Tests**: `git diff --check`, docs/search verification, and focused tests only if code surfaces change.
**Status**: Complete
