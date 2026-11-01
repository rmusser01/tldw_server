# PR 3319 Stage 3A Integration

Tracking: TASK-15400 (formerly TASK-13900); design TASK-15401 and planning TASK-15402. Historical archives have unique identities TASK-15403 through TASK-15405. Governing architecture: accepted ADR-060 and the approved Stage 3 durable-workflow spec. No new architecture or production activation is authorized.

Requester explicitly waived the human-written Change summary requirement for this PR and requested rebase, conflict resolution, independent requesting-code-review review, validated fixes, and auto-merge after issues are addressed. This does not change repository-wide policy.

## Stage 1: Current Dev Integration
**Goal**: Rebase the isolated PR branch onto fetched current dev, preserving upstream behavior and unrelated tracking history.
**Success Criteria**: Conflicts resolved; original head retained in backup; workstream tasks have unique IDs; no domain/provider/startup activation.
**Tests**: Diff/range-diff, canonical task/ID and ADR gates, workflow and existing Jobs compatibility regression matrix.
**Status**: Complete

## Stage 2: Independent Review And Verification
**Goal**: Review the full rebased PR with the requested skill, address validated findings, and certify changed behavior.
**Success Criteria**: No outstanding actionable review findings; explicit treatment of declined scope and baseline debt; no new scoped Bandit findings.
**Tests**: Both real database backends and acquisition modes, retained proofs/fences/encryption/recovery, legacy lifecycle/UUID/WorkerSDK, inline/endpoint/media compatibility, relevant CI contracts, scoped lint/format/compile/Bandit.
**Status**: Complete

## Stage 3: Publish And Arm Auto-Merge
**Goal**: Publish the verified rebase and fixes, record the PR-specific waiver and final evidence, and enable auto-merge according to repository rules.
**Success Criteria**: Remote head matches verified branch; conflicts gone; CI/review issues addressed; auto-merge explicitly enabled or actual external blocker reported without bypass.
**Tests**: GitHub PR head/base/mergeability, required check status and review threads; configured hooks and tracking gates before final commit.
**Status**: In Progress

## Readiness Snapshot

The full-PR review found two P1 lifecycle integration defects. Both are fixed in `5431b598e3`; an independent scoped re-review of the full combined fix at `d6e187e72e` found no new Critical, Important, or Minor findings. Single and batch enrolled renewals now retain strict database-clock authority; generic release rejects enrolled roots before mutation and leaves capability continuation or bounded expiry recovery available.

Final-source lifecycle verification: 200 selected passes, including 112 new focused cases and 88 existing cases. Controller independent final selection: 40 passes. The broader 1,913-case matrix passed before the final metadata-only release projection refinement; it was not repeated in full after that refinement. Scoped line coverage was 95.39%. Docs: 212 passes; full backend ratchet group: 63 passes; final scope/tracking: 17 passes. No new scoped Bandit findings. Full-project collection still fails at the unchanged `Workspaces/test_workspaces_api.py:12` import of unavailable `httpx2` after collecting 75,930 items, with 12 collection skips; this is not a whole-project-green claim.

Related CI fixes publish the ADR mirror, preserve accepted ADR rationale while representing its private spec as a repository-only reference, recognize real imported Hypothesis state machines without increasing the quality baseline, and refresh only existing tenant-predicate line coordinates. A separate test-only baseline UI setup correction is tracked as TASK-15410 and committed in `d9465bba5a`; production UI and assertions are unchanged.

Fresh remote CI and real-server UX validation remain the last external gate. Auto-merge is off at this committed readiness snapshot. The live PR timeline records subsequent CI and activation outcomes; these task records stay open until the external integration outcome is confirmed. No production activation or repository-wide policy waiver is included.
