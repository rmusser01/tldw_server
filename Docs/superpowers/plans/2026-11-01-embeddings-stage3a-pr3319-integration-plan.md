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
**Status**: In Progress

## Stage 3: Publish And Arm Auto-Merge
**Goal**: Publish the verified rebase and fixes, record the PR-specific waiver and final evidence, and enable auto-merge according to repository rules.
**Success Criteria**: Remote head matches verified branch; conflicts gone; CI/review issues addressed; auto-merge explicitly enabled or actual external blocker reported without bypass.
**Tests**: GitHub PR head/base/mergeability, required check status and review threads; configured hooks and tracking gates before final commit.
**Status**: Not Started
