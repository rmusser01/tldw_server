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

Check `MERGE_QUEUE` before activation. When `on`, arm the queue and do not rebase,
update, or hand-merge an armed PR. When unset, `off`, or `dry`, qualify only this
PR against current dev and wait for required statuses. The requester's explicit
auto-merge instruction applies after that qualification; it does not authorize
changing queue settings, dispatching replacement checks, or bypassing protection.

## Additional Review Intake

The automated review posted additional comments after the scoped technical review.
Stage 2 was reopened for verified dispositions, regressions, and a final independent
re-review of subsequent fixes. The earlier readiness snapshot below applies only
to `d6e187e72e` and the source-identical metadata head `8501d897f0`, not to later
unverified edits. Auto-merge remains off during this additional review.

## Earlier Readiness Snapshot

The full-PR review found two P1 lifecycle integration defects. Both are fixed in `5431b598e3`; an independent scoped re-review of the full combined fix at `d6e187e72e` found no new Critical, Important, or Minor findings. Single and batch enrolled renewals now retain strict database-clock authority; generic release rejects enrolled roots before mutation and leaves capability continuation or bounded expiry recovery available.

Final-source lifecycle verification: 200 selected passes, including 112 new focused cases and 88 existing cases. Controller independent final selection: 40 passes. The broader 1,913-case matrix passed before the final metadata-only release projection refinement; it was not repeated in full after that refinement. Scoped line coverage was 95.39%. Docs: 212 passes; full backend ratchet group: 63 passes; final scope/tracking: 17 passes. No new scoped Bandit findings. Full-project collection still fails at the unchanged `Workspaces/test_workspaces_api.py:12` import of unavailable `httpx2` after collecting 75,930 items, with 12 collection skips; this is not a whole-project-green claim.

Related CI fixes publish the ADR mirror, preserve accepted ADR rationale while representing its private spec as a repository-only reference, recognize real imported Hypothesis state machines without increasing the quality baseline, and refresh only existing tenant-predicate line coordinates. A separate test-only baseline UI setup correction is tracked as TASK-15410 and committed in `d9465bba5a`; production UI and assertions are unchanged.

Fresh remote CI and real-server UX validation remain the last external gate. Auto-merge is off at this committed readiness snapshot. The live PR timeline records subsequent CI and activation outcomes; these task records stay open until the external integration outcome is confirmed. No production activation or repository-wide policy waiver is included.

## Final Local Verification

All additional review findings have verified dispositions. The runtime fixes
reject NUL identities, bound controls to ordinary Jobs storage, generate retained
trace IDs, use database-time enrolled ordering, and scope legacy archive
correlations by owner without changing active Jobs uniqueness. PostgreSQL tests
now install ordinary Jobs FORCE RLS and establish/restore caller owner context;
runtime support inspection checks explicit grants and returns typed unavailable
without granting privileges or masking unrelated errors. The missing-role claim
did not reproduce: the existing typed boundary handles actual SQLSTATE `22023`.

The exact frozen runtime source passed 2,215 selected cases with zero skips:
SQLite 547, real PostgreSQL 536, Jobs compatibility 409, lifecycle/RLS/quarantine
231, inline 48, endpoint/media 155, CI 77, and Docs 212. Combined six-module line
coverage is 95.64%. Two later CI test-double repairs passed 30 additional distinct
cases, for 2,245 selected passes. Their final 43-case run also repeats all 13
quality tests; those repeated cases and 186 overlapping contract checks are not
added to the total. The test-quality baseline was not increased. Scope baseline
remains 28 entries with only two more line-coordinate updates. Task-ID, canonical
tracking, shard coverage, configured hooks, scoped lint/compile, and production
Bandit checks pass with the previously documented unchanged baseline debt.

Independent read-only reviews cover all 25 additional fix files and the separate
two-file CI repair delta; no Critical, Important, or Minor findings remain. The
reviewers verified patch/source correspondence and preservation of existing
assertions. A captured SQL assertion lives in the actual test, not its double.
Unchanged PR behavior was already fully reviewed. Stage 3B domain/global capacity
and retention, Stage 3C runner/adapters/accounting/trace mapping, Stage 3D whole
foundation certification and activation remain separately gated. Whole-project
success and universal legacy-debt cleanup are not certified; the unchanged
`httpx2` collection blocker remains. Remote qualification is still parent-owned.

Real-server UX run `41333443746` passed on published head `8501d897f0`; UI source
is unchanged by the subsequent fixes. Fresh publication-head CI, including UX and
the repaired Jobs suite, remains required before auto-merge. Auto-merge is still
off at this snapshot. The requester approved narrow task metadata repairs where
the vendored editor lacks support; the migration/tooling gap is filed as
`rmusser01/backlog-md-py#197`. No production activation or global policy change.

## Publication Base Update

While publishing, dev advanced from `54b321dcec` to `61d02ed484` through the
isolated Moderation compiler cleanup in PR #3396. All 34 PR commits rebased
without conflicts; range-diff reports every patch equivalent, and all 17 reviewed
source/test/tool hashes remain identical. The complete plan's current reachable
ancestor is `9eb0e2f81f`. Source verification above is retained, not represented as
a repeated whole matrix on the new base. Targeted Moderation, API/inline, CI and
tracking checks passed: 365 cases, zero skips, in 194.85 seconds. Task-ID and all
17 source hash checks also pass. These qualify the integration delta; final-head remote gates remain
required before auto-merge. The pre-rebase reviewed head is retained in
`codex/embeddings-stage3a-pre-dev61d-rebase`.
