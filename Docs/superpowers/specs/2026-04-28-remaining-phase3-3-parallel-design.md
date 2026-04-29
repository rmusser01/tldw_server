# Remaining Phase 3.3 Parallel Design

## Goal

Finish the remaining `Phase 3.3` sanitizer work in the `phase3.3-error-handler-adoption` worktree by running the rest of the safe, covered tranches in parallel, without broadening scope into Phase 4 or Phase 5 work.

The purpose of this design is not to reopen all prior Phase 3.3 candidate space. It is to define how to complete the remaining work from the current branch state efficiently, while preserving the conservative-plus rule set already agreed for this branch:

- remove only raw fallback/log leaks that are covered by tests or cheaply coverable with direct red/green tests
- preserve validation-facing `400/422`, not-found `404`, and conflict `409` behavior
- avoid giant files unless they already have a dedicated error-contract test harness
- keep source ownership disjoint across concurrent workers

## Current Branch State

The active worktree is:

`/Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption`

This branch already contains many completed local Phase 3.3 tranches, including recent cleanup/scheduler/service sanitizers. At the moment this design is written, there is uncommitted local work in:

- `tldw_Server_API/app/services/outputs_purge_scheduler.py`
- `tldw_Server_API/app/services/media_files_cleanup_service.py`
- `tldw_Server_API/app/services/file_artifacts_export_gc_service.py`
- `tldw_Server_API/app/services/ingestion_sources_cleanup_service.py`
- `tldw_Server_API/tests/Services/test_outputs_purge_scheduler_truthiness.py`
- `tldw_Server_API/tests/Services/test_media_files_cleanup_service.py`
- `tldw_Server_API/tests/Services/test_file_artifacts_export_gc_service.py`
- `tldw_Server_API/tests/Ingestion_Sources/test_ingestion_sources_cleanup_service.py`
- `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`

Those changes form the current local checkpoint and should be treated as the integration baseline for the remaining work.

Before any new parallel lane work begins, the parent must first checkpoint these already-verified local edits into a dedicated implementation commit. That checkpoint commit becomes the practical baseline for all remaining Phase 3.3 lane work, and no new worker lane may start from a dirty or partially integrated baseline.

## Scope

This design covers only the remaining `Phase 3.3` work.

In scope:

- remaining small, covered service fallback/log sanitizer tranches
- remaining medium endpoint tranches only if they already have focused error-contract coverage
- parent-side coordination, integration, and verification for concurrent worker output

Out of scope:

- new architecture, refactors, or extraction work
- large uncovered modules
- public payload contract changes that are not already pinned by tests or directly justified by a new focused regression
- later Phase 4 or Phase 5 work

## Candidate Tiers

### Tier 1: High-Confidence Small Services

These are the preferred parallel candidates.

Selection rules:

- roughly small file size and narrow responsibility, usually a source file under about `150-200 LOC`
- direct unit tests already exist, or cheap direct tests can be added
- remaining work is classic fallback/log sanitization
- no public response-shape change is needed

These candidates should populate the majority of worker lanes.

### Tier 2: Medium Endpoints With Dedicated Error Tests

These are acceptable for exactly one isolated lane at a time.

The current primary example is:

- `tldw_Server_API/app/api/v1/endpoints/sync.py`

Reason:

- it is larger than the service tranches, but still has a dedicated focused error-contract harness
- it already has dedicated error-contract coverage in `tldw_Server_API/tests/MediaDB2/test_sync_endpoint_errors.py`
- the remaining Phase 3.3-shaped work appears to be log/fallback sanitization rather than broad behavior redesign

This lane must remain narrowly bounded and must not expand into unrelated endpoint cleanup. For `sync.py`, that means no helper extraction, no query/path refactors, no schema changes, no response-shape changes, and no opportunistic cleanup outside the covered fallback/log branches and their directly adjacent tests.

### Tier 3: Borderline Or Deferred Candidates

These are candidates where the remaining issue is not a straightforward fallback leak, for example:

- success-path observability choices
- public payload fields that still intentionally contain file paths or diagnostics
- giant files with no focused coverage

These items are not assigned implementation lanes by default. They should either be explicitly rejected for this phase or promoted only after a scout review determines that a narrow covered branch exists.

## Parallel Execution Model

The remaining Phase 3.3 work should run as an aggressive wave-based fanout.

### Wave Structure

Run four concurrent lanes:

- `Lane A`: one medium endpoint tranche, currently `sync.py`
- `Lane B`: one high-confidence small service tranche
- `Lane C`: one additional high-confidence small service tranche
- `Lane D`: scout lane that inspects the next candidate set and either blesses or rejects them for Phase 3.3

When a small-service lane completes, it is replaced by the next scout-approved candidate. `Lane A` stays isolated until its tranche is complete.

### Ownership Rules

Each lane owns a disjoint write scope:

- one source family
- its corresponding focused test file(s)
- no overlap with other active lanes

No two lanes may edit the same source file, the same direct regression file, or the shared phase plan at the same time. No two lanes may edit the same shared helper, shared utility, shared fixture, shared test helper, or adjacent support module at the same time either. If a tranche requires changing a shared helper to land cleanly, that helper becomes part of the shard ownership and blocks other concurrent lanes from touching it.

Workers must not:

- edit the shared parent plan file
- stage or commit from the parent worktree
- push
- broaden their tranche after discovering adjacent work

### Scout Lane Responsibilities

The scout lane exists to keep the aggressive execution model disciplined.

Its input candidate set should come from:

1. the current remaining grep/audit output for raw fallback leaks, captured in the parent coordination notes at the start of the wave
2. the existing Phase 3.3 plan backlog and prior deferred items
3. nearby files suggested by completed worker tranches, but only when they remain disjoint and Phase-3.3-shaped

Its output for each candidate must be one of:

- `approve now`
- `defer for later in Phase 3.3`
- `reject from Phase 3.3`

Each decision should include:

- source file
- exact branch or fallback site
- existing test file reviewed
- why it is or is not conservative enough

The scout lane must return its `approve/defer/reject` decisions in its own handoff artifact or completion message, not by editing the shared plan directly.
Every `defer` or `reject` decision must then be copied by the parent into the active Phase 3.3 plan before the scout lane is recycled onto a new candidate. This prevents the same borderline files from being repeatedly re-triaged while preserving the parent-only plan ownership rule.

## Integration Model

The parent remains the only integrator.

After each wave:

1. review returned diffs for scope drift
2. integrate only completed, independently verified shards
3. confirm the merged touched-file set still matches the approved shard ownership and did not silently broaden into adjacent files
4. run one parent verification sweep across all touched files in the merged wave
5. run one touched-scope Bandit scan
6. run `git diff --check`
7. update the Phase 3.3 plan with `**Recent Update**` entries
8. commit locally in one logical batch

The parent should not commit partially merged overlapping work.

## Verification Contract

Every lane must provide:

- red proof for the added or adjusted focused regression
- green proof for the same focused regression
- green result for the full touched test file
- list of files changed
- note of any skipped nearby candidates

Parent verification for each merged wave must include:

- combined pytest run across all touched files in that wave
- Bandit over the touched source files
- `git diff --check`
- `git status --short --branch`

## Commit Strategy

Commit by merged wave, not by individual micro-edit, unless a wave contains only one isolated tranche.

The remaining branch should favor:

1. current local checkpoint commit
2. next wave commit containing `sync.py` plus any concurrently integrated small-service tranches that verified cleanly together
3. one or more follow-up wave commits until no acceptable Phase 3.3 candidates remain

No push should happen unless explicitly requested.

## Skip Rules

Workers and the parent must skip candidates when any of the following are true:

- changing the branch would alter a validation-facing `400/422` contract
- changing the branch would alter a `404` or `409` contract
- the remaining leak is primarily a success-path logging or observability-policy decision
- the file is large and lacks focused error-contract tests
- the new test needed would require broad integration setup instead of a narrow direct regression
- the worker cannot explain in one sentence why the candidate is still truly `Phase 3.3-shaped`

## Success Criteria

- The remaining Phase 3.3 work is decomposed into independent lanes without overlapping write scopes.
- `sync.py` is handled in its own isolated lane.
- All additional source changes are backed by focused red/green regressions.
- Parent merged-wave verification remains green.
- No later-phase architecture or broad refactor work is pulled in.
- The branch ends with Phase 3.3 reduced as far as coverage and conservative scope reasonably allow.
