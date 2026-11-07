# PR3434 Review And Audit Repair

Tracking: TASK-15466. User authorizes all validated PR fixes, Audit failure
repair, reverification, independent requesting-code-review, and gated
merge-commit auto-merge. The user explicitly waived Change summaries.
Work only in the isolated Claims worktree; preserve the main checkout.
ADR required: no new decision for narrow corrections/test isolation;
ADR094 governs shared Jobs resource/publication ownership. ADR095 remains
Proposed. Reassess if a validated issue changes a durable rule.

## Stage 1: Diagnose And Inventory
**Goal**: Reconcile complete PR feedback and reproduce the Audit failure.
**Success Criteria**: Feedback disposition list and measured settings/import
boundary explain the original shared-versus-per-user assertion.
**Tests**: Minimal collection-order experiments; original failing case with
root collection; inspect actual HTTP/backup settings bindings.
**Status**: Complete

Measured root failure: HTTP and backup share an older LazySettings while the
test mutates a replacement. Complete root collection reproduces the original
assertion; deterministic rebound variants reproduce both backup and restore
(2 failed/2 passed). Correct the test's per-test consumer bindings, retaining
all real refusal, no-artifact, file-integrity and rollback checks.

## Stage 2: Correct And Verify
**Goal**: Apply minimal test-first fixes for validated issues.
**Success Criteria**: Deterministic regression red/green without weakening
shared-store refusal, Jobs ownership, UUID fences or existing-only access.
**Tests**: Original failing test and implicated Admin/Audit/Config selection;
affected Claims/Jobs backend regressions; one root attempt with default
gates; full configured Ruff comparison, Bandit and applicable hooks.
**Status**: Complete

Both production findings have focused red/green proof: native lifetime9pass;
collision E2E/calculation33pass. Combined Audit/backup/Config91pass; full
vector/native/manager147pass with legacy1skip15xfail2xpass. Independent
correction review clean. Compile102pass, configured Ruff113vs116dev/no new,
Bandit57production one identical inherited finding/no errors, applicable
fullPRhooks128paths pass. Full Claims/Jobs is still running. Root default
attempt954pass/1missing-pg_dump prerequisite failure/12collection skips;
all original Audit variants pass. Do not claim full-root cleanliness.

## Stage 3: Review And Arm
**Goal**: Publish reviewed fixes and enable safe automatic integration.
**Success Criteria**: Independent whole-PR review and correction follow-up
have no unaddressed findings; all actionable GitHub threads resolved;
summary waiver accurately recorded; current exact head made ready and
merge-commit auto-merge armed without bypassing required checks.
**Tests**: Fresh PR/CI/thread state, exact remote/local head match, actual
autoMergeRequest readback. If CI reveals failures, fix and reverify before
arming. Never cancel CI, use admin merge or directly push dev.
**Status**: In Progress

Reviewed source is ready for publication and fresh CI, but auto-merge stays
off until final verification and new feedback dispositions complete.

The wide Claims/Jobs run completed5667pass/1stale-test-settingsfailure/
2PGsetupConnectionTimeout errors/3legacy skips. A deterministic current/
rebound ingestion test reproduces the input problem; three overrides now
configure the actual consumed proxy, preserving deadline/cancel/shutdown
assertions without production changes. All3exceptionalnodes passed untouched
in isolation; after the test correction, all Claims plus the complete affected
PG module passed1636tests with1legacy spaCy skip, zero errors/failures. This
closes observed outcomes, not a fresh all-green combined-suite claim.
Final static103compilepass/Ruff113vs116dev/no new/Bandit57production1inherited
finding/no errors. Independent final test-only review remains pending.

Review focus: stale imported settings after reload; shared Audit backup
refusal and rollback; no test-state leaks; cross-owner/stale-worker Jobs
authority; storage/publication locks retained through side effects.
Full independent review runs read-only while Audit repair is local. Each
validated fix receives regression evidence before normal commit/push;
remove only this plan when its stages are actually complete.
