---
id: TASK-13162
title: Activate existing Personal Context links with continuity fencing
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 13:40'
updated_date: '2026-09-05 17:03'
labels:
  - personal-context
  - sync
  - activation
  - security
dependencies:
  - TASK-13161
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
  - 'https://github.com/rmusser01/tldw_server/pull/2886'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the replayable activation journal that establishes a server baseline and publication checkpoint for existing and newly linked profiles before ongoing sync can run. Activation must preserve publication order, survive every cross-database interruption, and issue continuity proof that fails closed across capability gaps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preparation stores one encrypted exact-head baseline, digest, purge generation, and watermark at a whole publication-batch boundary.
- [x] #2 Deterministic Sync installation and receipt verification precede a leased Personalization CAS that marks all covered batches covered_by_activation and advances content-free covered-through proof before compaction.
- [x] #3 Per-device acknowledgments and activation state replay idempotently by activation ID and digest across Personalization and Sync without claiming cross-database atomicity.
- [x] #4 A random activation epoch and continuity token are durable, generation-bound, echoed and validated on every version-1 exchange, and invalidated or write-fenced when journaling cannot be guaranteed.
- [x] #5 Capability downgrade preserves links and queued work; restoration requires the same proven continuity pair or a fresh baseline.
- [x] #6 Restart tests cover preparation, Sync installation, coverage CAS, compaction, acknowledgment, racing server writes, and first post-watermark relay.
- [x] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs activation and continuity.
- [x] #8 Failed ingress repair can unblock baseline preparation only after exact canonical receipt verification; mismatched receipts and non-retryable states remain rejected.
- [x] #9 Concurrent activation and production ingress after-commit relay cannot deadlock; contended relay returns pending without losing publication debt.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement Task 1 of Docs/superpowers/plans/2026-09-03-personal-context-ongoing-sync-02-server-activation-conflict-purge.md against merged PR 2868. 1. Verify existing contract, publication/lease and exchange-proof baseline. 2. Test-first: durable encrypted exact-head preparation at whole-batch watermark, deterministic Sync installation and verified Personalization coverage CAS. 3. Add replayable per-device acknowledgment, durable continuity validation and fail-closed downgrade behavior through existing authenticated endpoints/factories. 4. Exercise restart/failure boundaries, competing writes, compaction proof, post-watermark relay, SQLite/PostgreSQL and protected canaries with targeted tests. 5. Run scoped formatting/lint/Bandit, independent specification then code/security reviews, update docs and task evidence. 6. Compatibility investigation addition: reproduce verified failed-ingress retry rejection, extend only its legal terminalization transition, and prove exact-receipt/disallowed-state rejection plus successful bootstrap after repair. Preserve ongoing_sync_version=0; do not implement TASK-13163–13165. ADR required: no new ADR. ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md. Reason: implementing the approved activation/continuity journal and custody decisions without changing protocol policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented encrypted exact-head activation preparation, protected deterministic Sync installation, leased canonical coverage/compaction, exact per-device acknowledgement, canonical continuity enforcement, delivery expiry and abandoned-preparation recovery. Existing authorized purge removes the new journals. Version-0 rollout and shared wire records remain unchanged. Updated API/developer/user docs and generated published copies; ADR-002 governs without a new decision.

Compatibility work preserved original cursor, race, projection and recovery assertions. Replaced metadata-only fixtures with explicit typed proof doubles where source publication state is the test subject; production certification and authenticated endpoints use real activation install/ack. Seven bootstrap failures also reproduced on unchanged dev (571 passed/7 failed at 3bc8c6a98c). Fixture repairs exposed a failed-ingress transition bug blocking bootstrap: added only failed to terminalization states after existing exact receipt verification. Regression RED: 2 failures/20 passes on SQLite/PostgreSQL; GREEN: 22 receipt/state cases plus all 48 bootstrap tests passed. This supporting fix was added to AC8 and the plan before implementation.

Final closing evidence, PostgreSQL required: 279 canonical/publication/authorization tests; 156 relay/recovery/shared-contract tests; 254 activation/API/transport/bootstrap/ingress tests. All three runs exited zero with no failures/errors/skips; 659 distinct cases across the runs. JUnit: /private/tmp/task13162-canonical-final-close.xml, /private/tmp/task13162-relay-final-close.xml, /private/tmp/task13162-integration-final-close.xml. No full repository sweep. Ruff passed all 23 changed Python files; five new files pass format checks; diff check clean; published docs match source. Bandit: zero findings in 13 changed production files. Compared test findings against unchanged dev: new low-severity hardcoded-token flags are reviewed synthetic proof fixtures, not production secrets. Independent specification and code/security reviews have no remaining P1/P2 findings.

Known separate issue: ordinary PostgreSQL ingress fixture insertion reached unchanged SyncDatabase._ensure_domain_state SQL translation with five placeholders and six parameters. No unrelated production fix was bundled. The new backend regression seeds valid ingress input rows with parameterized SQL and exercises real receipt/transition storage; it does not claim PostgreSQL ordinary-envelope insertion is verified. End-to-end bootstrap repair passes on SQLite. This transport insertion issue remains follow-up work. TASK-13163–13165 are not implemented; ongoing_sync_version remains 0.

PR handoff 2026-09-05: rebased without conflicts onto dev 63358431d799828cfb86511f7babd8490bbbdda9; range-diff confirms the implementation patch is unchanged (rebased commit adb741dfb10c77fc5b873abb809f19e7517f7b20). Fresh post-rebase integration gate: 254 passed, zero failures/errors/skips, PostgreSQL required, /private/tmp/task13162-pr-rebased-verification.xml. Scoped Ruff and production Bandit passed again. Opened PR 2886 against dev: https://github.com/rmusser01/tldw_server/pull/2886. Human-written Change summary is still required before merge; no auto-merge enabled.

PR 2886 review follow-up authorized 2026-09-05: user explicitly waived the human-written Change summary requirement for this PR only and authorized latest-dev rebase, all review/CI remediation and merge. Latest fetched dev remains 63358431d7, already the branch base. Plan: reproduce and address Qodo purge-generation/lock-order reports; move rollout dispatch into core, complete public docstrings/type hints and use domain exceptions; evaluate fixture feedback without introducing production test-only helpers or unrelated auth dependencies; reproduce/fix OpenAPI fingerprint CI drift; run focused regressions and independent review, reply to all threads, then wait for current-head Qodo/required checks and merge. ADR-002 continues to govern; no rollout or protocol-policy expansion.

Independent specification review traced a remaining preexisting Sync-to-profile-lock path through production ingress after-commit relay. Extend the review remediation plan to reproduce that interleaving and make relay process-lock acquisition nonblocking, preserving ordinary activation lease acquisition, durable fencing and retryable pending debt. Verify a deterministic production callback regression plus existing recovery budgets. This is a bounded lock-order correction under ADR-002, not a new storage or protocol policy.

Qodo remediation: core bootstrap now owns version dispatch; public activation contracts and fixture typing are explicit; activation errors use content-free domain subclasses with compatible HTTP translation. Activation takes profile lease before Sync and revalidates generation plus the exact link receipt inside the install guard. Production ingress after-commit relay skips a contended process lock before canonical SQL and retains durable publication debt. Deterministic regressions reproduced the races before the fixes and now pass, including later acknowledgment of the raced publication. Worker closing gate: 172 passed with PostgreSQL required; core dispatch gate: 124 passed. Independent specification and code/security review cleared the final behavioral changes. The two fixture suggestions were rebutted with concrete transaction and shared PostgreSQL fixture ownership evidence. Updated developer/published docs and regenerated the OpenAPI fingerprint; exact drift check passes (description-only difference, unchanged path/schema counts). Broad final targeted matrix and current-head CI/merge are still pending. Newly added activation files pass formatting; existing publication module has pre-existing whole-file formatter drift, so unrelated formatting is not bundled.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added replayable activation and canonical per-device continuity fencing for existing Personal Context links. Verified 659 distinct targeted cases, independent reviews, scoped Ruff/Bandit, and updated documentation. Ongoing-sync rollout remains disabled; separate PostgreSQL ordinary-ingress insertion issue is documented for follow-up.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
