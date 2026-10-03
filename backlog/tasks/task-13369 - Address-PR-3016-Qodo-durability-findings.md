---
id: TASK-13369
title: Address PR 3016 Qodo durability findings
status: In Progress
assignee: []
created_date: '2026-09-26 00:52'
updated_date: '2026-10-03 01:30'
labels:
  - vn-assets
  - review
  - durability
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/3016'
  - 'https://github.com/rmusser01/tldw_server/issues/2021'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR 3016 onto dev; address all verified Qodo and independent-review VN generation durability findings with regression coverage, preserve API and Jobs contracts, then merge only after exact-head review, required CI and human summary gates pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Enqueue failure retries the original batch and deterministic parent Job.
- [x] #2 Storage handoff failures replay without terminalizing the variant.
- [x] #3 Concurrent deliveries use a fenced claim and cannot publish duplicate assets.
- [x] #4 Cancellation clears outstanding reservation capacity and preserves counters.
- [ ] #5 All remaining review comments are addressed with tests or reasoned thread replies.
- [ ] #6 PR checks and human summary gate pass before merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
IMPLEMENTATION_PLAN_vn_pr_3016_review.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Human execution2026-10-02: "address all the ci failures and review findings then!" approves the four presented child test/docs fixes; human separately explicitly approved Task67's bounded timeout-test correction. Tasks1-63 complete/reviewed/published; no redispatch.

Tasks64/65/67 are LOCAL COMPLETE, independently SPEC/QUALITY PASS/no actionable findings; all implementer/reviewer sessions CLOSED. Task64 privately relocates isolated SQL under core DB_Management, replaces caller-frame assertions with sensitive real outcome/rollback checks and removes raw runtime copy/finally hooks. Task65 condenses tracking while preserving exact old plan/task bytes. Task67 observes only intended read-only VN handles, preserving responsiveness/off-thread/closure/10000 assertions and native unrelated connections.

Verification: Task64 focused45pass/4warnings110.61s, native rollback3pass and copied omission3intendedfail; Task67 native/adjacent3pass/4warnings10.85s and copied missing-timeout1intendedfail/4warnings4.47s. Counts overlap, not additive. Initial130pass1busy-timeout failure remains historical RED, not rewritten. Compile/Ruff pass; same6 inherited B106 non-B101 findings, new core helper unfiltered0. LocalPython3.11/SQLGlot29 evidence is not nativeCI30/PG/whole-repo green.

Task66 is diagnosed, NOT IMPLEMENTED: exact CI SQLGlot30.20.0 isolated real SQLite setup/bootstrap succeeds but UsersDB initialization fails; standalone AUTOINCREMENT rendering is empty although AST args are empty/full DDL valid. Current guard rejects it. One narrow AST-validation fix/regression/rejection-control approval question remains SEPARATELY UNANSWERED; Task67 approval does not answer it. No guard/schema/workflow/dependency-policy change.

Fresh human-followup-final poll19manifest verified: both PR heads68861/393c, whole bodies and complete review/check/status/thread arrays unchanged; both OPEN/unmerged. Parent97complete with known JobsSQLite failure; child15complete with known macOS/Ubuntu E2E failures. Actual protecteddevd81c13fddd1dac1948b30401af0388932a0af8f2 read twice stable. Parentfalse/dirty is GitHub conflict status, not local reproduction. Child native393c Jobs already verifies published Task59 fix:1297pass4skip577deselect4211warnings1402.64s, four fixture-registration outer tests pass; not parent/current-delta/PG green.

Plan IMPLEMENTATION_PLAN_vn_pr_3016_review.md; design Docs/Design/2026-10-02-vn-pr-3067-review-tests.md; ledger .superpowers/sdd/IMPLEMENTATION_PLAN_vn_pr_3016_review/progress.md. New reports/reviews/logs/XML/diagnostics/verification under its human-fixes-20261002 directory. Exact plan-before.md/task-before.md archived (SHA256 af4f0f4761466cb4ad02cf1a3fd218b0280d8091eaed60e5dc94ddbfb1b9707f / 9ec96ad970be40fe77711b0577a97ce44c5b5625d0117e10e1ab6ece4e7bc1a8). Official CLI frontmatter formatting changes qualified; metadata/body outside notes preserved. Prior Git history and all old frozen records remain. Historical raw loss535/1077references remains qualified; no recovery or old manifest refresh.

Uncommitted/unpublished fixes; no commit/push/public replies/resolutions/body changes/fetch/rebase/merge/cleanup. Parent guarded Verification-only approval, child OWN human Change summary and safe current-dev integration/full exact-head review/7requiredpasses remain separate. AC5/AC6/DoD pending. Preserve checkout/main/backups/already-applied stashes. ADR required:no; ADR002/004/006 govern. Automation native prompt condensed84725to6860characters with exact before/after TOML archives and equality/status/schedule/target verified; ACTIVE/QUIET on known blockers.

Human authorization 2026-10-02: after the presented next actions (publication of reviewed Tasks64/65/67, held Task66 bootstrap fix, parent current-dev integration, then fresh exact-head review/CI), requester explicitly replied APPROVED. This supersedes the separate Task66 implementation/publication/integration hold. Execute the narrow exact-AST/empty-arguments SQLite AUTOINCREMENT guard compatibility fix with real UsersDB regression and rejection controls, independent SPEC/QUALITY review, scoped verification, publication and safe parent integration. Preserve existing requester Change summary, frozen evidence, backups and applied stashes. Child OWN human-written Change summary and guarded parent Verification-only body edit remain separate. Merge normally only after genuine exact-head review, seven required passes, strict current-dev integration and human gate; AC5/AC6/DoD remain pending.

Task66 implemented under newest explicit approval: exact SQLite AutoIncrementColumnConstraint with empty AST args; no other guard/DDL/dependency/workflow changes. Matching SQLGlot30 RED2 intended failures then127pass; SQLGlot29 same127pass. Independent P2 observation ordering fixed, affected native bootstrap1pass on30/29 (18.45s/16.47s); managed-boundaries30 21pass. Ruff/format/compile/diff pass. Unfiltered Bandit49/49 normalized equality, inherited8 B105 and3 B608 plus38 pytest B101; not blanket clean. Dalton independent SPEC/QUALITY PASS on scoped re-review, closed. Evidence task-66-approved-20261002; frozen reports preserved. Approved Tasks64/65/67 committed9a5271aa2e; source publication follows Task66 commit. Parent/dev merge-tree now reproduces7 conflicts including authored retry and reload recovery; no safe integration credit yet. AC5/AC6/DoD and child OWN summary remain pending.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
