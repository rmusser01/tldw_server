---
id: TASK-13594
title: Atomically adopt prepared native fork candidates into original child claims
status: Done
labels:
- chat
- backend
- attachments
dependencies:
- TASK-13592
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue approved H2 Task2.1 adoption inside the existing native asset store. Install only the exact prepared candidate into the protected original child under operation/workspace/conversation/candidate locks and confirmed unreleased quota; reject stale, foreign, released or reclaiming state. No byte or external quota work enters the caller transaction, and public H2 remains disabled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Real prepared uploaded bytes install one exact live claim in the original protected child without another allocation or byte mutation.
- [x] #2 Same-attempt replay is exact, foreign target/owner/generation and reclaimed or unconfirmed quota refuse before new claims; caller rollback preserves preparation.
- [x] #3 Causal red/green, affected native transaction/I/O/fork checks, scoped security/static, two bounded reviews and normal exact-owned hooks pass while broader retention/read/send gates remain open.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. ADR049/050, proposed070 and approved H2 section8/Task2.1 govern private claim adoption. Reuse existing operation, candidate, claim, protected conversation and quota-intent tables; no schema, parser, dependency, public protocol or capability change. Root sole runner/editor; accepted two independent reviewers remain read-only.
Corrected actual adoption suite:19 passed,19 official unavailable PostgreSQL skips. Initial17 missing-method failures are causal RED; one wrong-scope FK setup failure is excluded. Initial connected13 schema/fixture failures are diagnostic, corrected by removing the nonexistent conversation system_operation_state column and using a real workspace FK. Both authorized independent source reviews CLEAR; final five-file connected suite is running. New-test formatting preserves exact AST. Review Docs/Reviews/CHAT_CANDIDATE_ADOPTION_13594_2026_10_11.md; plan Docs/Plans/IMPLEMENTATION_PLAN_chat_candidate_adoption_13594.md. No public H2, references/readiness receipt or production caller added.
Final supported3.12 connected187 passed/186 official PG skips; final21 adoption controls/21 official PG skips includes deterministic actual cleanup/adoption ordering. Production unchanged after connected run. Initial post-BEGIN test marker timeout excluded; corrected barrier retains original transaction manager. Ruff0/0, two-production Bandit0/0/errors0 and supported three-file compilation pass. Both source reviews CLEAR; test-only clarification preserves real locks and records expired-lease overlap. Normal exact-owned hooks pending; public H2/retention/ref/read/send/client gates stay open.
Eight initial exact-owned normal hooks exit0 with three Python ASTs exact. Both final test-only reviews CLEAR; corrected barrier observes transaction-entry intent, not completed lock acquisition, and cleanup-winner also includes expired lease. All bounded stages complete; only own completed plan retired to /private/tmp/IMPLEMENTATION_PLAN_chat_candidate_adoption_13594_completed.md. Final seven-file normal metadata hooks/local commit follow. Broader master and public H2 criteria remain open.
Evidence classification correction during dependent TASK13595: connected187 passed/186 skips comprise185 unavailable official PostgreSQL cases and one SQLite control explicitly requiring PostgreSQL row locks. Initial connected125 skips includes that SQLite control. Actual execution totals are unchanged; original source/static/hooks/commit qualification remains bounded. Adoption-only21 skips are all official unavailable PostgreSQL. Review record corrected; no new physical PostgreSQL acceptance.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Installs one exact prepared candidate claim and adopted state inside the original protected-child caller transaction; exact replay and rollback preserve bytes and the original single quota charge. Connected187 passed/186 skips (185 unavailable official PostgreSQL plus one SQLite PostgreSQL-lock-only control), final21 adoption controls/21 official PG skips, scoped Ruff0/0, Bandit0/0/errors0, supported three-file compile0, two production/test-only CLEAR and normal owned hooks0. No public H2, production caller, references/readiness receipt, physical PostgreSQL, client/build/AT/study acceptance is claimed. Bounded review: Docs/Reviews/CHAT_CANDIDATE_ADOPTION_13594_2026_10_11.md.
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
