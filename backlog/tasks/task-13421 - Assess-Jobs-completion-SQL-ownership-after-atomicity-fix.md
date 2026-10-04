---
id: TASK-13421
title: Assess Jobs completion SQL ownership after atomicity fix
status: In Progress
created_date: 2026-10-03 01:33
references:
- TASK-13215
- https://github.com/rmusser01/tldw_server/pull/3092#discussion_r4171067327
documentation:
- Docs/ADR/058-jobs-completion-row-identity.md
- Docs/superpowers/specs/2026-09-07-jobs-completion-row-identity-atomicity-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review Qodo PR #3092 finding dd3ff113-e31c-4ca9-a49d-48c73a4273eb against the existing Jobs facade and the repository DB_Management-only SQL guidance. The focused TASK-13215 lock/UUID fix intentionally preserves the current transaction owner and excludes extraction. Evaluate a separately designed boundary for completion SQL without splitting the authoritative lock, identity guard, RLS context, counters or outbox transaction. This task does not authorize broad migration or behavioral fixes during strict extraction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Inventory existing completion SQL ownership and approved strict-extraction boundaries before proposing changes.
- [ ] #2 Propose an explicit transaction/RLS-preserving ownership decision with risks and verification gates, for separate human approval.
- [x] #3 Keep behavior changes, caller migration and historical repair separate from extraction.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-04 read-only design refresh (implementation not authorized yet):
- Isolated managed worktree: /Users/appledev/.codex/worktrees/jobs-completion-ownership/tldw_server; branch codex/jobs-completion-ownership; baseline live dev bf8f2ad6a42ad6396376020876a5f6a709ec6b34. Root dirty checkout and historical foundation branch remain untouched.
- Prerequisite PR #3092 merged as 6c5da178252d14dc0d997d97988ccaefdf86bc89. git diff --quiet from that merge to baseline shows Jobs runtime and tests unchanged.
- Context exploration complete: historical 2026-08-23 foundation design, accepted ADR-058, current complete_job, SLA staging, RLS cursor setup, shared persistence/commit helpers, existing backend operations and DB_Management Jobs fragments/queue repository plus ledger reviewed.
- Strict-extraction scope reconfirmed from human directions: foundation then routing PR, no behavioral fixes, fail/cancel/batch extraction, generic terminal engine, schema/config/API changes, broad caller migration (TASK-13216) or historical repair (TASK-13217). Ordinary defects receive separate follow-ups; critical isolation/data-loss/atomicity defects stop the slice.
- Stale-design corrections required: include optional expected_uuid without eager coercion; preserve SQLite BEGIN IMMEDIATE and PostgreSQL RLS-aware FOR UPDATE; identity mismatch precedes token replay; retain raw null/empty/noncanonical stored UUID and null-safe mutation/replay predicates. Current queued completion classifies the locked row directly (no second domain lookup or processing-update fallback); final result serialization occurs once after transformation. Do not restore old redundant SQL/serialization.
- Ownership inventory: complete_job owns authoritative lookup, mutation, counter SQL and explicit applied commit inside facade-owned connection/cursor contexts. _stage_completion_sla_breach owns attachment/savepoint SQL; _insert_lifecycle_event and _reconcile_lifecycle_counter_row own shared event/reconciliation SQL; _commit_postgres_transaction rejects INERROR. Non-durable effects execute after explicit commit but before context exit/close. Early false/replay returns still pass through context exit. SLA policy is separate from SLA persistence; established-savepoint write failure is best effort, savepoint control failure fatal.
- Architecture proposal, NOT approved: A (recommended) put completion-specific contracts, backend atomic persistence operations and required SQL helpers under DB_Management; manager supplies the already-open connection/RLS cursor plus lazy policy callbacks, retains connection contexts and non-durable dispatch. No repository-owned connection, nested transaction, SQL-by-SQL facade orchestration or DB-to-JobManager runtime import. B: Jobs orchestration over granular DB_Management methods is consistent with older operations but increases split-protocol/ordering risk. C: leave SQL in Jobs operations needs an explicit policy exception and retains the ownership debt.
- Retain two PRs: foundation supplies characterization, completion-only contracts and mechanical shared-helper moves, with complete_job routing unchanged; routing follows only after foundation merges. Contracts introduce no eager result copy/serialization/UUID normalization; optional metric-preparation failures must remain suppressed exactly as now rather than becoming mandatory fact-validation errors. Full Jobs and required real PG plus Ruff/Black/compile/whitespace and baseline-compared Bandit are implementation gates; focused baseline is not full extraction qualification.
- ADR check: ADR required yes if A is approved because it establishes a durable completion persistence module boundary. A new complementary ADR should link ADR-058, not rewrite or supersede its unchanged locking/identity rationale; allocate its number against then-live index.
- Fresh baseline: activated shared project venv; RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest -q on completion_row_identity_atomicity, lifecycle_event_atomicity, completion_idempotent_sqlite/postgres and rls_postgres: 86 passed, 227 warnings, no skips, exit 0. XML /tmp/task13421_completion_baseline.xml. No source/test edits; Bandit not applicable to tracking-only checkpoint.
- Current repository task-editor cutover ADR-059/AGENTS reviewed; use backlog-py for canonical task updates, not legacy Node/MCP writes. TASK-13421 uniquely resolves to this assessment; obsolete foundation TASK-13112 collides with unrelated live work and will not be reused.
Brainstorming checklist: context exploration complete; visual companion not needed for text-only boundary choice; existing intent/constraints established; approaches prepared for human selection; architecture approval pending; written spec, spec self-review, human file review and implementation plan remain pending. Task stays In Progress, with no completion claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
