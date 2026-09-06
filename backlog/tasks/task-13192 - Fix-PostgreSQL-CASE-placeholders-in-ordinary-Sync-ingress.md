---
id: TASK-13192
title: Fix PostgreSQL CASE placeholders in ordinary Sync ingress
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:50'
updated_date: '2026-09-06 00:18'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ordinary PostgreSQL Sync envelope insertion fails because the shared placeholder converter treats CASE result binds as JSONB operators. Restore backend parity without changing Sync authority or conflict policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CASE-expression placeholders convert correctly while JSONB operators and quoted text remain unchanged
- [x] #2 Real PostgreSQL and SQLite ordinary Personal Context ingress and monotonic domain watermarks are verified
- [x] #3 Ingress repair tests use ordinary envelope insertion and targeted tests and static checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md. Reason: preserve existing SQL and Sync contracts; correct shared parameter translation only. First add failing CASE and ordinary-ingress regressions, then minimally correct converter keyword handling, remove the raw-seed workaround, and run targeted backend plus PostgreSQL-required ingress tests. Record verification and review before closing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected directional CASE placeholder recognition in the shared backend converter; preserved JSONB operators before CASE and after END. Ordinary SyncV2Store insertion now seeds receipt repair coverage, with real PostgreSQL/SQLite successive insertion, replay and nondecreasing domain-watermark assertions. Plan: Docs/superpowers/plans/2026-09-05-postgres-case-ingress.md. ADR check: no new decision; existing ADR-002 governs unchanged Personal Context authority. RED: 11 parser failures and real PostgreSQL ordinary insertion failure, SQLite passed. Final focused verification: 109 passed (placeholder helpers, backend utilities, dual-backend ingress repair, bootstrap), TLDW_TEST_POSTGRES_REQUIRED=1. Ruff touched files and test formatting passed; production Bandit no findings; diff check passed. Independent plan and correctness review approved (reviewer additionally ran 18 parser tests). Broader selected store run: 94 passed, one pre-existing fake-backend receipt test failed with personal_context_link_binding_stale; reproduced after loading unchanged HEAD converter in memory, and neither that test nor Sync_DB changed. No full sweep, capability enablement, PR or merge. Added incident-based testing lesson. Backlog MCP search hung; CLI fallback used.

Publication: opened https://github.com/rmusser01/tldw_server/pull/2909 against dev from reviewed source 6363466d07. Fresh pre-publication placeholder check: 18 passed, exit 0, existing warnings; diff check passed. Dependent conflict PR: https://github.com/rmusser01/tldw_server/pull/2910. No rebase or merge; current dev integration and human Change summary remain pre-merge steps.

PR review round: rebase on current dev; verify all six Qodo findings, add missing test annotations/docstrings/category markers, preserve meaningful lower-sequence coverage, and document fixture/abstraction decisions with evidence. Run PostgreSQL-required ingress and parser tests, static checks and diff review before publication. Existing ADR-002 remains applicable; no new ADR required for test compliance fixes.

Review evidence: rebased onto server dev 946e591ee9. Added CASE test type hints, docstrings and unit markers plus envelope/watermark test docs. Replaced direct private domain updater with public dataset re-enrollment; retained narrow read-only SQL assertion of persisted watermark so an envelope cursor cannot hide domain-state regression. No production inspection helper added solely for a test. The global pg_database_config fixture depends on function-scoped pg_temp_db with unique create/drop lifecycle; AuthNZ isolated_test_environment is a separate AuthNZ-only fixture, so its substitution is not applicable. Fresh two-file parser/ingress selection: 42 passed, 6 existing warnings, PostgreSQL required, exit 0. Ruff, test formatting, production Bandit and diff check passed. Independent scoped review pending.

Independent scoped spec and quality review approved all review fixes and the documented inspection/fixture rationale with no actionable findings. All six Qodo items have a fix or evidence-backed disposition. Publishing this reviewed round only; no merge or capability change.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
