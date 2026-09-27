---
id: TASK-13240
title: Prevent synchronous RBAC lookup from stalling live PostgreSQL requests
status: Done
assignee: []
created_date: '2026-09-10 05:46'
updated_date: '2026-09-10 06:22'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2939'
documentation:
  - Docs/Reviews/ISSUES_2935_2938_LIVE_UAT_2026_09_10.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Discovered during TASK-13239 live UAT for PR2939: WebUI periodic requests trigger PostgreSQL DDL while a synchronous RBAC role SELECT blocks the API event loop, so even /health and API-key Save time out. Native stack sample shows psycopg wait_c on uvloop main thread; pg_stat_activity in the isolated UAT database shows the SELECT waiting behind an asyncpg schema transaction. Investigate the exact async caller and repair with regression coverage, then repeat the browser Save/reload scenario.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RBAC lookups used by asynchronous request handling do not block the event loop while PostgreSQL locks are held
- [x] #2 A deterministic regression reproduces the stall before the fix and verifies concurrent event-loop progress afterward
- [x] #3 Relevant regression tests, lint, Bandit, review, and live WebUI Save/reload acceptance checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the live synchronous RBAC lock stall with an isolated PostgreSQL regression.
2. Move blocking request-path RBAC/database reads off the event loop while preserving sync repository contracts.
3. Run regressions and security/lint checks, review, then repeat WebUI Save/reload and document live results in the PR UAT report.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Native sample identified psycopg wait_c on the uvloop main thread. Isolated pg_stat_activity: role SELECT waits on a relation lock held by an asyncpg schema transaction after CREATE rbac_user_rate_limits. Cancelling that read alone did not restore health within5s; investigate all synchronous RBAC helper reads. Related quota fix is committed as74f86e5c76.

Live replay with the fix passed: real roles-table lock observed while authenticated /api/v1/buddies waited; /health200 in7ms before unlocking; authenticated request200 after unlock. WebUI Save received profile200, showed success, and reloaded successfully; final sanitized persistence evidence being captured. Scope is complete _enrich_user_with_rbac offload at JWT/API-key callers.

Independent review found concurrent cold-cache construction and configuration publication races after RBAC moved to worker threads. Added shared initialization/reset RLock and public-entrypoint concurrency regressions; four initially failed, then42 related tests and4 PostgreSQL lock tests passed with clean exits. Final review identified a reset_lazy gap before the cold getter, now receiving its own red-green regression. Fresh live API: four simultaneous authenticated requests200, exactly one shared DB construction, profile200, health200. Real role-lock replay: health200 in8ms while authenticated request waits, then200 after unlock. Exact Node20.20.2 runtime also passed fresh-login JWT and invalid-cookie matrix.

Final review also reproduced a lazy-reset gap between configuration lookup and cold database creation. Added a fifth public-entrypoint regression, observed its stale-path failure, then rechecked initialization inside the existing RLock. The final 26-test configuration/backend batch passed with exit0; previous 42-test batch and four real PostgreSQL lock regressions also passed. Final fresh API replay: four concurrent authenticated requests200, one DB construction, profile200, health200; held-role-lock health200 in7ms, waiting authentication200 after unlock. Reviewer reran its original probe and found no remaining actionable issues. Ruff, compilation, formatting of new tests, whitespace checks, and scoped Bandit pass; three pre-existing low Bandit credential-type literals in User_DB_Handling.py are unchanged. Standard UAT fixture exited0, its database was confirmed absent, all owned ports and browsers stopped. No external LLM or full-suite qualification claimed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved complete synchronous RBAC enrichment for JWT and API-key requests to worker threads so PostgreSQL lock waits cannot stall the API event loop. Protected shared database/configuration construction and test resets with a reentrant lock, including lazy-reset reinitialization, while preserving lock-free warm cache reads. Added four real PostgreSQL contention regressions and five public-entrypoint concurrency regressions; red-green, live UAT, independent review, lint, compilation, and scoped security checks completed. Evidence and limitations: Docs/Reviews/ISSUES_2935_2938_LIVE_UAT_2026_09_10.md. PR: https://github.com/rmusser01/tldw_server/pull/2939. The completed task-specific execution plan was removed after verification.
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
