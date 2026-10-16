---
id: TASK-13818
title: Fix intermittent workflow WebSocket rejection
status: Done
labels:
- backend
- workflows
- testing
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate the backend test_ws_events_stream rejection first observed in PR #3250 CI run 38867120384. Reproduce the cause against latest dev, implement the smallest verified repair without weakening authentication or ownership, independently review a new PR, and merge after checks pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root cause is demonstrated by a deterministic regression or bounded diagnostic evidence
- [x] #2 Repair preserves token validation, run ownership, and WebSocket event behavior
- [x] #3 Connected tests, scoped security and repository checks pass
- [x] #4 Independent exact-head review and new human Change summary complete before merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Baseline origin/dev f87446f05d442146ee180277418b07a2e45b4683. Failed original shard: test_ws_events_stream rejected before accept; rerun passed 60/60. Prior isolated Python 3.11 standalone and ordered-prefix probes passed 1/1 and 40/40, so root cause remains unconfirmed. Investigation will compare Python 3.12, full collection/order, token manager identity, close code, database identity and owner without logging secrets. ADR required: no for anticipated local correctness/fixture repair preserving current auth policy; reassess if a durable security rule changes. ADR-059 governs task editing.
Plan: Docs/Design/IMPLEMENTATION_PLAN_workflows_ws_rejection_13818.md. Reusing clean isolated managed worktree on new branch codex/workflows-ws-rejection-20261016. Primary checkout remains untouched. Existing Python 3.12.13 environment allows version comparison with CI 3.12.15. Prior private-copy diagnostic logs retained; no auth or owner cause inferred from successful rerun.
Reproduced real root defect on Python3.12.13 with exact CI FastAPI0.142.4/Starlette1.8.0: concurrent real WorkflowsDatabase.get_run calls return other run IDs or empty/None owners (private log /private/tmp/ws13818-query-race-red.log). Actual authenticated WS probe then rejects its correct owner before accept with4403 on iteration0 (/private/tmp/ws13818-ws-concurrent-red.log). Matches confirmed CPython shared statement-cache issue https://github.com/python/cpython/issues/118172. This demonstrates the defect can cause the original symptom; exact original CI attribution remains unprovable because its close code was omitted. Full original shard60/60,30cold and100warm WS checks pass without contention. Add real DB identity and owned/foreign/missing WS concurrency regressions before minimal two-site cached_statements=0 repair. Independent source review confirms only primary and optional pooled connections are reusable; independent read stays thread-confined; PostgreSQL/auth rules unchanged. ADR required:no, local correctness repair preserves existing policy.
Repair and regression verification: new tests fail 4/5 before repair and pass 5/5 after; 116 connected workflow tests pass, final lint-clean rerun 64/64 passes. Linux Python3.12.14/SQLite3.46.1 real DB comparison: baseline 72 identity mismatches in1600 reads, fixed zero. Scoped Ruff and production Bandit clean; test Bandit clean excluding assertion-only B101. Independent working-tree review found no actionable issues; exact-commit confirmation, hosted checks and human Change summary remain integration gates.
PR https://github.com/rmusser01/tldw_server/pull/3252 created. Human requester supplied Change summary explaining incorrect run/owner reads, reusable SQLite cache disable, preserved auth and behavior regressions. Exact-head independent review of3913015403 found no findings. Hosted backend-required exposed scope-predicate baseline line numbers shifted by the new explanatory comment, with all three predicate ASTs unchanged; refresh recorded locations and verify ratchet before final integration.
Implementation and review complete for PR #3252. Independent final source/baseline review at91347b931b found no findings. All13 scope-ratchet tests pass after refreshing exactly3 moved locations; predicate ASTs match dev. Human Change summary is recorded verbatim in PR body. Local plan is retired into this task/history; hosted exact-head gates and merge are operational follow-through tracked by PR #3252 and remain pending, not claimed successful here. Extra statement preparation is not benchmarked; original historical failure cannot be attributed conclusively because its close code was absent.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Disabled cached statements on both reusable workflow SQLite connection sites, fixing reproduced cross-run/cross-owner read corruption and false owner WebSocket refusal. Added real concurrent DB identity coverage for pooled/unpooled connections and actual owned/foreign/missing WebSocket authorization tests. Regression-first failures turn green;116 connected tests,64 final source tests,13 scope-ratchet tests, Linux72-to-zero identity comparison, Ruff, hooks and scoped Bandit pass. Independent code review is clear and requester rationale is recorded. Hosted checks and merge remain required before integration; PostgreSQL and auth policy are unchanged.
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
