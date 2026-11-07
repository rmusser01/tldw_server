---
id: TASK-15429
title: Report confirmed monitoring alert acknowledgments
status: In Progress
references:
- https://github.com/rmusser01/tldw_server/pull/3429
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The dashboard and monitoring view need to distinguish reading an alert from acknowledging it, and report confirmed bulk write outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A read-only alert remains actionable until it has a confirmed acknowledgment.
- [x] #2 The dashboard interprets the API list response and reports failed or partial bulk results truthfully.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: exercise normalizer and dashboard with synthetic API responses, implement confirmed bulk results, run focused tests and review. ADR not required; the public monitoring API contract is already established.
The monitoring endpoint exposes separate read and acknowledgment fields. Bulk results validate the response status, ID and merged item timestamp, then refresh the list. Dashboard refresh now uses the shared in-flight guard. Reviewed implementation and scoped diff; no API or database changes.
CI follow-up for head ff4c413b680039fdfb2e0b5441835e9a620bc442: frontend-required run 42315377159 job 128236102165 passed the Admin Vitest ratchet with inherited=40 regressions=0 (head: 40 failed/786 passed; exact base 2f402bd7e34015ca6823793fd05cee68a16092b2 replay: 41 failed/88 passed). The job failed later during real-backend JWT global setup: POST /api/v1/test-support/admin-e2e/seed returned 403 CSRF token validation failed before browser tests. Inherited commit 24ef09e7b1a1bc54f5380ea1e1c47a98f7b1ad27 enabled CSRF by default even in TEST_MODE, while the unchanged seed helper sends no CSRF cookie/header. The entire backend, E2E harness, package/lock, Playwright config and workflows are unchanged from that exact base. An isolated invocation of the real middleware loaded from each commit reproduced the same 403 and confirmed identical source bytes. Current public dev f93397c17dfe6d00e295d0031d0db111d33d1333 has no intervening changes to these paths, so rebasing does not resolve this inherited blocker. No production code or test-gate changes made; user scope is failures introduced by this PR. Fresh focused alert tests: 17 passed; Admin TypeScript passed. CI run: https://github.com/rmusser01/tldw_server/actions/runs/42315377159/job/128236102165
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Original verification: 24 focused checks passed and Admin TypeScript passed; wider monitoring/dashboard run passed 116 of 119 with the three failures reproduced on dev. Fresh CI follow-up: 17 focused alert tests and Admin TypeScript pass; CI production build passes, and the unit ratchet reports zero regressions. frontend-required remains blocked by the inherited real-backend seed CSRF403, reproduced on exact base and head. No local full backend/browser UAT; no unrelated inherited fix or gate weakening. Bandit is not applicable to this notes-only follow-up. The requester waived the human-written change-summary requirement for this PR; merge review and required gates remain pending.
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
