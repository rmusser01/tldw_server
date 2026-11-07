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
Follow-up scope approved for merge readiness: repair the related Admin real-backend validation harness in postAdminE2EJson, without changing production authentication or CI gates. Plan: add failing helper regression, obtain the server-issued token from the existing public liveness GET and send the matching csrf_token cookie plus X-CSRF-Token on support writes, verify focused tests and actual local middleware/seed/reset with isolated synthetic data, then request review before commit. This is an inherited test setup compatibility repair required to exercise the Admin PR, not an alert-code regression. ADR not required: uses the existing CSRF public contract. Existing callers are global setup, SeededSession, monitoring-authority and backup-schedules; the shared helper reaches all callers without call-site changes.
Harness repair verified: postAdminE2EJson now GETs the public /health route, reads the server-issued X-CSRF-Token and echoes that exact value as csrf_token cookie plus X-CSRF-Token header. Node fetch has no automatic cookie jar. The middleware explicitly emits the same URL-safe token through both Set-Cookie and X-CSRF-Token; no token is fabricated and no API-key/Bearer auth exemption is used. Both requests use the existing configured backend origin. No reusable Node-side bootstrap was present; the shared helper fixes all setup callers. A live probe revealed /api/v1/healthz requires authentication, so bootstrap uses the actual public /health registration. HTTP status/body errors remain visible, and a server explicitly configured without CSRF remains supported. Regression red: seed/reset rejected with CSRF403 before implementation; green: 27 focused tests passed, Admin TypeScript and scoped ESLint passed. Node 26 native-fetch probe against a local full backend with isolated synthetic databases, CSRF_ENABLED=true and CSRF_BIND_TO_USER=true verified missing-pair seed rejection (403), matching real Set-Cookie/header token, successful seed/JWT bootstrap/bodyless reset, and wrong-support-key rejection (403). No beta or deployed server touched. Independent reviewer passed the final /health diff; production auth, CI gates and inherited UI assertions are unchanged. Bandit not applicable: TypeScript test helper/test and task notes only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Alert behavior verification remains green (17 focused tests); the related inherited Admin E2E setup incompatibility now has an 11-line shared-helper repair and four regressions. Fresh combined run: 27 tests passed; Admin TypeScript, scoped ESLint and diff checks passed. A Node native-fetch probe on a local synthetic full backend passed seed, JWT bootstrap and reset with CSRF enabled and user binding enabled; omitted CSRF and wrong support keys still returned 403. Independent review passed. Original CI unit ratchet reported inherited=40 regressions=0; required CI must rerun on the repaired head. No full browser UAT, deployment, production-auth change or gate weakening. The requester waived the human-written change-summary requirement; merge remains subject to required gates.
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
