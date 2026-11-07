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
Required CI follow-up, 2026-11-07: frontend-required job 128319043150 reports inherited=40 and one extra failure in unchanged AIOpsPage renders top agents table. Its static heading renders before Promise.allSettled returns data; the test then synchronously queries rows. Plan: reproduce the original assertions with delayed synthetic metrics on exact base and head, change only the row readiness query to await real rendered data, run focused Admin tests/typecheck/lint and independent review, then push. All assertions, timeouts, product code and protected gate logic remain intact. ADR not required for this related validation-harness synchronization repair.
The entire AI Ops source/test tree is identical at exact base 2f402bd7 and head 7e2de2e2. A controlled 50 ms synthetic metrics response made the original top-agents assertions fail identically on both revisions, matching the CI missing research-assistant error. Waiting with findAllByText instead of getAllByText passes both controlled reproductions; all original names, counts, prices, headings and timeout settings remain intact. Final focused validation: 29 tests passed (8 AI Ops, 11 alert normalization, 6 dashboard refresh, 4 CSRF support helper); full Admin tsc and scoped ESLint passed. Independent review found no actionable issue and confirmed that session badges alone cannot satisfy the preserved distinct agent-price assertions. Bandit not applicable to this TypeScript test-only change. CI reported 40 inherited failures and this one race; no broad baseline repairs, product changes or gate weakening. Required CI must run on the new commit.
November 7 requester follow-up explicitly authorizes fixing inherited CI failures. Port the independently reviewed shared corrections from TASK-15450: explicit trusted bootstrap membership context for storage-resolver setup, module-scoped registration of the already-declared pytest-mock plugin under disabled autoload, and regenerated stale WebScraping import inventory. Current auth shard reproduces the baseline six errors; identical MCP/inventory paths also need the deterministic fixture/artifact corrections. Preserve all assertions and Admin behavior. Validate the exact port with targeted tests, changed-file mypy, Ruff, Bandit comparison and canonical task checks, then push for fresh protected CI. ADR required: no; restores existing fixture and generated-artifact contracts. No merge or recovery operations.
The reviewed four-file shared repair is byte-identical to TASK-15450: resolver setup includes trusted context, the accounting module registers pytest-mock, and import inventory artifacts have only three corrected source references. On this branch 12 resolver cases passed across SQLite and isolated PostgreSQL 18, and 73 combined MCP/inventory tests passed. Initial sandboxed PostgreSQL attempts could not reach the task-owned cluster; the authorized local-network run passed all 12 without skips. Exact mypy 2.4.1 and Ruff pass all two changed Python files; independent review verified the port byte identity and all existing assertions. Bandit accounting findings match baseline exactly (49 test assertions); resolver findings are baseline-identical. No Admin product changes in this repair; prior 29 focused Admin tests/typecheck/lint evidence remains applicable. Fresh hosted CI required after push. Waiver retained; no merge or recovery actions.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Confirmed alert acknowledgments, CSRF bootstrap and async AI Ops row readiness remain validated. The requester-authorized CI follow-up also repairs inherited backend resolver fixture context, MCP plugin registration and stale import inventory. This branch passes 12 SQLite/PostgreSQL resolver cases and 73 MCP/inventory checks, exact changed-file mypy and Ruff; all shared repair bytes and preserved assertions were independently reviewed. Prior 29 focused Admin tests and TypeScript/lint validation remain applicable. New-head hosted CI is pending; no merge, beta change or restore. Requester Change summary waiver retained.
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
