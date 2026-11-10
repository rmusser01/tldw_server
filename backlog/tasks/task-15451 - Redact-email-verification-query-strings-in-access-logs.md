---
id: TASK-15451
title: Redact email verification query strings in access logs
status: In Progress
references:
- https://github.com/rmusser01/tldw_server/pull/3428
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Protect the verification token in the Uvicorn request target while retaining the route, HTTP version and status.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The verification query is redacted with and without a trailing slash.
- [x] #2 Neighboring routes and existing redaction targets retain their behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Extends the existing access-log redaction contract. Plan: reproduce the formatted Uvicorn record, extend the sensitive-query matcher and run adjacent tests plus Bandit.
November 7 requester follow-up explicitly authorizes fixing inherited CI failures. Port the independently reviewed shared corrections from TASK-15450: explicit trusted bootstrap membership context for storage-resolver setup, module-scoped registration of the already-declared pytest-mock plugin under disabled autoload, and regenerated stale WebScraping import inventory. Preserve all assertions and runtime redaction behavior. Validate the exact port with targeted tests, changed-file mypy, Ruff, Bandit comparison and canonical task checks, then push for fresh protected CI. ADR required: no; restores existing fixture and generated-artifact contracts. No merge or recovery operations.
The reviewed four-file shared repair is byte-identical to TASK-15450: resolver setup includes trusted context, the accounting module registers pytest-mock, and inventory artifacts have only the three corrected source references. On this branch 12 resolver cases passed across SQLite and isolated PostgreSQL 18, and 73 combined MCP/inventory tests passed. Initial sandboxed PostgreSQL attempts could not reach the task-owned cluster; the authorized local-network runs passed all 12 without skips. The expanded changed-file gate also exposed an inherited access-log return typing error: dispatch now uses Starlette RequestResponseEndpoint instead of unparameterized Callable, with identical runtime body. All changed Python files pass exact mypy 2.4.1 and Ruff; access-log redaction/JSON tests pass. Independent reviewer verified the annotation and identical shared ports. Bandit accounting findings match baseline exactly (49 test assertions); no new runtime finding. Fresh hosted CI will verify this head. Historical additional AuthNZ teardown/GC errors remain unproven, not claimed fixed. Waiver retained; no merge or recovery actions.
Additional teardown investigation found a concrete detached-pool lifecycle in the original PostgreSQL bootstrap test: bootstrap replaces the registered pool, but the test reuses its old closed reference and lazily reopens five unowned sessions. PID-only instrumentation reproduced this while the original test passed. Plan: verify that reacquiring the current registered pool before the post-bootstrap assertions prevents that specific leak, preserve all assertions, validate the safe synthetic bootstrap cases and review before pushing. A separate startup background refresh also appears to retain a pool reference; neither the complete historical teardown failure nor the later pytest GC abort is yet claimed fixed. No restore, replay or production fixture guard changes.
PID-only before/after proof confirms the test-only reacquire removes all five test-origin detached sessions. The remaining independent startup path also retains a stale pool across the seed call: startup_auth.init_auth_services caches the pool, seeding replaces it, then provider-override refresh reopens the old reference. Expanded minimal plan: refresh against the currently registered pool after seeding, add a focused replacement-pool regression, preserve reset and offline/recovery guards, and validate that the instrumented synthetic bootstrap lifecycle no longer reopens either detached pool. Mechanical lint/type corrections are allowed in these newly touched files. Independent review and current-head CI remain required; no claimed cause for the later native GC abort without evidence.
Independent review caught a failure-path regression in the first draft of pool reacquisition: a RuntimeError from the second lookup could be treated as optional setup by the outer startup wrapper. Correcting before commit by sharing the existing mandatory AuthStartupError/None acquisition guard, with a regression through the real wrapper and sanitized failure behavior. No unreviewed draft was pushed.
Final detached-pool repair verified and independently approved. Startup reacquires the registered pool after seeding; both acquisitions share the original sanitized AuthStartupError/None guard. Bootstrap assertions read from the currently registered pool instead of reopening a retired reference. PID-only synthetic proof shows both detached reopen paths eliminated (14 cleanup clients before, 9 after test-only correction, 4 after both corrections; remaining clients are handled by normal fixture cleanup). Seven replacement-failure cases through the real startup wrapper failed before guarded reacquisition and pass afterward. Final 44 startup/runtime tests and all four official PostgreSQL bootstrap tests pass with clean process exit; independent reviewer ran 11 additional success/failure cases and reproduced the formerly swallowed RuntimeError as a fatal sanitized AuthStartupError. All seven changed Python files pass exact mypy 2.4.1 and Ruff; canonical task/diff checks pass. Bandit runtime findings remain zero, existing test findings unchanged except four new regression assertions. Earlier resolver12, MCP/inventory73 and accesslog13 results remain applicable. Historical ObjectInUseError/GC causality is not claimed proven; fresh hosted CI must validate the combined correction. No fixture termination/restore guards were changed, and no recovery operation was run.
Required E2E Critical run 42389342656/job128462508044 failed only the Rowan citation-membership journey (45 passed). Trace proves the random fixture UUID 67d9a83b-3374-4043-8077-7503cfcef1b4 contains phone-shaped 3374-4043-8077; the real existing RAG content-policy filter redacts it in both title and body, so the unchanged literal-source assertion correctly fails. Direct execution of apply_content_policy reproduces the identical redaction. Plan: change only UUID separators in this synthetic fixture namespace to underscores, preserving UUID entropy and all source/citation assertions. Validate exact CI UUID and numeric edge cases through the real privacy filter and grounding oracle, run existing grounding tests/lint/typecheck, independently review and push. ADR required: no; test fixture correction leaves privacy policy, product behavior and acceptance contract intact. No retries, timeout changes or recovery operations.
The exact CI UUID and a fully numeric valid UUID now pass the unchanged literal-source oracle after the real backend apply_content_policy; before the fixture edit the same harness failed with the exact Citation 1 excerpt is not in the canonical source assertion. A real phone-number control still redacts. All 50 existing UAT390 grounding tests and scoped ESLint pass. Independent read-only review found no issues, verified all downstream namespace uses, and checked 10004 UUIDs including numeric edge cases against the PII patterns; replacing separators is reversible and preserves uniqueness. Only fixture ID formatting and this task record change. Bandit not applicable to this TypeScript-only follow-up; previous Python security evidence remains. Complete browser journey awaits fresh current-head CI.
Final fixture verification: standard full WebUI TypeScript check passed with the repository 8192 MiB setting and Node20 after recreating missing workspace dependency links; all 50 grounding tests, scoped ESLint, canonical task formatting and diff checks pass. Initial local test/typecheck attempts were blocked by missing worktree dependency links rather than code diagnostics. Independent review approved the one-expression fixture fix. Temporary task-owned dependency links removed after validation. Full real-browser journey was not rerun locally; fresh GitHub current-head E2E remains required.
Serial dev rebase, November 9: retained current upstream WebScraping inventory, native MCP accounting doubles, and trusted membership context. The pool is reacquired before the upstream provider snapshot refresh. Changed-file mypy exposed inherited PostgreSQL schema annotations; typed the existing pool and migration call table without changing runtime validation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Email verification query redaction remains intact. Authorized CI follow-up fixes resolver setup, mock-plugin registration and stale import inventory, plus two proven AuthNZ detached-pool lifecycles. Startup and test readback reacquire the registered pool after seeding; original sanitized fail-closed acquisition guards protect both lookups. Local validation passes:12 SQLite/PostgreSQL resolver cases,73 MCP/inventory checks,13 access-log tests,44 startup/runtime tests and4 official PostgreSQL bootstrap tests. Seven changed Python files pass exact mypy2.4.1/Ruff; independent review found no remaining issue. No new runtime Bandit findings. Hosted CI is required on the new head; exact historical GC-abort causality remains unproven. Requester waiver retained; no merge, beta change or restore.
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
