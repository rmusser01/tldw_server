# UAT365 Knowledge QA scope cancellation implementation plan

> Execute inline with test-driven development. Root approved this bounded design and owns commits, the running UAT tracker, frozen sources, shared runtimes and native acceptance. Request independent review from diagnose_uat356_handoff; do not spawn.

**Task:** TASK-13260.277.15.
**Goal:** An unverified or revoked private QA operation settles safely without an error overlay, synthetic local thread, stale persistence, or automatic replay.
**Architecture:** Keep the existing authority hook and scoped-client enforcement. Gate private provider actions on a verified current snapshot; return explicit cancellation from thread creation; contain scope failures around the complete query startup lifecycle. Public shared-token reads retain their existing unauthenticated authority predicate, and genuine same-owner failures retain local fallback.
**Tech stack:** React provider/reducer, existing ServicePrompt scope errors, Vitest/Testing Library.

## Stage 1: Causal tests
**Goal:** Reproduce the exact handled-console-error path and late private-startup denial.
**Success Criteria:** Actual provider and scoped-client tests fail on error logging/local fallback during unverified submission, and distinguish scope denial during thread creation, user persistence and stream startup from genuine persistence failure. Explicit current-owner retry succeeds.
**Files:** components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.authority.test.tsx; existing cancel/persistence suites as controls.
**Tests:** Hold authority loading; invoke actual SearchBar or private actions; assert no rejected action, no console error, no transport or synthetic thread. Inject the actual scope-change error at create/user-save/stream startup; assert settled cancellation, no downstream request/write, no failed-search history. Keep public token read and genuine local fallback controls.
**Status:** Complete
- [x] Add causal regressions and retain RED output.

## Stage 2: Provider repair
**Goal:** Handle expected ownership cancellation without weakening ownership checks.
**Success Criteria:** `createNewThread` returns null for unavailable authority or scope denial; all callers handle cancellation. The query's existing try/finally includes thread creation and first-user persistence. Scope errors cancel without adding failure history or local fallback; abort references are cleaned up. Public share reads still work without a private snapshot.
**Files:** components/Option/KnowledgeQA/KnowledgeQAProvider.tsx, types.ts and knowledgeQaClient.ts; authority and services knowledge-qa.private-scope tests.
**Tests:** Stage1 regressions, existing authority/canonical-authority/cancel/persistence/branch-share suites.
**Status:** Complete
- [x] Add private readiness checks and explicit nullable thread cancellation.
- [x] Include startup in query cancellation cleanup and contain expected scope failures in private-action callers.
- [x] Run targeted tests to GREEN, preserving same-owner error fallback.

## Stage 3: Verify and review
**Goal:** Produce a bounded independently reviewed candidate.
**Success Criteria:** Focused tests pass; scoped lint/type diagnostics add no errors; source review confirms no stale account writes or new public-share restriction.
**Tests:** Local Vitest, scoped ESLint and TypeScript diagnostics. No Python changes: Bandit is not applicable.
**Status:** In Progress
- [x] Run surrounding suites and inspect changed source.
- [ ] Complete root's independent review and resolve concrete findings (reviewer reassigned from diagnose_uat356_handoff).
- [x] Update task15 via official CLI with evidence; leave native acceptance to root.

## Retained evidence

- `.tmp/post2970-full-20260920/sqlite-multi-121-qa-error-overlay.txt`: exact console.error plus empty pageErrors; original task's unhandled-rejection characterization was inaccurate.
- `/private/tmp/uat365-diagnostic/result.log`: read-only probe reproduces resolved search, zero remote calls, console error and synthetic local thread while authority is unverified.
- `/private/tmp/uat365-existing-guards.log`: existing26 authority/cancellation tests pass and omit the unverified submission window.
- `/private/tmp/uat365-causal-red.log`: six new unverified/private startup and create/user-save/stream regressions fail against the original implementation.
- `/private/tmp/uat365-startup-red.log`: character search/list and conversation version/tag scope denials are swallowed by optional startup fallbacks (four causal failures). An additional initialization injection was removed: the real scoped client's initialize is a local guard, so mocking the underlying transport initializer did not exercise that path.
- `/private/tmp/uat365-result-persistence-red.log`: post-answer context scope denial logs an error and marks local fallback (one causal failure, eight controls pass).
- `/private/tmp/uat365-green.log`: four actual-provider authority/cancellation/persistence/branch-share suites pass, 51 tests. Explicit retries, genuine same-owner local fallback, retained answers and public-token reads remain covered.
- Initial surrounding sweep: `/private/tmp/uat365-surrounding.log`, 12 suites148 tests pass. Initial scoped ESLint: zero errors and11 unchanged provider warnings. UI TypeScript reports409 repository diagnostics; no touched-file diagnostics in the initial three-file scope.
- Root took independent review from diagnose_uat356_handoff because that agent was reassigned to UAT367. Root's initial provider/null-caller review found no issue.
- Additional actual-transport self-review found that raw server412 scope responses bypassed thrown-error handling. `/private/tmp/uat365-server412-red.log`: four causal failures (two real transport, two real provider) and two unrelated412/503 controls pass. Root approved QA-client-only normalization of exact scope-code412; other raw response contracts and public shares stay intact.
- `/private/tmp/uat365-final-suites.log`: final13 suites209 tests pass, including33 authority and59 real private-transport cases. Scope covers provider/canonical-authority/streaming/cancel/persistence/history/branch/focus/defaults, SearchBar and FollowUpInput.
- `/private/tmp/uat365-eslint-final.json`: scoped ESLint exits0, zero errors and11 unchanged provider warnings; QA client/types/both touched tests have zero warnings.
- `/private/tmp/uat365-typescript-final.log`: UI compiler exits2 with412 repository diagnostics; no production or authority-test diagnostics. The private transport test retains its pre-existing resolveApiPath mock return-type diagnostic (previous line46, now48); there are no new diagnostics in changed scope. This is not a full typecheck pass.
- Final source snapshot for root: `/private/tmp/uat365-review-final.patch` and `/private/tmp/uat365-review-final-manifest.txt`. Source frozen pending independent review/native acceptance; no native, runtime, server, Docker or git mutations performed.
