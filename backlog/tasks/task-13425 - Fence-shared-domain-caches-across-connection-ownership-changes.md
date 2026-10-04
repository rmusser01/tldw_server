---
id: TASK-13425
title: Fence shared domain caches across connection ownership changes
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current dev character and chat-message singleton caches are keyed only by resource IDs and query parameters. Fence cached values and in-flight reads with existing native connection authority and account-boundary contracts, without changing authentication protocols.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Same resource IDs cannot reuse cached values or in-flight reads after native JWT, API-key, server, organization, or cookie-session account boundaries
- [x] #2 Late completion cannot repopulate or remove the current owner cache or in-flight entry
- [x] #3 Same-owner JWT refresh and existing requestScope/fresh behavior remain compatible
- [ ] #4 Real domain and client regressions pass; focused typecheck and security assessment recorded; upstream PR targets dev
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR check: ADR required: no. Reuses existing connectionAuthoritiesMatch, watchChatAccountChanges, and request-scope contracts; no new authentication, persistence, or public API architecture. Governing task workflow: Docs/ADR/059-backlog-py-task-editor-cutover.md. Spec and plan will be linked before implementation.
Design: Docs/superpowers/specs/2026-10-04-domain-cache-ownership-design.md. Plan: IMPLEMENTATION_PLAN_domain_cache_ownership_20261004.md. Open upstream PR and task searches found no duplicate cache-ownership fix; task ID chosen above open-PR maximum 13424. Read-only dependency symlinks reuse available installed packages; no lockfile changes.
TDD evidence on current dev 502da5bf0c: initial ownership suite 84 failing and 12 passing tests; first fix green at 96. Self-review exposed post-await cache read/publication races; 12 additional tests failed before making the final revision assertion synchronous with cache access. The 108-case ownership suite and five related suites then passed (193 tests). Covers real public/base/domain implementations. Parent integration mechanism: shared module account epoch, per-client connection authority revision, effective-config checks before hits/joins and after async work, and identity-checked in-flight cleanup.
Focused TypeScript compiler comparison used current files and git-show baseline sources without altering any checkout: identical seven existing errors on baseline and branch, zero introduced errors. Full UI typecheck exhausted the default 4 GiB heap; focused check completed at 8 GiB. ESLint touched-scope reports zero errors; existing warning baseline comparison in progress. Bandit attempt in the project venv reports No module named bandit; no Python files are touched, so Bandit is not applicable to this TypeScript-only change. Security self-review confirms no new cookies, credential serialization, storage, or auth protocol; only native cache ownership fencing. Independent reviewer tools are not available; self-review performed.
Final verification before review/publication: all 417 tests passed in ten focused suites, including 114 ownership regressions, real background proxy transport, native refresh, and quickstart auth. Added six RED regressions for older failed config verification clearing a newer owner flight; failure cleanup now checks its starting revision. Focused TypeScript baseline comparison remains identical (7 existing errors; 0 introduced). ESLint has 0 errors and exactly the baseline 830 warnings; the new regression file has 0 warnings. git diff --check passed. Latest origin/dev fetched before publication remains 502da5bf0ccd1bc3aa4323e0d0fc430f36821a78. Only generic upstream changes will be staged; dependency symlinks and external compiler/review artifacts are excluded. Independent upstream-diff review is requested before closeout.
Parent reviewed the generic upstream diff and confirmed the native cache approach, with promise-identity finally cleanup explicitly preserved. Addressed nullable-revision feedback by using a numeric unused revision when shared cache is bypassed; the three ownership/scope suites passed again (162 tests) and focused TypeScript still has only the identical seven baseline diagnostics. Task ID 13425 is absent from current origin/dev; highest current dev ID is 13443 after normalization. Extra standalone read-only reviewer attempt failed before review because the app model alias is unsupported by the CLI account; retrying a recognized model without touching the dirty original checkout.
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
