---
id: TASK-13237
title: Regress edge JWT buffer handling and harden malformed tokens (issue 2935)
status: Done
assignee: []
created_date: '2026-09-10 03:36'
updated_date: '2026-09-10 03:59'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2935'
documentation:
  - Docs/Design/ISSUES_2935_2938_REGRESSION_REPAIR.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Protect the existing typed-array JWT verification fix with cross-realm runtime regression coverage, audit nearby WebCrypto buffers and invalid-token handling, and fix confirmed malformed-token failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Valid and invalid JWTs are handled without cross-realm ArrayBuffer failures
- [x] #2 Malformed signature encoding fails closed without unhandled middleware exceptions
- [x] #3 Tests detect reintroducing the old raw ArrayBuffer argument
- [x] #4 Nearby WebCrypto calls audited and relevant checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Completed JWT regression and malformed-token hardening; design, audit, and verification retained in Docs/Design/ISSUES_2935_2938_REGRESSION_REPAIR.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Regression detects the historical raw ArrayBuffer crash on Node 20 and a strict typed-array boundary. Confirmed malformed base64url signatures threw and added fail-closed handling. Initial focused middleware run: 10 passed; Node 20 and 24 JWT runs: 5 passed each; lint and TypeScript pass. Independent review found a related non-string alg header exception; adding regression and guard before final checks.

Final independent re-review cleared the prior non-string alg finding and the existing typed-array regression. Fresh focused Vitest: 11 tests across 4 files passed; Node 20 and Node 24 JWT tests: 6 passed each. Scoped ESLint, production TypeScript, and git diff checks passed. Bandit is not applicable to TypeScript. Full admin UI suite was not run; Vitest reports an existing Node 26 deprecation warning. Nearby WebCrypto and malformed JSON type probes found no additional uncaught exceptions.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Preserved the existing Uint8Array JWT fix and added a regression that rejects the historical raw ArrayBuffer on real Node 20 WebCrypto and a strict boundary on newer Node. Fixed malformed signature base64url and non-string algorithm headers returning uncaught middleware exceptions; both now fail closed. Added valid, rotated-secret, invalid-signature, expired, and malformed-token tests.
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
