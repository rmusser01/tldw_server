---
id: TASK-2406
title: Address AuthNZ review hardening findings
status: Done
assignee: []
created_date: '2026-06-23 18:10'
updated_date: '2026-06-24 01:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and remediate validated AuthNZ review findings: OIDC authorization URL validation, OIDC JSON response caps, OAuth callback base URL hardening, mock email token redaction, and any confirmed low-risk cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused tests fail before fixes and pass after fixes; validated findings are addressed or documented as not applicable; Bandit runs over touched AuthNZ scope; Backlog task records verification.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified and addressed AuthNZ review findings. Implemented HTTPS/no-fragment validation for OIDC provider URLs, bounded OIDC discovery/token/JWKS JSON fetches, actual-body max_bytes enforcement in HTTP JSON helpers, PUBLIC_WEB_BASE_URL-based federation callback construction, mock redaction for token-bearing auth emails, and replacement of billing_repo production assert with an explicit ValueError. Verification: OIDC service focused tests 6 passed; HTTP JSON size-limit focused tests 3 passed; mock email redaction focused tests 2 passed; federation callback helper test 1 passed; production files py_compile passed. Bandit on touched source files reported only existing low-confidence B106 literal-string noise in auth.py; rerun with B106 skipped reported 0 findings. Full FastAPI endpoint redirect test was attempted with a longer timeout but remained in app startup for several minutes and was interrupted before the test body; helper-level coverage verifies the callback URL construction behavior directly.

Follow-up: rebase PR #2444 onto latest dev and address all actionable PR comments/check findings.

PR #2444 follow-up: rebased onto fetched latest dev, dropped unrelated Claims_Extraction commit from branch diff, and addressed actionable review comments. Fixes include docstrings on new helpers, typed new tests/stubs, async-safe mock email test file reads, OIDC authorization query key de-duplication, robust PUBLIC_WEB_BASE_URL callback URI construction, and callback token exchange redirect_uri validation against the helper-built URI.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2444 is rebased onto latest dev and review feedback has been addressed with focused verification. Known note remains: full app endpoint startup is too slow locally, so callback behavior is covered by helper-level and focused AuthNZ tests.
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
