---
id: TASK-12139
title: Harden audit Integrations weather provider egress policy
status: Done
created_date: 2026-07-04 01:50
labels:
- audit
- remediation
- integrations
- http-policy
priority: medium
references:
- AUDIT-2026-06-27-INTEGRATIONS-003
- https://github.com/rmusser01/tldw_server/pull/2610
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/integrations-providers.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- tldw_Server_API/app/core/Integrations/weather_providers.py
- tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py
updated_date: 2026-07-10 05:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate AUDIT-2026-06-27-INTEGRATIONS-003 by routing weather-provider outbound API requests through the central HTTP defaults or an explicitly safe client configuration, while preserving existing provider behavior and sanitized error handling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Weather provider outbound requests no longer use a raw client that bypasses central HTTP policy defaults.
- [x] #2 Weather provider tests verify the request path uses central HTTP behavior or explicitly safe client configuration.
- [x] #3 Existing weather provider input validation and sanitized error behavior remain intact.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the AUDIT-2026-06-27-INTEGRATIONS-003 weather-provider egress remediation. TDD evidence: `test_openweather_uses_central_http_fetch_for_requests` first failed because the provider still used the raw client path; after routing through central `fetch`, it passed. Added `test_openweather_central_policy_denial_returns_sanitized_error`; it first failed with `EgressPolicyError` bubbling out, then passed after adding central HTTP exception handling. Added a retry-preservation assertion that first failed on missing `retry`, then passed after using `RetryPolicy(attempts=1)` to preserve prior single-request weather timeout behavior.

Current-dev refresh (2026-07-04): rebased `codex/audit-weather-egress-2026-07-04` onto `origin/dev` `09d9ec901e1d4548f7924f1c6bcefa963fadd9bd`; merge-base matches `origin/dev`. Current validation after the HTTP redirect hardening merge: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py -q` passed with 36 tests; `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Integrations/weather_providers.py -f json -o /tmp/bandit_weather_egress_origin_dev_09d9ec.json` reported 0 findings over 257 LOC; `git diff --check HEAD~1..HEAD` passed. PR review feedback about spaces in the Backlog filename was evaluated as non-actionable because this repository's Backlog.md convention uses `task-id - title.md` filenames.
2026-07-04 latest-dev refresh: rebased and validated PR #2610 on origin/dev 6b727b221e55646eba663a03571e38302f7fafc2. Tested head 2f2a02e8cdba. Verification: python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py -q => 36 passed, 80 warnings; bandit -r tldw_Server_API/app/core/Integrations/weather_providers.py => 0 findings over 257 LOC; git diff --check HEAD~1..HEAD => clean.
2026-07-09 PR follow-up: reopened task to rebase PR #2610 onto current origin/dev and re-evaluate all review threads and CI failures before merge readiness.
2026-07-09: This weather task ID collided with a different TASK-12139 added to dev. The weather remediation record is superseded by TASK-12945 and archived to preserve history without an ambiguous active ID.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened weather provider egress handling and command-router coverage. Final refresh validated against origin/dev 6b727b221e55646eba663a03571e38302f7fafc2 with focused tests passing, Bandit clean on touched production scope, and whitespace check clean.
Superseded by TASK-12945 after a latest-dev rebase exposed an active task ID collision.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused weather provider tests pass.
- [x] #2 Bandit runs clean over touched production weather provider code.
- [x] #3 git diff --check passes.
- [x] #4 AUDIT-2026-06-27-INTEGRATIONS-003 closure evidence is recorded in the task notes.
<!-- DOD:END -->
