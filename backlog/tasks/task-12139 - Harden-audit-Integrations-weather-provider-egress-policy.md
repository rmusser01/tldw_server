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
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/integrations-providers.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- tldw_Server_API/app/core/Integrations/weather_providers.py
- tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py
updated_date: 2026-07-04 01:58
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
Implemented the AUDIT-2026-06-27-INTEGRATIONS-003 weather-provider egress remediation. TDD evidence: `test_openweather_uses_central_http_fetch_for_requests` first failed because the provider still used the raw client path; after routing through central `fetch`, it passed. Added `test_openweather_central_policy_denial_returns_sanitized_error`; it first failed with `EgressPolicyError` bubbling out, then passed after adding central HTTP exception handling. Added a retry-preservation assertion that first failed on missing `retry`, then passed after using `RetryPolicy(attempts=1)` to preserve prior single-request weather timeout behavior. Verification: `python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py -q` passed with 36 passed; `python -m bandit -r tldw_Server_API/app/core/Integrations/weather_providers.py -f json -o /tmp/bandit_integrations_weather.json` produced 0 errors and 0 results; `git diff --check` passed; raw client scan for `http_client_factory`, `httpx.Client`, `httpx.AsyncClient`, and direct `requests` usage returned no matches in the touched weather files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Routed OpenWeather API requests through the central synchronous HTTP helper, preserving the provider timeout as a single-attempt request and expanding sanitized handling to central HTTP policy/transport exceptions. Updated weather provider tests to patch the central fetch seam and added regressions for central fetch usage, sanitized policy denials, and unchanged validation-before-network behavior. This closes AUDIT-2026-06-27-INTEGRATIONS-003 for this focused branch.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused weather provider tests pass.
- [x] #2 Bandit runs clean over touched production weather provider code.
- [x] #3 git diff --check passes.
- [x] #4 AUDIT-2026-06-27-INTEGRATIONS-003 closure evidence is recorded in the task notes.
<!-- DOD:END -->
