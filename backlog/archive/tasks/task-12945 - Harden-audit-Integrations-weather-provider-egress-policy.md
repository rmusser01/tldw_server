---
id: TASK-12945
title: Harden audit Integrations weather provider egress policy
status: Done
created_date: 2026-07-10 05:28
labels:
- audit
- remediation
- integrations
- http-policy
- pr-followup
priority: medium
references:
- AUDIT-2026-06-27-INTEGRATIONS-003
- https://github.com/rmusser01/tldw_server/pull/2610
- Supersedes colliding weather task record TASK-12139
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/integrations-providers.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
- Docs/Operations/Env_Vars.md
- Docs/User_Guides/WebUI_Extension/Chatbook_Tools_Getting_Started.md
modified_files:
- tldw_Server_API/app/core/Integrations/weather_providers.py
- tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py
- Docs/Operations/Env_Vars.md
- Docs/User_Guides/WebUI_Extension/Chatbook_Tools_Getting_Started.md
- tldw_Server_API/app/core/Integrations/README.md
updated_date: 2026-07-10 05:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate AUDIT-2026-06-27-INTEGRATIONS-003 and complete PR #2610 follow-up by routing OpenWeather through central HTTP policy without redirecting credentials, documenting production egress configuration, and preserving sanitized provider behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Weather provider outbound requests use the central HTTP policy and never forward the OpenWeather API key across redirects.
- [ ] #2 Tests cover central HTTP routing, redirect refusal, strict-profile allow/deny behavior, and sanitized errors.
- [ ] #3 Production documentation explains the OpenWeather egress allowlist requirement.
- [ ] #4 The PR uses a unique active Backlog task ID and preserves the superseded task history.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: Rebase on latest dev and reconcile comments/checks. Stage 2: Add failing redirect and strict-policy regression tests. Stage 3: Apply the minimal redirect fix and update production configuration documentation. Stage 4: Run focused and security verification, independent review, push, and reconcile fresh PR checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created on 2026-07-09 because latest dev introduced a different active TASK-12139, colliding with the original weather remediation task. The original task record will be archived as superseded after this unique replacement is verified.
TDD evidence (2026-07-09): added a real central-HTTP redirect regression using httpx.MockTransport plus a central-fetch call contract assertion. Before the production fix, the focused suite failed twice: the redirect target returned a successful forged weather payload and the central call omitted allow_redirects. The captured second request was https://example.com/capture?appid=secret-key&units=metric&lang=en&q=Boston. Adding allow_redirects=False produced one request to api.openweathermap.org and the 12-test weather suite passed. Added real strict-profile tests proving denial before network I/O without an allowlist and successful access when EGRESS_ALLOWLIST includes api.openweathermap.org. Updated operator and user documentation for the production allowlist requirement.
2026-07-09: origin/dev advanced during verification and introduced unrelated active TASK-12945 records. This weather record is superseded by unique TASK-12946 and archived before the second rebase.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Superseded by TASK-12946 after a later latest-dev refresh exposed additional active TASK-12945 collisions.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Focused weather and command-router tests pass on latest dev.
- [ ] #2 Ruff passes for touched Python files.
- [ ] #3 Bandit reports no findings in touched production code.
- [ ] #4 git diff --check passes.
- [ ] #5 Independent specification and code-quality reviews have no unresolved actionable findings.
- [ ] #6 PR review threads and fresh CI state are reconciled.
<!-- DOD:END -->
