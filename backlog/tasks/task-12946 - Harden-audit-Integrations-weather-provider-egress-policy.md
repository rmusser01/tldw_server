---
id: TASK-12946
title: Harden audit Integrations weather provider egress policy
status: Done
created_date: 2026-07-10 05:35
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
- Supersedes weather records TASK-12139 and TASK-12945 after latest-dev ID collisions
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
updated_date: 2026-07-10 06:07
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate AUDIT-2026-06-27-INTEGRATIONS-003 and complete PR #2610 follow-up by routing OpenWeather through central HTTP policy without redirecting credentials, documenting production egress configuration, and preserving sanitized provider behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Weather provider outbound requests use the central HTTP policy and never forward the OpenWeather API key across redirects.
- [x] #2 Tests cover central HTTP routing, redirect refusal, strict-profile allow/deny behavior, and sanitized errors.
- [x] #3 Production documentation explains the OpenWeather egress allowlist requirement.
- [x] #4 The PR uses a unique active Backlog task ID and preserves superseded task history.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: Rebase on latest dev and reconcile comments/checks. Stage 2: Add failing redirect and strict-policy regression tests. Stage 3: Apply the minimal redirect fix and update production configuration documentation. Stage 4: Run focused and security verification, independent review, push, and reconcile fresh PR checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created on 2026-07-09 after origin/dev advanced during verification and introduced unrelated active TASK-12945 records. Earlier weather records TASK-12139 and TASK-12945 are archived as superseded. TDD evidence: the real central-HTTP redirect regression failed before the fix after requesting https://example.com/capture?appid=secret-key&units=metric&lang=en&q=Boston; adding allow_redirects=False reduced execution to one OpenWeather request and made the 12-test weather suite pass. Strict-profile tests prove denial before network I/O without an allowlist and success when EGRESS_ALLOWLIST includes api.openweathermap.org. Production documentation now records that requirement.
Latest-dev verification on 2026-07-09: branch rebased onto origin/dev 38bc70fd02ad4b55ed7ffc414642a936b6f28b0e and merge-base matched. Focused pytest command for weather providers plus command router passed 38 tests with 83 warnings. Ruff passed touched Python files. Bandit reported 0 findings, 0 skipped tests, and no errors over 258 production LOC. Both committed-range and working-tree git diff --check commands passed. Active TASK-12946 is unique; superseded weather records TASK-12139 and TASK-12945 are archived.
Review gates: independent specification review approved after confirming merge-base 38bc70fd02 and TASK-12946 plan references. Independent final code-quality/security review approved with no actionable P0-P3 findings. Residual risk is limited to live DNS/TLS behavior because deterministic tests use an in-memory transport and pytest's DNS shortcut.
PR refresh: force-pushed head 84bb98d57b5add518889cb271ef6c59d7ec43369, updated the PR body, and resolved the only review thread after its convention-based response. Fresh GitHub workflows were triggered with no failures, but repository-wide Actions capacity was unavailable: 113 runs queued and 0 in progress. This external queue state is recorded rather than misclassified as a branch failure; PR #2610 remains draft pending the required human-written Change summary and eventual CI execution.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2610 onto origin/dev 38bc70fd02, fixed the independently discovered cross-origin OpenWeather API-key redirect leak by disabling redirects, added real central-policy redirect and strict allow/deny tests, documented the production egress allowlist, and replaced colliding active task IDs with unique TASK-12946 while archiving superseded records. Focused tests passed 38/38, Ruff passed, Bandit reported zero findings over 258 production LOC, whitespace checks passed, independent spec and quality reviews approved, and the existing GitHub thread was resolved. Fresh GitHub checks are queued without failures because repository-wide Actions currently has no running capacity.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused weather and command-router tests pass on latest dev.
- [x] #2 Ruff passes for touched Python files.
- [x] #3 Bandit reports no findings in touched production code.
- [x] #4 git diff --check passes.
- [x] #5 Independent specification and code-quality reviews have no unresolved actionable findings.
- [x] #6 PR review threads and fresh CI state are reconciled.
<!-- DOD:END -->
