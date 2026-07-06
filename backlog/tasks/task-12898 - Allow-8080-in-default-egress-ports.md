---
id: TASK-12898
title: Allow 8080 in default egress ports
status: Done
assignee: []
created_date: '2026-07-06 00:12'
updated_date: '2026-07-06 00:57'
labels:
  - security
  - docs
  - setup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fresh installs should allow outbound inference/provider calls on port 8080 without custom port configuration, while documenting why other hosted inference ports or private addresses may still need egress settings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default egress ports include 8080 in code and shipped config.
- [x] #2 Single-user and multi-user setup docs explain provider connectivity failures caused by server-side egress policy.
- [x] #3 Config and design docs stay consistent with the new default.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed default egress port set from 80,443 to 80,443,8080 in the evaluator fallback and shipped config. Updated config, ADR, workflow, Docker single-user, and Docker multi-user setup docs including published copies. Verification: stale-default rg returned no hits; evaluator check printed egress-default-ok; Bandit on tldw_Server_API/app/core/Security/egress.py reported 0 results and 0 errors.

Review follow-up: centralized DEFAULT_ALLOWED_PORTS in the egress module, reused it from the Workflows config endpoint, and added regressions for 8080 default allowance, empty/invalid port env fallback, and /workflows/config reported defaults. Verification: focused pytest for security/workflows config defaults passed; stale-default rg returned no hits; Bandit on touched Python files reported 0 results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Default egress ports now include 8080 permanently, and setup docs explain why hosted inference servers on other ports or private/LAN/loopback addresses may need WORKFLOWS_EGRESS_ALLOWED_PORTS and WORKFLOWS_EGRESS_BLOCK_PRIVATE configuration.
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
