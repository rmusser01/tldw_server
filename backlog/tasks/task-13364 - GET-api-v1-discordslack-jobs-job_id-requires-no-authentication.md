---
id: TASK-13364
title: 'GET /api/v1/{discord,slack}/jobs/{job_id} requires no authentication'
status: To Do
assignee: []
created_date: '2026-09-23 18:06'
labels:
  - security
  - chatops
  - authz
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both routes (now _chatops/ingress.job_status_payload) are mounted without any auth dependency (router_groups/content.py mounts the integration routers with no dependencies; the handlers take no principal) and there is no global auth middleware. Any caller can enumerate job ids and learn status, domain, queue and job_type for every Discord/Slack job. The only guard is 'job.domain == provider', which is the 'IDOR fix' ADR-050 mentions being written four times: it stops cross-integration reads, not unauthenticated ones. The routing tests (tests/Discord/test_discord_command_routing.py, tests/Slack/test_slack_command_routing.py) call it with no credentials and expect 200. The in-platform 'status' command is properly scoped (tenant + optional owner) - only the HTTP route is open. Decide who legitimately calls it (web UI user, admin, platform callback) and require that principal; for users, apply the same tenant/owner scoping as the status command. Found during TASK-13347.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The route requires an authenticated principal
- [ ] #2 A user can only read jobs they own or that belong to their tenant scope
- [ ] #3 Routing tests updated to authenticate
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
