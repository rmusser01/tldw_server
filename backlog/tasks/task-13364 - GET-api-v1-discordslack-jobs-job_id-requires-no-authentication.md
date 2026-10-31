---
id: TASK-13364
title: 'GET /api/v1/{discord,slack}/jobs/{job_id} requires no authentication'
status: Done
assignee: []
created_date: '2026-09-23 18:06'
updated_date: '2026-09-23 18:28'
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
- [x] #1 The route requires an authenticated principal
- [x] #2 A user can only read jobs they own or that belong to their tenant scope
- [x] #3 Routing tests updated to authenticate
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Decision (owner): the caller is the web UI user. Route now requires get_request_user; allowed = job owner, or active member of an org whose workspace_provider_installations include the job's guild/team, unless the tenant policy is owner-only (*_and_user); single-user mode sees all jobs of the integration; else 404. Shared implementation in _chatops/ingress.job_status_payload. Tests: tests/Discord/test_chatops_job_status_authz.py (10 cases, 4 fail on the old code); routing tests authenticate. ChatOps set: 742 passed, 14 pre-existing failures unchanged. ADR-050 addendum. Bandit: no new issues.
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
