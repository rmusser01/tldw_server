---
id: TASK-12007
title: Address PR 2505 runtime auth review comments
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-24 15:07'
labels:
  - docker
  - webui
  - auth
  - review
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-06-24-docker-webui-runtime-auth-bootstrap-design.md
  - >-
    Docs/superpowers/plans/2026-06-24-docker-webui-runtime-auth-bootstrap-implementation-plan.md
modified_files:
  - Dockerfiles/docker-compose.webui.yml
  - Dockerfiles/README.md
  - Docs/Getting_Started/Profile_Docker_Single_User.md
  - Docs/Published/Getting_Started/Profile_Docker_Single_User.md
  - Docs/Getting_Started/TROUBLESHOOTING.md
  - Docs/superpowers/specs/2026-06-24-docker-webui-runtime-auth-bootstrap-design.md
  - Docs/superpowers/plans/2026-06-24-docker-webui-runtime-auth-bootstrap-implementation-plan.md
  - README.md
  - apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts
  - apps/tldw-frontend/__tests__/pages/api/runtime-config.test.ts
  - apps/tldw-frontend/__tests__/pr-916-review-followups.test.ts
  - tldw_Server_API/cli/wizard/profiles.py
  - tldw_Server_API/tests/wizard/test_cli_profiles.py
  - backlog/tasks/task-12007 - Address-PR-2505-runtime-auth-review-comments.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address all actionable PR #2505 review threads after rebasing the Docker WebUI runtime-auth bootstrap branch. Current scope: change the runtime-auth compose overlay to fail closed by default, keep local quickstart bootstrap explicit, update docs/tests, and resolve outdated review threads that no longer apply to the current diff.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 WebUI compose overlay defaults TLDW_WEBUI_EXPOSE_RUNTIME_AUTH to 0 unless explicitly supplied.
- [ ] #2 Docker single-user WebUI setup writes TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=1 for local quickstart and preserves operator overrides.
- [ ] #3 Runtime-config endpoint, compose, and wizard tests cover fail-closed default and explicit setup opt-in.
- [ ] #4 Docs and PR-local design notes describe explicit local quickstart opt-in and disabling runtime auth for non-loopback WebUI exposure.
- [ ] #5 Current PR review thread is replied to and resolved; outdated unrelated threads are replied to and resolved after confirming they are no longer in the PR diff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation in progress: changed shared WebUI compose overlay default to TLDW_WEBUI_EXPOSE_RUNTIME_AUTH:-0; added Docker single-user WebUI setup-profile env generation for explicit TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=1 while preserving existing overrides; updated endpoint/compose/wizard tests and docs/design notes.
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
