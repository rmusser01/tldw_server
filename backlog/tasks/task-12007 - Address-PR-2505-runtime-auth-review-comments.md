---
id: TASK-12007
title: Address PR 2505 runtime auth review comments
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-24 19:05'
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
- [x] #1 WebUI compose overlay defaults TLDW_WEBUI_EXPOSE_RUNTIME_AUTH to 0 unless explicitly supplied.
- [x] #2 Docker single-user WebUI setup writes TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=1 for local quickstart and preserves operator overrides.
- [x] #3 Runtime-config endpoint, compose, and wizard tests cover fail-closed default and explicit setup opt-in.
- [x] #4 Docs and PR-local design notes describe explicit local quickstart opt-in and disabling runtime auth for non-loopback WebUI exposure.
- [x] #5 Current PR review thread is replied to and resolved; outdated unrelated threads are replied to and resolved after confirming they are no longer in the PR diff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation in progress: changed shared WebUI compose overlay default to TLDW_WEBUI_EXPOSE_RUNTIME_AUTH:-0; added Docker single-user WebUI setup-profile env generation for explicit TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=1 while preserving existing overrides; updated endpoint/compose/wizard tests and docs/design notes.

Verification: `bunx vitest run __tests__/pages/api/runtime-config.test.ts __tests__/frontend-quickstart-networking.test.ts __tests__/pr-916-review-followups.test.ts` passed 58 tests; `python -m pytest tldw_Server_API/tests/wizard/test_cli_profiles.py tldw_Server_API/tests/MCP_unified/test_packaging_shape.py tldw_Server_API/tests/Security/test_setup_access_guard.py -v` passed 75 tests; `docker compose` config rendered runtime auth as 0 by default and 1 with explicit env; `python -m compileall tldw_Server_API/cli/wizard/profiles.py` passed; docs tests passed 13 tests; production Bandit scope `tldw_Server_API/cli/wizard/profiles.py` passed. Full touched Python Bandit report only contained existing pytest `B101 assert_used` findings in `test_cli_profiles.py`.

GitHub review: replied to and resolved all five review threads on PR #2505. The runtime-auth default thread was fixed in commit 2af096ef39. Four older plan-file threads were confirmed outdated after the rebase and outside the current PR diff before resolving.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #2505 review comments by making the shared WebUI compose overlay fail closed for runtime-auth exposure, moving the local quickstart opt-in into `docker-single-webui` setup-generated `.env`, and documenting the security boundary. Added regression coverage for omitted exposure flags and setup-profile opt-in preservation. Replied to and resolved the current runtime-auth review thread plus four outdated unrelated plan-file threads.
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
