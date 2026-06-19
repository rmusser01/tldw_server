---
id: TASK-2388
title: ACP release caveat live-backend browser E2E closeout
status: Done
labels:
- ACP
- release-caveat
- E2E
references:
- https://github.com/rmusser01/tldw_server/issues/2404
- https://github.com/rmusser01/tldw_server/issues/2398
- https://github.com/rmusser01/tldw_server/pull/2405
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track GitHub issue #2404: run and document the final live-backend ACP browser E2E gate against a seeded backend/API key so release claims are backed by real server behavior, not only mocked UI coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Live backend E2E command is reproducible from recorded evidence.
- [x] #2 Results distinguish pass, skip, and external-runtime blocker.
- [x] #3 ACP Playground, Agent Registry, and Agent Tasks are covered or explicitly scoped out with rationale.
- [x] #4 Parent issue #2398 and child issue #2404 are updated with evidence and final status.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Live backend command setup:

- Started the API in single-user mode with `SINGLE_USER_API_KEY=test-api-key-12345`, test DB/output paths under `/private/tmp`, `TLDW_ACP_HOST_HOME=/Users/macbook-dev`, and `GOCACHE=/tmp/tldw-acp-go-cache`.
- Important runner setup detail: do not override `ACP_RUNNER_ENV` for this gate. The product default `runner_env = HOME=./acp_runner_home,PYTHONUNBUFFERED=1` resolves to the bundled runner home at `tldw_Server_API/Config_Files/acp_runner_home`, which contains the downstream `opencode`, `goose`, and `hermes` runner profiles.

Evidence:

- API smoke checks passed: `/api/v1/health` returned `status: ok`; `/api/v1/acp/sessions` returned `200` with an empty list before the browser run.
- Initial misconfigured run with `ACP_RUNNER_ENV=HOME=/Users/macbook-dev,...` produced `19 passed, 1 skipped`. The focused JSON rerun showed the skip reason: `Dispatch did not create a diagnostics-linked ACP run; dispatch status was HTTP 502`. Backend logs showed `ACPResponseError: unknown agent type: opencode`, proving the server registry and runner registry had diverged due to the env override.
- Corrected run without `ACP_RUNNER_ENV` showed `/api/v1/acp/agents` reporting `opencode`, `goose`, and `hermes` as configured.
- Focused rerun passed: `binds a Research Workspace to a real ACP run history and diagnostics path` -> `1 passed (18.0s)`.
- Final full run passed: `20 passed (29.6s)`.
- GitHub evidence posted to #2404 at https://github.com/rmusser01/tldw_server/issues/2404#issuecomment-4748491132; #2404 was closed as completed. Parent #2398 was updated at https://github.com/rmusser01/tldw_server/issues/2398#issuecomment-4748493270. Evidence-only PR: https://github.com/rmusser01/tldw_server/pull/2405.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the ACP live-backend browser E2E closeout for GitHub #2404. The reproducible command uses the product's bundled ACP runner home rather than a host HOME override. Final verification: `env TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=test-api-key-12345 TLDW_WEB_URL=http://localhost:18080 TLDW_WEB_CMD='bun run dev -- -p 18080' TLDW_E2E_ACP_WORKSPACE_ROOT_BASE=/private/tmp npx playwright test e2e/workflows/tier-3-automation/acp-playground.spec.ts e2e/workflows/tier-3-automation/agent-registry.spec.ts e2e/workflows/tier-3-automation/agent-tasks.spec.ts --project=tier-3 --reporter=line --workers=1` -> `20 passed (29.6s)`. No Python/application code changed; Bandit is not applicable for this evidence-only closeout.
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
