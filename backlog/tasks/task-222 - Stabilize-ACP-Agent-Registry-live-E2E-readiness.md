---
id: TASK-222
title: Stabilize ACP Agent Registry live E2E readiness
status: In Progress
assignee: []
created_date: '2026-05-10 05:44'
labels:
  - acp
  - e2e
  - release-signoff
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1505'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The ACP release-signoff run for GitHub issue #1505 found that the Agent Registry E2E can assert before the live agent list finishes loading. The page eventually renders the expected registered agents, so the task is to make the E2E page object wait for the real settled state before counting cards, without masking genuine backend or UI errors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Agent Registry page object waits for either rendered agent entries or the no-agents empty state before list assertions run.
- [x] #2 Live Agent Registry E2E no longer fails while the page is still showing loading spinners.
- [x] #3 The focused ACP tier-3 browser specs pass against the seeded live backend.
- [x] #4 GitHub issue #1505 is updated with the before/after validation evidence.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

- Reproduced the live Agent Registry race with the seeded backend: the focused `should show agent cards or empty state` test failed before the agent list left its loading state.
- Added explicit health-card and agent-list settled waits to the Agent Registry E2E page object.
- Removed an existing unused `expect` import from the touched page object so the focused lint check is clean.
- Bandit is not applicable because this task only touches TypeScript E2E page-object code and a Backlog task record.

## Verification

- Red: `TLDW_WEB_URL=http://localhost:8080 TLDW_WEB_AUTOSTART=false TLDW_SERVER_URL=http://127.0.0.1:8000 TLDW_API_KEY=<seeded> TLDW_E2E_ALLOW_OFFLINE=0 bunx playwright test e2e/workflows/tier-3-automation/agent-registry.spec.ts -g 'should show agent cards or empty state' --reporter=line` failed with the original assertion.
- Focused green: same command passed after the agent-list wait fix.
- Health focused green: `TLDW_WEB_URL=http://localhost:18083 TLDW_WEB_CMD='bun run dev -- -p 18083' ... bunx playwright test e2e/workflows/tier-3-automation/agent-registry.spec.ts -g 'should show health status indicators or health unavailable warning' --reporter=line` passed.
- Final live slice: `TLDW_WEB_URL=http://localhost:18083 TLDW_WEB_CMD='bun run dev -- -p 18083' TLDW_SERVER_URL=http://127.0.0.1:8000 TLDW_API_KEY=<seeded> TLDW_E2E_ALLOW_OFFLINE=0 bunx playwright test e2e/workflows/tier-3-automation/acp-playground.spec.ts e2e/workflows/tier-3-automation/agent-registry.spec.ts e2e/workflows/tier-3-automation/agent-tasks.spec.ts --reporter=line` passed `19 passed (2.2m)`.
- Focused lint: `bunx eslint e2e/utils/page-objects/AgentRegistryPage.ts` passed with no warnings.

## Final Summary

Stabilized the ACP Agent Registry E2E page object by waiting for the health card and registered-agent list to finish loading before sampling their state. The full live ACP tier-3 browser slice passes against the seeded backend after the fix. Evidence was posted to GitHub issue #1505 and parent tracker #1500; #1505 remains open pending PR #1507 merge or explicit acceptance of local branch validation.
