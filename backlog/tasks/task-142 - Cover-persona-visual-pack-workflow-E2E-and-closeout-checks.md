---
id: TASK-142
title: Cover persona visual pack workflow E2E and closeout checks
status: Done
assignee: []
created_date: '2026-05-09 01:37'
updated_date: '2026-05-09 01:49'
labels:
  - e2e
  - frontend
  - persona
  - visuals
  - verification
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1388'
  - 'https://github.com/rmusser01/tldw_server/issues/1389'
documentation:
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
  - Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 11 from Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md: add Persona Live visual-state E2E fixture coverage, run the focused backend/frontend persona visual verification set, run Bandit over the touched backend scope, document any environment blockers, and commit the closeout test/docs slice. GitHub tracker: #1388. Related completed MCP sub-issue: #1389.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona Live E2E fixture covers active visual-pack idle state, speaking/tool/error state transitions, and broken-pack fallback without blocking live controls
- [x] #2 Focused backend persona visual tests pass or each blocker is documented with exact command output
- [x] #3 Focused frontend persona visual tests pass or each blocker is documented with exact command output
- [x] #4 Persona Live E2E workflow passes or an environment/setup blocker is documented with exact command output
- [x] #5 Bandit runs over the planned touched backend scope with no new production findings or documented non-new findings
- [x] #6 Implementation plan and Backlog task record all verification results and known skips/blockers
- [x] #7 Closeout changes are committed in the persona visual packs worktree
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Persona Live visual workflow E2E coverage in apps/tldw-frontend/e2e/workflows/persona-live.spec.ts. Added mocked persona API/session/profile/visual-pack fixtures, active-pack idle state assertion, speaking/tool_running/error visual_state_override assertions, and broken-pack fallback coverage that still verifies live controls connect.

Verification passed: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py tldw_Server_API/tests/Persona/test_persona_visual_jobs.py tldw_Server_API/tests/Services/test_persona_visual_jobs_worker_startup.py tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py -v (36 passed).

Verification passed: bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx (33 passed).

Verification passed with escalated local-server permission: bunx playwright test e2e/workflows/persona-live.spec.ts --reporter=line (2 passed, 1 skipped; skipped test was the existing live-backend smoke path because backend capability was unavailable for that proof). Initial sandboxed run failed with listen EPERM on 0.0.0.0:8080; rerun with escalation was required for Next dev server binding.

Verification passed: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/services/startup_optional_workers.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py -f json -o /tmp/bandit_persona_visuals.json. Bandit results: 0 findings, report at /tmp/bandit_persona_visuals.json.

Verification passed: git diff --check.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Change summary:
- Added Persona Live E2E coverage for active persona visual packs in apps/tldw-frontend/e2e/workflows/persona-live.spec.ts. The fixture mocks persona health, capability, profile, session, visual-pack, and websocket behavior so the test can validate the visual runtime without requiring a live backend response path.
- Covered the main runtime states required by the plan: initial data-visual-state="idle", visual_state_override transitions for speaking/tool_running/error, and a broken visual-pack API response that falls back to the text buddy shell while keeping live controls usable.
- Updated the persona visual implementation plan and Backlog record with verification outcomes.

Why:
- The prior coverage proved the live persona websocket path but did not assert that the active 2D visual pack participates in the Persona Live surface or that a failed pack load remains non-blocking. The mocked fixture makes those UI contracts deterministic while leaving the existing live-backend smoke test intact.

Verification:
- Backend: 36 focused persona visual tests passed.
- Frontend unit/component: 33 focused tests passed.
- E2E: bunx playwright test e2e/workflows/persona-live.spec.ts --reporter=line passed for the 2 mocked visual workflow tests; the existing live-backend smoke test skipped because backend persona capability was unavailable for that proof.
- Security/format: Bandit reported 0 findings at /tmp/bandit_persona_visuals.json; git diff --check passed.
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
