---
id: TASK-12130
title: 'Issue #2605: certify Research Workspace final UAT browser runner'
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-04 18:32'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2605'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address GitHub issue #2605 by rerunning the Research Workspace final UAT browser runner against the full application, fixing product issues exposed by that run, and recording evidence that separates product failures from environment blockers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Full FastAPI backend and full Next.js WebUI are used for the final UAT runner path.
- [x] #2 Standalone Playwright browser-runner result is recorded with product vs environment classification.
- [x] #3 Product issues exposed by the full-app runner are fixed or documented with focused evidence.
- [x] #4 UAT matrix links the command, evidence artifacts, resulting status, and remaining environment blockers.
- [x] #5 No /workspace-playground alias or redirect is reintroduced.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started the live FastAPI app on http://127.0.0.1:8000 and the live Next.js quickstart WebUI proxy on http://127.0.0.1:8080. The first full-app UAT exposed a product-side auth assumption in the ACP handoff path: the spec and ACP helpers still expected the single-user key to remain directly in tldwConfig even though runtime bootstrap can scrub it into a session override. Added ACP runtime single-user key fallback coverage and updated the UAT assertions to accept the supported runtime auth storage. Hardened ACP/Sandbox menu activation in the real-backend spec to use the stable workspace settings test id and keyboard fallback. Re-ran the full final UAT against the full app; standalone Chromium launched and executed all 25 tests with 19 expected passes, 6 skips, 0 flaky, and 0 unexpected product failures. The wrapper status remains environment_blocked because this local backend does not expose POST /api/v1/sandbox/runs and does not advertise a runnable chat model for live-generation paths.

Follow-up llama.cpp UAT: confirmed http://127.0.0.1:9099/v1 exposes gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf. Started the full backend with the temp config pointing llama_api_IP at that endpoint plus WORKFLOWS_EGRESS_ALLOWED_PORTS=80,443,9099 and WORKFLOWS_EGRESS_BLOCK_PRIVATE=false so backend provider discovery enabled llama. The first llama-backed full-app run exposed one product-side E2E assertion race in the Flashcards scope-move check; the backend and manual full-app probe showed the moved card/deck were visible, so the spec now reopens the moved general deck directly before asserting the preserved flashcard UUID. Focused rerun of that real-backend case passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Issue #2605 was revalidated with the full application rather than a reduced harness. The live backend/WebUI runner now gets past browser launch and product execution. The initial no-model run classified the missing runnable chat model separately from product failures. Product changes are limited to ACP single-user runtime auth fallback plus UAT spec robustness. Evidence artifacts are apps/tldw-frontend/test-results/research-workspace-final-uat-evidence-2026-07-04.json and apps/tldw-frontend/test-results/research-workspace-final-uat-report-2026-07-04.json. Verification recorded: ACP auth Vitest regression passed; focused full-app UAT handoff cases passed; full final UAT executed 25 tests with 19 passed, 6 skipped, and 0 unexpected failures; Bandit is not applicable because no Python code was touched.

Follow-up with the provided local llama.cpp model removed the no_runnable_chat_model blocker. Final full-app UAT evidence is apps/tldw-frontend/test-results/research-workspace-final-uat-evidence-2026-07-04-llamacpp9099-rerun.json and report is apps/tldw-frontend/test-results/research-workspace-final-uat-report-2026-07-04-llamacpp9099-rerun.json. Result: 25 executed, 24 passed, 1 skipped, 0 unexpected failures, 0 flaky; wrapper status remains environment_blocked only for sandbox_run_api_unavailable. Bandit remains not applicable because no Python code was touched.
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
