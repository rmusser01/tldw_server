---
id: TASK-2231
title: Address PR 2250 CodeRabbit follow-up comments
status: Done
assignee: []
created_date: '2026-06-03 06:45'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2250'
  - TASK-2230
priority: high
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace ACP history modal surfaces direct ACP session fetch failures when agent-task history is empty.
- [x] #2 TASK-2227 verification commands are portable and do not hardcode local absolute paths.
- [x] #3 workspace-live-e2e review-loop detection requires structured reviewer evidence, not raw substring matches.
- [x] #4 workspace-live-e2e HTTP failures are labeled as workspace_live_backend_acp_e2e.
- [x] #5 ACP workspace context redacts local adapter_source paths while preserving repository-style identifiers.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a Workspace ACP history modal fallback branch so direct ACP session fetch errors are shown when Agent Tasks history has no runs to display.
- Replaced workspace-live-e2e reviewer-loop substring detection with recursive structured-key detection for `review_loop`, `reviewer`, and `review_decision`.
- Labeled workspace-live-e2e HTTP and evidence failures as `workspace_live_backend_acp_e2e` while keeping the generic backend live runner on `live_backend_acp_e2e`.
- Added ACP workspace context `adapter_source` redaction for local filesystem paths while preserving package/repository-style and HTTP(S) identifiers.
- Updated branch-local TASK-2227 closeout commands to avoid hardcoded absolute checkout paths and to list `ACP_E2E_WORKSPACE_ID` as a required live-run environment variable.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the PR 2250 CodeRabbit follow-up comments with focused UI, helper, endpoint, tests, and backlog updates. Verification passed with the focused regressions, the full Research Workspace header test file, the affected ACP backend suites, Bandit on touched Python helper/endpoint scope, `git diff --check`, and the opt-in live harness prerequisite refusal.

Verification:
- `bun run test src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx -t "surfaces direct ACP session fetch errors when Agent Tasks history has no runs"` -> 1 passed, 40 skipped.
- `bun run test src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx` -> 41 passed.
- `python -m pytest -q tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py::test_payload_contains_review_evidence_requires_structured_signal tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py::test_backend_workspace_live_e2e_uses_workspace_failure_label` -> 2 passed, 6 warnings.
- `python -m pytest -q tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py::test_acp_workspace_context_adapter_source_redacts_local_paths` -> 1 passed, 6 warnings.
- `python -m pytest -q tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py` -> 128 passed, 6 warnings.
- `python -m bandit -q Helper_Scripts/Testing-related/acp_certification_smoke.py tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` -> exit 0.
- `python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile workspace-live-e2e --run` -> exit 2 safe refusal because live backend/profile/workspace environment is not configured.
- `git diff --check` -> exit 0.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused regression tests cover behavioral changes.
- [x] #8 Affected tests and Bandit verification recorded.
- [x] #9 PR branch pushed.
<!-- DOD:END -->
