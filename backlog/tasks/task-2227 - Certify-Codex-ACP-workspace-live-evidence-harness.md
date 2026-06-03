---
id: TASK-2227
title: Certify Codex ACP workspace live evidence harness
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-03 05:58'
labels: []
dependencies: []
references:
  - Docs/superpowers/specs/2026-06-01-acp-codex-orchestration-design.md
  - TASK-606
documentation:
  - >-
    Docs/superpowers/plans/2026-06-03-codex-acp-workspace-live-certification-plan.md
  - Docs/Development/ACP_Certification_Checklist.md
  - Docs/Development/ACP_Compatibility_Matrix.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an opt-in workspace-live-e2e certification profile for Codex/ACP that drives the backend REST session flow with Research Workspace context, non-empty MCP injection, redacted support views, artifact/sandbox/reviewer-loop evidence reporting, and strict expectation flags without making normal CI depend on a live Codex runner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 workspace-live-e2e manifest exists and refuses to run without live backend/profile environment.
- [x] #2 Backend workspace live runner creates a session with workspace_id and non-empty MCP server config, then queries detail/events/artifacts/diagnostics and workspace-filtered session history.
- [x] #3 Harness reports pass/skip/fail capability states for workspace_env, mcp_injection, artifacts, sandbox, review_loop, diagnostics, redacted support views, and close cleanup.
- [x] #4 Strict expectation flags make optional artifacts/sandbox/reviewer-loop gaps fail without double-closing already closed sessions.
- [x] #5 Focused helper tests, Bandit, and live prerequisite check are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-03-codex-acp-workspace-live-certification-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented in TDD slices: workspace-live-e2e manifest contract, backend workspace live runner contract, strict optional artifact expectation regression, and documentation updates. Main-checkout MCP-created TASK-509 was not used for this branch because the worktree already had task-509 collisions; TASK-2227 is the canonical branch-local task.

Verification:
- PASS: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py` -> 43 passed, 6 warnings.
- PASS: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -q Helper_Scripts/Testing-related/acp_certification_smoke.py` -> exit 0.
- LIVE SKIP/REFUSAL: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile workspace-live-e2e --run` -> exit 2, missing `TLDW_E2E_SERVER_URL`, `TLDW_E2E_API_KEY`, `ACP_AGENT_PROFILE`. No live certification claim made.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added an opt-in `workspace-live-e2e` ACP certification profile and backend runner for Research Workspace-scoped Codex/ACP validation. The helper now creates a workspace-bound session with a non-empty MCP server config, prompts for artifact-like output, queries redacted detail/events/artifacts plus diagnostics and workspace-filtered session history, and emits bounded pass/skip/fail capability evidence. Strict `ACP_E2E_EXPECT_ARTIFACTS`, `ACP_E2E_EXPECT_SANDBOX`, and `ACP_E2E_EXPECT_REVIEWER_LOOP` flags turn skipped optional evidence into failure without double-closing already closed sessions. Documentation now distinguishes plain host live E2E from workspace live evidence and keeps the Codex matrix caveats unupgraded until a real workspace run passes.

Verification:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py` -> 43 passed, 6 warnings.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -q Helper_Scripts/Testing-related/acp_certification_smoke.py` -> exit 0.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile workspace-live-e2e --run` -> exit 2 with safe refusal because `TLDW_E2E_SERVER_URL`, `TLDW_E2E_API_KEY`, and `ACP_AGENT_PROFILE` were not set. No live certification claim was made.
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
