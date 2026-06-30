---
id: TASK-497
title: Run Goose ACP backend live E2E certification
status: Done
labels:
- ACP
- certification
- Goose
references:
- https://github.com/rmusser01/tldw_server/issues/1563
- https://github.com/rmusser01/tldw_server/issues/1532
documentation:
- Docs/Development/ACP_Compatibility_Matrix.md
- Docs/Development/ACP_Certification_Checklist.md
- tldw_Server_API/Config_Files/agents.yaml
modifiedFiles:
- Helper_Scripts/Testing-related/acp_certification_smoke.py
- Docs/Development/ACP_Compatibility_Matrix.md
- tldw_Server_API/Config_Files/acp_runner_home/.tldw-agent/config.yaml
- tldw_Server_API/Config_Files/agents.yaml
- tldw_Server_API/Config_Files/config.txt
- tldw_Server_API/app/core/Agent_Client_Protocol/config.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_config_cwd.py
- tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py
- tools/tldw-agent/internal/acp/runner.go
- tools/tldw-agent/internal/acp/runner_test.go
- IMPLEMENTATION_PLAN_acp_goose_backend_live_e2e.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the next ACP certification slice for Goose through the backend live-E2E path, record evidence, and update compatibility metadata/docs only if the live backend run supports the claim. Scope includes focused registry/helper tests, compatibility matrix and checklist/task updates, and GitHub issue #1563 evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Shipped ACP runner config resolves to the in-repo `tools/tldw-agent` runner.
- [x] Certification manifest Python commands use the active interpreter instead of a bare `python`.
- [x] Bundled runner-home config exposes the Goose ACP profile used by the API registry.
- [x] Backend runner env preserves the operator HOME for downstream Goose while keeping the runner HOME isolated.
- [x] Goose backend live-E2E passes and compatibility metadata/docs are updated with explicit caveats.
- [x] Focused pytest, Go runner verification, Bandit, and diff checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Fixed default `[ACP] runner_cwd` from the stale sibling path to `../../tools/tldw-agent`, relative to `Config_Files`.
- Switched ACP certification manifest Python argv entries to `sys.executable`.
- Added `TLDW_ACP_HOST_HOME` propagation when a relative isolated runner HOME is resolved, and expanded registered-agent env placeholders in the Go ACP runner.
- Replaced the bundled runner-home config's legacy default Codex command with registered `custom`, `goose`, `hermes`, and `opencode` entries.
- Verified Goose backend live-E2E on the macOS host runner with session `20260524_5`; result summary was `stop_reason=end_turn`, `events_total=2`, `artifacts_total=0`, `diagnostics_total=0`.
- Updated Goose support metadata to `supported_with_caveats` / `live_e2e_tested`; caveats remain for sandbox, non-empty MCP injection, artifact-producing workflows, and reviewer-loop behavior.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Goose now has backend live-E2E evidence through the server ACP API and the bundled runner config is aligned with the API registry. The implementation also fixes the two blockers found during certification: stale runner cwd resolution and manifest commands depending on a bare `python` executable.

Verification recorded: focused ACP pytest `22 passed`; `go test ./internal/acp -count=1`; `tools/tldw-agent/scripts/verify-local-build.sh`; Bandit on touched Python with zero findings; `git diff --check`.

Known remaining caveats: the Goose support claim is limited to the verified macOS host runner with configured Goose provider state. Sandbox behavior, non-empty MCP server injection, artifact-producing workflows, and reviewer-loop behavior still need separate certification before those capabilities can be claimed.
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
