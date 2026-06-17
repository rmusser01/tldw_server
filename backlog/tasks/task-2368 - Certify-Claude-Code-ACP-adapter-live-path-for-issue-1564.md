---
id: TASK-2368
title: Certify Claude Code ACP adapter live path for issue 1564
status: Done
labels:
- ACP
- certification
- Claude Code
priority: high
references:
- https://github.com/rmusser01/tldw_server/issues/1564
- https://github.com/rmusser01/tldw_server/pull/2248
- https://agentclientprotocol.com/get-started/agents
- https://github.com/rmusser01/tldw_server/issues/1564#issuecomment-4730743066
- https://github.com/rmusser01/tldw_server/pull/2374
modified_files:
- Docs/superpowers/plans/2026-06-17-claude-acp-live-certification.md
- Docs/Development/ACP_Compatibility_Matrix.md
- Docs/Published/User_Guides/Integrations_Experiments/Anthropic_ClaudeCode_ClaudeSDK_Setup.md
- Docs/Published/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md
- tldw_Server_API/Config_Files/agents.yaml
- tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py
- tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the evidence-first ACP work to validate the pinned Claude Code ACP adapter, run direct and backend E2E certification where available, and update support claims/docs only when live evidence passes. GitHub issue: https://github.com/rmusser01/tldw_server/issues/1564
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Direct Claude Code ACP initialize probe passes with pinned adapter on PATH.
- [x] Backend live E2E passes through `tldw_server` ACP endpoints for `claude_code`.
- [x] Registry/docs only claim `supported_with_caveats` for the verified macOS host profile and retain unverified caveats.
- [x] Focused ACP tests and security scan results are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-17-claude-acp-live-certification.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Baseline focused ACP registry/smoke tests passed before support-claim edits: `82 passed, 6 warnings`.
- Installed pinned `@agentclientprotocol/claude-agent-acp@0.40.0` in a disposable `/tmp` workspace and verified `node v26.0.0`, `npm 11.12.1`, `npx 11.12.1`, and `claude 2.1.177 (Claude Code)`.
- Direct ACP stdio certification passed: `PASS acp_initialize_probe` with the adapter on `PATH`.
- Backend live E2E passed on local loopback server `127.0.0.1:18004` for `ACP_AGENT_PROFILE=claude_code`: session `44b71fdb-c014-41e1-8b56-14fa310039e6`, `stop_reason=end_turn`, `events_total=2`, `artifacts_total=0`, `diagnostics_total=0`.
- Updated Claude Code registry/docs from `documented_unverified` / `documented_only` to `supported_with_caveats` / `live_e2e_tested` only for the verified macOS host runner profile.
- Preserved caveats for the external adapter requirement, artifact-producing workflows, non-empty MCP injection, sandbox behavior, reviewer-loop behavior, diagnostics payloads, and unverified host profiles.
- Added regression coverage for the seeded Claude profile and registry manifest metadata.
- PR maintenance pass rebased on `origin/dev` (branch was already up to date) and addressed all current Gemini/Qodo review threads: moved Claude Code ACP under the supported setup section, replaced machine-specific plan venv paths with `source .venv/bin/activate`, and revised `agents.yaml` notes to include reproducible command/runtime setup while moving ephemeral branch/commit/session details to the compatibility matrix.
- Review-pass verification: focused ACP registry/smoke suite passed with `83 passed, 6 warnings`; ACP health suite passed with `21 passed, 6 warnings`. No Python files changed in the review pass, so no new Bandit run was required beyond the existing PR Bandit record.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Claude Code ACP via pinned @agentclientprotocol/claude-agent-acp@0.40.0 is now documented and seeded as supported_with_caveats/live_e2e_tested for the verified macOS host runner profile. Remaining caveats are explicitly preserved for the external adapter requirement, artifact-producing workflows, non-empty MCP injection, sandbox behavior, reviewer-loop behavior, diagnostics payloads, and unverified host profiles. PR review feedback was addressed and the branch was checked against latest origin/dev. Verification: focused ACP registry/smoke suite passed; ACP health suite passed; Bandit on touched Python tests produced only pytest B101 assert findings and passed with B101 skipped.
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
