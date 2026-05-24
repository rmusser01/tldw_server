---
id: TASK-498
title: Configure OpenCode and Aider ACP profiles for local llama.cpp certification
status: Done
labels:
- ACP
- certification
- OpenCode
- Aider
- llama.cpp
references:
- https://github.com/rmusser01/tldw_server/issues/1563
documentation:
- Docs/Development/ACP_Compatibility_Matrix.md
- Docs/Development/ACP_Certification_Checklist.md
- tldw_Server_API/Config_Files/agents.yaml
modified_files:
- Docs/Plans/IMPLEMENTATION_PLAN_acp_opencode_aider_llamacpp.md
- Docs/Development/ACP_Compatibility_Matrix.md
- tldw_Server_API/Config_Files/agents.yaml
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Configure the newly installed OpenCode and Aider tooling for the local llama.cpp server at 127.0.0.1:9099, determine which profiles have ACP-compatible entrypoints, run certification where possible, and update compatibility metadata with explicit evidence and caveats.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/Plans/IMPLEMENTATION_PLAN_acp_opencode_aider_llamacpp.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Configured OpenCode v1.15.7 and Aider v0.86.2 for the local llama.cpp server at 127.0.0.1:9099. Direct OpenCode and Aider prompts reached the local model. OpenCode passed direct ACP stdio certification and backend live-E2E as ACP_AGENT_PROFILE=opencode with stop_reason=end_turn, events_total=2, artifacts_total=0, diagnostics_total=0. Aider remains documented_unverified because the installed CLI has no ACP-compatible stdio server entrypoint; the certification manifest reports entrypoint_strategy_missing. Updated agents.yaml, the ACP compatibility matrix, and focused registry tests to preserve those support boundaries.
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
