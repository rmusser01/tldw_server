---
id: TASK-2362
title: Define ACP custom profile evidence contract
status: In Progress
references:
- https://github.com/rmusser01/tldw_server/issues/2052
- https://github.com/rmusser01/tldw_server/pull/2367
modified_files:
- Docs/Development/ACP_Certification_Checklist.md
- Docs/Development/ACP_Compatibility_Matrix.md
- Docs/Development/ACP_OSS_Custom_Certification_2026_05_11.md
- Helper_Scripts/Testing-related/acp_certification_smoke.py
- tldw_Server_API/Config_Files/agents.yaml
- tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py
- tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py
- tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py
- tools/tldw-agent/internal/acp/runner.go
- tools/tldw-agent/internal/acp/runner_test.go
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track GitHub issue #2052: define and enforce the minimum evidence bundle for certifying concrete custom ACP profiles while keeping the generic custom template caveated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Minimum custom-profile evidence requirements are documented in ACP compatibility/certification surfaces.
- [ ] #2 Compatibility matrix distinguishes generic template support from concrete live-certified custom profiles.
- [ ] #3 Setup/registry language avoids generic custom support claims without evidence.
- [ ] #4 Parent GitHub issue #1563 is updated with the result without closing it unless all child work is complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-16-acp-custom-profile-evidence-contract.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the custom-profile evidence contract in the ACP certification manifest, tightened custom-template wording across Python/API/Go surfaces, and updated ACP compatibility/checklist docs to reserve the seeded custom profile as template-only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Opened PR https://github.com/rmusser01/tldw_server/pull/2367 for GitHub issue #2052. The PR defines the concrete custom ACP profile evidence contract, keeps the seeded custom profile template-only, aligns Python/API/Go setup wording, and updates ACP compatibility/checklist docs. Verification recorded: focused ACP/smoke pytest passed (78 passed, 6 warnings), Go config/acp package tests passed, Bandit returned 0 results/0 errors on touched Python scope, generated custom manifest JSON validates and contains the expected contract markers, and git diff --check passed. Posted tracker updates on #2052 (https://github.com/rmusser01/tldw_server/issues/2052#issuecomment-4719678486) and #1563 (https://github.com/rmusser01/tldw_server/issues/1563#issuecomment-4719681257). Parent issue #1563 remains open for the broader ACP release tracker.
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
