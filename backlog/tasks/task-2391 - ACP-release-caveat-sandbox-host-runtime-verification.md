---
id: TASK-2391
title: ACP release caveat sandbox host-runtime verification
status: In Progress
labels:
- ACP
- release-caveat
- sandbox
- verification
references:
- https://github.com/rmusser01/tldw_server/issues/2400
- https://github.com/rmusser01/tldw_server/issues/2398
modified_files:
- Docs/Development/ACP_Sandbox_Host_Runtime_Verification_2026_06_19.md
- Docs/Development/ACP_Production_Readiness.md
- Docs/Development/Agent_Client_Protocol.md
- Docs/Development/ACP_Certification_Checklist.md
- Docs/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md
- IMPLEMENTATION_PLAN_acp_sandbox_host_runtime_verification_2400.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track GitHub issue #2400: resolve or explicitly accept the ACP sandbox host-runtime release caveat by verifying a selected host runtime or documenting that sandbox support is not claimed for this release.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
IMPLEMENTATION_PLAN_acp_sandbox_host_runtime_verification_2400.md

Stages:
1. Host runtime evidence: record macOS host details and Docker/Lima/macOS virtualization availability.
2. Release posture: select verified runtime support or explicitly decline sandbox-backed ACP support for this release host.
3. Documentation updates: add durable #2400 evidence and link it from ACP readiness/setup surfaces.
4. Verification and handoff: run doc checks and targeted ACP sandbox fail-closed tests where feasible.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added durable ACP sandbox host-runtime verification evidence for #2400. Docker-backed sandbox lifecycle passed on the recorded macOS/Docker Desktop host; Lima and VZ remain unverified; named downstream-agent sandbox support remains gated on agent-specific workspace-live-e2e evidence with ACP_E2E_EXPECT_SANDBOX=1. Verification: Docker lifecycle pytest 1 passed; workspace allowlist helper pytest 17 passed; ACP runtime policy/sandbox runner pytest 33 passed; docker cleanup probes returned no tldw containers/networks; git diff --check passed. PR: https://github.com/rmusser01/tldw_server/pull/2410. Parent #2398 comment: https://github.com/rmusser01/tldw_server/issues/2398#issuecomment-4753238301.
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
