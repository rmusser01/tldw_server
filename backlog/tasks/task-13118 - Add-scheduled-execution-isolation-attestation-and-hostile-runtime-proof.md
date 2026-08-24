---
id: TASK-13118
title: Add scheduled execution isolation attestation and hostile runtime proof
status: To Do
created_date: 2026-08-24 17:38
dependencies:
- TASK-13117
labels:
- scheduled-tasks
- phase-4d
- security
- sandbox
- attestation
priority: High
references:
- Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md
- Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-feasibility-gate-implementation-plan.md
updated_date: 2026-08-24 17:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the server-verified isolation attestation and hostile-agent proof required by Phase 4D. Bind tenant, workspace, runtime, image, mounts, egress, credential policy, signer, validity, isolation profile, and dispatch subject to a live trust root with signer revocation. Prove host file, uncontrolled network, subprocess, direct MCP/tool, inherited-secret, and ambient-credential bypass attempts fail for each claimed deployment class.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Attestations are signature-verified against configured trust roots and reject forged, stale, future, revoked, wrong-tenant, wrong-workspace, wrong-runtime, wrong-image, wrong-policy, and fingerprint-mismatched evidence.
- [ ] #2 The deterministic hostile probe covers host files, uncontrolled network, subprocess bypass, direct MCP/tool access, inherited secrets, and ambient credentials under the exact attested profile.
- [ ] #3 No runtime is promoted by static metadata, unit mocks, self-assertion, or generic sandbox lifecycle evidence.
- [ ] #4 Evidence output is bounded and contains no prompt, credentials, secret values, raw logs, or local absolute paths.
- [ ] #5 Watchlists, standalone Agent Tasks, and non-scheduled Sandbox users retain their existing behavior.
- [ ] #6 Runtime installation, upgrade invalidation, health, attestation freshness, signer revocation, and fail-closed outage behavior are tested and surfaced through bounded operator evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
