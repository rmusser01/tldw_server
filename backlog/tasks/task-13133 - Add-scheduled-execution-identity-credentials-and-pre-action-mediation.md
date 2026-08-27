---
id: TASK-13133
title: Add scheduled execution identity credentials and pre-action mediation
status: To Do
created_date: 2026-08-24 17:39
dependencies:
- TASK-13129
labels:
- scheduled-tasks
- phase-4d
- security
- credentials
- governance
- rbac
priority: High
references:
- Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md
- Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-feasibility-gate-implementation-plan.md
updated_date: 2026-08-24 17:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement scheduled execution identity, version-bound act-as and credential-use grants, brokered per-action credential issuance, exact action/argument mediation, and live revocation checks. Remove ambient credential channels from scheduled mode and ensure ACP/MCP policy may only narrow Scheduled Tasks authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Execution subject, authorizer, delegation, credential owner/reference/version/scope/expiry, target, and policy fingerprints are bound into immutable scheduled authority.
- [ ] #2 Every mediated action and credential issue rechecks live grant, principal, credential, deny-policy, signer, and kill-switch state.
- [ ] #3 Scheduled mode receives brokered per-action credentials and cannot inherit process or session environment secrets.
- [ ] #4 ACP remembered allows, session/batch approvals, adapter defaults, wildcard permissions, and model-selected tiers cannot expand exact Scheduled Tasks authority.
- [ ] #5 Pre-action approval is durable and is created before side effects when exact authority is absent but approval is permitted.
- [ ] #6 Grant/broker installation and upgrade, credential-provider outage, trust/revocation health, and mediation outage behavior fail closed and publish bounded operator evidence.
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
