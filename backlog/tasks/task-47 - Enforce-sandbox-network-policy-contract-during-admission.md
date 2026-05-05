---
id: TASK-47
title: Enforce sandbox network policy contract during admission
status: Done
assignee: []
created_date: '2026-05-05 00:38'
updated_date: '2026-05-05 00:43'
labels:
  - sandbox
  - runtime-admission
  - network-policy
dependencies: []
documentation:
  - Docs/Sandbox/sandbox-architecture-doctrine.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/Sandbox/sandbox-security-policy-matrix.md
  - Docs/superpowers/specs/2026-05-05-sandbox-network-policy-admission-design.md
  - >-
    Docs/superpowers/plans/2026-05-05-sandbox-network-policy-admission-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Use the sandbox runtime network_policy_contract as the shared admission source of truth for session and run requests. Admission must fail closed for unsupported or non-strict runtime/network-policy combinations while keeping runtime preflight as the dynamic readiness layer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SandboxPolicy rejects unsupported or non-strict runtime/network-policy combinations for sessions and direct runs after trust/profile defaults are applied.
- [x] #2 Invalid network_policy values still fail with unsupported_network_policy before runtime execution.
- [x] #3 Host-local runtimes seatbelt and worktree are rejected for strict deny_all and allowlist admission rather than implying strict network enforcement from availability.
- [x] #4 vz_linux deny_all remains admissible at the static contract layer while allowlist is rejected.
- [x] #5 Focused tests cover session and run admission, default profile network_policy application, host-local negative claims, and static contract helper behavior.
- [x] #6 Documentation or implementation notes explain that static contract admission is separate from dynamic preflight readiness.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation complete. Added static network_policy_contract admission in SandboxPolicy after trust-profile/default network policy normalization for sessions and direct runs. RED: new admission tests failed before implementation with host-local deny_all/allowlist, invalid policy, and vz_linux allowlist admitted. GREEN: new focused tests passed after implementation.

Verification: pytest test_network_policy_contract_admission.py test_runtime_inventory_contract.py test_lima_strict_admission.py => 25 passed, 2 warnings. pytest test_macos_runtime_admission.py test_macos_runtime_service_dispatch.py test_worktree_runner.py test_seatbelt_runner.py => 40 passed, 2 warnings. Bandit policy.py => 0 findings. git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Enforced the runtime network_policy_contract during SandboxPolicy session/run admission. Unsupported, scaffold, non-strict, and invalid network policy combinations now fail closed before enqueue/session creation while dynamic runtime preflight remains the readiness layer.
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
