---
id: TASK-54
title: Stabilize sandbox network policy effective support reporting
status: Done
assignee: []
created_date: '2026-05-05 02:11'
updated_date: '2026-05-05 02:34'
labels:
  - sandbox
  - network-policy
  - runtime-discovery
  - security-contract
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1278'
documentation:
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/Sandbox/sandbox-security-policy-matrix.md
  - Docs/Sandbox/sandbox-architecture-doctrine.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make sandbox runtime discovery and admission report effective network policy support consistently so scaffold or unsupported allowlist paths cannot be advertised as currently usable. Scope is the Phase 2 sandbox gap for inconsistent allowlist support: centralize effective support from static network_policy_contract plus current readiness/config, keep unsupported and scaffold policies fail-closed, and preserve existing runtime execution behavior unless the contract already admits it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime discovery exposes consistent effective network policy support for deny_all and allowlist without over-claiming scaffold or unsupported policies.
- [x] #2 Sandbox admission accepts allowlist only when the runtime contract supports strict enforcement and the configured or preflight readiness source is ready.
- [x] #3 Docker allowlist discovery reflects configured enforcement readiness and does not treat deny-all fallback as true allowlist support.
- [x] #4 Firecracker scaffold allowlist, Lima allowlist, VZ allowlist, seatbelt allowlist, and worktree allowlist remain rejected with stable normalized reasons.
- [x] #5 Focused tests cover discovery consistency and admission behavior for the affected runtimes, and sandbox network policy docs are updated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED tests for effective network-policy support and Docker readiness-gated allowlist admission. 2. Add a shared runtime_network_policy_effective_support() helper derived from network_policy_contract plus preflight readiness. 3. Feed Docker and Firecracker readiness into collect_runtime_preflights(). 4. Use effective support for SandboxPolicy admission and SandboxService discovery booleans. 5. Update sandbox network-policy docs and run focused tests, py_compile, Bandit, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented network policy effective support gating. RED: Docker allowlist admission with allowlist readiness false did not raise, and discovery tests failed because runtime_network_policy_effective_support() did not exist. GREEN: added the helper, Docker/Firecracker readiness facts, policy admission wiring, and discovery wiring. Docker allowlist is now effectively supported only with Docker available plus egress enforcement plus granular enforcement; Docker network=none fallback is treated as deny_all, not allowlist. Firecracker allowlist remains scaffold and is never advertised as effective support. Verification: focused sandbox admission/discovery/Lima tests passed (33 tests), py_compile passed for touched production modules, Bandit reported 0 findings for touched production modules, and git diff --check passed.

PR review follow-up: verifying and fixing Qodo findings for _settings_flag() observability and missing-preflight SandboxPolicy admission compatibility.

PR review fixes completed. Added regression coverage for missing-preflight Docker deny_all admission and _settings_flag() observability. Implemented static supported-only fallback for direct SandboxPolicy callers without preflights and warning logging for readiness flag read failures. Verification: 44 sandbox policy/discovery tests passed, py_compile passed, Bandit reported 0 findings for touched Sandbox modules, and git diff --check passed.

Additional PR review follow-up: verifying Gemini cleanup suggestions for shared noncritical exceptions, _settings_flag() signature redundancy, and redundant service effective-support calculations.

Additional Gemini review fixes completed. Centralized the shared config noncritical exception tuple for policy/runtime capability config parsing, simplified _settings_flag() to a single setting name, and reused _preflight_fields() results for Docker/Firecracker/Lima effective support in feature discovery. Verification rerun: 44 sandbox policy/discovery tests passed, py_compile passed, Bandit reported 0 findings for touched Sandbox modules, and git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Centralized sandbox network-policy effective support so discovery and admission use the same static contract plus current readiness calculation. Docker allowlist now requires granular egress enforcement readiness; scaffold/unsupported allowlist paths remain fail-closed and cannot be over-advertised through legacy discovery booleans. Updated network-policy docs and focused tests.

PR review follow-up: fixed Qodo findings by adding operator-visible logging for readiness flag read failures and preserving no-preflight direct SandboxPolicy callers for statically supported Docker deny_all while keeping host-gated/scaffold policies fail-closed.

Additional PR review follow-up: addressed Gemini cleanup comments by centralizing shared config exception handling, simplifying readiness flag lookup, and removing redundant service effective-support calculations.
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
