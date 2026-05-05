---
id: TASK-44
title: Expose sandbox runtime network policy metadata
status: Done
assignee: []
created_date: '2026-05-04 14:58'
updated_date: '2026-05-05 00:16'
labels:
  - sandbox
  - runtime-discovery
  - network-policy
dependencies: []
documentation:
  - Docs/Sandbox/sandbox-architecture-doctrine.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/Sandbox/sandbox-security-policy-matrix.md
  - Docs/superpowers/specs/2026-05-04-sandbox-network-policy-metadata-design.md
  - >-
    Docs/superpowers/plans/2026-05-04-sandbox-network-policy-metadata-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a structured, machine-readable network policy contract to sandbox runtime discovery. The contract must separate static security guarantees from current host readiness, preserve existing compatibility booleans, and align with the sandbox architecture doctrine, runtime capability inventory, and security policy matrix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each RuntimeType has static network policy metadata with import-time completeness validation.
- [x] #2 Runtime discovery includes required network_policy_contract fields for deny_all and allowlist without removing existing compatibility fields.
- [x] #3 Host-local runtimes seatbelt and worktree machine-report strict deny_all and allowlist as unsupported.
- [x] #4 vz_linux reports deny_all as host-gated strict and allowlist as unsupported without implying network readiness from availability.
- [x] #5 Public API docs, published API docs, runtime capability inventory, and security policy matrix describe the new contract and warn that availability is not a security guarantee.
- [x] #6 Focused tests cover metadata completeness, discovery payload shape, host-local negative claims, and schema requirements.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing contract/schema tests for network policy metadata. 2. Add runtime metadata map, validation, safe accessor, discovery projection, and schema models. 3. Update public docs and sandbox contract docs. 4. Run focused pytest, Bandit, and git diff --check before finalizing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved design reviewed for risk before implementation. Main adjustments: use network_policy_contract with support_state, strict_enforcement, and readiness_source; keep existing readiness booleans for compatibility; avoid implying availability is a security guarantee.

Verification: focused runtime inventory/docs tests passed: 19 passed, 2 warnings. Bandit on touched Python sandbox/schema files reported 0 findings. git diff --check passed.

PR review pass: Qodo opened four active threads on PR #1269. Verified against current branch: missing class docstrings, two PEP8 wrapping issues, and runtime_network_policy_metadata signature mismatch all apply and will be fixed in this branch.

PR review fixes applied: added class docstrings for the new Pydantic models, wrapped the long metadata declaration and ValueError line, changed runtime_network_policy_metadata to accept RuntimeType | str, and removed the now-unneeded test type ignore. Verification after fixes: focused tests 19 passed, 2 warnings; Bandit 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a structured sandbox runtime network_policy_contract to runtime discovery, with complete runtime metadata, schema coverage, focused tests, and public/operator documentation. The contract separates static runtime network-policy posture from current host readiness while preserving existing compatibility booleans.
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
