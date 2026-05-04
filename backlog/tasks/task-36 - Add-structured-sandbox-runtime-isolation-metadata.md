---
id: TASK-36
title: Add structured sandbox runtime isolation metadata
status: Done
assignee: []
created_date: '2026-05-04 05:19'
updated_date: '2026-05-04 05:33'
labels:
  - sandbox
  - runtime-discovery
  - security
dependencies: []
documentation:
  - Docs/Sandbox/sandbox-architecture-doctrine.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/Sandbox/sandbox-security-policy-matrix.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose machine-readable isolation posture in sandbox runtime discovery so clients and operators do not infer security guarantees from prose notes. The slice should stay additive and align with the sandbox architecture doctrine and security policy matrix: VM-backed runtimes can advertise VM-grade posture, host-local runtimes must be explicit as not VM-grade and not untrusted-eligible, and scaffold runtimes must not overclaim readiness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sandbox runtime discovery includes explicit structured isolation fields for every RuntimeType without removing existing fields.
- [x] #2 seatbelt and worktree discovery report host-local isolation, not VM-grade isolation, and not untrusted-eligible in machine-readable fields.
- [x] #3 VM-backed runtimes report their boundary class and untrusted eligibility consistently with the security policy matrix without treating availability as proof of readiness.
- [x] #4 API schema, focused tests, and public/runtime contract docs are updated to cover the new metadata.
- [x] #5 Focused sandbox tests and diff whitespace checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation added structured runtime isolation metadata fields to sandbox runtime discovery: boundary_class, vm_grade_isolation, and untrusted_eligible. Metadata is centralized in runtime_capabilities.py and wired through SandboxService.feature_discovery() plus the Pydantic response schema.

Verification passed: python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py tldw_Server_API/tests/Docs/test_sandbox_public_docs_contract.py -q --timeout=60; Bandit on touched Python files reported 0 findings; git diff --check produced no output.

Known verification note: tldw_Server_API/tests/sandbox/test_feature_discovery_flags.py timed out in existing FastAPI TestClient teardown / background worker shutdown when run alone and in the combined suite. The timeout occurred in TestClient context manager exit after startup/background jobs, not in the new metadata assertions; no pytest process remained afterward.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added additive, machine-readable sandbox runtime isolation posture metadata so clients can distinguish container, VM-grade, scaffold, and host-local runtimes without parsing prose notes. Updated public and internal sandbox docs, schema, focused contract tests, and the Backlog task record.
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
