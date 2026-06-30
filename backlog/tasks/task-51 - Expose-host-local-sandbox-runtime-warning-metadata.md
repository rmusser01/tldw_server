---
id: TASK-51
title: Expose host-local sandbox runtime warning metadata
status: Done
assignee: []
created_date: '2026-05-05 01:14'
updated_date: '2026-05-05 02:03'
labels:
  - sandbox
  - runtime-discovery
  - security-contract
dependencies: []
documentation:
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/Sandbox/sandbox-security-policy-matrix.md
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
  - >-
    Docs/superpowers/specs/2026-05-05-sandbox-host-local-warning-metadata-design.md
  - >-
    Docs/superpowers/plans/2026-05-05-sandbox-host-local-warning-metadata-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add additive runtime discovery metadata that explicitly warns clients when a runtime is host-local and not VM-grade, starting with seatbelt and worktree. This follows the Phase 2 sandbox security-contract roadmap and should not change runtime admission or execution behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /api/v1/sandbox/runtimes exposes additive machine-readable warning metadata for host-local runtimes without removing existing fields.
- [x] #2 seatbelt and worktree advertise explicit not-VM-grade/not-untrusted-eligible warnings while VM-grade runtimes do not receive host-local warnings.
- [x] #3 Schema/docs describe the new warning field as advisory discovery metadata rather than an admission decision.
- [x] #4 Focused tests cover service discovery and API schema behavior for the new field.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing tests for additive isolation warning metadata in SandboxService.feature_discovery() and the /api/v1/sandbox/runtimes response shape. 2. Add a small warning-code type/helper in runtime_capabilities.py derived only from RuntimeIsolationMetadata. 3. Wire the helper into SandboxService._preflight_fields() and SandboxRuntimeInfo without changing admission/preflight behavior. 4. Update sandbox runtime inventory/security docs plus local design/plan docs to define warnings as advisory discovery metadata. 5. Run focused tests, py_compile/Bandit on touched Python scope, and git diff --check before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented additive isolation_warnings discovery metadata. RED: focused service/API tests failed because isolation_warnings was absent. GREEN: 14 focused sandbox discovery/API tests passed after adding RuntimeIsolationWarningCode, runtime_isolation_warnings(), service wiring, schema field, and docs. Verification: py_compile on touched Python modules passed; Bandit on touched Python modules produced 0 findings; git diff --check passed. Broader test_feature_discovery_flags.py file was attempted but timed out in existing FastAPI lifecycle/job-worker shutdown while running unrelated lifecycle-heavy tests, so final verification used the focused service/API tests that cover this slice.

PR #1278 review fix pass: Qodo identified that public Sandbox API guides still omit the new isolation_warnings response field; Gemini identified non-portable absolute local Python paths in the implementation plan. Plan: update both public guides and published guide examples/narrative, replace absolute verification commands with portable activated-environment python commands, rerun focused docs/diff checks, commit, push, and resolve review threads.

PR #1278 review fixes completed: public and published Sandbox API guides now include isolation_warnings in the runtime discovery field list, example response, advisory semantics, and posture mapping. The implementation plan now uses portable python -m commands instead of local absolute venv paths. Verification: focused sandbox discovery/API pytest passed (14 tests), git diff --check passed, and the reviewed docs/plan absolute-path scan returned no hits.

Additional PR #1278 review pass: CodeRabbit requested test hardening so non-host-local warning assertions cover every discovered runtime and API response shape checks validate every runtime entry rather than only the first. Verified current tests still have those narrower assertions; updating tests only.

Additional CodeRabbit review hardening completed: service discovery warning test now verifies every discovered non-host-local runtime lacks host_local_boundary, and the API shape test verifies required fields plus list-typed isolation_warnings for every runtime entry. Verification: focused sandbox discovery/API pytest passed (14 tests) and git diff --check passed. Bandit was not rerun for this final test/docs-only hardening slice; prior production-code Bandit run for the feature had 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added advisory host-local isolation warning metadata to sandbox runtime discovery. The new isolation_warnings field is derived from existing RuntimeIsolationMetadata and currently flags seatbelt and worktree with host_local_boundary, not_vm_grade_isolation, and not_untrusted_eligible. This is additive discovery metadata only; admission, preflight, and runtime execution behavior are unchanged. Updated schema and sandbox docs, plus focused tests for service discovery and API response shape. Verification passed for focused pytest, py_compile, Bandit with 0 findings, and git diff --check; the broader feature_discovery_flags file timed out in unrelated lifecycle-heavy shutdown code and is documented as a known verification skip for this slice.

PR review follow-up documented isolation_warnings in both public Sandbox API guides and removed local absolute Python paths from the implementation plan. Focused sandbox discovery/API pytest passed again, git diff --check passed, and the reviewed docs/plan path scan returned no hits.

Final PR review hardening expanded test coverage so runtime warning absence and API response shape are checked across all discovered runtimes, not only selected entries. Focused sandbox discovery/API pytest passed and git diff --check passed.
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
