---
id: TASK-183
title: Add shared sandbox audit metadata contract
status: In Progress
assignee:
  - Codex
created_date: '2026-05-09 19:31'
updated_date: '2026-05-09 19:58'
labels:
  - sandbox
  - security
  - audit
  - runtime-policy
dependencies: []
references:
  - tldw_Server_API/app/core/Sandbox/service.py
  - tldw_Server_API/app/api/v1/endpoints/sandbox.py
  - tldw_Server_API/app/core/Sandbox/limits.py
  - tldw_Server_API/tests/sandbox/test_sandbox_run_limit_audit.py
documentation:
  - Docs/Sandbox/sandbox-security-policy-matrix.md
  - Docs/Sandbox/sandbox-architecture-doctrine.md
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the sandbox module roadmap Phase 2 security-hardening work by making sandbox audit metadata contract-shaped instead of duplicating ad hoc dictionaries in endpoint and background-service paths. The slice should centralize safe run-completion metadata for lifecycle/policy/artifact-limit fields, keep path/secret minimization guarantees, and preserve existing audit event behavior. Do not broaden this into a generic audit subsystem rewrite or runtime execution changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A shared sandbox audit metadata helper/schema exists for run completion metadata and is used by both endpoint and background-service run-completion audit paths.
- [x] #2 The helper includes requested/effective runtime where available, trust/network/policy identifiers where available, spec version, exit/outcome fields, status reason where available, and existing bounded limit metadata without exposing raw artifact paths or environment values.
- [x] #3 Focused tests cover metadata shape, path minimization, and parity between endpoint-style and background-service audit metadata construction.
- [x] #4 Existing limit audit behavior remains compatible with current audit tests.
- [x] #5 Documentation or inline contract notes connect the helper to the sandbox security policy matrix audit expectations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the current endpoint and background-service sandbox run-completion audit metadata to identify overlapping fields and existing limit metadata behavior.
2. Add focused failing tests in the sandbox audit test module for a shared safe metadata builder, including path minimization and endpoint/background parity.
3. Implement a small shared helper under tldw_Server_API/app/core/Sandbox/ that builds run-completion audit metadata from RunStatus plus optional request/spec context.
4. Replace the duplicated metadata dictionaries in SandboxService._audit_run_completion and the sandbox endpoint completion audit path with the helper while preserving existing event types/actions/results.
5. Run focused pytest, py_compile or import checks as needed, git diff --check, and Bandit on touched Python paths; update acceptance criteria and final task notes with evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
User asked to continue after PR #1438 merged. Existing trust-level and runtime capability contracts already cover the initially considered security-policy slice, so this task targets the remaining Phase 2 audit metadata contract gap from the sandbox security matrix.

Implemented shared run-completion audit metadata helper and wired endpoint/background service paths. Red check: focused audit test file failed on missing helper and missing spec argument. Green checks: python -m pytest tldw_Server_API/tests/sandbox/test_sandbox_run_limit_audit.py -q passed 4 tests; endpoint smoke test passed 1 test; py_compile passed for touched production modules; git diff --check passed; Bandit on touched production files reported 0 results and 0 errors at /tmp/bandit_sandbox_audit_metadata_contract.json.

PR review follow-up verified against current branch. Plan: add focused regression tests and run them red; update audit metadata and call sites minimally; run focused pytest, endpoint smoke, py_compile, diff check, Bandit; update the task, commit, and push.

PR review fixes applied and verified. Red check failed on Windows drive-relative base_image redaction and omitted requested_runtime. Green checks passed: audit pytest 8 passed, endpoint smoke 1 passed, py_compile passed for touched production modules, git diff --check passed, Bandit reported 0 results and 0 errors in /tmp/bandit_sandbox_audit_metadata_contract_review.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented a shared sandbox run-completion audit metadata contract and wired it into both REST endpoint and background-service completion audit paths. The helper centralizes runtime, requested/effective runtime, trust/network policy, spec version, outcome, status reason, policy/image identifiers, and bounded limit metadata while omitting raw artifact paths, raw capture patterns, and host-path base image values. Focused tests cover path minimization and service metadata parity; verification passed for focused pytest, endpoint smoke, py_compile, git diff --check, and Bandit. PR: https://github.com/rmusser01/tldw_server/pull/1441

PR review follow-up: added helper docstrings, redacted Windows drive-relative base_image values, preserved omitted requested_runtime as None, split the broad metadata contract test into focused cases, and removed redundant reason_code merges from endpoint/service call sites.
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
