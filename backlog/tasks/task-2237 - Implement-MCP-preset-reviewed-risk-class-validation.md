---
id: TASK-2237
title: Implement MCP preset reviewed risk-class validation
status: Done
labels:
- mcp-unified
- profiles
- risk
- implementation
priority: medium
modified_files:
- mcp_unified/profiles/presets.py
- tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 2 from the MCP default profile tooling implementation plan: add tests and validator support for reviewed high-risk classes including browser_mutation, deployment_mutation, git_mutation, memory_mutation, and test_execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Reviewed risk classes are accepted when each has explicit approval and high-risk provenance.
- [x] `browser_mutation` without approval returns `browser_mutation_requires_approval`.
- [x] Unknown future risk classes still return `unknown_high_risk_requires_review`.
- [x] Existing safety behavior remains covered by the preset safety test file.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added TDD coverage for `browser_mutation`, `deployment_mutation`, `git_mutation`, `memory_mutation`, and `test_execution`.
- RED evidence: the first targeted pytest run failed with `unknown_high_risk_requires_review` for the reviewed classes and no `browser_mutation_requires_approval` violation.
- Added an extra regression asserting generic `mutating`/`write` approval does not satisfy reviewed high-risk approval; that test failed before tightening `_approval_required_for()`.
- Final verification evidence: targeted pytest passed, `git diff --check` passed, and Bandit reported zero findings for `mcp_unified/profiles/presets.py`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extended preset safety validation so reviewed high-risk classes are known, require exact approval_policy.required_for entries, and require high-risk provenance. Verified with targeted pytest, git diff --check, and Bandit on mcp_unified/profiles/presets.py.
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
