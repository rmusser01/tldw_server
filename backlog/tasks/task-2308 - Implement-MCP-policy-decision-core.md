---
id: TASK-2308
title: Implement MCP policy decision core
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 18:52'
labels:
  - mcp
  - profiles
  - policy
  - implementation
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-06-07-mcp-profile-policy-decision-model-design.md
  - >-
    Docs/superpowers/plans/2026-06-07-mcp-policy-decision-core-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first MCP/profile policy decision-model slice from the approved plan: package-level deny/ask/allow decision primitives, rule compilation from existing profile policy fields, resolution decision metadata, public exports, redacted explain/simulation helper, tests, Bandit, and validation. Defer catalog visibility changes, path matcher compiler, external MCP wildcard enforcement, shell alias runtime hardening, and hook enforcement to later slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented MCP policy decision core per the plan. Added package-owned PolicyDecision/PolicyDecisionRule/PolicyExplanation models, safe command-rule compilation, tool-only decision evaluation, EffectivePolicyResult.decision metadata, redacted explanation helpers, and public profile package exports.

Review/quality notes:
- Subagent workflow was used for Tasks 1-3 until usage limits blocked further subagents; remaining work was completed inline with local review.
- Review findings addressed during implementation: command argv wildcard/string/set validation, precompiled command-rule revalidation, capability-only decision correctness, command/MCP validation isolation from tool decisions, legacy Bash isolation from tool decisions, and formatting hygiene.

Touched files:
- mcp_unified/profiles/decisions.py
- mcp_unified/profiles/resolution.py
- mcp_unified/profiles/__init__.py
- tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py
- tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first MCP/profile policy decision-model slice. The package now has structured deny/ask/allow decision primitives, safe rule compilation from legacy and structured profile policy fields, optional decision metadata on effective policy resolution, redacted explain/simulation payloads, and public exports from mcp_unified.profiles.

Validation:
- source ../../.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py::test_gateway_profile_runtime_filters_and_allows_default_profile_tools tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py::test_protocol_tools_call_blocks_tool_denied_by_effective_policy -q -> 77 passed, 5 warnings.
- source ../../.venv/bin/activate && python -m black --check mcp_unified/profiles/decisions.py mcp_unified/profiles/resolution.py mcp_unified/profiles/__init__.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py -> pass.
- source ../../.venv/bin/activate && python -m bandit -r mcp_unified/profiles/decisions.py mcp_unified/profiles/resolution.py mcp_unified/profiles/__init__.py -f json -o /tmp/bandit_mcp_policy_decision_core.json -> pass, JSON output written.
- git diff --check -> pass.

Known skips/blockers: final subagent review could not run because the subagent tool hit a usage limit; the remaining review was completed locally with focused tests and diff inspection.
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
