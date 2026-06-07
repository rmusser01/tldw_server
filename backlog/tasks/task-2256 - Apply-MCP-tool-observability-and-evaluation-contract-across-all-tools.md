---
id: TASK-2256
title: Apply MCP tool observability and evaluation contract across all tools
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-05 01:37'
labels:
  - mcp
  - observability
  - evals
  - tools
dependencies: []
references:
  - Docs/superpowers/specs/2026-06-04-mcp-git-read-tools-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define and apply the shared MCP tool observability/evaluation metadata and execution event contract across built-in modules, governed virtual commands, and external/federated tool surfaces so standalone MCP operators can compare model tool-use quality, tool prompt variants, and profile grant effectiveness across all tools.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented shared MCP tool observability/eval metadata helpers; create_tool_definition now fills sanitized metadata.eval defaults while preserving safe explicit eval blocks.

Protocol tools/list enriches copied descriptors, prepare_tool_call normalizes resolved definitions, and tools/call responses now include safe execution eval metadata with structured results receiving embedded eval when absent.

External federated virtual tools now attach local external_federated eval metadata and strip upstream eval blocks to prevent untrusted prompt-id/metadata override.

Docs updated in tldw_Server_API/app/core/MCP_unified/README.md; plan recorded in Docs/superpowers/plans/2026-06-04-mcp-tool-observability-contract-implementation-plan.md.

Verification: 126 targeted MCP tests passed; Bandit on touched production files exited 0 with zero findings.

PR review fixes: addressed valid Gemini/CodeRabbit/Qodo feedback by allowlisting eval profile IDs, merging safe partial explicit eval metadata over inferred defaults, rejecting non-string explicit eval scalar fields, guarding list cleanup for null/scalar strings, logging non-critical eval enrichment failures, keeping top-level execution eval canonical, and documenting the profile-id constraint. Review-fix verification: 131 targeted MCP tests passed; Bandit on touched production MCP files exited 0 with zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the cross-tool MCP observability/evaluation contract with sanitized definition metadata, protocol execution eval enrichment, direct external federation coverage, documentation, and targeted verification.
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
