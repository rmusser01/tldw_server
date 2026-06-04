---
id: TASK-2232
title: Design MCP default profile tooling presets
status: Done
assignee: []
created_date: 2026-06-03T07:43:00Z
updated_date: 2026-06-03 07:43
labels:
- mcp-unified
- design
- profiles
- tools
dependencies: []
priority: medium
modified_files:
- Docs/superpowers/specs/2026-06-03-mcp-default-profile-tooling-presets-design.md
- backlog/tasks/task-2232 - Design-MCP-default-profile-tooling-presets.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved design spec for default MCP role profile tooling, progressive disclosure, external binding categories, and native tool backlog for tldw_server MCP/ACP workspaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- MCP default profile tooling preset scope is documented, including role coverage,
  progressive disclosure, native/default tool candidates, external binding
  categories, and approval/risk defaults.
- The spec records review outcomes and validation evidence for the design pass.
- Documentation-only security validation is explicitly recorded with the Bandit
  skip rationale.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Drafted the approved brainstorming design into the repo's superpowers spec format. The spec covers researched agent-harness prior art, the default tool substrate, progressive disclosure, role preset matrix, metadata.tooling schema, native tool backlog, external binding categories, runtime enforcement, ACP implications, and testing requirements.

Spec review pass 1 found four issues: unresolved public/internal tool_call semantics, web-search default conflict with vendor-neutral/external-network policy, missing risk-class validator compatibility details, and underspecified safe test-runner command-source constraints. Revised the spec to make tool_call a public profile-scoped bridge with fixed schema and authorization path; make web search recommended-unavailable until configured with grants/provenance; add a risk-class compatibility table and validator requirements; and define safe test runner command identity, args/env/cwd, approval, and audit constraints.

Spec review pass 2 returned APPROVED.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed and spec-review-approved the MCP default profile tooling presets design. Verification: git diff --check passed; spec review pass 2 returned APPROVED. Bandit skipped because this is documentation-only design work.
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
