---
id: TASK-2234
title: Resolve MCP tooling preset spec open questions
status: Done
labels:
- mcp-unified
- design
- profiles
- tools
- pr
priority: medium
modified_files:
- Docs/superpowers/specs/2026-06-03-mcp-default-profile-tooling-presets-design.md
- backlog/tasks/task-2234 - Resolve-MCP-tooling-preset-spec-open-questions.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the MCP default profile tooling presets draft spec with user decisions for browser inspection path, tool-search ranking, recommendation catalog mutability, and exact external MCP install targets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Updated the draft spec with the user decisions:

- CDP is the first browser inspection path.
- Tool search does not use semantic search initially; it filters by profile
  grants and workspace assignment, partitions by installation status, applies
  category filters, then BM25 scores the allowed catalog.
- Operators can patch recommendation catalog metadata separately from
  executable policy; recommendation patches do not grant authority.
- `ChromeDevTools/chrome-devtools-mcp` is the initial `exact_target` external
  MCP binding for the browser/CDP category.

Also moved these items out of open questions into resolved implementation
decisions and added a test requirement for the tool-search ranking behavior.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the spec's browser path, tool-search ranking, recommendation catalog
mutability, and initial exact external MCP target decisions. Verification: git
diff --check passed; rg confirmed the resolved-decision and ranking-test
sections. Bandit skipped because this is documentation-only work.
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
