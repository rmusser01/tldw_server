---
id: TASK-12014
title: Document MCP Unified standalone custom profile setup
status: Done
labels:
- docs
- mcp-unified
modified_files:
- apps/mcp-unified/USER_GUIDE.md
- apps/mcp-unified/src/mcp_unified/USER_GUIDE.md
- apps/mcp-unified/README.md
- apps/mcp-unified/src/mcp_unified/README.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a clear user-facing path for discovering available tools and creating or configuring a custom standalone MCP Unified profile.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Standalone MCP Unified docs explain how to discover tools and presets before profile authoring.
- [x] #2 Docs include a minimal custom profile JSON example for create-profile.
- [x] #3 Docs explain preview-profile-tools and explain-policy as validation steps after profile creation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- PR #2555 review follow-up: changed unscoped profile examples to use a generic `<profile-id>` placeholder and switched simple JSON stdin examples from `printf` to `echo` in both USER_GUIDE copies.
- PR #2555 Qodo follow-up: clarified that `explain-policy` explains tool/permission/TTL decisions but does not fully validate authored `policy_document.path_grants`, and split the README quick profile workflow into duplicate-preset and create-from-JSON options.
- Rebase follow-up: synced pre-existing USER_GUIDE drift between the root and packaged documentation copies so package-boundary notes and notebook file-tool guidance are present in both places.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated standalone MCP Unified docs to add a clear custom profile workflow: tool/preset discovery, stored profile preview, minimal create-profile JSON, explain-policy guidance, and troubleshooting guidance for missing tools. Mirrored changes into the packaged src/mcp_unified README and USER_GUIDE copies. Rebased PR #2555 on latest dev, addressed review comments by using generic `<profile-id>` placeholders for unscoped profile examples, `echo` for simple JSON stdin examples, clearer README duplicate/create alternatives, and explicit `explain-policy` path-grant limitations. Also resynced pre-existing USER_GUIDE drift exposed by the rebase. Verification: cmp confirmed README and USER_GUIDE root/src copies match; MCPProfile.model_validate accepted the documented JSON example; git diff --check passed for touched docs and backlog task. Bandit skipped because only Markdown/backlog documentation changed.
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
