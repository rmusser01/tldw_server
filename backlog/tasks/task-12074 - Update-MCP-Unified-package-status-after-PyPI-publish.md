---
id: TASK-12074
title: Update MCP Unified package status after PyPI publish
status: Done
assignee: []
created_date: '2026-06-30 05:43'
updated_date: '2026-06-30 06:06'
labels:
  - mcp
  - packaging
  - pypi
  - docs
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reflect the successful first PyPI publish of mcp-unified 0.1.1 by updating package metadata, package-local docs, gateway status expectations, and focused tests from not-published to published while keeping the package status internal-experimental.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 package metadata reports publishing_status published
- [x] #2 README and USER_GUIDE install/status guidance no longer says the package is not published
- [x] #3 gateway status no longer emits package_not_published for default package metadata
- [x] #4 focused package metadata, CLI, and status tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR review follow-up: Qodo found the repo-wide MCP docs contract still enforced old not-published wording; Gemini found the CLI subprocess test overwrote PYTHONPATH. Verified both comments against current code and reopening the task for the minimal PR follow-up commit.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated MCP Unified metadata and package-local docs to reflect the successful PyPI publish while keeping package_status internal-experimental. Updated gateway/CLI/package-boundary tests and RC fixture evidence from not-published to published. PR review follow-up also updated the repo-wide MCP docs contract to allow PyPI-published guidance while continuing to forbid unsupported standalone server promises, and changed the CLI subprocess test to prepend the standalone source path without discarding an inherited PYTHONPATH. Verified with the focused MCP package/docs suite, Ruff, diff checks, package-info CLI, publish dry-run, and Bandit touched-scope scans.
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
