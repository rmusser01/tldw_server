---
id: TASK-2243
title: Update MCP filesystem helper design with governed shell follow-up
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-04 00:43'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Revise the MCP filesystem helper spec and implementation plan to incorporate design-review findings, including bounded traversal, regex hardening, cross-platform hidden semantics, symlink coverage, and a follow-up task for a governed bash/shell alias facade over the existing run command runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents the governed shell facade boundary and leaves bash/shell aliases as follow-up work.
- [x] #2 Implementation plan includes traversal-cap, regex-hardening, hidden-path, and symlink test requirements.
- [x] #3 Implementation plan names the separate governed shell facade alias follow-up task with command-runtime files and tests.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reviewed the existing RunCommandModule and command_runtime parser/adapter/registry against the pasted backend-lead note. The repo already has the safe shell-shaped execution model, so the plan now keeps typed filesystem helpers in the current slice and scopes bash/shell compatibility aliases to a separate governed runtime follow-up. Added requirements for traversal caps, regex pattern guards, explicit dot-segment hidden semantics, and symlink outside-workspace/loop coverage.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the MCP filesystem helper spec and implementation plan with the approved governed shell facade direction. Bash/shell aliases are documented as a separate follow-up over the existing run command runtime, not as raw shell execution. Validation: git diff --check passed. Bandit skipped because no Python/source files changed.
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
