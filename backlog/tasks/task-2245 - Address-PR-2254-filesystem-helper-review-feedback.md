---
id: TASK-2245
title: Address PR 2254 filesystem helper review feedback
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-04 01:40
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR 2254 onto latest dev and address still-valid review findings: secure no-follow path resolution through parent symlinks, cap double-star wildcard expansion, return symlink directory entries from fs.glob without traversing, count directories in grep walk caps, and bump builtin preset version for the tool-surface change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch rebased cleanly onto latest origin/dev.
- [x] #2 Still-valid PR review comments fixed with regression coverage.
- [x] #3 Preset bundle version bumped for the helper-tool surface change.
- [x] #4 Focused filesystem/profile/gateway/package tests, Bandit, and git diff --check pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified all PR 2254 review findings against the rebased code and found them still valid. Rebased codex/mcp-filesystem-helper-plan onto origin/dev without conflicts. Added regression coverage for no-follow parent symlink escapes, excessive **/ patterns, symlink directory entries in fs.glob without traversal, grep directory-only walk caps, and the preset version bump.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR 2254 onto origin/dev after PR 2251 merged, addressed all still-valid unresolved Qodo comments, and validated the filesystem helper changes. Validation: focused red tests failed before implementation, then passed; test_filesystem_module.py passed 33 tests; adjacent MCP profile/discovery/package suite passed 90 tests; Ruff passed for touched Python files; Bandit report /tmp/bandit_pr2254_filesystem_qodo.json had zero findings; git diff --check passed.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Second PR review pass: verified Qodo's three new findings against current code and found all still valid. Fixed fs.glob to return size=null and size_unavailable=true when best-effort file size metadata cannot be read. Added grep_allow_regex gating so regex mode is disabled by default, and added grep_max_total_bytes / grep_max_files aggregate scan budgets with truncation_reasons and remaining_count_known metadata. Updated packaged docs and design/plan artifacts for the new settings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
