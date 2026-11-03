---
id: TASK-15418
title: ACP runner writes a read-only Go module cache into the repo tree
status: To Do
labels:
- backend
- acp
- dev-experience
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found 2026-10-31 and 2026-11-02 while removing worktrees after test runs. config.txt [ACP] starts the runner as 'go run ./cmd/tldw-agent-acp' with runner_env HOME=./acp_runner_home, so Go puts its module cache at tldw_Server_API/Config_Files/acp_runner_home/go/pkg/mod. Go makes that cache read-only by design. The path is gitignored, but 'git worktree remove' then fails with 'Permission denied' and leaves a half-deleted directory, and 'rm -rf' of the directory fails too. The same directory holds a tracked file (.tldw-agent/config.yaml), so deleting the whole directory to clean up also deletes a tracked file. Likely fix: add GOFLAGS=-modcacherw to runner_env (Go then writes the module cache writable), or point GOMODCACHE outside the repo tree. Check first which tests start the runner, and whether a deployed server relies on the cache living under acp_runner_home.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After the tests that start the ACP runner, the files under tldw_Server_API/Config_Files/acp_runner_home are writable, so 'git worktree remove' and 'git clean -ffdx' succeed without a chmod
- [ ] #2 The tracked acp_runner_home/.tldw-agent/config.yaml is unchanged, and a deployed runner still starts with the documented config
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
