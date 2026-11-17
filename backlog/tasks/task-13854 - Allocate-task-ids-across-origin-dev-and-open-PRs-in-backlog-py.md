---
id: TASK-13854
title: Allocate task ids across origin/dev and open PRs in backlog-py
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Root-cause fix for the concurrent-session task-id collisions (13403-13408 clobbered; 12905/12946 pre-existing; even cross-branch self-collision observed). _next_task_id now takes the max across local + origin/dev (ls-tree) + open-PR files (gh pr list); explicit --id colliding with a remote source is rejected with the source named; unreachable sources degrade to local-only with a stderr warning. 7 new tests; full backlog-py suite green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An id is allocated above the max of local tasks, origin/dev (fetched first) and open PRs' backlog files, including PRs over 100 files
- [x] #2 An explicit --id held by a remote source is rejected, naming the source
- [x] #3 Each unreachable source warns and allocation falls back to the remaining sources
- [x] #4 The suite never runs the real git or gh
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review fix commit: warns per failing source (open PRs gh, origin/dev) and continues with the rest; git fetch origin dev (30 s) before reading origin/dev, warning 'as of the last fetch' on failure; gh timeout 30 s (local git ls-tree stays 5 s); PRs whose files list is shorter than changedFiles (GitHub caps it at 100) are paged via gh api pulls/N/files, with a warning if that fails; only backlog/ paths count; RemoteIdState built once; _reject_remote_id extracted for TASK-13961; child ids skip the remote lookup; tests/conftest.py stubs collect_remote_ids.
Measured: backlog-py suite 179 passed; with logging git/gh shims first on PATH the suite spawned 0 git and 0 gh (review measured 44 git + 22 gh before). Bandit run on remote_ids.py and repository.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
backlog-py now allocates task ids above the max of local tasks, origin/dev (fetched first) and open PR backlog files, including PRs past GitHub's 100-file listing cap, rejects an explicit --id that a remote source holds (naming the source), and warns per unreachable source while falling back to the remaining ones. The test suite stubs the remote lookup by default so it never runs the real git or gh.
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
