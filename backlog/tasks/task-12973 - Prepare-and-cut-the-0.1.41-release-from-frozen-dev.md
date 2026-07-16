---
id: TASK-12973
title: Prepare and cut the 0.1.41 release from frozen dev
status: In Progress
labels:
- release
- documentation
- operations
priority: High
references:
- origin/dev@4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8
- https://github.com/rmusser01/tldw_server/pull/2744
- Docs/Development/Release_Process.md
- Docs/Release_Checklist.md
documentation:
- Docs/superpowers/specs/2026-07-15-release-0.1.41-design.md
- Docs/superpowers/plans/2026-07-15-release-0.1.41-implementation-plan.md
modified_files:
- CHANGELOG.md
- README.md
- pyproject.toml
- tldw_Server_API/app/main.py
- Docs/mkdocs.yml
- Docs/RELEASE_NOTES.md
- Docs/Published/RELEASE_NOTES.md
- Docs/superpowers/specs/2026-07-15-release-0.1.41-design.md
- Docs/superpowers/plans/2026-07-15-release-0.1.41-implementation-plan.md
- backlog/tasks/task-12973 - Prepare-and-cut-the-0.1.41-release-from-frozen-dev.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare release 0.1.41 from frozen origin/dev commit 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8 (through PR #2744), excluding all open PRs. Update authoritative version metadata, CHANGELOG.md, README.md, and release-note entry points; verify the release diff and required gates; merge the reviewed release branch into main; tag the final main merge commit as v0.1.41; publish the GitHub Release; verify publication workflows; and sync main back into dev. Preserve the user's dirty primary checkout by working only in the isolated release worktree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Release scope is frozen at origin/dev 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8 and open PRs are excluded
- [ ] #2 Authoritative project, FastAPI, MkDocs, README, changelog, and release-note version surfaces consistently report 0.1.41
- [ ] #3 CHANGELOG.md contains a curated 0.1.41 rollup for merged PRs included after 0.1.40 through PR #2744
- [ ] #4 README.md current status and What's New accurately summarize 0.1.41
- [ ] #5 Focused release metadata/docs tests, diff checks, and Bandit policy are satisfied
- [ ] #6 Release PR into main is merged only after required checks are green
- [ ] #7 Annotated v0.1.41 tag and GitHub Release point at the final main merge commit
- [ ] #8 Release artifact workflows are verified and main is synchronized back into dev
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-15-release-0.1.41-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Preflight (2026-07-15): release source is frozen at 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8 through PR #2744. Refreshed origin refs without merging or rebasing: HEAD=7c2c7d07e5396fdaf9f4e0dd4d9c9076e8f22e8d, origin/dev=4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8, origin/main=7273cca4926abaa242a682f558fe1d3173f230e7. The frozen SHA is an ancestor of HEAD (git merge-base --is-ancestor exit 0) and the exact merge-base is 4c2ad2070ed63992dac8a97a6c4cf3c7d75f6de8. Post-freeze first-parent history contains only reviewed release design/gate/plan commits 6e1fc05b637933d6fb279ae2222c0e374c42d43a, 8841419f6965a488415fcf00da82bdf75ebd82cf, and 7c2c7d07e5396fdaf9f4e0dd4d9c9076e8f22e8d; the post-freeze first-parent merge query is empty. The frozen inventory from 7273cca4926abaa242a682f558fe1d3173f230e7 through the frozen SHA is exactly 37 first-parent merge commits and ends at PR #2744. v0.1.41 was absent at check time: origin tag query was empty; GitHub Release lookup returned "release not found" (exit 1); PyPI returned HTTP 404 (status-only request 404, HTTP/1.1 fail-mode exit 22; the environment's default HTTP fail-mode command also surfaced the 404 with curl exit 56). Open PRs and all post-freeze PRs/commits are excluded.
Scope clarification: Open PRs and all post-freeze `dev` PRs/commits are excluded; the reviewed 0.1.41 release design, gate, plan, and preflight evidence commits are intentionally present on the isolated release branch.
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
