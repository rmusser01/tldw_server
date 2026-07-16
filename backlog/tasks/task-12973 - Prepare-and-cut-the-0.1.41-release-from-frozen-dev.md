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
modified_files:
- CHANGELOG.md
- README.md
- pyproject.toml
- tldw_Server_API/app/main.py
- Docs/mkdocs.yml
- Docs/RELEASE_NOTES.md
- Docs/Published/RELEASE_NOTES.md
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
