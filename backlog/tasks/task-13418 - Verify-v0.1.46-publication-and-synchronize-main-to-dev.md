---
id: TASK-13418
title: Verify v0.1.46 publication and synchronize main to dev
status: In Progress
assignee: []
created_date: '2026-10-02 07:36'
updated_date: '2026-10-02 12:10'
labels:
  - release
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/3074'
  - 'https://github.com/rmusser01/tldw_server/releases/tag/v0.1.46'
documentation:
  - Docs/superpowers/plans/2026-10-01-release-0.1.46-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the approved v0.1.46 release under a unique task ID because late dev contains a separate cookie-model TASK-13408. PR3074 merged as 75cc3eed14d550aee08d47cd6373c0509dfa84a8; v0.1.46 and GitHub changelog release are published. Verify existing server PyPI/GHCR publication and synchronize main into latest dev, preserving later dev changes and the unrelated dirty primary checkout. No new infrastructure or certification scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PyPI 0.1.46 and app/worker/audio-worker GHCR artifacts are verified against the reviewed main merge
- [ ] #2 A normal synchronization PR preserves latest dev changes and dev contains the released main merge
- [ ] #3 Release tracking is closed, only this release plan is removed and its managed worktree is cleaned up safely
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify existing PyPI/GHCR publication for main75cc3eed14. 2. Prepare and verify a normal main-to-dev synchronization PR preserving later dev. 3. Merge when normal checks pass; finish tracking, owned-plan cleanup and managed-worktree cleanup.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Continuation of the release task13408 prepared/published in PR3074. Preserve its historical notes via official archive before merging dev, which already contains a different task13408 for cookie model discovery. Frozen protected source3304f6cf0c2cab8836c6a73308d7210534a759d9, manifest2f10da6c5c91e30356d15568d427e3e8dc2750123ebd48d1e5c2580cf958ab47. Requester selected the updated agent-authored changelog and approved release2026-10-01/Countdown2028-10-01T12:00:00Z. Release six gates, trusted license and CodeQL pass on3b5051d9fb2a301a0cfda33fc425846a1be6cc9e; repaired Sync shard449passed. Existing initializer18passed and relay/activation207passed6fixture-reportedPostgresunavailable skips; metadata80passed; scoped production Ruff/Bandit clean. No full certification claim. PyPI workflow36978827799 and formal Docker workflow36978977010 target released merge75cc3eed14 and are queued.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Prepared codex/release-0.1.46-sync at17eb0e35a3716e2431e0286caa431feb6e75b170 by merging latest dev3caebcfc1e16596f4a352744a9f017017d766165 into released main75cc3eed14. No conflicts; both parent revisions are ancestors and later cookie-model/route-auth fixes are preserved. Officially archived only the historical release TASK13408 before merge; dev cookie TASK13408 remains active unchanged, release continuation TASK13418 is unique. Combined-tree existing release/docs/helper/licensing, initializer and relay/activation checks298passed6fixture-reportedPostgresunavailable skips. Native trusted temporary directory used. Ruff clean for initializer/relay/recovery tests; scoped Bandit(main/init/relay)0findings0errors; diffcheckclean. Frozen protected checkout equality was already verified on tagged release; sync intentionally preserves later dev UI. Publication workflows remain queued/building; do not claim artifact publication or dev sync complete yet.

Registry verification: worker and audio-worker jobs in formal workflow36978977010 completed successfully. Docker buildx registry inspection confirms version0.1.46 and revision75cc3eed14d550aee08d47cd6373c0509dfa84a8 on both published configurations. Worker index sha256:61d9260ff384538f0a3dfabc8ccd491eb6ea9c3521c311553a4c81742e9723db; audio-worker index sha256:2cfcf7ca3eb516dba3cca60ef2a13b762fbdfdd9a329b88ff5c4677281300815. App image remains building; PyPI Build Distributions queued after successful Detect PyPI Version and Release Contract Gate. Local desktop credential lookup stalled; stopped only this inspection and reused installed docker-buildx with empty temporary DOCKER_CONFIG for public registry reads. No publication failure reported.

Formal GHCR workflow36978977010 completed successfully for all three images. Registry inspection verifies app version0.1.46, sourcehttps://github.com/rmusser01/tldw_server and revision75cc3eed14d550aee08d47cd6373c0509dfa84a8; app index digest sha256:dc477950177edd4f39153b08b12edf74e979515499d89a44f3d56df5a852fa34. Worker/audio digests and revision were already verified. PyPI Build Distributions still queued on ubuntu-latest, no failure or environment-approval gate at this job; preserve existing workflow and wait for runner. Artifact AC remains incomplete until published wheel/sdist verified. No sync PR yet.

PyPI workflow36978827799 Build Distributions completed successfully on released revision75cc3eed14. Wheel tldw_server-0.1.46-py3-none-any.whl and sdist tldw_server-0.1.46.tar.gz built; both Twine checks passed and existing backend/API-only artifact validation passed. Publish to PyPI job110790681336 is queued on ubuntu-latest, no runner assigned and pending_deployments empty. PyPI version0.1.46 JSON still404; publication is not yet verified. Build log /tmp/release046-pypi-build.log.

PyPI workflow36978827799 completed successfully at reviewed main75cc3eed14. Published PyPI0.1.46 metadata requires Python>=3.12; wheel and sdist not yanked. Downloaded artifacts verified against official SHA256: wheel6e1f49a89461d093b9bb42a08cbef50cdbb8497098cd3d5cc605cfdade2b13b3; sdist3940be2404e28a168a945a03b0530948755215c7cc56defa7a95552e4a4698e7. Existing check_pypi_artifacts.py passes on published archives; main entrypoint and AuthNZ initializer/Sync relay source bytes in both archives match released main75cc exactly. All three formal GHCR outputs previously verified. Artifact acceptance criterion complete. Latest remote dev remains3caebcfc; normal synchronization PR is next.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
