# Claims Stage2C Publication

Tracking: TASK-15465. User authorized rebase, verification and PR creation;
merging is not authorized. Work only in the existing isolated Claims worktree.
ADR check: no new durable rule; ADR094 (resource admission/publication),
ADR003/058/082 govern. SDK ADR095 remains Proposed. Renumbering preserves
accepted rationale and other owners' task/ADR records.

## Stage 1: Integrate Current Dev
**Goal**: Rebase the completed branch and repair private namespace collisions.
**Success Criteria**: Current fetched dev is an ancestor; both resource and
enrolled-workflow guards retained, unique canonical tasks/ADRs.
**Tests**: Conflict inspection, task/ADR audit, compile and ID guard.
**Status**: Complete

## Stage 2: Fresh Qualification
**Goal**: Qualify Claims/Jobs and affected storage/API/config paths on the rebase.
**Success Criteria**: Official backend fixtures, scoped regressions, full-file
configured lint comparison, Bandit and applicable hooks recorded honestly.
**Tests**: Claims/Jobs; publication/snapshot/existing-only Media; Watchlists;
Config/router; Backlog; Chroma; full repository attempt with default gates.
**Status**: Complete

Fresh qualification: initial full Claims/Jobs 5608 passed, 46 failed, 3 legacy
skips; all 46 failures plus a forced-high-FD regression subsequently passed
(47 focused; 1238 affected, zero skips). Storage/MediaPG/Watchlists/Config/
Chroma selection: 710 passed, 1 legacy skip, 15 xfailed, 2 xpassed. Backlog:
179 passed. Final compile100 passed; configured Ruff113 versus115dev, no new;
Bandit57 production files one identical inherited B608, zero errors/new.
Both independent integration reviews found no actionable findings. All
applicable full PR-scoped hooks passed on125 paths. The single root attempt
stopped after840 passed/1Audit failure/12collection skips; isolated case passed,
order/config interaction not diagnosed. No full-suite green claim.

## Stage 3: Publish The PR
**Goal**: Push the verified feature branch and create a PR against dev.
**Success Criteria**: Exact published head matches; no merge, gate bypass,
fabricated human Change summary or unrelated main-checkout change.
**Tests**: GitHub PR head/base/state verification, clean local worktree.
**Status**: In Progress

Publication will be Draft: the human-written Change summary remains missing,
root qualification is incomplete, and published-head CI has not run yet.
Verification and namespace evidence are recorded in TASK-15465. No merge or
auto-merge is authorized.
